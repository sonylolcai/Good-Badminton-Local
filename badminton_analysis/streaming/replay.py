"""Bounded-memory media adapters for segment replay and decoded uploads."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator, Optional

import cv2

from .models import FramePacket, FrameSegment, SegmentDescriptor


class ReplayDecodeError(RuntimeError):
    """Raised when OpenCV cannot open or fully decode the requested media."""


class ReplayUsageError(RuntimeError):
    """Raised when a lazy replay segment is not consumed before requesting the next."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class OpenCVSegmentDecoder:
    """Decode one independently playable uploaded segment lazily.

    The transport/storage layer must verify the descriptor SHA-256 against the
    uploaded bytes before calling this adapter.  Frames receive absolute source
    timestamps derived from the descriptor, never from upload wall-clock time.
    """

    def decode(self, segment_path, descriptor: SegmentDescriptor) -> FrameSegment:
        path = Path(segment_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        def frames() -> Iterator[FramePacket]:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ReplayDecodeError(f"unable to open segment: {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                capture.release()
                raise ReplayDecodeError(f"unable to read segment FPS: {path}")
            source_frame_start = (
                descriptor.source_frame_start_index
                if descriptor.source_frame_start_index is not None
                else int(round(descriptor.source_start_time_sec * fps))
            )
            local_index = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    yield FramePacket(
                        frame=frame,
                        source_frame_index=source_frame_start + local_index,
                        source_time_sec=descriptor.source_start_time_sec + local_index / fps,
                        segment_index=descriptor.segment_index,
                    )
                    local_index += 1
                if (
                    descriptor.source_frame_count is not None
                    and local_index != descriptor.source_frame_count
                ):
                    raise ReplayDecodeError(
                        f"decoded {local_index} frames, expected {descriptor.source_frame_count}"
                    )
            finally:
                capture.release()

        return FrameSegment(descriptor=descriptor, frames=frames())


class FileReplayAdapter:
    """Replay a complete local video as lazy 1–2 second analysis segments.

    This adapter is an analysis test source, not an fMP4 transport segmenter.
    It keeps only one decoded frame in memory at a time and derives a stable
    virtual digest from the source-file digest plus frame range.  Task C owns
    production FFmpeg/fMP4 creation and supplies the real byte digest.
    """

    def __init__(self, video_path, *, segment_duration_sec: float = 2.0) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(self.video_path)
        if not 0 < float(segment_duration_sec) <= 10:
            raise ValueError("segment_duration_sec must be greater than 0 and no more than 10")
        self.segment_duration_sec = float(segment_duration_sec)
        self._capture: Optional[cv2.VideoCapture] = None
        self._fps = 0.0
        self._total_frames = 0
        self._next_frame_index = 0
        self._segment_open = False
        self._source_digest: Optional[str] = None

    def __enter__(self) -> "FileReplayAdapter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def fps(self) -> float:
        self._ensure_open()
        return self._fps

    @property
    def total_frames(self) -> int:
        self._ensure_open()
        return self._total_frames

    def open(self) -> None:
        if self._capture is not None:
            return
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise ReplayDecodeError(f"unable to open replay video: {self.video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            capture.release()
            raise ReplayDecodeError(f"invalid FPS/frame count for replay video: {self.video_path}")
        self._capture = capture
        self._fps = fps
        self._total_frames = total_frames
        self._next_frame_index = 0
        self._segment_open = False
        self._source_digest = _file_sha256(self.video_path)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._segment_open = False

    def __iter__(self) -> Iterator[FrameSegment]:
        self._ensure_open()
        segment_index = 0
        frames_per_segment = max(1, int(round(self.segment_duration_sec * self._fps)))
        while self._next_frame_index < self._total_frames:
            if self._segment_open:
                raise ReplayUsageError("consume the current replay segment before requesting another")
            start_frame = self._next_frame_index
            end_frame = min(self._total_frames, start_frame + frames_per_segment)
            frame_count = end_frame - start_frame
            source_start_time_sec = start_frame / self._fps
            duration_sec = frame_count / self._fps
            digest_seed = f"replay:{self._source_digest}:{start_frame}:{end_frame}"
            digest = hashlib.sha256(digest_seed.encode("utf-8")).hexdigest()
            estimated_bytes = max(
                1,
                int(self.video_path.stat().st_size * frame_count / self._total_frames),
            )
            descriptor = SegmentDescriptor(
                segment_index=segment_index,
                source_start_time_sec=source_start_time_sec,
                duration_sec=duration_sec,
                sha256=digest,
                idempotency_key=f"replay-{self._source_digest[:16]}-{segment_index:06d}",
                content_type="video/mp4",
                content_length_bytes=estimated_bytes,
            )
            self._segment_open = True
            yield FrameSegment(
                descriptor=descriptor,
                frames=self._read_frame_range(segment_index, start_frame, end_frame),
            )
            if self._segment_open:
                raise ReplayUsageError("replay segment iterator was not fully consumed")
            segment_index += 1

    def _read_frame_range(
        self,
        segment_index: int,
        start_frame: int,
        end_frame: int,
    ) -> Iterator[FramePacket]:
        capture = self._capture
        if capture is None:
            raise ReplayUsageError("replay adapter is closed")
        try:
            if self._next_frame_index != start_frame:
                raise ReplayUsageError("replay source is not at the expected frame boundary")
            for frame_index in range(start_frame, end_frame):
                ok, frame = capture.read()
                if not ok:
                    raise ReplayDecodeError(
                        f"decode stopped at frame {frame_index}, expected {end_frame} frames"
                    )
                self._next_frame_index = frame_index + 1
                yield FramePacket(
                    frame=frame,
                    source_frame_index=frame_index,
                    source_time_sec=frame_index / self._fps,
                    segment_index=segment_index,
                )
        finally:
            self._segment_open = False

    def _ensure_open(self) -> None:
        if self._capture is None:
            raise ReplayUsageError("use FileReplayAdapter as a context manager or call open()")
