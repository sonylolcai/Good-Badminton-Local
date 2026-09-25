"""FFmpeg-backed creation of independently decodable short video segments."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional


@dataclass(frozen=True)
class SegmentArtifact:
    path: Path
    segment_index: int
    source_start_time_sec: float
    duration_sec: float
    content_type: str = "video/mp4"
    source_frame_start_index: int | None = None
    source_frame_count: int | None = None


class GrowingVideoSegmenter:
    """Convert a file/live input into finalized MP4 fragments as they appear.

    FFmpeg's segment muxer closes each fragment before the next fragment becomes
    visible to the caller.  Consequently this class never slices arbitrary MP4
    bytes and never publishes the file that FFmpeg is still writing.

    ``encoding_mode='copy'`` preserves the source codec and is preferred for
    cameras with a suitable 1–2 second GOP.  ``encoding_mode='h264'`` is an
    explicit compatibility fallback that inserts aligned keyframes and records
    the fact in the generated manifest.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        segment_duration_sec: float = 2.0,
        encoding_mode: str = "copy",
        preserve_audio: bool = False,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
        poll_interval_sec: float = 0.1,
        sleep_fn: Callable[[float], None] = time.sleep,
        declare_frame_sequence: bool = False,
    ) -> None:
        if not 0.5 <= float(segment_duration_sec) <= 10:
            raise ValueError("segment_duration_sec must be between 0.5 and 10 seconds")
        if encoding_mode not in {"copy", "h264"}:
            raise ValueError("encoding_mode must be copy or h264")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_duration_sec = float(segment_duration_sec)
        self.encoding_mode = encoding_mode
        self.preserve_audio = bool(preserve_audio)
        self.ffmpeg_path = ffmpeg_path or _find_ffmpeg()
        self.ffprobe_path = ffprobe_path or _find_ffprobe(self.ffmpeg_path)
        self.poll_interval_sec = max(0.01, float(poll_interval_sec))
        self.sleep_fn = sleep_fn
        self.declare_frame_sequence = bool(declare_frame_sequence)

    def segment_file(self, source: Path, *, overwrite: bool = False) -> list[SegmentArtifact]:
        """Segment a complete local file using the same incremental path as live input."""

        return list(self.iter_input(str(Path(source)), overwrite=overwrite, realtime=False))

    def iter_input(
        self,
        input_uri: str,
        *,
        overwrite: bool = False,
        realtime: bool = False,
        input_args: Optional[list[str]] = None,
    ) -> Iterator[SegmentArtifact]:
        """Yield each fragment once FFmpeg has closed it.

        ``input_uri`` may be a local file, RTSP URL or another FFmpeg-supported
        source.  ``realtime=True`` rate-limits local-file replay so tests mimic a
        growing camera stream instead of producing all segments immediately.
        """

        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg is required for stream segmentation")
        pattern = self.output_dir / "segment_%06d.mp4"
        manifest_path = self.output_dir / "segment_manifest.json"
        existing = sorted(self.output_dir.glob("segment_*.mp4"))
        if existing and not overwrite:
            if manifest_path.is_file():
                yield from self._read_manifest(manifest_path, input_uri)
                return
            raise RuntimeError(
                "segment output contains an unfinished prior run; retain it for ledger recovery "
                "or rerun explicitly with overwrite=True"
            )
        if overwrite:
            for path in existing:
                path.unlink()
            if manifest_path.exists():
                manifest_path.unlink()

        command = self._command(input_uri, pattern, realtime=realtime, input_args=input_args)
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        yielded: set[Path] = set()
        artifacts: list[SegmentArtifact] = []
        source_start = 0.0
        source_frame_start = 0
        try:
            while True:
                return_code = process.poll()
                paths = sorted(self.output_dir.glob("segment_*.mp4"))
                # While FFmpeg is running, the newest file may still be open.
                finalized = paths if return_code is not None else paths[:-1]
                for path in finalized:
                    resolved = path.resolve()
                    if resolved in yielded:
                        continue
                    duration = self._duration(path)
                    if not 0 < duration <= 10:
                        raise RuntimeError(
                            f"segment {path.name} duration {duration:.3f}s violates stream-session.v1; "
                            "use encoding_mode='h264' or configure a shorter camera GOP"
                        )
                    frame_count = self._count_decoded_frames(path) if self.declare_frame_sequence else None
                    if frame_count is None:
                        self._assert_decodable(path)
                    artifact = SegmentArtifact(
                        path=path.resolve(),
                        segment_index=len(artifacts),
                        source_start_time_sec=round(source_start, 6),
                        duration_sec=round(duration, 6),
                        source_frame_start_index=(source_frame_start if frame_count is not None else None),
                        source_frame_count=frame_count,
                    )
                    artifacts.append(artifact)
                    yielded.add(resolved)
                    source_start += duration
                    if frame_count is not None:
                        source_frame_start += frame_count
                    yield artifact
                if return_code is not None:
                    stderr = process.stderr.read() if process.stderr else ""
                    if return_code != 0:
                        raise RuntimeError(
                            f"FFmpeg segmentation failed with exit code {return_code}: {stderr[-2000:]}"
                        )
                    break
                self.sleep_fn(self.poll_interval_sec)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            if process.stderr:
                process.stderr.close()

        if not artifacts:
            raise RuntimeError("FFmpeg produced no decodable video segments")
        self._write_manifest(manifest_path, input_uri, command, artifacts)

    def _command(
        self,
        input_uri: str,
        pattern: Path,
        *,
        realtime: bool,
        input_args: Optional[list[str]],
    ) -> list[str]:
        command = [self.ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        command.extend(input_args or [])
        if realtime:
            command.append("-re")
        command.extend(["-i", input_uri, "-map", "0:v:0"])
        if self.preserve_audio:
            command.extend(["-map", "0:a?", "-c:a", "copy"])
        else:
            command.append("-an")
        if self.encoding_mode == "copy":
            command.extend(["-c:v", "copy"])
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-force_key_frames",
                    f"expr:gte(t,n_forced*{self.segment_duration_sec:g})",
                ]
            )
            if self.declare_frame_sequence:
                command.extend(["-fps_mode", "passthrough"])
        command.extend(
            [
                "-f",
                "segment",
                "-segment_time",
                f"{self.segment_duration_sec:g}",
                "-reset_timestamps",
                "1",
                "-segment_format",
                "mp4",
                "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof",
                str(pattern),
            ]
        )
        return command

    def _duration(self, path: Path) -> float:
        if self.ffprobe_path:
            command = [
                self.ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode == 0:
                try:
                    duration = float(completed.stdout.strip())
                    if duration > 0:
                        return duration
                except ValueError:
                    pass
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps <= 0 or frames <= 0:
                raise RuntimeError(f"cannot determine duration for {path}")
            return frames / fps
        finally:
            capture.release()

    @staticmethod
    def _assert_decodable(path: Path) -> None:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if not capture.isOpened() or not ok or frame is None:
                raise RuntimeError(f"segment is not independently decodable: {path}")
        finally:
            capture.release()

    @staticmethod
    def _count_decoded_frames(path: Path) -> int:
        import cv2

        capture = cv2.VideoCapture(str(path))
        count = 0
        try:
            if not capture.isOpened():
                raise RuntimeError(f"segment is not independently decodable: {path}")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None:
                    raise RuntimeError(f"segment contains an invalid decoded frame: {path}")
                count += 1
            if count == 0:
                raise RuntimeError(f"segment is not independently decodable: {path}")
            return count
        finally:
            capture.release()

    def _write_manifest(
        self,
        path: Path,
        input_uri: str,
        command: list[str],
        artifacts: list[SegmentArtifact],
    ) -> None:
        payload = {
            "schema_version": "business-segment-manifest.v1",
            "input_uri": input_uri,
            "source_fingerprint": _source_fingerprint(input_uri),
            "segment_duration_target_sec": self.segment_duration_sec,
            "encoding_mode": self.encoding_mode,
            "preserve_audio": self.preserve_audio,
            "declare_frame_sequence": self.declare_frame_sequence,
            "ffmpeg_command": command,
            "segments": [
                {
                    **asdict(artifact),
                    "path": str(artifact.path),
                }
                for artifact in artifacts
            ],
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _read_manifest(self, path: Path, input_uri: str) -> Iterator[SegmentArtifact]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "business-segment-manifest.v1":
            raise ValueError("unsupported segment manifest version")
        expected = {
            "input_uri": input_uri,
            "source_fingerprint": _source_fingerprint(input_uri),
            "segment_duration_target_sec": self.segment_duration_sec,
            "encoding_mode": self.encoding_mode,
            "preserve_audio": self.preserve_audio,
            "declare_frame_sequence": self.declare_frame_sequence,
        }
        actual = {
            key: bool(payload.get(key, False)) if key == "declare_frame_sequence" else payload.get(key)
            for key in expected
        }
        if actual != expected:
            raise RuntimeError(
                "completed segment output belongs to a different source or configuration; "
                "use a separate work directory or overwrite=True"
            )
        for item in payload.get("segments", []):
            segment_path = Path(item["path"])
            if not segment_path.is_file():
                raise FileNotFoundError(segment_path)
            yield SegmentArtifact(
                path=segment_path,
                segment_index=int(item["segment_index"]),
                source_start_time_sec=float(item["source_start_time_sec"]),
                duration_sec=float(item["duration_sec"]),
                content_type=str(item.get("content_type") or "video/mp4"),
                source_frame_start_index=item.get("source_frame_start_index"),
                source_frame_count=item.get("source_frame_count"),
            )


def _find_ffmpeg() -> Optional[str]:
    configured = os.environ.get("GOOD_BADMINTON_FFMPEG", "").strip()
    if configured:
        return configured
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _find_ffprobe(ffmpeg_path: Optional[str]) -> Optional[str]:
    configured = os.environ.get("GOOD_BADMINTON_FFPROBE", "").strip()
    if configured:
        return configured
    executable = shutil.which("ffprobe")
    if executable:
        return executable
    if ffmpeg_path:
        candidate = Path(ffmpeg_path).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if candidate.is_file():
            return str(candidate)
    return None


def _source_fingerprint(input_uri: str) -> Optional[dict]:
    """Cheap replay guard without hashing an entire long recording up front."""

    path = Path(input_uri)
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
