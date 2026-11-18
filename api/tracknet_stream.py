"""Bounded, checkpointable TrackNet observations for ordered video segments.

Uses non-overlapping 8-frame chunk inference (``sliding_step=8``) with bounded
batched execution.  At the end of every segment, any trailing ``1..7`` frames
are settled within the segment boundary (using preceding context frames when
available, or tail-padding for an initial short span) so shuttle events never
lag behind an already-published event cursor.
"""

from __future__ import annotations

import base64
import math
from collections import deque
from typing import Any

import cv2
import numpy as np

from badminton_analysis.streaming.models import ProcessorEvent


STATE_VERSION = "tracknet-stream.v2"
WINDOW_FRAMES = 8
WINDOW_STEP = 8
MAX_BATCH_CHUNKS = 8
MODEL_WIDTH = 512
MODEL_HEIGHT = 288


class TrackNetStreamProcessor:
    """Emit only measured ball evidence using batched 8-frame chunk inference."""

    def __init__(
        self,
        detector: Any,
        *,
        model_sha256: str,
        model_checkpoint: str = "TrackNet_best.pt",
        max_frame_interval_ratio: float = 1.5,
        max_batch_chunks: int = MAX_BATCH_CHUNKS,
    ):
        if len(str(model_sha256)) != 64 or any(
            char not in "0123456789abcdef" for char in str(model_sha256)
        ):
            raise ValueError("TrackNet model SHA-256 must be a lowercase digest")
        if int(detector.seq_len) != WINDOW_FRAMES:
            raise ValueError("TrackNet streaming requires an eight-frame model")
        if not math.isfinite(float(max_frame_interval_ratio)) or float(max_frame_interval_ratio) < 1.0:
            raise ValueError("max_frame_interval_ratio must be finite and at least 1")
        if int(max_batch_chunks) < 1:
            raise ValueError("max_batch_chunks must be at least 1")
        self.detector = detector
        self.model_sha256 = str(model_sha256)
        self.model_checkpoint = str(model_checkpoint)
        self.max_frame_interval_ratio = float(max_frame_interval_ratio)
        self.max_batch_chunks = int(max_batch_chunks)
        self.context_frames: deque[np.ndarray] = deque(maxlen=WINDOW_FRAMES - 1)
        self.pending_entries: list[dict[str, Any]] = []
        self.background: np.ndarray | None = None
        self.last_frame_index: int | None = None
        self.last_source_time_sec: float | None = None
        self.frame_interval_sec: float | None = None
        self.frame_shape: tuple[int, int] | None = None
        self.last_segment_index: int | None = None
        self.last_source_frame_identity_declared = False

    @staticmethod
    def _resized_rgb(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("TrackNet requires a three-channel decoded source frame")
        return cv2.cvtColor(
            cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )

    @classmethod
    def preprocess_frame(cls, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Downscale and convert to RGB once so callers need not retain raw frames."""
        return cls._resized_rgb(frame), (int(frame.shape[0]), int(frame.shape[1]))

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("unable to encode TrackNet checkpoint frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _decode_image(value: str) -> np.ndarray:
        raw = base64.b64decode(value, validate=True)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (MODEL_HEIGHT, MODEL_WIDTH):
            raise ValueError("invalid TrackNet checkpoint frame")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _reset_window(self) -> None:
        self.context_frames.clear()
        self.pending_entries.clear()
        self.background = None
        self.frame_interval_sec = None

    def _detect_gap(self, frame_index: int, source_time_sec: float, shape: tuple[int, int]) -> str | None:
        if self.last_frame_index is None:
            return None
        if shape != self.frame_shape:
            return "source_resolution_changed"
        if frame_index != self.last_frame_index + 1:
            return "source_frame_index_gap"
        elapsed = source_time_sec - float(self.last_source_time_sec)
        if elapsed <= 0 or not math.isfinite(elapsed):
            return "source_time_not_increasing"
        if self.frame_interval_sec is not None and not (
            1.0 / self.max_frame_interval_ratio
            <= elapsed / self.frame_interval_sec
            <= self.max_frame_interval_ratio
        ):
            return "source_timestamp_gap"
        if self.frame_interval_sec is None:
            self.frame_interval_sec = elapsed
        return None

    @staticmethod
    def _context_dict(context: Any, shape: tuple[int, int]) -> dict[str, Any]:
        return {
            "source_frame_index": int(context.source_frame_index),
            "source_time_sec": float(context.source_time_sec),
            "segment_index": int(context.segment_index),
            "measurement_bucket": int(context.measurement_bucket),
            "shape": [int(shape[0]), int(shape[1])],
        }

    def _build_event(
        self,
        heatmap_np: np.ndarray,
        entry: dict[str, Any],
    ) -> ProcessorEvent:
        shape = entry["shape"]
        visible, x, y, peak = self.detector._extract_coordinate_from_heatmap(
            heatmap_np,
            (shape[1] / MODEL_WIDTH, shape[0] / MODEL_HEIGHT),
        )
        measurement = {
            "status": "detected" if visible else "missing",
            "visible": bool(visible),
            "accepted": bool(visible),
            "image": [float(x), float(y)] if visible else None,
            "source": "tracknet_v3_stream",
            "measurement_kind": "temporal_heatmap",
            "confidence": float(peak) if visible else None,
            "confidence_status": "uncalibrated_heatmap_peak",
            "heatmap_peak": float(peak),
        }
        return ProcessorEvent(
            event_type="shuttle_observation",
            evidence_state="detected" if visible else "missing",
            confidence=max(0.0, min(1.0, float(peak))) if visible else 0.0,
            source_time_sec=float(entry["source_time_sec"]),
            segment_index=int(entry["segment_index"]),
            data={
                "detector": "tracknet_v3",
                "model_identity": {
                    "sport_id": "badminton",
                    "model_kind": "ball",
                    "model_checkpoint": self.model_checkpoint,
                    "model_sha256": self.model_sha256,
                },
                "measurement": measurement,
                "source_frame_index": int(entry["source_frame_index"]),
                "measurement_bucket": int(entry["measurement_bucket"]),
                "temporal_window_frames": WINDOW_FRAMES,
                "window_step": WINDOW_STEP,
            },
        )

    def _run_chunks(
        self,
        chunks: list[tuple[list[np.ndarray], list[tuple[int, dict[str, Any]]]]],
    ) -> list[tuple[dict[str, Any], ProcessorEvent]]:
        import torch

        if not chunks:
            return []
        if self.background is None:
            self.background = np.median(np.stack(chunks[0][0]), axis=0).astype(np.uint8)

        tensor_cache: dict[int, torch.Tensor] = {}

        def to_chw_tensor(image: np.ndarray) -> torch.Tensor:
            cached = tensor_cache.get(id(image))
            if cached is None:
                cached = torch.from_numpy(np.moveaxis(image, -1, 0).copy()).float().div_(255.0)
                tensor_cache[id(image)] = cached
            return cached

        bg_tensor = to_chw_tensor(self.background)
        emitted: list[tuple[dict[str, Any], ProcessorEvent]] = []
        for start in range(0, len(chunks), self.max_batch_chunks):
            chunk_slice = chunks[start:start + self.max_batch_chunks]
            batch_tensors = [
                torch.cat([bg_tensor, *(to_chw_tensor(img) for img in window_images)], dim=0)
                for window_images, _ in chunk_slice
            ]
            batch_x = torch.stack(batch_tensors, dim=0).to(self.detector.device)
            with torch.no_grad():
                pred_heatmaps = self.detector.model(batch_x).detach().cpu().numpy()

            for b_idx, (_, targets) in enumerate(chunk_slice):
                for slot_idx, entry in targets:
                    emitted.append((entry, self._build_event(pred_heatmaps[b_idx, slot_idx], entry)))
        return emitted

    def _flush_ready_chunks(
        self,
        *,
        flush_boundary: bool,
        pad_short_tail: bool = False,
    ) -> list[tuple[dict[str, Any], ProcessorEvent]]:
        chunks: list[tuple[list[np.ndarray], list[tuple[int, dict[str, Any]]]]] = []
        while len(self.pending_entries) >= WINDOW_FRAMES:
            group = self.pending_entries[:WINDOW_FRAMES]
            del self.pending_entries[:WINDOW_FRAMES]
            window_images = [item["image"] for item in group]
            targets = [(slot, item["meta"]) for slot, item in enumerate(group)]
            chunks.append((window_images, targets))
            for img in window_images[-(WINDOW_FRAMES - 1):]:
                self.context_frames.append(img)

        if flush_boundary and self.pending_entries:
            needed = WINDOW_FRAMES - len(self.pending_entries)
            if len(self.context_frames) >= needed:
                left = list(self.context_frames)[-needed:]
                tail = self.pending_entries
                self.pending_entries = []
                window_images = [*left, *(item["image"] for item in tail)]
                targets = [(needed + offset, item["meta"]) for offset, item in enumerate(tail)]
                chunks.append((window_images, targets))
                for item in tail:
                    self.context_frames.append(item["image"])
            elif pad_short_tail:
                left = list(self.context_frames)
                tail = self.pending_entries
                self.pending_entries = []
                if self.background is None:
                    self.background = np.median(
                        np.stack([*left, *(item["image"] for item in tail)]), axis=0
                    ).astype(np.uint8)
                pad_needed = WINDOW_FRAMES - len(left) - len(tail)
                last_img = tail[-1]["image"]
                window_images = [*left, *(item["image"] for item in tail), *([last_img] * pad_needed)]
                targets = [
                    (len(left) + offset, item["meta"]) for offset, item in enumerate(tail)
                ]
                chunks.append((window_images, targets))
                for item in tail:
                    self.context_frames.append(item["image"])

        return self._run_chunks(chunks)

    def _ingest_frame(
        self, frame: Any, context: Any, *, allow_mid_flush: bool
    ) -> list[tuple[dict[str, Any] | None, ProcessorEvent]]:
        if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[1], tuple):
            resized_rgb, shape = frame
        else:
            shape = (int(frame.shape[0]), int(frame.shape[1]))
            resized_rgb = self._resized_rgb(frame)

        frame_index = int(context.source_frame_index)
        source_time_sec = float(context.source_time_sec)
        segment_index = int(context.segment_index)
        identity_declared = bool(getattr(context, "source_frame_identity_declared", False))
        crossed_segment = (
            self.last_frame_index is not None
            and (self.last_segment_index is None or segment_index != self.last_segment_index)
        )
        gap = (
            "source_frame_identity_unverified"
            if crossed_segment
            and not (identity_declared and self.last_source_frame_identity_declared)
            else self._detect_gap(frame_index, source_time_sec, shape)
        )
        emitted: list[tuple[dict[str, Any] | None, ProcessorEvent]] = []
        if gap is not None:
            if self.pending_entries:
                emitted.extend(
                    self._flush_ready_chunks(flush_boundary=True, pad_short_tail=True)
                )
            self._reset_window()
            emitted.append(
                (
                    None,
                    ProcessorEvent(
                        event_type="session_status",
                        evidence_state="derived",
                        confidence=1.0,
                        source_time_sec=source_time_sec,
                        segment_index=segment_index,
                        data={
                            "ball_continuity": "unknown",
                            "reason": gap,
                            "source_frame_index": frame_index,
                        },
                    ),
                )
            )

        self.pending_entries.append(
            {
                "image": resized_rgb,
                "meta": self._context_dict(context, shape),
            }
        )
        self.last_frame_index = frame_index
        self.last_source_time_sec = source_time_sec
        self.frame_shape = shape
        self.last_segment_index = segment_index
        self.last_source_frame_identity_declared = identity_declared
        if allow_mid_flush and len(self.pending_entries) >= WINDOW_FRAMES:
            emitted.extend(self._flush_ready_chunks(flush_boundary=False))
        return emitted

    def process_frame(self, frame: np.ndarray, context: Any) -> list[ProcessorEvent]:
        emitted = self._ingest_frame(frame, context, allow_mid_flush=True)
        return [event for _, event in emitted]

    def process_segment_frames(
        self,
        frames: list[Any],
        contexts: list[Any],
        *,
        flush_boundary: bool = True,
    ) -> list[tuple[Any, list[ProcessorEvent]]]:
        ordered_events: list[ProcessorEvent] = []
        for frame, context in zip(frames, contexts):
            for _, event in self._ingest_frame(frame, context, allow_mid_flush=False):
                ordered_events.append(event)
        for _, event in self._flush_ready_chunks(
            flush_boundary=flush_boundary,
            pad_short_tail=flush_boundary,
        ):
            ordered_events.append(event)
        if not contexts:
            return []
        return [(contexts[-1], ordered_events)]

    def flush_segment(self) -> list[ProcessorEvent]:
        """Settle any remaining frames within the current segment boundary."""
        return [
            event
            for _, event in self._flush_ready_chunks(flush_boundary=True, pad_short_tail=True)
        ]

    def finalize(self, _context: Any) -> list[ProcessorEvent]:
        return self.flush_segment()

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "model_sha256": self.model_sha256,
            "model_checkpoint": self.model_checkpoint,
            "max_frame_interval_ratio": self.max_frame_interval_ratio,
            "context_frames_png": [self._encode_image(frame) for frame in self.context_frames],
            "pending_entries": [
                {
                    "image_png": self._encode_image(item["image"]),
                    "meta": dict(item["meta"]),
                }
                for item in self.pending_entries
            ],
            "background_png": self._encode_image(self.background) if self.background is not None else None,
            "last_frame_index": self.last_frame_index,
            "last_source_time_sec": self.last_source_time_sec,
            "frame_interval_sec": self.frame_interval_sec,
            "frame_shape": list(self.frame_shape) if self.frame_shape is not None else None,
            "last_segment_index": self.last_segment_index,
            "last_source_frame_identity_declared": self.last_source_frame_identity_declared,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if state.get("state_version") != STATE_VERSION:
            raise ValueError("unsupported TrackNet stream checkpoint")
        if state.get("model_sha256") != self.model_sha256:
            raise ValueError("TrackNet model changed during session")
        if state.get("model_checkpoint") != self.model_checkpoint:
            raise ValueError("TrackNet model checkpoint identity changed during session")
        if float(state.get("max_frame_interval_ratio", 0.0)) != self.max_frame_interval_ratio:
            raise ValueError("TrackNet frame continuity configuration changed during session")

        encoded_context = state.get("context_frames_png")
        if not isinstance(encoded_context, list) or len(encoded_context) > WINDOW_FRAMES - 1:
            raise ValueError("invalid TrackNet checkpoint context window")
        encoded_pending = state.get("pending_entries")
        if not isinstance(encoded_pending, list) or len(encoded_pending) >= WINDOW_FRAMES:
            raise ValueError("invalid TrackNet checkpoint pending entries")
        self.context_frames = deque(
            [self._decode_image(value) for value in encoded_context],
            maxlen=WINDOW_FRAMES - 1,
        )
        self.pending_entries = [
            {
                "image": self._decode_image(item["image_png"]),
                "meta": dict(item["meta"]),
            }
            for item in encoded_pending
        ]

        background = state.get("background_png")
        self.background = self._decode_image(background) if background is not None else None
        self.last_frame_index = (
            int(state["last_frame_index"]) if state.get("last_frame_index") is not None else None
        )
        self.last_source_time_sec = (
            float(state["last_source_time_sec"])
            if state.get("last_source_time_sec") is not None
            else None
        )
        self.frame_interval_sec = (
            float(state["frame_interval_sec"])
            if state.get("frame_interval_sec") is not None
            else None
        )
        shape = state.get("frame_shape")
        if shape is not None and (not isinstance(shape, list) or len(shape) != 2):
            raise ValueError("invalid TrackNet checkpoint source shape")
        self.frame_shape = tuple(int(value) for value in shape) if shape is not None else None
        self.last_segment_index = (
            int(state["last_segment_index"])
            if state.get("last_segment_index") is not None
            else None
        )
        self.last_source_frame_identity_declared = bool(
            state.get("last_source_frame_identity_declared", False)
        )
        if (self.last_frame_index is None) != (self.last_source_time_sec is None):
            raise ValueError("incomplete TrackNet checkpoint cursor")
