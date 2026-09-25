"""Bounded, checkpointable TrackNet observations for ordered video segments.

The model sees the most recent eight *source* frames on every inference.  A
window can cross an upload boundary without reusing or dropping a frame.  This
costs more inference than the offline non-overlap runner, but it lets each
observation retain its own source timestamp and preserves event ordering.
"""

from __future__ import annotations

import base64
import math
from collections import deque
from typing import Any

import cv2
import numpy as np

from badminton_analysis.streaming.models import ProcessorEvent


STATE_VERSION = "tracknet-stream.v1"
WINDOW_FRAMES = 8
MODEL_WIDTH = 512
MODEL_HEIGHT = 288


class TrackNetStreamProcessor:
    """Emit only measured ball evidence; keep image context across segments."""

    def __init__(
        self, detector: Any, *, model_sha256: str,
        model_checkpoint: str = "TrackNet_best.pt", max_frame_interval_ratio: float = 1.5,
    ):
        if len(str(model_sha256)) != 64 or any(char not in "0123456789abcdef" for char in str(model_sha256)):
            raise ValueError("TrackNet model SHA-256 must be a lowercase digest")
        if int(detector.seq_len) != WINDOW_FRAMES:
            raise ValueError("TrackNet streaming requires an eight-frame model")
        if not math.isfinite(float(max_frame_interval_ratio)) or float(max_frame_interval_ratio) < 1.0:
            raise ValueError("max_frame_interval_ratio must be finite and at least 1")
        self.detector = detector
        self.model_sha256 = str(model_sha256)
        self.model_checkpoint = str(model_checkpoint)
        self.max_frame_interval_ratio = float(max_frame_interval_ratio)
        self.frames: deque[np.ndarray] = deque(maxlen=WINDOW_FRAMES)
        self.background: np.ndarray | None = None
        self.last_frame_index: int | None = None
        self.last_source_time_sec: float | None = None
        self.frame_interval_sec: float | None = None
        self.frame_shape: tuple[int, int] | None = None

    @staticmethod
    def _resized_rgb(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("TrackNet requires a three-channel decoded source frame")
        return cv2.cvtColor(
            cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )

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
        self.frames.clear()
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

    def _infer(self, shape: tuple[int, int], measurement_bucket: int) -> ProcessorEvent:
        import torch

        if self.background is None:
            self.background = np.median(np.stack(self.frames), axis=0).astype(np.uint8)
        channels = [self.background, *self.frames]
        image = np.concatenate(channels, axis=2)
        tensor = torch.from_numpy(np.moveaxis(image, -1, 0).copy()).float().div_(255.0)
        with torch.no_grad():
            heatmap = self.detector.model(tensor.unsqueeze(0).to(self.detector.device))
        heatmap_np = heatmap[0, WINDOW_FRAMES - 1].detach().cpu().numpy()
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
            data={
                "detector": "tracknet_v3",
                "model_identity": {
                    "sport_id": "badminton",
                    "model_kind": "ball",
                    "model_checkpoint": self.model_checkpoint,
                    "model_sha256": self.model_sha256,
                },
                "measurement": measurement,
                "source_frame_index": self.last_frame_index,
                "measurement_bucket": measurement_bucket,
                "temporal_window_frames": WINDOW_FRAMES,
            },
        )

    def process_frame(self, frame: np.ndarray, context: Any) -> list[ProcessorEvent]:
        frame_index = int(context.source_frame_index)
        source_time_sec = float(context.source_time_sec)
        shape = tuple(int(value) for value in frame.shape[:2])
        gap = self._detect_gap(frame_index, source_time_sec, shape)
        events: list[ProcessorEvent] = []
        if gap is not None:
            self._reset_window()
            events.append(ProcessorEvent(
                event_type="session_status",
                evidence_state="derived",
                confidence=1.0,
                data={"ball_continuity": "unknown", "reason": gap,
                      "source_frame_index": frame_index},
            ))
        self.frames.append(self._resized_rgb(frame))
        self.last_frame_index = frame_index
        self.last_source_time_sec = source_time_sec
        self.frame_shape = shape
        if len(self.frames) == WINDOW_FRAMES:
            events.append(self._infer(shape, int(context.measurement_bucket)))
        return events

    def finalize(self, _context: Any) -> list[ProcessorEvent]:
        return []

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "model_sha256": self.model_sha256,
            "model_checkpoint": self.model_checkpoint,
            "max_frame_interval_ratio": self.max_frame_interval_ratio,
            "frames_png": [self._encode_image(frame) for frame in self.frames],
            "background_png": self._encode_image(self.background) if self.background is not None else None,
            "last_frame_index": self.last_frame_index,
            "last_source_time_sec": self.last_source_time_sec,
            "frame_interval_sec": self.frame_interval_sec,
            "frame_shape": list(self.frame_shape) if self.frame_shape is not None else None,
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
        encoded_frames = state.get("frames_png")
        if not isinstance(encoded_frames, list) or len(encoded_frames) > WINDOW_FRAMES:
            raise ValueError("invalid TrackNet checkpoint window")
        frames = [self._decode_image(value) for value in encoded_frames]
        background = state.get("background_png")
        self.frames = deque(frames, maxlen=WINDOW_FRAMES)
        self.background = self._decode_image(background) if background is not None else None
        self.last_frame_index = (
            int(state["last_frame_index"]) if state.get("last_frame_index") is not None else None
        )
        self.last_source_time_sec = (
            float(state["last_source_time_sec"])
            if state.get("last_source_time_sec") is not None else None
        )
        self.frame_interval_sec = (
            float(state["frame_interval_sec"])
            if state.get("frame_interval_sec") is not None else None
        )
        shape = state.get("frame_shape")
        if shape is not None and (not isinstance(shape, list) or len(shape) != 2):
            raise ValueError("invalid TrackNet checkpoint source shape")
        self.frame_shape = tuple(int(value) for value in shape) if shape is not None else None
        if (self.last_frame_index is None) != (self.last_source_time_sec is None):
            raise ValueError("incomplete TrackNet checkpoint cursor")
