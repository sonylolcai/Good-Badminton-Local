"""A small, explicit adapter around Ultralytics' official BYTETracker.

The pose detector owns person detection.  ByteTrack only associates those
person boxes over time and never invents a pose, a court position, a team, or
an identity.  Keeping that boundary explicit makes it possible to fall back to
the deterministic court-space association during evaluation without changing
the output schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import os
from types import SimpleNamespace
import tempfile
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class _DetectionBatch:
    """Minimal Results-like input consumed by Ultralytics BYTETracker."""

    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, index: Any) -> "_DetectionBatch":
        return _DetectionBatch(self.xyxy[index], self.conf[index], self.cls[index])

    @property
    def xywh(self) -> np.ndarray:
        """Return centre-x, centre-y, width, height for ByteTrack parsing."""
        if len(self.xyxy) == 0:
            return np.empty((0, 4), dtype=np.float32)
        left, top, right, bottom = self.xyxy.T
        return np.column_stack(((left + right) / 2.0, (top + bottom) / 2.0, right - left, bottom - top))


class ByteTrackAdapter:
    """Associate pose boxes with official ByteTrack IDs.

    ``update`` returns a mapping from the original pose-detection index to the
    association key.  The caller remains responsible for court-space tracking,
    confidence degradation and the public ``track_id`` contract.
    """

    BACKEND_NAME = "bytetrack"

    @staticmethod
    def _prepare_ultralytics_runtime():
        """Avoid Ultralytics writing its user config into a protected profile."""
        config_dir = os.path.join(tempfile.gettempdir(), "good-badminton-ultralytics")
        os.makedirs(config_dir, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", config_dir)

    def __init__(
        self,
        fps: float,
        track_buffer_frames: int = 30,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.30,
        # Far-court badminton poses change scale and overlap quickly between
        # 10 Hz samples. The generic 0.80 IoU gate fragments a player even
        # when the pose detector keeps seeing them; 0.50 retains that temporal
        # continuity while the fixed-court roster remains the second guard.
        match_thresh: float = 0.50,
        tracker_factory: Optional[Callable[[Any], Any]] = None,
    ):
        self.fps = float(fps)
        self.config = {
            "track_buffer": int(track_buffer_frames),
            "track_high_thresh": float(track_high_thresh),
            "track_low_thresh": float(track_low_thresh),
            "new_track_thresh": float(new_track_thresh),
            "match_thresh": float(match_thresh),
            "fuse_score": True,
        }
        args = SimpleNamespace(**self.config)
        if tracker_factory is None:
            if find_spec("lap") is None:
                raise RuntimeError(
                    "ByteTrack requires the optional 'lap>=0.5.12' package. "
                    "Install project requirements before enabling it."
                )
            self._prepare_ultralytics_runtime()
            try:
                from ultralytics.trackers.byte_tracker import BYTETracker
            except ModuleNotFoundError as exc:
                if exc.name == "lap":
                    raise RuntimeError(
                        "ByteTrack requires the optional 'lap>=0.5.12' package. "
                        "Install project requirements before enabling it."
                    ) from exc
                raise RuntimeError("Ultralytics ByteTrack is unavailable.") from exc
            tracker_factory = BYTETracker
        self._tracker = tracker_factory(args)

    @classmethod
    def availability(cls) -> tuple[bool, Optional[str]]:
        """Return whether the runtime has both Ultralytics and its LAP solver."""
        if find_spec("lap") is None:
            return False, "missing dependency: lap"
        cls._prepare_ultralytics_runtime()
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: F401
        except ModuleNotFoundError as exc:
            return False, f"missing dependency: {exc.name}"
        except Exception as exc:  # pragma: no cover - defensive runtime probe
            return False, str(exc)
        return True, None

    def update(self, observations: list[dict]) -> dict[int, str]:
        """Return ``pose_detection_index -> bytetrack association key``.

        ByteTrack only emits *confirmed* tracks.  An observation without a
        returned key is deliberately left to the court tracker as an unhinted
        detection; it is not assigned a guessed ByteTrack ID.
        """
        valid = []
        for original_index, observation in enumerate(observations):
            bbox = observation.get("bbox_xyxy") or observation.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
            if not (x2 > x1 and y2 > y1):
                continue
            valid.append((original_index, [x1, y1, x2, y2], float(observation.get("confidence") or 0.0)))

        if not valid:
            self._tracker.update(_DetectionBatch(
                xyxy=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.float32),
            ))
            return {}

        batch = _DetectionBatch(
            xyxy=np.asarray([item[1] for item in valid], dtype=np.float32),
            conf=np.asarray([item[2] for item in valid], dtype=np.float32),
            cls=np.zeros((len(valid),), dtype=np.float32),
        )
        tracks = np.asarray(self._tracker.update(batch), dtype=np.float32)
        if tracks.size == 0:
            return {}
        tracks = tracks.reshape((-1, tracks.shape[-1]))
        index_to_key = {}
        for track in tracks:
            # Ultralytics output: x1, y1, x2, y2, tracker_id, score, cls, input_idx.
            if len(track) < 8:
                continue
            batch_index = int(track[7])
            if 0 <= batch_index < len(valid):
                index_to_key[valid[batch_index][0]] = f"bytetrack_{int(track[4])}"
        return index_to_key
