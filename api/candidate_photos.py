"""Anonymous full-body candidate photos for post-match role claiming.

The GPU never identifies a person.  It only keeps one high-quality full-body
crop per anonymous ``track_id`` so the business service can later let a player
recognise themselves from a face, side view, or back view.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from badminton_analysis.streaming.models import FrameContext, ProcessorEvent


_TRACK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class CandidatePhotoCollector:
    """Persist the best anonymous player crop seen for each track.

    A crop is selected from the same pose observation that produced the public
    track event.  The collector stores no user identity, face embedding, or
    score; ``track_id`` remains anonymous until the business service accepts a
    later, user-confirmed claim.
    """

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self._best_quality: dict[str, float] = {}

    def attach(
        self,
        frame: Any,
        context: FrameContext,
        events: Iterable[ProcessorEvent],
    ) -> list[ProcessorEvent]:
        output: list[ProcessorEvent] = []
        for event in events:
            candidate = self._capture(frame, context, event)
            if candidate is None:
                output.append(event)
                continue
            data = deepcopy(dict(event.data))
            data["candidate_photo"] = candidate
            output.append(
                ProcessorEvent(
                    event_type=event.event_type,
                    evidence_state=event.evidence_state,
                    confidence=event.confidence,
                    data=data,
                    source_time_sec=event.source_time_sec,
                    segment_index=event.segment_index,
                )
            )
        return output

    def snapshot_state(self) -> dict[str, Any]:
        return {"state_version": "candidate-photo.v1", "best_quality": dict(self._best_quality)}

    def restore_state(self, state: dict[str, Any]) -> None:
        if state.get("state_version") != "candidate-photo.v1":
            raise ValueError("unsupported candidate photo checkpoint")
        self._best_quality = {
            str(track_id): float(quality)
            for track_id, quality in dict(state.get("best_quality") or {}).items()
            if _TRACK_ID_PATTERN.fullmatch(str(track_id)) and float(quality) >= 0.0
        }

    def _capture(self, frame: Any, context: FrameContext, event: ProcessorEvent) -> dict[str, Any] | None:
        if event.event_type != "person_observation" or event.evidence_state != "detected":
            return None
        track = (event.data or {}).get("track")
        if not isinstance(track, dict):
            return None
        track_id = str(track.get("track_id") or "")
        if not _TRACK_ID_PATTERN.fullmatch(track_id):
            return None
        bbox = ((track.get("location_evidence") or {}).get("bbox_xyxy"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        image = np.asarray(frame)
        if image.ndim < 2:
            return None
        height, width = image.shape[:2]
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return None
        pad_x = max(4, int(round((x2 - x1) * 0.12)))
        pad_y = max(4, int(round((y2 - y1) * 0.08)))
        left, top = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
        right, bottom = min(width, int(x2) + pad_x), min(height, int(y2) + pad_y)
        if right - left < 24 or bottom - top < 48:
            return None
        confidence = max(0.0, min(1.0, float(track.get("confidence") or 0.0)))
        size_factor = min(1.0, (bottom - top) / max(1.0, height * 0.28))
        quality = round(confidence * (0.65 + 0.35 * size_factor), 4)
        if quality <= self._best_quality.get(track_id, -1.0):
            return None
        crop = image[top:bottom, left:right]
        encoded, payload = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not encoded:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{track_id}.jpg"
        temporary = destination.with_suffix(".jpg.tmp")
        temporary.write_bytes(payload.tobytes())
        os.replace(temporary, destination)
        self._best_quality[track_id] = quality
        return {
            "track_id": track_id,
            "source_time_sec": round(float(context.source_time_sec), 6),
            "capture_quality": quality,
            "media_type": "image/jpeg",
        }
