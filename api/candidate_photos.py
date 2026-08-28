"""Anonymous full-body candidate photos for post-match role claiming.

The GPU never identifies a person.  It keeps one high-quality full-body crop
per anonymous ``track_id`` and prefers a pose-supported front view, so the
business service can later let a player recognise themselves without face
recognition or biometric identity storage.
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
_POSE_KEYPOINT_CONFIDENCE = 0.25
_NOSE, _LEFT_EYE, _RIGHT_EYE, _LEFT_EAR, _RIGHT_EAR = range(5)


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
        self._best_rank: dict[str, float] = {}

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
        return {
            "state_version": "candidate-photo.v2",
            "best_quality": dict(self._best_quality),
            "best_rank": dict(self._best_rank),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        version = state.get("state_version")
        if version not in {"candidate-photo.v1", "candidate-photo.v2"}:
            raise ValueError("unsupported candidate photo checkpoint")
        self._best_quality = {
            str(track_id): float(quality)
            for track_id, quality in dict(state.get("best_quality") or {}).items()
            if _TRACK_ID_PATTERN.fullmatch(str(track_id)) and float(quality) >= 0.0
        }
        if version == "candidate-photo.v2":
            rank_source = dict(state.get("best_rank") or {})
        else:
            # A v1 checkpoint has no pose-orientation evidence. Preserve its
            # old quality ordering until a newly scored candidate is observed.
            rank_source = self._best_quality
        self._best_rank = {
            str(track_id): float(rank)
            for track_id, rank in rank_source.items()
            if _TRACK_ID_PATTERN.fullmatch(str(track_id)) and float(rank) >= 0.0
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
        crop = image[top:bottom, left:right]
        sharpness = self._sharpness(crop)
        quality = round(confidence * (0.55 + 0.30 * size_factor + 0.15 * sharpness), 4)
        frontal_score, view_label = self._front_view_score(track, bbox_width=right - left)
        # Front-facing evidence is intentionally weighted above marginal
        # detection-quality differences. A very poor crop still cannot win,
        # because quality remains nearly half of the final rank.
        rank = round(0.55 * frontal_score + 0.45 * quality, 4)
        if rank <= self._best_rank.get(track_id, -1.0):
            return None
        encoded, payload = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not encoded:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{track_id}.jpg"
        temporary = destination.with_suffix(".jpg.tmp")
        temporary.write_bytes(payload.tobytes())
        os.replace(temporary, destination)
        self._best_quality[track_id] = quality
        self._best_rank[track_id] = rank
        return {
            "track_id": track_id,
            "source_time_sec": round(float(context.source_time_sec), 6),
            "capture_quality": quality,
            "frontal_score": frontal_score,
            "view_label": view_label,
            "selection_score": rank,
            "selection_policy": "pose_front_priority_v1",
            "media_type": "image/jpeg",
        }

    @staticmethod
    def _sharpness(crop: np.ndarray) -> float:
        """Return a bounded non-biometric sharpness signal for crop ranking."""

        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(max(0.0, min(1.0, variance / 180.0)), 4)

    @classmethod
    def _front_view_score(cls, track: dict[str, Any], *, bbox_width: int) -> tuple[float, str]:
        """Estimate face orientation from anonymous COCO17 pose points.

        This is not facial recognition: it only asks whether both eyes and a
        centered nose were visible in the current measurement. When the camera
        is too distant or the model lacks head points, callers get an explicit
        ``not_assessed`` fallback instead of a fabricated front-view claim.
        """

        pose = track.get("pose") if isinstance(track.get("pose"), dict) else {}
        points = pose.get("keypoints_image") or track.get("keypoints_image")
        scores = pose.get("keypoint_scores") or track.get("keypoint_scores")

        def point(index: int) -> tuple[float, float] | None:
            if not isinstance(points, (list, tuple)) or index >= len(points):
                return None
            value = points[index]
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                return None
            try:
                x, y = float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
            if not np.isfinite([x, y]).all() or x <= 1 or y <= 1:
                return None
            if isinstance(scores, (list, tuple)) and index < len(scores):
                try:
                    if float(scores[index]) < _POSE_KEYPOINT_CONFIDENCE:
                        return None
                except (TypeError, ValueError):
                    return None
            return x, y

        nose = point(_NOSE)
        left_eye, right_eye = point(_LEFT_EYE), point(_RIGHT_EYE)
        left_ear, right_ear = point(_LEFT_EAR), point(_RIGHT_EAR)
        visible_head_points = sum(item is not None for item in (nose, left_eye, right_eye, left_ear, right_ear))
        if not (left_eye and right_eye and nose):
            return (0.0, "side_or_back" if visible_head_points else "not_assessed")

        eye_span = abs(right_eye[0] - left_eye[0])
        minimum_span = max(3.0, float(bbox_width) * 0.06)
        if eye_span < minimum_span:
            return 0.0, "not_assessed"
        eye_midpoint = (left_eye[0] + right_eye[0]) / 2.0
        nose_offset = abs(nose[0] - eye_midpoint) / eye_span
        symmetry = max(0.0, 1.0 - nose_offset / 0.65)
        ears_visible = sum(item is not None for item in (left_ear, right_ear)) / 2.0
        score = round(max(0.0, min(1.0, 0.80 * symmetry + 0.20 * ears_visible)), 4)
        if score >= 0.72:
            return score, "front"
        if score >= 0.42:
            return score, "three_quarter"
        return score, "side_or_back"
