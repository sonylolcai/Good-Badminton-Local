"""Normalize fixed-sport GPU evidence for shared business-side processing.

The badminton and tennis GPU processes may use unrelated model stacks.  This
adapter is the stable seam consumed by movement statistics, synchronized video
rendering, and later vision-language analysis.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


SCHEMA_VERSION = "visual-observation.v1"
SUPPORTED_SPORTS = {"badminton", "tennis"}
SUPPORTED_EVIDENCE_STATES = {"detected", "predicted", "missing"}


def _point(value, *, size=2):
    if value is None:
        return None
    try:
        result = [float(item) for item in value[:size]]
    except (TypeError, ValueError, IndexError):
        return None
    return result if len(result) == size else None


def _confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _coordinate_system(sport_id, coordinate_system_id, court_dimensions_m):
    dimensions = _point(court_dimensions_m)
    if dimensions is None or any(value <= 0 for value in dimensions):
        raise ValueError("court_dimensions_m must contain two positive numbers")
    return {
        "id": str(coordinate_system_id),
        "unit": "meter",
        "court_dimensions_m": dimensions,
    }


def from_stream_event(
    event: Mapping[str, Any],
    *,
    sport_id: str = "badminton",
    coordinate_system_id: str = "standard_badminton_court_m",
    court_dimensions_m=(6.10, 13.40),
):
    """Return one ``visual-observation.v1`` record or ``None``.

    Session lifecycle and roster-review events intentionally remain transport
    concerns.  Only person and physical-ball evidence crosses this interface.
    """
    sport_id = str(sport_id).strip().lower()
    if sport_id not in SUPPORTED_SPORTS:
        raise ValueError(f"unsupported sport_id: {sport_id}")
    event_type = str(event.get("event_type") or "")
    if event_type not in {"person_observation", "shuttle_observation", "ball_observation"}:
        return None
    evidence_state = str(event.get("evidence_state") or "missing")
    if evidence_state not in SUPPORTED_EVIDENCE_STATES:
        evidence_state = "missing"
    payload = dict(event.get("data") or {})
    source_frame_index = payload.get("source_frame_index")

    if event_type == "person_observation":
        track = dict(payload.get("track") or {})
        source_frame_index = track.get("source_frame_index", source_frame_index)
        pose = dict(track.get("pose") or {})
        pose_current = bool(pose.get("is_current_measurement")) and pose.get("keypoints_image") is not None
        data = {
            "kind": "person",
            "track_id": str(track.get("track_id") or "unknown"),
            "role": track.get("court_end"),
            "bbox_xyxy": _point(
                (track.get("location_evidence") or {}).get("bbox_xyxy"), size=4
            ),
            "ground_image_xy": _point(track.get("image_xy")),
            "court_xy_m": _point(track.get("court_xy_m")),
            "pose": {
                "format": "coco17_image_v1",
                "evidence_state": "detected" if pose_current else "missing",
                "keypoints_image": deepcopy(pose.get("keypoints_image")) if pose_current else None,
                "keypoint_scores": deepcopy(pose.get("keypoint_scores")) if pose_current else None,
            },
        }
    else:
        measurement = dict(payload.get("measurement") or {})
        source_frame_index = payload.get("source_frame_index", source_frame_index)
        kind = "tennis_ball" if event_type == "ball_observation" else "shuttlecock"
        data = {
            "kind": kind,
            "image_xy": _point(measurement.get("image")),
            "court_xy_m": _point(measurement.get("court")),
            "model_checkpoint": payload.get("model_checkpoint"),
        }

    if source_frame_index is None:
        raise ValueError("visual observation requires source_frame_index")
    return {
        "schema_version": SCHEMA_VERSION,
        "sport_id": sport_id,
        "observation_type": "person" if event_type == "person_observation" else "ball",
        "source_frame_index": int(source_frame_index),
        "source_time_sec": max(0.0, float(event.get("source_time_sec") or 0.0)),
        "coordinate_system": _coordinate_system(
            sport_id, coordinate_system_id, court_dimensions_m
        ),
        "evidence_state": evidence_state,
        "confidence": _confidence(event.get("confidence")),
        "data": data,
    }
