"""Read anonymous spatial measurements from the frozen analysis contract.

This package is intentionally neutral: both the GPU-side legacy visualizer and
the business interpretation service may consume it without either service
importing the other. Predicted, missing, low-confidence and invalid rows remain
auditable exclusions instead of being converted into movement facts.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.50
DEFAULT_MIN_LOCATION_CONFIDENCE = 0.50
DEFAULT_MIN_IDENTITY_CONFIDENCE = 0.70


def collect_track_position_evidence(
    detections_path,
    *,
    min_detection_confidence=DEFAULT_MIN_DETECTION_CONFIDENCE,
    min_location_confidence=DEFAULT_MIN_LOCATION_CONFIDENCE,
    min_identity_confidence=DEFAULT_MIN_IDENTITY_CONFIDENCE,
):
    """Collect auditable per-track court positions from v2 detections."""

    path = Path(detections_path)
    result = {
        "has_spatial_tracks": False,
        "source_frames": 0,
        "match_mode": None,
        "thresholds": {
            "min_detection_confidence": float(min_detection_confidence),
            "min_location_confidence": float(min_location_confidence),
            "min_identity_confidence": float(min_identity_confidence),
        },
        "tracks": {},
    }
    if not path.is_file():
        return result

    with path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sampling = record.get("sampling") or {}
            # Shuttle-only rows preserve higher-rate ball evidence. They must
            # not dilute player-measurement coverage or movement statistics.
            if sampling.get("pose_sampled") is not False:
                result["source_frames"] += 1
            spatial = record.get("spatial") or {}
            tracks = spatial.get("tracks")
            if not isinstance(tracks, list):
                continue
            result["has_spatial_tracks"] = True
            if result["match_mode"] is None:
                result["match_mode"] = (spatial.get("match") or {}).get("mode")
            frame = _integer_or_none(record.get("frame"))
            time_sec = _float_or_none(record.get("time_sec"))
            for track in tracks:
                if not isinstance(track, dict) or not track.get("track_id"):
                    continue
                track_id = str(track["track_id"])
                entry = result["tracks"].setdefault(track_id, _empty_track_entry(track_id))
                entry["track_rows"] += 1
                status = str(track.get("status") or "missing")
                entry["state_counts"][status] += 1
                if status != "detected":
                    reason = status if status in {"predicted", "missing"} else "non_detected"
                    entry["excluded"][reason] += 1
                    continue

                point = _court_point(track.get("court_xy_m"))
                if point is None:
                    entry["excluded"]["invalid_coordinate"] += 1
                    continue
                detection_confidence = _float_or_none(track.get("confidence"))
                location_confidence = _float_or_none(
                    (track.get("location_evidence") or {}).get("confidence")
                )
                identity_confidence = _association_identity_confidence(
                    track.get("association") or {}
                )
                if detection_confidence is None or detection_confidence < min_detection_confidence:
                    entry["excluded"]["low_detection_confidence"] += 1
                    continue
                if location_confidence is None or location_confidence < min_location_confidence:
                    entry["excluded"]["low_location_confidence"] += 1
                    continue
                if identity_confidence < min_identity_confidence:
                    entry["excluded"]["low_identity_confidence"] += 1
                    continue
                entry["usable_points"].append(
                    {
                        "frame": frame,
                        "time_sec": time_sec,
                        "court_xy_m": point,
                        "detection_confidence": detection_confidence,
                        "location_confidence": location_confidence,
                        "identity_confidence": identity_confidence,
                    }
                )
    return result


def _empty_track_entry(track_id):
    return {
        "track_id": track_id,
        "track_rows": 0,
        "state_counts": Counter(),
        "excluded": Counter(),
        "usable_points": [],
    }


def _association_identity_confidence(association):
    value = _float_or_none(association.get("identity_confidence"))
    if value is not None:
        return value
    return {
        "bytetrack": 0.95,
        "roster_bootstrap": 0.95,
        "court_association": 0.85,
        "roster_reassociation": 0.72,
        "roster_end_recovery": 0.55,
    }.get(str(association.get("source") or ""), 0.60)


def _court_point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x_value = _float_or_none(value[0])
    y_value = _float_or_none(value[1])
    if x_value is None or y_value is None:
        return None
    if not (0.0 <= x_value <= COURT_WIDTH_M and 0.0 <= y_value <= COURT_LENGTH_M):
        return None
    return (x_value, y_value)


def _float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _integer_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "COURT_WIDTH_M",
    "COURT_LENGTH_M",
    "DEFAULT_MIN_DETECTION_CONFIDENCE",
    "DEFAULT_MIN_LOCATION_CONFIDENCE",
    "DEFAULT_MIN_IDENTITY_CONFIDENCE",
    "collect_track_position_evidence",
]
