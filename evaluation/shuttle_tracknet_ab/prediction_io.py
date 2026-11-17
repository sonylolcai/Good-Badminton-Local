"""Adapters between existing Good-Badminton evidence and TrackNetV3 CSVs."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Iterable, Mapping


def baseline_predictions(rows: Iterable[Mapping]) -> dict[int, dict]:
    """Read current immutable detections without treating predictions as observations."""
    predictions: dict[int, dict] = {}
    for row in rows:
        frame_index = int(row.get("frame", row.get("frame_index", -1)))
        if frame_index < 0:
            continue
        shuttle = dict(row.get("shuttlecock") or {})
        point = shuttle.get("image")
        status = shuttle.get("status", "missing")
        if status == "detected" and shuttle.get("accepted") is False:
            status = "missing"
        predictions[frame_index] = {
            "status": status,
            "image_xy": point,
            "confidence": shuttle.get("confidence"),
            "source": shuttle.get("source") or "existing_yolo",
        }
    return predictions


def load_tracknet_csv(path: Path, *, status: str) -> dict[int, dict]:
    """Load the official TrackNetV3 ``Frame,Visibility,X,Y`` CSV contract."""
    required = {"Frame", "Visibility", "X", "Y"}
    predictions: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{path}: missing TrackNet CSV columns {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            try:
                frame_index = int(float(row["Frame"]))
                visible = int(float(row["Visibility"])) != 0
                x = float(row["X"])
                y = float(row["Y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid TrackNet prediction") from exc
            if frame_index < 0:
                raise ValueError(f"{path}:{line_number}: Frame must be non-negative")
            if frame_index in predictions:
                raise ValueError(f"{path}:{line_number}: duplicate Frame {frame_index}")
            predictions[frame_index] = {
                "status": status if visible else "missing",
                "image_xy": [x, y] if visible else None,
                # Official output has a binary visibility flag, not a calibrated
                # confidence.  Never manufacture a confidence value here.
                "confidence": None,
                "source": "tracknet_v3_raw" if status == "detected_tracknet" else "tracknet_v3_rectified",
            }
    return predictions


def materialize_tracknet_detections(
    baseline_rows: Iterable[Mapping],
    predictions_by_frame: Mapping[int, Mapping],
    *,
    kind: str,
) -> list[dict]:
    """Create a derived candidate stream while retaining all player evidence.

    ``kind=raw`` writes detector measurements in the existing shuttlecock
    contract so the offline pipeline can consume them.  ``kind=rectified`` is
    intentionally marked unaccepted and cannot become score evidence.
    """
    if kind not in {"raw", "rectified"}:
        raise ValueError("kind must be 'raw' or 'rectified'")
    result: list[dict] = []
    for source_row in baseline_rows:
        row = copy.deepcopy(dict(source_row))
        frame_index = int(row.get("frame", row.get("frame_index", -1)))
        prediction = predictions_by_frame.get(frame_index)
        point = prediction.get("image_xy") if prediction else None
        is_visible = bool(prediction and prediction.get("status") != "missing" and _valid_point(point))
        if kind == "raw" and is_visible:
            shuttle = {
                "image": [float(point[0]), float(point[1])],
                "status": "detected",
                # TrackNet's published CSV exposes a binary heatmap visibility
                # decision.  ``0.5`` records that threshold, not a calibrated
                # detector probability; downstream candidate evidence keeps
                # that distinction in confidence_status.
                "confidence": 0.5,
                "source": "tracknet_v3_raw",
                "accepted": True,
                "visible": True,
                "candidate_count": 1,
                "raw_candidate_count": 1,
                "filtered_rejections": {},
                "gap_frames": 0,
                "rejection_reason": None,
                "measurement_kind": "temporal_heatmap",
                "confidence_status": "uncalibrated_binary_visibility_threshold_0.5",
            }
        elif kind == "rectified" and is_visible:
            shuttle = {
                "image": [float(point[0]), float(point[1])],
                "status": "rectified",
                "confidence": None,
                "source": "tracknet_v3_rectified",
                "accepted": False,
                "visible": False,
                "candidate_count": 0,
                "raw_candidate_count": 0,
                "filtered_rejections": {},
                "gap_frames": 0,
                "rejection_reason": "trajectory_rectification_is_not_raw_measurement",
                "measurement_kind": "trajectory_rectification",
                "confidence_status": "inferred_not_score_evidence",
            }
        else:
            shuttle = {
                "image": None,
                "status": "missing",
                "confidence": None,
                "source": None,
                "accepted": False,
                "visible": False,
                "candidate_count": 0,
                "raw_candidate_count": 0,
                "filtered_rejections": {},
                "gap_frames": 0,
                "rejection_reason": "tracknet_no_visible_measurement",
            }
        row["shuttlecock"] = shuttle
        result.append(row)
    return result


def _valid_point(point: object) -> bool:
    return isinstance(point, (list, tuple)) and len(point) == 2 and all(
        isinstance(value, (int, float)) for value in point
    )
