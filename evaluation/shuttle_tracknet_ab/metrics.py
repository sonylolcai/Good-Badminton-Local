"""Pure, confidence-preserving metrics for the shuttlecock A/B benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import mean, median
from typing import Mapping, Sequence


MEASURED_STATUSES = {"detected", "detected_tracknet"}
INFERRED_STATUSES = {"predicted", "rectified_tracknet"}


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict
    misses: list[dict]
    false_positives: list[dict]
    localization_errors: list[dict]


def evaluate_shuttle_predictions(
    annotations: Sequence[Mapping],
    predictions_by_frame: Mapping[int, Mapping],
    *,
    tolerance_px: float,
) -> EvaluationResult:
    """Evaluate raw measurements only; inferred points stay visible in reports.

    A rectified or constant-velocity point may be useful to a reviewer, but it
    cannot be credited as a detector hit.  This prevents a gap-filling model
    from appearing better merely because it fabricates a continuous line.
    """
    if tolerance_px <= 0:
        raise ValueError("tolerance_px must be greater than zero")

    visible_total = 0
    true_positives = 0
    inferred_visible = 0
    non_visible_total = 0
    misses: list[dict] = []
    false_positives: list[dict] = []
    localization_errors: list[dict] = []
    visible_outcomes: list[tuple[int, float, bool]] = []

    for row in sorted(annotations, key=lambda item: int(item["frame_index"])):
        frame_index = int(row["frame_index"])
        time_sec = float(row["time_sec"])
        shuttle = row["shuttle"]
        visibility = shuttle["visibility"]
        prediction = dict(predictions_by_frame.get(frame_index) or {})
        status = prediction.get("status", "missing")
        point = prediction.get("image_xy")
        is_measurement = status in MEASURED_STATUSES and _valid_point(point)
        is_inferred = status in INFERRED_STATUSES and _valid_point(point)

        if visibility == "ambiguous":
            continue
        if visibility == "visible":
            visible_total += 1
            target = shuttle["image_xy"]
            matched = False
            if is_measurement:
                error_px = _distance(point, target)
                localization_errors.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "error_px": round(error_px, 6),
                        "prediction": point,
                        "ground_truth": target,
                        "status": status,
                    }
                )
                matched = error_px <= tolerance_px
            elif is_inferred:
                inferred_visible += 1

            visible_outcomes.append((frame_index, time_sec, matched))
            if matched:
                true_positives += 1
            else:
                misses.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "ground_truth": target,
                        "prediction": point if _valid_point(point) else None,
                        "status": status,
                    }
                )
        elif visibility == "not_visible":
            non_visible_total += 1
            if is_measurement:
                false_positives.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "prediction": point,
                        "status": status,
                        "confidence": prediction.get("confidence"),
                        "source": prediction.get("source"),
                    }
                )

    error_values = [item["error_px"] for item in localization_errors]
    longest_samples, longest_seconds = _longest_miss_run(visible_outcomes)
    recall = true_positives / visible_total if visible_total else None
    false_positive_rate = len(false_positives) / non_visible_total if non_visible_total else None
    metrics = {
        "visible_ground_truth_count": visible_total,
        "true_positive_count": true_positives,
        "raw_detection_recall": recall,
        "raw_detection_miss_rate": (1.0 - recall) if recall is not None else None,
        "not_visible_ground_truth_count": non_visible_total,
        "false_positive_count": len(false_positives) if non_visible_total else None,
        "false_positive_rate": false_positive_rate,
        "inferred_points_on_visible_ground_truth": inferred_visible,
        "mean_localization_error_px": round(mean(error_values), 6) if error_values else None,
        "median_localization_error_px": round(median(error_values), 6) if error_values else None,
        "longest_consecutive_raw_miss_samples": longest_samples,
        "longest_consecutive_raw_miss_seconds": round(longest_seconds, 6),
        "matching_tolerance_px": tolerance_px,
    }
    return EvaluationResult(
        metrics=metrics,
        misses=misses,
        false_positives=false_positives,
        localization_errors=localization_errors,
    )


def _longest_miss_run(outcomes: Sequence[tuple[int, float, bool]]) -> tuple[int, float]:
    if not outcomes:
        return 0, 0.0
    times = [time_sec for _, time_sec, _ in outcomes]
    cadence = median([later - earlier for earlier, later in zip(times, times[1:])]) if len(times) > 1 else 0.0
    longest_samples = 0
    longest_seconds = 0.0
    current: list[tuple[int, float, bool]] = []
    for outcome in [*outcomes, (-1, 0.0, True)]:
        if not outcome[2]:
            current.append(outcome)
            continue
        if current:
            longest_samples = max(longest_samples, len(current))
            longest_seconds = max(longest_seconds, current[-1][1] - current[0][1] + cadence)
            current = []
    return longest_samples, longest_seconds


def _valid_point(point: object) -> bool:
    return isinstance(point, (list, tuple)) and len(point) == 2 and all(
        isinstance(value, (int, float)) for value in point
    )


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
