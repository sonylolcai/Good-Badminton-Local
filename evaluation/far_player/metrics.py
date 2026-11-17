"""Pure metric functions for the far-player benchmark.

This module deliberately has no model or OpenCV dependency so the acceptance
logic can be tested independently from the detector runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping, Sequence


VISIBLE_STATES = {"visible", "partial"}


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Return intersection-over-union for two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_in_regions(box: Sequence[float], regions: Iterable[Sequence[float]]) -> bool:
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return any(rx1 <= cx <= rx2 and ry1 <= cy <= ry2 for rx1, ry1, rx2, ry2 in regions)


def greedy_match(
    ground_truth: Sequence[Mapping],
    detections: Sequence[Mapping],
    iou_threshold: float,
) -> tuple[dict[int, int], set[int]]:
    """Greedily match detections to ground truth in descending IoU order."""
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for det_index, det in enumerate(detections):
            overlap = iou_xyxy(gt["bbox_xyxy"], det["bbox_xyxy"])
            if overlap >= iou_threshold:
                candidates.append((overlap, gt_index, det_index))
    candidates.sort(reverse=True)
    matches: dict[int, int] = {}
    used_detections: set[int] = set()
    for _, gt_index, det_index in candidates:
        if gt_index not in matches and det_index not in used_detections:
            matches[gt_index] = det_index
            used_detections.add(det_index)
    return matches, used_detections


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict
    misses: list[dict]
    false_positives: list[dict]


def evaluate_predictions(
    annotations: Sequence[Mapping],
    predictions_by_frame: Mapping[int, Sequence[Mapping]],
    iou_threshold: float = 0.3,
) -> EvaluationResult:
    """Evaluate one method against complete, human-reviewed annotations.

    The recall denominator is every visible/partially visible person whose
    ``role`` is ``far_player``. False positives are only valid when every
    person in the frame has been annotated (``people_annotation_complete``).
    """
    far_total = 0
    far_detected = 0
    misses: list[dict] = []
    false_positives: list[dict] = []
    frame_missed: list[tuple[int, float, bool]] = []
    fp_evaluable_frames = 0

    for row in sorted(annotations, key=lambda item: int(item["frame_index"])):
        frame_index = int(row["frame_index"])
        time_sec = float(row.get("time_sec", 0.0))
        visible_people = [
            person
            for person in row.get("people", [])
            if person.get("visibility", "visible") in VISIBLE_STATES
        ]
        # A detected umpire/spectator is a false player candidate, not a true
        # positive. Such distractors can still be explicitly annotated for
        # visual review using role=non_player_person.
        people = [person for person in visible_people if person.get("role") in {"far_player", "other_player"}]
        detections = list(predictions_by_frame.get(frame_index, []))
        matches, used_detections = greedy_match(people, detections, iou_threshold)

        far_indices = [index for index, person in enumerate(people) if person.get("role") == "far_player"]
        far_total += len(far_indices)
        detected_in_frame = sum(index in matches for index in far_indices)
        far_detected += detected_in_frame
        has_far_miss = detected_in_frame < len(far_indices)
        frame_missed.append((frame_index, time_sec, has_far_miss))
        for gt_index in far_indices:
            if gt_index not in matches:
                person = people[gt_index]
                misses.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "person_id": person.get("id"),
                        "bbox_xyxy": person["bbox_xyxy"],
                    }
                )

        if row.get("people_annotation_complete") is True:
            fp_evaluable_frames += 1
            ignore_regions = row.get("ignore_regions", [])
            for det_index, detection in enumerate(detections):
                if det_index in used_detections:
                    continue
                if _center_in_regions(detection["bbox_xyxy"], ignore_regions):
                    continue
                false_positives.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": time_sec,
                        "bbox_xyxy": detection["bbox_xyxy"],
                        "confidence": detection.get("confidence"),
                        "source": detection.get("source"),
                    }
                )

    longest_samples = 0
    longest_seconds = 0.0
    current: list[tuple[int, float, bool]] = []
    positive_times = [time_sec for _, time_sec, _ in frame_missed]
    cadence = median([b - a for a, b in zip(positive_times, positive_times[1:])]) if len(positive_times) > 1 else 0.0
    for sample in frame_missed + [(-1, 0.0, False)]:
        if sample[2]:
            current.append(sample)
        elif current:
            longest_samples = max(longest_samples, len(current))
            longest_seconds = max(longest_seconds, current[-1][1] - current[0][1] + cadence)
            current = []

    recall = far_detected / far_total if far_total else None
    metrics = {
        "far_player_ground_truth_count": far_total,
        "far_player_detected_count": far_detected,
        "far_player_recall": recall,
        "far_player_miss_rate": (1.0 - recall) if recall is not None else None,
        "longest_consecutive_miss_samples": longest_samples,
        "longest_consecutive_miss_seconds": round(longest_seconds, 6),
        "false_positive_count": len(false_positives) if fp_evaluable_frames else None,
        "false_positive_evaluable_frames": fp_evaluable_frames,
        "matching_iou_threshold": iou_threshold,
    }
    return EvaluationResult(metrics=metrics, misses=misses, false_positives=false_positives)
