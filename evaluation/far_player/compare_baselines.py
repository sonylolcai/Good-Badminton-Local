#!/usr/bin/env python
"""Compare completed per-video reports and evaluate the ByteTrack entry gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RECALL_TIE_TOLERANCE = 0.01
MIN_RECALL = 0.95
MAX_MISS_SECONDS = 0.30
MIN_FAR_GT_PER_VIDEO = 20


def build_summary(reports: list[dict]) -> dict:
    if len(reports) < 2:
        raise ValueError("at least two completed video reports are required")
    if any(report.get("status") != "complete" for report in reports):
        raise ValueError("all input reports must have status=complete")
    allowed_tiers = {"human_reviewed", "preliminary_reviewed"}
    if any(report.get("annotations", {}).get("review_tier") not in allowed_tiers for report in reports):
        raise ValueError("all reports must use complete reviewed ground truth")
    common_methods = set(reports[0].get("methods", {}))
    for report in reports[1:]:
        common_methods &= set(report.get("methods", {}))
    if not common_methods:
        raise ValueError("reports have no common evaluated method")

    method_summaries = {}
    for method in sorted(common_methods):
        total_gt = 0
        total_detected = 0
        weighted_time = 0.0
        timed_frames = 0
        fp_total = 0
        fp_available = True
        per_video = []
        for report in reports:
            result = report["methods"][method]
            metrics = result["metrics"]
            timing = result["timing"]
            gt_count = int(metrics["far_player_ground_truth_count"])
            detected_count = int(metrics["far_player_detected_count"])
            frame_count = int(timing["evaluated_frame_count"])
            total_gt += gt_count
            total_detected += detected_count
            if timing.get("mean_ms_per_frame") is not None:
                weighted_time += float(timing["mean_ms_per_frame"]) * frame_count
                timed_frames += frame_count
            if metrics.get("false_positive_count") is None:
                fp_available = False
            else:
                fp_total += int(metrics["false_positive_count"])
            per_video.append(
                {
                    "video_sha256": report["video"]["sha256"],
                    "resolution": [report["video"]["width"], report["video"]["height"]],
                    "far_player_ground_truth_count": gt_count,
                    "far_player_recall": metrics.get("far_player_recall"),
                    "longest_consecutive_miss_seconds": metrics.get("longest_consecutive_miss_seconds"),
                    "false_positive_count": metrics.get("false_positive_count"),
                }
            )
        recall = total_detected / total_gt if total_gt else None
        method_summaries[method] = {
            "far_player_ground_truth_count": total_gt,
            "far_player_detected_count": total_detected,
            "far_player_recall": recall,
            "far_player_miss_rate": 1.0 - recall if recall is not None else None,
            "max_longest_consecutive_miss_seconds": max(
                float(item["longest_consecutive_miss_seconds"] or 0.0) for item in per_video
            ),
            "weighted_mean_ms_per_frame": weighted_time / timed_frames if timed_frames else None,
            "false_positive_count": fp_total if fp_available else None,
            "per_video": per_video,
        }

    measurable = [(method, result) for method, result in method_summaries.items() if result["far_player_recall"] is not None]
    if not measurable:
        raise ValueError("no far-player ground truth was evaluated")
    best_recall = max(result["far_player_recall"] for _, result in measurable)
    near_best = [
        (method, result)
        for method, result in measurable
        if best_recall - result["far_player_recall"] <= RECALL_TIE_TOLERANCE
    ]
    recommended_method, recommended = min(
        near_best,
        key=lambda item: (
            item[1]["weighted_mean_ms_per_frame"] if item[1]["weighted_mean_ms_per_frame"] is not None else float("inf"),
            item[1]["false_positive_count"] if item[1]["false_positive_count"] is not None else float("inf"),
        ),
    )

    resolutions = [(int(report["video"]["width"]), int(report["video"]["height"])) for report in reports]
    gate_checks = {
        "at_least_two_videos": len({report["video"]["sha256"] for report in reports}) >= 2,
        "contains_852x480_video": any(width == 852 and height == 480 for width, height in resolutions),
        "contains_higher_resolution_video": any(width > 852 and height > 480 for width, height in resolutions),
        "minimum_20_far_gt_per_video": all(
            item["far_player_ground_truth_count"] >= MIN_FAR_GT_PER_VIDEO for item in recommended["per_video"]
        ),
        "recall_at_least_0_95_each_video": all(
            item["far_player_recall"] is not None and item["far_player_recall"] >= MIN_RECALL
            for item in recommended["per_video"]
        ),
        "longest_miss_at_most_0_30s_each_video": all(
            item["longest_consecutive_miss_seconds"] is not None
            and item["longest_consecutive_miss_seconds"] <= MAX_MISS_SECONDS
            for item in recommended["per_video"]
        ),
        "false_positives_are_evaluable": recommended["false_positive_count"] is not None,
        "human_ground_truth_reviewed": all(
            report.get("annotations", {}).get("review_tier") == "human_reviewed" for report in reports
        ),
    }
    return {
        "schema_version": "1.0",
        "status": "complete",
        "report_count": len(reports),
        "selection_rule": {
            "primary": "highest pooled far-player recall",
            "tie_tolerance": RECALL_TIE_TOLERANCE,
            "tie_breaker": "lowest weighted mean inference time, then fewest false positives",
        },
        "methods": method_summaries,
        "recommended_default_method": recommended_method,
        "bytetrack_entry_gate": {
            "ready": all(gate_checks.values()),
            "checks": gate_checks,
            "interpretation": (
                "ready for a separately evaluated ByteTrack integration"
                if all(gate_checks.values())
                else "not ready; fix failed checks before integrating ByteTrack"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
        summary = build_summary(reports)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
