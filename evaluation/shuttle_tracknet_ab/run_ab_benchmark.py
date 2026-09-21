#!/usr/bin/env python
"""Evaluate current YOLO evidence against TrackNetV3 without mutating matches.

The runner compares detector measurements using human-reviewed labels.  It also
creates isolated derived TrackNet evidence so existing offline rally-candidate
logic can be compared under the same player observations.  No generated file
is written into the source analysis directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from badminton_analysis.analysis.offline_shot_reconstruction import (
    evaluate_rally_terminal_predictions,
    generate_offline_artifacts,
)
from evaluation.shuttle_tracknet_ab.annotations import (
    load_jsonl,
    sha256_file,
    validate_annotations,
    write_jsonl,
)
from evaluation.shuttle_tracknet_ab.metrics import evaluate_shuttle_predictions
from evaluation.shuttle_tracknet_ab.prediction_io import (
    baseline_predictions,
    load_tracknet_csv,
    materialize_tracknet_detections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--baseline-detections", required=True, type=Path, help="Existing immutable detections.jsonl")
    parser.add_argument("--tracknet-raw-csv", required=True, type=Path, help="TrackNet-only Frame,Visibility,X,Y CSV")
    parser.add_argument("--tracknet-rectified-csv", type=Path, help="Optional TrackNet+InpaintNet CSV for B* display/review")
    parser.add_argument("--annotations", required=True, type=Path, help="Completed human-reviewed shuttle annotations.jsonl")
    parser.add_argument("--metadata", required=True, type=Path, help="Annotation-set metadata.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tolerance-px", type=float, help="Defaults to 4 pixels at 512px source width, scaled to the video")
    parser.add_argument("--reference-terminals", type=Path, help="Optional human-only rally terminal JSONL for candidate evaluation")
    parser.add_argument("--terminal-tolerance-sec", type=float, default=0.6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        video = _video_metadata(args.video)
        annotations, annotation_metadata = _validated_human_annotations(args, video)
        baseline_rows = load_jsonl(args.baseline_detections)
        if not baseline_rows:
            raise ValueError("baseline detections.jsonl is empty")
        raw_predictions = load_tracknet_csv(args.tracknet_raw_csv, status="detected_tracknet")
        rectified_predictions = (
            load_tracknet_csv(args.tracknet_rectified_csv, status="rectified_tracknet")
            if args.tracknet_rectified_csv
            else None
        )
        if args.reference_terminals is not None and not args.reference_terminals.is_file():
            raise ValueError(f"reference terminals does not exist: {args.reference_terminals}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _write_blocker(args.output_dir, str(exc))
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2

    available_frames = {int(row.get("frame", row.get("frame_index", -1))) for row in baseline_rows}
    missing_baseline_frames = [
        int(row["frame_index"]) for row in annotations if int(row["frame_index"]) not in available_frames
    ]
    if missing_baseline_frames:
        message = (
            "annotations include frames not present in baseline detections; rerun the full current pipeline "
            f"or annotate only the same valid court interval. First missing frames: {missing_baseline_frames[:10]}"
        )
        _write_blocker(args.output_dir, message)
        print(f"BLOCKED: {message}", file=sys.stderr)
        return 2

    tolerance_px = args.tolerance_px or 4.0 * video["width"] / 512.0
    if tolerance_px <= 0 or args.terminal_tolerance_sec <= 0:
        message = "tolerance values must be greater than zero"
        _write_blocker(args.output_dir, message)
        print(f"BLOCKED: {message}", file=sys.stderr)
        return 2

    baseline_result = evaluate_shuttle_predictions(
        annotations, baseline_predictions(baseline_rows), tolerance_px=tolerance_px
    )
    raw_result = evaluate_shuttle_predictions(annotations, raw_predictions, tolerance_px=tolerance_px)
    rectified_result = (
        evaluate_shuttle_predictions(annotations, rectified_predictions, tolerance_px=tolerance_px)
        if rectified_predictions is not None
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    derived = _write_isolated_derived_artifacts(
        args.output_dir,
        args.baseline_detections,
        baseline_rows,
        raw_predictions,
        rectified_predictions,
    )
    terminal_evaluation = _evaluate_terminals(args, derived)
    report = {
        "schema_version": "1.0",
        "status": "complete_human_reviewed_ab_benchmark",
        "policy": {
            "source_detections_immutable": True,
            "tracknet_raw_is_measurement": True,
            "tracknet_rectified_is_inferred_not_score_evidence": True,
            "human_references_evaluation_only": True,
        },
        "video": video,
        "annotation_set": {
            "path": str(args.annotations.resolve()),
            "sha256": sha256_file(args.annotations),
            "ground_truth_source": annotation_metadata.get("ground_truth_source"),
            "reviewer": annotation_metadata.get("reviewer"),
            "sample_count": len(annotations),
        },
        "inputs": {
            "baseline_detections": str(args.baseline_detections.resolve()),
            "tracknet_raw_csv": str(args.tracknet_raw_csv.resolve()),
            "tracknet_rectified_csv": str(args.tracknet_rectified_csv.resolve()) if args.tracknet_rectified_csv else None,
            "tolerance_px": round(tolerance_px, 6),
            "terminal_tolerance_sec": args.terminal_tolerance_sec,
        },
        "methods": {
            "a_current_yolo": baseline_result.metrics,
            "b_tracknet_v3_raw": raw_result.metrics,
            "b_star_tracknet_v3_rectified": rectified_result.metrics if rectified_result else None,
        },
        "delta_b_minus_a": _metric_delta(baseline_result.metrics, raw_result.metrics),
        "derived_artifacts": derived,
        "terminal_evaluation": terminal_evaluation,
        "review_artifacts": {
            "a_current_yolo_misses": _write_json(args.output_dir / "a_yolo_misses.json", baseline_result.misses),
            "a_current_yolo_false_positives": _write_json(args.output_dir / "a_yolo_false_positives.json", baseline_result.false_positives),
            "b_tracknet_raw_misses": _write_json(args.output_dir / "b_tracknet_raw_misses.json", raw_result.misses),
            "b_tracknet_raw_false_positives": _write_json(args.output_dir / "b_tracknet_raw_false_positives.json", raw_result.false_positives),
            "b_star_rectified_misses": _write_json(args.output_dir / "b_star_rectified_misses.json", rectified_result.misses) if rectified_result else None,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(args.output_dir / "summary.csv", report["methods"])
    print(report_path)
    return 0


def _validated_human_annotations(args: argparse.Namespace, video: dict) -> tuple[list[dict], dict]:
    if not args.annotations.exists() or not args.metadata.exists():
        raise ValueError("annotations and metadata must both exist")
    annotations = load_jsonl(args.annotations)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    annotation_video = metadata.get("video") or {}
    if annotation_video.get("sha256") != video["sha256"]:
        raise ValueError("annotation metadata SHA-256 does not match --video")
    if metadata.get("ground_truth_source") != "human_manual" or metadata.get("review_status") != "complete":
        raise ValueError("benchmark requires human_manual, complete ground truth; model proposals cannot be promoted to truth")
    if not metadata.get("reviewer"):
        raise ValueError("human-reviewed metadata must name the reviewer")
    errors = validate_annotations(annotations, video["width"], video["height"], require_complete=True)
    if errors:
        raise ValueError("invalid annotations: " + "; ".join(errors[:8]))
    return annotations, metadata


def _video_metadata(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"video does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if min(width, height, fps, frame_count) <= 0:
        raise ValueError("video dimensions, FPS, or frame count are invalid")
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps,
    }


def _write_isolated_derived_artifacts(
    output_dir: Path,
    baseline_path: Path,
    baseline_rows: list[dict],
    raw_predictions: dict[int, dict],
    rectified_predictions: dict[int, dict] | None,
) -> dict:
    baseline_dir = output_dir / "a_current_yolo"
    raw_dir = output_dir / "b_tracknet_raw"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _copy_metadata_if_present(baseline_path.parent, baseline_dir)
    _copy_metadata_if_present(baseline_path.parent, raw_dir)
    baseline_copy = baseline_dir / "detections.jsonl"
    raw_copy = raw_dir / "detections.jsonl"
    write_jsonl(baseline_copy, baseline_rows)
    write_jsonl(raw_copy, materialize_tracknet_detections(baseline_rows, raw_predictions, kind="raw"))
    result = {
        "a_current_yolo": generate_offline_artifacts(baseline_copy, output_dir=baseline_dir / "derived"),
        "b_tracknet_v3_raw": generate_offline_artifacts(raw_copy, output_dir=raw_dir / "derived"),
    }
    if rectified_predictions is not None:
        rectified_dir = output_dir / "b_star_tracknet_rectified"
        rectified_dir.mkdir(parents=True, exist_ok=True)
        _copy_metadata_if_present(baseline_path.parent, rectified_dir)
        rectified_copy = rectified_dir / "detections.jsonl"
        write_jsonl(
            rectified_copy,
            materialize_tracknet_detections(baseline_rows, rectified_predictions, kind="rectified"),
        )
        # Do not execute scoring-adjacent derivation on inferred TrackNet points.
        result["b_star_tracknet_v3_rectified"] = {
            "detections_path": str(rectified_copy),
            "status": "stored_for_visual_review_only",
            "reason": "rectified points are explicitly unaccepted and excluded from score/rally evidence",
        }
    return result


def _copy_metadata_if_present(source_dir: Path, destination_dir: Path) -> None:
    source = source_dir / "metadata.json"
    if source.is_file():
        (destination_dir / "metadata.json").write_bytes(source.read_bytes())


def _evaluate_terminals(args: argparse.Namespace, derived: dict) -> dict | None:
    if args.reference_terminals is None:
        return None
    references = load_jsonl(args.reference_terminals)
    result: dict = {"reference_path": str(args.reference_terminals.resolve())}
    for method, artifacts in (("a_current_yolo", derived["a_current_yolo"]), ("b_tracknet_v3_raw", derived["b_tracknet_v3_raw"])):
        rallies = load_jsonl(Path(artifacts["rallies_path"]))
        result[method] = evaluate_rally_terminal_predictions(
            rallies, references, tolerance_sec=args.terminal_tolerance_sec
        )
    return result


def _metric_delta(baseline: dict, candidate: dict) -> dict:
    keys = (
        "raw_detection_recall",
        "false_positive_rate",
        "mean_localization_error_px",
        "median_localization_error_px",
        "longest_consecutive_raw_miss_seconds",
    )
    return {
        key: (round(float(candidate[key]) - float(baseline[key]), 6) if candidate.get(key) is not None and baseline.get(key) is not None else None)
        for key in keys
    }


def _write_summary_csv(path: Path, methods: dict) -> None:
    keys = [
        "raw_detection_recall",
        "false_positive_rate",
        "mean_localization_error_px",
        "median_localization_error_px",
        "longest_consecutive_raw_miss_seconds",
        "inferred_points_on_visible_ground_truth",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *keys])
        writer.writeheader()
        for method, metrics in methods.items():
            if metrics is None:
                continue
            writer.writerow({"method": method, **{key: metrics.get(key) for key in keys}})


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _write_blocker(output_dir: Path, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked_missing_or_invalid_human_ground_truth",
        "reason": reason,
        "next_step": "Create and complete a human-reviewed annotation set before using the model outputs as evidence.",
    }
    (output_dir / "blocked.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
