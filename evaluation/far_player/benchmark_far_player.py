#!/usr/bin/env python
"""Run the reproducible fixed-camera far-player detection benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from statistics import mean, median

# Ultralytics otherwise tries to create a settings file under AppData/Roaming,
# which is not writable in restricted/offline benchmark environments.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "good-badminton-ultralytics"))

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.far_player.annotations import load_jsonl, validate_annotations, write_jsonl
from evaluation.far_player.detector_runner import METHODS, UltralyticsBenchmarkRunner, parse_roi, roi_to_pixels
from evaluation.far_player.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path, help="Human-reviewed annotations.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="weights/yolo11n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--inference-iou", type=float, default=0.7)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument("--matching-iou", type=float, default=0.3)
    parser.add_argument("--far-roi", default="0.0,0.0,1.0,0.55", help="Normalized x1,y1,x2,y2")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max-fp-images", type=int, default=20)
    return parser.parse_args()


def video_metadata(video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return {
        "path": str(video_path.resolve()),
        "filename": video_path.name,
        "size_bytes": video_path.stat().st_size,
        "sha256": sha256_file(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps if fps > 0 else None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def model_metadata(model_path: str, runner: UltralyticsBenchmarkRunner) -> dict:
    import ultralytics

    possible_path = Path(model_path)
    if not possible_path.exists():
        ckpt_path = getattr(runner.model, "ckpt_path", None)
        if ckpt_path:
            possible_path = Path(ckpt_path)
    return {
        "requested_path": model_path,
        "resolved_path": str(possible_path.resolve()) if possible_path.exists() else None,
        "sha256": sha256_file(possible_path) if possible_path.exists() and possible_path.is_file() else None,
        "ultralytics_version": ultralytics.__version__,
    }


def read_frame(capture: cv2.VideoCapture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return frame


def draw_false_positive_sample(frame, annotation: dict, detections: list[dict], false_positive_boxes: list[list], roi_px, path: Path):
    output = frame.copy()
    rx1, ry1, rx2, ry2 = roi_px
    cv2.rectangle(output, (rx1, ry1), (rx2, ry2), (255, 180, 0), 2)
    for person in annotation.get("people", []):
        x1, y1, x2, y2 = map(round, person["bbox_xyxy"])
        color = (0, 255, 255) if person.get("role") == "far_player" else (0, 200, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, f"GT:{person.get('id', '?')}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for detection in detections:
        box = detection["bbox_xyxy"]
        x1, y1, x2, y2 = map(round, box)
        is_false_positive = any(all(abs(a - b) < 0.01 for a, b in zip(box, false_box)) for false_box in false_positive_boxes)
        color = (0, 0, 255) if is_false_positive else (255, 255, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1 if not is_false_positive else 3)
        cv2.putText(output, f"{detection['confidence']:.2f}:{detection['source']}", (x1, min(output.shape[0] - 5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)


def write_blocker(output_dir: Path, reason: str, details: list[str] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked_missing_or_invalid_human_ground_truth",
        "reason": reason,
        "details": details or [],
        "next_command": "python evaluation/far_player/init_annotation_set.py --video <video.mp4> --output-dir <annotation_dir>",
    }
    (output_dir / "blocked.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        roi_norm = parse_roi(args.far_roi)
        metadata = video_metadata(args.video)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.annotations.exists():
        reason = f"human annotation file does not exist: {args.annotations}"
        write_blocker(args.output_dir, reason)
        print(f"BLOCKED: {reason}", file=sys.stderr)
        return 2
    try:
        annotations = load_jsonl(args.annotations)
    except ValueError as exc:
        write_blocker(args.output_dir, str(exc))
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    errors = validate_annotations(annotations, metadata["width"], metadata["height"], require_complete=True)
    far_ground_truth_count = sum(
        person.get("role") == "far_player" and person.get("visibility", "visible") in {"visible", "partial"}
        for row in annotations
        for person in row.get("people", [])
    )
    if far_ground_truth_count == 0:
        errors.append("no visible/partially visible far_player ground-truth boxes")
    for row_number, row in enumerate(annotations, 1):
        frame_index = row.get("frame_index")
        if isinstance(frame_index, int) and frame_index >= metadata["frame_count"]:
            errors.append(f"row {row_number}: frame_index {frame_index} is outside the video")
        expected_time = frame_index / metadata["fps"] if isinstance(frame_index, int) and metadata["fps"] > 0 else None
        if expected_time is not None and abs(float(row.get("time_sec", expected_time)) - expected_time) > 1.0 / metadata["fps"]:
            errors.append(f"row {row_number}: time_sec does not match frame_index/video FPS")

    annotation_metadata_path = args.annotations.parent / "metadata.json"
    annotation_metadata = None
    if not annotation_metadata_path.exists():
        errors.append(f"annotation metadata is missing: {annotation_metadata_path}")
    else:
        try:
            annotation_metadata = json.loads(annotation_metadata_path.read_text(encoding="utf-8"))
            expected_video_sha = annotation_metadata["video"]["sha256"]
            if expected_video_sha != metadata["sha256"]:
                errors.append("annotation metadata video SHA-256 does not match --video")
            metadata_review_status = annotation_metadata.get("review_status", annotation_metadata.get("status"))
            if metadata_review_status not in {"complete", "reviewed", "reviewed_preliminary"}:
                errors.append("annotation metadata review_status/status must be complete, reviewed, or reviewed_preliminary")
            if not annotation_metadata.get("ground_truth_source"):
                errors.append("annotation metadata ground_truth_source is required")
            if not annotation_metadata.get("reviewer"):
                errors.append("annotation metadata reviewer is required")
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"invalid annotation metadata: {exc}")
    if errors:
        write_blocker(args.output_dir, "annotations did not pass validation", errors)
        print("BLOCKED: annotations did not pass validation", file=sys.stderr)
        for error in errors[:20]:
            print(f"  - {error}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "blocked.json").unlink(missing_ok=True)
    runner = UltralyticsBenchmarkRunner(args.model, args.device, args.confidence, args.inference_iou, args.merge_iou)
    capture = cv2.VideoCapture(str(args.video))
    roi_px = roi_to_pixels(roi_norm, metadata["width"], metadata["height"])
    warmup_frame = read_frame(capture, int(annotations[0]["frame_index"]))
    runner.warmup(warmup_frame, roi_norm)
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "video": metadata,
        "annotations": {
            "path": str(args.annotations.resolve()),
            "sha256": sha256_file(args.annotations),
            "metadata_path": str(annotation_metadata_path.resolve()),
            "metadata_sha256": sha256_file(annotation_metadata_path),
            "reviewed_frame_count": len(annotations),
            "ground_truth_source": annotation_metadata.get("ground_truth_source"),
            "reviewer": annotation_metadata.get("reviewer"),
            "review_status": annotation_metadata.get("review_status", annotation_metadata.get("status")),
            "review_tier": (
                "human_reviewed"
                if annotation_metadata.get("ground_truth_source") in {"human_manual", "human_reviewed"}
                else "preliminary_reviewed"
            ),
            "sampling": annotation_metadata.get("sampling") if annotation_metadata else None,
        },
        "model": model_metadata(args.model, runner),
        "configuration": {
            "requested_device": args.device,
            "resolved_device": str(runner.device),
            "confidence_threshold": args.confidence,
            "inference_iou_threshold": args.inference_iou,
            "merge_iou_threshold": args.merge_iou,
            "matching_iou_threshold": args.matching_iou,
            "far_roi_normalized": list(roi_norm),
            "far_roi_xyxy": list(roi_px),
        },
        "methods": {},
    }

    annotation_by_frame = {int(row["frame_index"]): row for row in annotations}
    try:
        for method in args.methods:
            prediction_rows = []
            predictions_by_frame: dict[int, list[dict]] = {}
            elapsed_values = []
            for annotation in annotations:
                frame_index = int(annotation["frame_index"])
                frame = read_frame(capture, frame_index)
                output = runner.run(frame, method, roi_norm)
                predictions_by_frame[frame_index] = output.detections
                elapsed_values.append(output.elapsed_ms)
                prediction_rows.append(
                    {
                        "frame_index": frame_index,
                        "time_sec": annotation.get("time_sec"),
                        "elapsed_ms": output.elapsed_ms,
                        "detections": output.detections,
                    }
                )
            safe_method = method.replace("+", "_")
            write_jsonl(args.output_dir / f"predictions_{safe_method}.jsonl", prediction_rows)
            evaluated = evaluate_predictions(annotations, predictions_by_frame, args.matching_iou)
            timing = {
                "evaluated_frame_count": len(elapsed_values),
                "mean_ms_per_frame": mean(elapsed_values) if elapsed_values else None,
                "median_ms_per_frame": median(elapsed_values) if elapsed_values else None,
                "p95_ms_per_frame": percentile(elapsed_values, 0.95),
            }
            fp_dir = args.output_dir / "false_positives" / safe_method
            saved_samples = []
            fp_by_frame: dict[int, list[list]] = {}
            for false_positive in evaluated.false_positives:
                fp_by_frame.setdefault(false_positive["frame_index"], []).append(false_positive["bbox_xyxy"])
            for frame_index, boxes in list(fp_by_frame.items())[: args.max_fp_images]:
                frame = read_frame(capture, frame_index)
                path = fp_dir / f"frame_{frame_index:08d}.jpg"
                draw_false_positive_sample(frame, annotation_by_frame[frame_index], predictions_by_frame[frame_index], boxes, roi_px, path)
                saved_samples.append(str(path.relative_to(args.output_dir)))
            report["methods"][method] = {
                "inference_plan": (
                    [{"source": "full_frame", "imgsz": 640}]
                    if method == "full_640"
                    else [{"source": "full_frame", "imgsz": 1280}]
                    if method == "full_1280"
                    else [
                        {"source": "full_frame", "imgsz": 640},
                        {"source": "far_roi", "imgsz": 640, "roi_xyxy": list(roi_px)},
                    ]
                ),
                "metrics": evaluated.metrics,
                "timing": timing,
                "misses": evaluated.misses,
                "false_positives": evaluated.false_positives,
                "false_positive_sample_images": saved_samples,
            }
    finally:
        capture.release()

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "method", "far_player_recall", "far_player_miss_rate", "longest_consecutive_miss_samples",
            "longest_consecutive_miss_seconds", "mean_ms_per_frame", "p95_ms_per_frame", "false_positive_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in report["methods"].items():
            writer.writerow({
                "method": method,
                **{key: result["metrics"].get(key) for key in fields if key in result["metrics"]},
                "mean_ms_per_frame": result["timing"]["mean_ms_per_frame"],
                "p95_ms_per_frame": result["timing"]["p95_ms_per_frame"],
            })
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
