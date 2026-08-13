#!/usr/bin/env python
"""Extract reproducible review frames and create an empty annotation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.far_player.annotations import write_jsonl
from evaluation.far_player.detector_runner import parse_roi, roi_to_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, help="Limit extraction to a continuous time window")
    parser.add_argument("--max-samples", type=int, help="Hard cap after time-window sampling")
    parser.add_argument(
        "--frame-indices",
        help="Explicit comma-separated source frame indices; overrides start/duration/sample-fps",
    )
    parser.add_argument("--far-roi", default="0.0,0.0,1.0,0.55")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    annotation_path = args.output_dir / "annotations.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    if not args.video.exists():
        print(f"ERROR: video does not exist: {args.video}", file=sys.stderr)
        return 2
    if (annotation_path.exists() or metadata_path.exists()) and not args.overwrite:
        print(f"ERROR: {args.output_dir} already contains an annotation set; use --overwrite intentionally", file=sys.stderr)
        return 2
    if args.sample_fps <= 0:
        print("ERROR: --sample-fps must be greater than zero", file=sys.stderr)
        return 2
    if args.start_sec < 0 or (args.duration_sec is not None and args.duration_sec <= 0):
        print("ERROR: --start-sec must be >= 0 and --duration-sec must be > 0", file=sys.stderr)
        return 2
    if args.max_samples is not None and args.max_samples <= 0:
        print("ERROR: --max-samples must be greater than zero", file=sys.stderr)
        return 2
    try:
        roi_norm = parse_roi(args.far_roi)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"ERROR: cannot open video: {args.video}", file=sys.stderr)
        return 2
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        print("ERROR: video FPS/frame count is invalid", file=sys.stderr)
        return 2
    step = max(1, round(fps / args.sample_fps))
    if args.frame_indices:
        try:
            frame_indices = sorted({int(value.strip()) for value in args.frame_indices.split(",") if value.strip()})
        except ValueError:
            capture.release()
            print("ERROR: --frame-indices must be comma-separated integers", file=sys.stderr)
            return 2
        if not frame_indices or frame_indices[0] < 0 or frame_indices[-1] >= frame_count:
            capture.release()
            print(f"ERROR: --frame-indices must be within 0..{frame_count - 1}", file=sys.stderr)
            return 2
        sampling_mode = "explicit_frame_indices"
    else:
        start_frame = min(frame_count, round(args.start_sec * fps))
        end_frame = frame_count
        if args.duration_sec is not None:
            end_frame = min(frame_count, start_frame + round(args.duration_sec * fps))
        frame_indices = list(range(start_frame, end_frame, step))
        sampling_mode = "continuous_time_window"
    if args.max_samples is not None:
        frame_indices = frame_indices[: args.max_samples]
    if not frame_indices:
        capture.release()
        print("ERROR: selected time window contains no frames", file=sys.stderr)
        return 2
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            print(f"ERROR: failed to read frame {frame_index}", file=sys.stderr)
            return 2
        image_name = f"frame_{frame_index:08d}.jpg"
        cv2.imwrite(str(frames_dir / image_name), frame)
        rows.append(
            {
                "frame_index": frame_index,
                "time_sec": round(frame_index / fps, 6),
                "image": f"frames/{image_name}",
                "label_status": "unlabeled",
                "people_annotation_complete": False,
                "people": [],
                "ignore_regions": [],
            }
        )
    capture.release()
    write_jsonl(annotation_path, rows)
    metadata = {
        "schema_version": "1.0",
        "ground_truth_source": "unreviewed_template",
        "reviewer": None,
        "review_status": "unreviewed",
        "video": {
            "path": str(args.video.resolve()),
            "filename": args.video.name,
            "size_bytes": args.video.stat().st_size,
            "sha256": sha256_file(args.video),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": frame_count / fps,
        },
        "sampling": {
            "mode": sampling_mode,
            "requested_fps": args.sample_fps if not args.frame_indices else None,
            "source_frame_step": step if not args.frame_indices else None,
            "actual_fps": fps / step if not args.frame_indices else None,
            "sample_count": len(rows),
            "start_sec": args.start_sec if not args.frame_indices else None,
            "duration_sec": args.duration_sec if not args.frame_indices else None,
            "max_samples": args.max_samples,
            "frame_indices": frame_indices,
        },
        "far_roi_normalized": list(roi_norm),
        "far_roi_xyxy": list(roi_to_pixels(roi_norm, width, height)),
        "instructions": "Review every image, annotate every person box, identify far_player, then set label_status=complete and people_annotation_complete=true.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(annotation_path)
    print("Ground truth is intentionally UNLABELED; benchmark will refuse to run until human review is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
