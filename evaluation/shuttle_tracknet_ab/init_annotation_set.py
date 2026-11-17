#!/usr/bin/env python
"""Extract reproducible shuttlecock-review frames for the TrackNet A/B test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.shuttle_tracknet_ab.annotations import sha256_file, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, help="Continuous time window to extract")
    parser.add_argument("--max-samples", type=int, help="Hard cap after sampling")
    parser.add_argument("--frame-indices", help="Explicit source frame indices; overrides time-window sampling")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations_path = args.output_dir / "annotations.jsonl"
    metadata_path = args.output_dir / "metadata.json"
    if not args.video.exists():
        return _error(f"video does not exist: {args.video}")
    if (annotations_path.exists() or metadata_path.exists()) and not args.overwrite:
        return _error(f"{args.output_dir} already has an annotation set; use --overwrite intentionally")
    if args.sample_fps <= 0 or args.start_sec < 0:
        return _error("--sample-fps must be > 0 and --start-sec must be >= 0")
    if args.duration_sec is not None and args.duration_sec <= 0:
        return _error("--duration-sec must be > 0")
    if args.max_samples is not None and args.max_samples <= 0:
        return _error("--max-samples must be > 0")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        return _error(f"cannot open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        capture.release()
        return _error("video metadata is invalid")

    try:
        frame_indices, sampling = _frame_indices(args, fps, frame_count)
    except ValueError as exc:
        capture.release()
        return _error(str(exc))
    if args.max_samples is not None:
        frame_indices = frame_indices[: args.max_samples]
    if not frame_indices:
        capture.release()
        return _error("selected video window contains no frames")

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            return _error(f"failed to read source frame {frame_index}")
        image_name = f"frame_{frame_index:08d}.jpg"
        if not cv2.imwrite(str(frames_dir / image_name), frame):
            capture.release()
            return _error(f"failed to write review image {image_name}")
        rows.append(
            {
                "frame_index": frame_index,
                "time_sec": round(frame_index / fps, 6),
                "image": f"frames/{image_name}",
                "label_status": "unlabeled",
                "shuttle": {"visibility": "ambiguous", "image_xy": None},
            }
        )
    capture.release()
    write_jsonl(annotations_path, rows)
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
        "sampling": {**sampling, "sample_count": len(rows), "frame_indices": frame_indices},
        "instructions": (
            "Review each image yourself. visible requires the shuttle centre in original-video pixels; "
            "not_visible means it is absent, hidden, or outside the frame; ambiguous is excluded. "
            "When every row is reviewed set label_status=complete, ground_truth_source=human_manual, "
            "review_status=complete, and reviewer to the human reviewer."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(annotations_path)
    print("Ground truth is intentionally unlabeled; the A/B benchmark will block until human review is complete.")
    return 0


def _frame_indices(args: argparse.Namespace, fps: float, frame_count: int) -> tuple[list[int], dict]:
    if args.frame_indices:
        try:
            indices = sorted({int(value.strip()) for value in args.frame_indices.split(",") if value.strip()})
        except ValueError as exc:
            raise ValueError("--frame-indices must be comma-separated non-negative integers") from exc
        if not indices or indices[0] < 0 or indices[-1] >= frame_count:
            raise ValueError(f"--frame-indices must be within 0..{frame_count - 1}")
        return indices, {"mode": "explicit_frame_indices", "requested_fps": None}
    step = max(1, round(fps / args.sample_fps))
    start = min(frame_count, round(args.start_sec * fps))
    end = frame_count if args.duration_sec is None else min(frame_count, start + round(args.duration_sec * fps))
    return list(range(start, end, step)), {
        "mode": "continuous_time_window",
        "requested_fps": args.sample_fps,
        "actual_fps": fps / step,
        "source_frame_step": step,
        "start_sec": args.start_sec,
        "duration_sec": args.duration_sec,
    }


def _error(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
