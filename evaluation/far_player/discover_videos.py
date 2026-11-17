#!/usr/bin/env python
"""Discover candidate videos and report resolution without claiming benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


SKIP_DIRS = {".git", ".venv", "__pycache__"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidates = []
    for path in args.root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts) or not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            continue
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if width <= 0 or height <= 0 or frames <= 0:
            continue
        candidates.append(
            {
                "path": str(path.resolve()),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frames,
                "duration_sec": frames / fps if fps > 0 else None,
                "is_852x480_candidate": width == 852 and height == 480,
                "is_higher_resolution_candidate": width > 852 and height > 480,
                "ground_truth_status": "unknown; human annotation required",
            }
        )
    candidates.sort(key=lambda item: (not item["is_852x480_candidate"], -(item["width"] * item["height"]), item["path"]))
    payload = {"candidate_count": len(candidates), "videos": candidates}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
