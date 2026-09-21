#!/usr/bin/env python
"""Run the official TrackNetV3 checkout twice: raw first, rectified second.

This project intentionally does not vendor TrackNetV3 or its checkpoints.  It
keeps upstream model code, its GPU-oriented environment, and all model weights
outside the Good-Badminton repository while retaining reproducible commands.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--tracknet-root", required=True, type=Path, help="Official TrackNetV3 checkout")
    parser.add_argument("--tracknet-python", default=sys.executable, help="Python in TrackNetV3's compatible environment")
    parser.add_argument("--tracknet-checkpoint", required=True, type=Path)
    parser.add_argument("--inpaint-checkpoint", type=Path, help="Optional InpaintNet checkpoint for B*")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-mode", choices=("nonoverlap", "average", "weight"), default="weight")
    parser.add_argument(
        "--fast-predictor",
        type=Path,
        help="Good-Badminton TrackNet adapter with sampled background and visible progress.",
    )
    parser.add_argument("--background-sample-count", type=int, default=120)
    parser.add_argument("--chunk-frames", type=int, default=96)
    parser.add_argument(
        "--run-rectified",
        action="store_true",
        help="Run the slow upstream InpaintNet B* pass after raw B has been validated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predict_script = args.tracknet_root / "predict.py"
    for path, name in ((args.video, "video"), (predict_script, "TrackNetV3 predict.py"), (args.tracknet_checkpoint, "TrackNet checkpoint")):
        if not path.exists():
            return _error(f"{name} does not exist: {path}")
    if args.inpaint_checkpoint and not args.inpaint_checkpoint.exists():
        return _error(f"Inpaint checkpoint does not exist: {args.inpaint_checkpoint}")
    if args.batch_size <= 0:
        return _error("--batch-size must be greater than zero")
    if args.background_sample_count <= 0:
        return _error("--background-sample-count must be greater than zero")
    if args.chunk_frames <= 0:
        return _error("--chunk-frames must be greater than zero")
    if args.fast_predictor and not args.fast_predictor.is_file():
        return _error(f"Fast TrackNet predictor does not exist: {args.fast_predictor}")

    raw_dir = args.output_dir / "tracknet_raw"
    rectified_dir = args.output_dir / "tracknet_rectified"
    raw_csv = raw_dir / _csv_name(args.video)
    rectified_csv = rectified_dir / _csv_name(args.video)
    if not args.overwrite and (raw_csv.exists() or rectified_csv.exists()):
        return _error(f"{args.output_dir} already contains predictions; use --overwrite intentionally")

    _run_predict(args, raw_dir, inpaint_checkpoint=None)
    if not raw_csv.exists():
        return _error(f"TrackNetV3 returned successfully but did not write {raw_csv}")
    print(f"raw_csv={raw_csv}")

    if args.inpaint_checkpoint and args.run_rectified:
        _run_predict(args, rectified_dir, inpaint_checkpoint=args.inpaint_checkpoint)
        if not rectified_csv.exists():
            return _error(f"TrackNetV3 returned successfully but did not write {rectified_csv}")
        print(f"rectified_csv={rectified_csv}")
    else:
        print("rectified_csv=not_requested")
    return 0


def _run_predict(args: argparse.Namespace, output_dir: Path, *, inpaint_checkpoint: Path | None) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_fast_predictor = args.fast_predictor is not None and inpaint_checkpoint is None
    predictor = args.fast_predictor if use_fast_predictor else args.tracknet_root / "predict.py"
    command = [
        args.tracknet_python,
        str(predictor.resolve()),
        "--video-file" if use_fast_predictor else "--video_file",
        args.video.resolve().as_posix(),
        "--tracknet-file" if use_fast_predictor else "--tracknet_file",
        str(args.tracknet_checkpoint.resolve()),
        "--save-dir" if use_fast_predictor else "--save_dir",
        str(output_dir.resolve()),
        "--batch-size" if use_fast_predictor else "--batch_size",
        str(args.batch_size),
        "--eval-mode" if use_fast_predictor else "--eval_mode",
        args.eval_mode,
    ]
    if use_fast_predictor:
        command.extend((
            "--background-sample-count", str(args.background_sample_count),
            "--chunk-frames", str(getattr(args, "chunk_frames", 96)),
        ))
    if inpaint_checkpoint:
        command.extend(("--inpaintnet_file", str(inpaint_checkpoint.resolve())))
    runtime_env = None
    if use_fast_predictor:
        # Python uses the directory of an absolute script path as sys.path[0].
        # Keep the official TrackNetV3 checkout explicitly importable so the
        # adapter can reuse its predict/dataset/model modules.
        runtime_env = os.environ.copy()
        upstream_path = str(args.tracknet_root.resolve())
        inherited_path = runtime_env.get("PYTHONPATH", "")
        runtime_env["PYTHONPATH"] = os.pathsep.join(
            value for value in (upstream_path, inherited_path) if value
        )
    print("Running:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=args.tracknet_root, check=True, env=runtime_env)


def _csv_name(video: Path) -> str:
    return f"{video.stem}_ball.csv"


def _error(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
