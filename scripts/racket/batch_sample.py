"""批量抽帧：按每视频目标帧数自动计算间隔，均匀覆盖整场视频（任务 L）。

对完整比赛视频（可能长达数十分钟）用固定小间隔会抽出上千帧，
本脚本按「每视频目标帧数」反推间隔（interval = 时长 / 目标帧数），
保证均匀覆盖整场，不会只在开头取样。

用法：
    python scripts/racket/batch_sample.py videos_list.txt -o <输出目录> --per-video 60

videos_list.txt 为 UTF-8 文本，每行一个视频路径，支持 # 注释。
输出：
    <out>/<视频源>__<时间秒>s.jpg   抽帧图片（原分辨率）
    <out>/manifest_raw.csv          帧清单：source_video, frame, time_sec
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sample_frames import VIDEO_EXTS, extract_video

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def read_list_file(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return [Path(p) for p in paths]


def probe(video: Path) -> tuple[float, int] | None:
    """返回 (fps, 总帧数)；无法打开返回 None。"""
    if cv2 is None:  # pragma: no cover
        raise SystemExit("缺少 opencv-python，请先安装：pip install opencv-python")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), total


def main() -> None:
    parser = argparse.ArgumentParser(description="按目标帧数批量抽帧，均匀覆盖整场")
    parser.add_argument("list_file", help="UTF-8 文本文件，每行一个视频路径")
    parser.add_argument("-o", "--out", required=True, help="输出目录")
    parser.add_argument("--per-video", type=int, default=60, help="每视频目标帧数，默认 60")
    parser.add_argument("--min-interval", type=float, default=1.0, help="最小间隔（秒），默认 1.0")
    parser.add_argument("--quality", type=int, default=95, help="JPEG 质量 1-100，默认 95")
    args = parser.parse_args()

    if args.per_video < 1:
        raise SystemExit("--per-video 必须 >= 1。")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality 必须在 1-100。")

    list_file = Path(args.list_file)
    if not list_file.is_file():
        raise SystemExit(f"清单文件不存在: {list_file}")

    videos = read_list_file(list_file)
    missing = [p for p in videos if not p.is_file()]
    for m in missing:
        print(f"[warn] 文件不存在，跳过: {m}", file=sys.stderr)
    videos = [p for p in videos if p.is_file()]
    if not videos:
        raise SystemExit("清单中没有可用视频。")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    total_frames = 0
    print(f"共 {len(videos)} 个视频，目标每场 {args.per_video} 帧，输出到 {out_dir}")
    for video in videos:
        info = probe(video)
        if info is None:
            print(f"[warn] 无法打开视频，跳过: {video}", file=sys.stderr)
            continue
        fps, total = info
        duration = total / fps if total > 0 else 0.0
        if duration <= 0:
            print(f"[warn] 无法读取时长，跳过: {video}", file=sys.stderr)
            continue
        interval = max(args.min_interval, duration / max(1, args.per_video))
        print(f"{video.name}: 时长 {duration:.0f}s -> 间隔 {interval:.1f}s")
        total_frames += extract_video(
            video,
            out_dir,
            interval=interval,
            max_frames=0,
            quality=args.quality,
            manifest_rows=manifest_rows,
        )

    manifest_path = out_dir / "manifest_raw.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_video", "frame", "time_sec"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"完成：共 {total_frames} 张，清单 -> {manifest_path}")


if __name__ == "__main__":
    main()
