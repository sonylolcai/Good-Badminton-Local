"""球拍数据集抽帧工具（任务 L / 标注规范 v1.1 §4）。

从比赛视频中按时间间隔均匀采样帧，供球拍 5 关键点标注使用。

规则（对齐标注规范）：
- 帧间隔 >= 0.5 秒（默认 2.0 秒，可用 --interval 调整，禁止相邻帧）；
- 保留原始分辨率，不降采样；
- 输出文件名携带视频源与时间戳（如 matchA__0012.50s.jpg），
  后续按视频切分（split_by_video）时以视频源分组，防相邻帧泄漏。

用法示例：
    python scripts/racket/sample_frames.py videos/ -o datasets/racket_v1/images/raw --interval 2
    python scripts/racket/sample_frames.py matchA.mp4 matchB.mp4 -o raw --max-frames 60

输出：
    <out>/<视频源>__<时间秒>s.jpg         抽帧图片（JPEG，原分辨率）
    <out>/manifest_raw.csv                帧清单：source_video, frame, time_sec

批量按目标帧数抽帧（自动算间隔，均匀覆盖整场）用 batch_sample.py。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".m2ts", ".webm"}

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def collect_videos(paths: list[str]) -> list[Path]:
    """展开文件/目录输入为视频文件列表，按路径排序保证可复现。"""
    videos: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            videos.extend(
                sorted(child for child in p.rglob("*") if child.suffix.lower() in VIDEO_EXTS)
            )
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
        else:
            print(f"[warn] 跳过非视频输入: {p}", file=sys.stderr)
    if not videos:
        raise SystemExit("未找到任何视频文件。")
    return sorted(set(videos), key=lambda p: str(p).lower())


def extract_video(
    video: Path,
    out_dir: Path,
    *,
    interval: float,
    max_frames: int,
    quality: int,
    manifest_rows: list[dict[str, str]],
) -> int:
    """从单个视频按间隔抽帧，写入 out_dir，并追加清单行。返回抽帧数。

    cv2.imwrite 在 Windows 上不支持中文/非 ASCII 路径，
    因此用 imencode + 字节写入，兼容任意合法文件名。
    """
    if cv2 is None:  # pragma: no cover
        raise SystemExit("缺少 opencv-python，请先安装：pip install opencv-python")

    stem = video.stem
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[warn] 无法打开视频，跳过: {video}", file=sys.stderr)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, round(interval * fps))
    print(f"  {video.name}: fps={fps:.2f}, 总帧数={total}, 步长={frame_step} 帧")

    count = 0
    idx = 0
    while True:
        if max_frames and count >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        time_sec = idx / fps
        name = f"{stem}__{time_sec:07.2f}s.jpg"
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            print(f"[warn] 编码失败: {name}", file=sys.stderr)
        else:
            (out_dir / name).write_bytes(encoded.tobytes())
        manifest_rows.append(
            {"source_video": stem, "frame": name, "time_sec": f"{time_sec:.2f}"}
        )
        count += 1
        idx += frame_step

    cap.release()
    print(f"  {video.name}: 抽帧 {count} 张")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="按时间间隔从比赛视频均匀抽帧（球拍数据集）")
    parser.add_argument("videos", nargs="+", help="视频文件或包含视频的目录")
    parser.add_argument("-o", "--out", default="datasets/racket_v1/images/raw", help="输出目录")
    parser.add_argument("--interval", type=float, default=2.0, help="抽帧间隔（秒），默认 2.0，必须 >= 0.5")
    parser.add_argument("--max-frames", type=int, default=0, help="每视频最大抽帧数，0=不限")
    parser.add_argument("--quality", type=int, default=95, help="JPEG 质量 1-100，默认 95")
    args = parser.parse_args()

    if args.interval < 0.5:
        raise SystemExit("--interval 必须 >= 0.5 秒（标注规范禁止相邻帧）。")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality 必须在 1-100。")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(args.videos)
    print(f"发现 {len(videos)} 个视频，间隔 {args.interval}s，输出到 {out_dir}")

    manifest_rows: list[dict[str, str]] = []
    total_frames = 0
    for video in videos:
        total_frames += extract_video(
            video,
            out_dir,
            interval=args.interval,
            max_frames=args.max_frames,
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
