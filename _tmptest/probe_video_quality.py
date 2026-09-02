"""诊断：抽帧是否损伤画质 + 源视频清晰度量化（一次性脚本）。

对每个视频：
1. 报告实际分辨率 / fps / 时长 / 估算码率（文件大小×8/时长）
2. 对若干抽帧：计算 Laplacian 方差（清晰度指标，越小越糊）
3. 同一时间点：视频原帧 vs 已抽出的 jpg 做 PSNR（>40dB 视为抽帧无损）
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2

VIDEOS = [
    r"S:\Software Tool\bilibili-download\DownKyi-1.6.1\Media\9分逆转国际球员！什么叫世界冠军！什么叫法拉利啊！.mp4",
    r"S:\Software Tool\bilibili-download\DownKyi-1.6.1\Media\【4k60帧】王正行，南京萝卜双打全场回放！.mp4",
    r"S:\Software Tool\bilibili-download\DownKyi-1.6.1\Media\低视角（韩国女双）2016 羽毛球比赛外录现场 女双 女子双打 羽毛球比赛 体育运动.mp4",
]
FRAMES_DIR = Path(r"S:\Software Tool\bilibili-download\DownKyi-1.6.1\Media\editedFrames")


def lap_var(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 99.9
    return 10 * np.log10(255.0 ** 2 / mse)


print("=== 1) 源视频实际参数 ===")
for v in VIDEOS:
    p = Path(v)
    if not p.is_file():
        print(f"[skip] 不存在 {p}")
        continue
    cap = cv2.VideoCapture(v)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    dur = n / fps if fps else 0
    mbps = p.stat().st_size * 8 / (dur * 1_000_000) if dur else 0
    print(f"{p.name[:28]:30s} {w}x{h} {fps:.0f}fps 时长{dur:.0f}s 码率{mbps:.1f}Mbps")

print()
print("=== 2) 抽帧清晰度(Laplacian方差,越大越清晰) + 3) 与源帧PSNR(>40=抽帧无损) ===")
for v in VIDEOS:
    p = Path(v)
    if not p.is_file():
        continue
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stem = p.stem
    times = [0.0, 5.0, 10.0, 15.0, 20.0]  # 每视频取5个时间点
    print(f"\n-- {p.name[:28]}")
    for t in times:
        jpg = FRAMES_DIR / f"{stem}__{t:07.2f}s.jpg"
        if not jpg.is_file():
            continue
        idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, src = cap.read()
        if not ok:
            continue
        ext = cv2.imdecode(np.fromfile(str(jpg), dtype=np.uint8), cv2.IMREAD_COLOR)
        if ext is None:
            continue
        src = cv2.resize(src, (ext.shape[1], ext.shape[0])) if src.shape != ext.shape else src
        print(f"  t={t:5.1f}s 帧: 清晰度={lap_var(ext):7.1f}  源帧清晰度={lap_var(src):7.1f}  PSNR={psnr(ext, src):5.1f}dB")
    cap.release()
