#!/usr/bin/env python3
"""Render TrackNetV3 shuttlecock detections and YOLO-Pose + ByteTrack player tracking.

Combines:
1. TrackNetV3 temporal heatmap ball measurements (CSV).
2. YOLO11n-Pose + ByteTrack persistent player tracking (IDs, skeletons, footprints, movement paths).
3. Professional HUD dashboard and player ID color legend.
4. Muxed original audio into browser-compatible H.264 MP4.
"""

import argparse
import csv
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# High-contrast distinctive colors for players (BGR)
PLAYER_PALETTE = [
    (102, 255, 0),    # Bright Green (ID 1)
    (255, 229, 0),    # Bright Cyan (ID 2)
    (0, 145, 255),    # Bright Orange (ID 3)
    (251, 64, 224),   # Bright Magenta (ID 4)
    (0, 255, 255),    # Yellow (ID 5)
    (255, 128, 0),    # Dodger Blue (ID 6)
    (200, 200, 200),  # Light Gray
]

def get_player_color(track_id):
    if track_id is None:
        return (180, 180, 180)
    idx = (int(track_id) - 1) % len(PLAYER_PALETTE)
    return PLAYER_PALETTE[idx]


def load_tracknet_csv(csv_path):
    measurements = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(float(row["Frame"]))
            vis = int(float(row["Visibility"])) != 0
            x = float(row["X"])
            y = float(row["Y"])
            measurements[f_idx] = (vis, x, y)
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-file", required=True)
    parser.add_argument("--tracknet-csv", required=True)
    parser.add_argument("--pose-model", default="weights/yolo11n-pose.pt")
    parser.add_argument("--output-dir", default="outputs/tracknet_eval")
    parser.add_argument("--output-scale", type=float, default=0.5)
    parser.add_argument("--pose-imgsz", type=int, default=960)
    parser.add_argument("--pose-conf", type=float, default=0.15)
    args = parser.parse_args()

    video_path = Path(args.video_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_mp4 = output_dir / f"eval_{video_path.stem}_tracknet_bytetrack.mp4"
    raw_video_tmp = output_dir / f"tmp_{video_path.stem}_render.mp4"
    metrics_json = output_dir / f"eval_{video_path.stem}_tracknet_metrics.json"

    print("=" * 60)
    print(f"Rendering TrackNet + ByteTrack Video for {video_path.name}")
    print("=" * 60)

    # 1. Load TrackNet CSV
    print(f"Loading TrackNet measurements from {args.tracknet_csv}...")
    tracknet_data = load_tracknet_csv(args.tracknet_csv)
    vis_count = sum(1 for vis, _, _ in tracknet_data.values() if vis)
    print(f"  Loaded {len(tracknet_data)} frames, {vis_count} visible ball frames ({vis_count/len(tracknet_data)*100:.1f}%)")

    # 2. Load YOLO Pose
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading YOLO Pose model on {dev}...")
    pose_model = YOLO(args.pose_model)
    # Warmup
    pose_model(np.zeros((720, 1280, 3), dtype=np.uint8), device=dev, imgsz=args.pose_imgsz, verbose=False)

    cap = cv2.VideoCapture(str(video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.57
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = int(orig_w * args.output_scale)
    out_h = int(orig_h * args.output_scale)
    out_w = out_w if out_w % 2 == 0 else out_w - 1
    out_h = out_h if out_h % 2 == 0 else out_h - 1

    scale_x = out_w / orig_w
    scale_y = out_h / orig_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_video_tmp), fourcc, fps, (out_w, out_h))

    # Ball trajectory queue (last 25 points in output resolution)
    ball_trail = deque(maxlen=25)

    # Player movement path history: track_id -> deque of (x, y)
    player_paths = {}

    # Skeleton connections
    skeleton_pairs = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    t_render_start = time.perf_counter()
    frame_idx = 0

    print(f"\nProcessing {total_frames} frames to {out_w}x{out_h}...")

    # Statistics accumulators
    track_id_counts = {}
    frames_with_ball = 0

    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run YOLO-Pose with ByteTrack
        pose_res = pose_model.track(
            frame,
            persist=True,
            device=dev,
            imgsz=args.pose_imgsz,
            conf=args.pose_conf,
            verbose=False,
        )[0]

        # Resize frame for drawing
        draw_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # 2. Draw TrackNet Ball & Trajectory
        ball_info = tracknet_data.get(frame_idx, (False, 0, 0))
        ball_vis, bx_orig, by_orig = ball_info

        if ball_vis and bx_orig > 0 and by_orig > 0:
            bx = int(bx_orig * scale_x)
            by = int(by_orig * scale_y)
            ball_trail.append((bx, by, frame_idx))
            frames_with_ball += 1
        else:
            # If not detected for several frames, decay trail
            if ball_trail and frame_idx - ball_trail[-1][2] > 6:
                ball_trail.popleft()

        # Draw ball trail
        if len(ball_trail) > 1:
            pts = list(ball_trail)
            for i in range(1, len(pts)):
                p1 = (pts[i - 1][0], pts[i - 1][1])
                p2 = (pts[i][0], pts[i][1])
                progress = i / len(pts)
                # Color gradient: Orange to Neon Yellow
                trail_color = (
                    int(30 + 100 * progress),
                    int(150 + 105 * progress),
                    int(255)
                )
                thick = max(1, int(1 + progress * 3.5))
                cv2.line(draw_frame, p1, p2, trail_color, thick, cv2.LINE_AA)

        # Draw current ball marker
        if ball_vis and bx_orig > 0 and by_orig > 0:
            bx = int(bx_orig * scale_x)
            by = int(by_orig * scale_y)
            # Glowing outer ring
            cv2.circle(draw_frame, (bx, by), 8, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.circle(draw_frame, (bx, by), 4, (0, 255, 255), -1, cv2.LINE_AA)
            # Crosshair
            cv2.line(draw_frame, (bx - 12, by), (bx + 12, by), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(draw_frame, (bx, by - 12), (bx, by + 12), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(draw_frame, "SHUTTLE", (bx + 12, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Players with ByteTrack ID & Skeletons
        current_active_ids = []
        if pose_res.boxes is not None and len(pose_res.boxes) > 0:
            boxes = pose_res.boxes.xyxy.cpu().numpy()
            confs = pose_res.boxes.conf.cpu().numpy()
            track_ids = (
                pose_res.boxes.id.int().cpu().tolist()
                if pose_res.boxes.id is not None
                else [None] * len(boxes)
            )
            kpts = (
                pose_res.keypoints.xy.cpu().numpy()
                if pose_res.keypoints is not None
                else None
            )
            kpts_conf = (
                pose_res.keypoints.conf.cpu().numpy()
                if pose_res.keypoints is not None and pose_res.keypoints.conf is not None
                else None
            )

            for p_idx in range(len(boxes)):
                box = boxes[p_idx]
                p_conf = float(confs[p_idx])
                tid = track_ids[p_idx]
                if tid is not None:
                    current_active_ids.append(tid)
                    track_id_counts[tid] = track_id_counts.get(tid, 0) + 1

                p_color = get_player_color(tid)

                bx1 = int(box[0] * scale_x)
                by1 = int(box[1] * scale_y)
                bx2 = int(box[2] * scale_x)
                by2 = int(box[3] * scale_y)

                # Draw player bounding box
                cv2.rectangle(draw_frame, (bx1, by1), (bx2, by2), p_color, 2, cv2.LINE_AA)

                # Player label pill with background
                label_txt = f"ID #{tid}" if tid is not None else "Player"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(draw_frame, (bx1, max(0, by1 - th - 8)), (bx1 + tw + 10, max(th + 8, by1)), p_color, -1)
                cv2.putText(draw_frame, label_txt, (bx1 + 5, max(th + 4, by1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

                # Draw skeleton
                if kpts is not None and p_idx < len(kpts):
                    pts = kpts[p_idx]
                    pts_scaled = []
                    feet_x = []
                    feet_y = []

                    for i in range(17):
                        px = int(pts[i][0] * scale_x)
                        py = int(pts[i][1] * scale_y)
                        vis = (pts[i][0] > 0 and pts[i][1] > 0)
                        if kpts_conf is not None and i < len(kpts_conf[p_idx]):
                            if kpts_conf[p_idx][i] < 0.20:
                                vis = False
                        pts_scaled.append((px, py, vis))
                        if vis and i in (15, 16): # Left/Right ankle
                            feet_x.append(px)
                            feet_y.append(py)

                    # Draw bones in player's unique color
                    for p1_i, p2_i in skeleton_pairs:
                        x1, y1, v1 = pts_scaled[p1_i]
                        x2, y2, v2 = pts_scaled[p2_i]
                        if v1 and v2:
                            cv2.line(draw_frame, (x1, y1), (x2, y2), p_color, 2, cv2.LINE_AA)

                    # Draw joints (white center with player color ring)
                    for px, py, vis in pts_scaled:
                        if vis:
                            cv2.circle(draw_frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(draw_frame, (px, py), 4, p_color, 1, cv2.LINE_AA)

                    # Footprint position & trajectory
                    if feet_x and feet_y and tid is not None:
                        ground_x = int(np.mean(feet_x))
                        ground_y = int(np.mean(feet_y))
                        cv2.ellipse(draw_frame, (ground_x, ground_y), (14, 6), 0, 0, 360, p_color, 2, cv2.LINE_AA)

                        if tid not in player_paths:
                            player_paths[tid] = deque(maxlen=30)
                        player_paths[tid].append((ground_x, ground_y))

        # Draw player movement trails on court floor
        for tid, path in player_paths.items():
            if len(path) > 1:
                p_col = get_player_color(tid)
                path_pts = list(path)
                for pi in range(1, len(path_pts)):
                    cv2.line(draw_frame, path_pts[pi-1], path_pts[pi], p_col, 1, cv2.LINE_AA)

        # 4. HUD Top Bar
        hud_h = 38
        overlay = draw_frame.copy()
        cv2.rectangle(overlay, (0, 0), (out_w, hud_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, draw_frame, 0.25, 0, draw_frame)

        time_sec = frame_idx / fps
        m = int(time_sec // 60)
        s = int(time_sec % 60)
        ms = int((time_sec - int(time_sec)) * 100)

        left_str = f"🏸 TrackNetV3 (Temporal Heatmap) + YOLO-Pose (ByteTrack)"
        mid_str = f"Frame: {frame_idx:04d}/{total_frames:04d} ({m:02d}:{s:02d}.{ms:02d})"
        ids_str = ", ".join(f"#{i}" for i in sorted(current_active_ids)) if current_active_ids else "None"
        right_str = f"Track IDs: [{ids_str}] | Ball: {'IN FLIGHT' if ball_vis else 'LOST'}"

        cv2.putText(draw_frame, left_str, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(draw_frame, mid_str, (out_w // 2 - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(draw_frame, right_str, (out_w - 420, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (50, 255, 100) if ball_vis else (200, 200, 200), 1, cv2.LINE_AA)

        # 5. Bottom Legend Panel
        leg_w = 340
        leg_h = 32
        leg_x = 15
        leg_y = out_h - leg_h - 15
        leg_overlay = draw_frame.copy()
        cv2.rectangle(leg_overlay, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (20, 20, 20), -1)
        cv2.addWeighted(leg_overlay, 0.7, draw_frame, 0.3, 0, draw_frame)

        # Legend circles
        curr_x = leg_x + 12
        for i in range(1, 5):
            col = get_player_color(i)
            cv2.circle(draw_frame, (curr_x, leg_y + 16), 6, col, -1, cv2.LINE_AA)
            cv2.putText(draw_frame, f"ID #{i}", (curr_x + 10, leg_y + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            curr_x += 65

        cv2.circle(draw_frame, (curr_x + 5, leg_y + 16), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(draw_frame, "Ball", (curr_x + 15, leg_y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        writer.write(draw_frame)
        frame_idx += 1

        if frame_idx % 150 == 0 or frame_idx == total_frames:
            cur_elapsed = time.perf_counter() - t_render_start
            print(f"  Rendered {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%) "
                  f"at {frame_idx/cur_elapsed:.1f} FPS (elapsed: {cur_elapsed:.1f}s)")

    cap.release()
    writer.release()

    render_elapsed = time.perf_counter() - t_render_start
    print(f"  Rendering & Pose Inference finished in {render_elapsed:.2f}s ({frame_idx/render_elapsed:.1f} FPS)")

    # 4. Mux original audio
    print(f"  Muxing original audio into final MP4: {final_mp4.name}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_video_tmp),
        "-i", str(video_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "21",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(final_mp4)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio muxing failed: "
            + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    raw_video_tmp.unlink(missing_ok=True)

    print(f"  Successfully finalized {final_mp4}")

    # Top tracked player IDs
    sorted_ids = sorted(track_id_counts.items(), key=lambda x: x[1], reverse=True)
    top_players = [f"ID #{tid}: {cnt} frames ({cnt/total_frames*100:.1f}%)" for tid, cnt in sorted_ids[:6]]

    summary = {
        "video_name": video_path.name,
        "total_frames": total_frames,
        "duration_seconds": round(total_frames / fps, 2),
        "tracknet_visible_ball_frames": vis_count,
        "tracknet_visible_ratio": round(vis_count / total_frames, 4),
        "rendering_elapsed_seconds": round(render_elapsed, 2),
        "top_tracked_players": top_players,
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
