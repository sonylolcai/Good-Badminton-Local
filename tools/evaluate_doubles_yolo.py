#!/usr/bin/env python3
"""Evaluate YOLO Pose and YOLO Ball capabilities on real badminton doubles videos.

Uses local GPU (Apple Silicon MPS / CUDA) for accelerated inference.
Decoupled from court lines, court templates, and homography calibration.
Generates:
1. Annotated MP4 videos with 17-keypoint skeleton tracking and shuttlecock trajectory.
2. Frame-by-frame and summary evaluation metrics (JSON).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def resolve_device(requested="auto"):
    req = str(requested or "auto").strip().lower()
    if req not in {"", "auto"}:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class DoublesEvaluator:
    def __init__(
        self,
        pose_model_path="weights/yolo11n-pose.pt",
        ball_model_path="weights/yolo11s-ball.pt",
        device="auto",
        pose_imgsz=960,
        ball_imgsz=1280,
        pose_conf=0.20,
        ball_conf=0.15,
        output_scale=0.5,
    ):
        self.device = resolve_device(device)
        self.pose_imgsz = int(pose_imgsz)
        self.ball_imgsz = int(ball_imgsz)
        self.pose_conf = float(pose_conf)
        self.ball_conf = float(ball_conf)
        self.output_scale = float(output_scale)

        from ultralytics import YOLO

        print(f"Loading YOLO Pose model from {pose_model_path} (device: {self.device})...")
        self.pose_model = YOLO(pose_model_path)
        print(f"Loading YOLO Ball model from {ball_model_path} (device: {self.device})...")
        self.ball_model = YOLO(ball_model_path)

        # Warm up models
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.pose_model(dummy, device=self.device, imgsz=self.pose_imgsz, verbose=False)
        self.ball_model(dummy, device=self.device, imgsz=self.ball_imgsz, verbose=False)
        print("Models successfully initialized and warmed up.")

    def process_video(self, video_path, output_dir, max_frames=None):
        video_path = Path(video_path).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        video_stem = video_path.stem
        final_mp4 = output_dir / f"eval_{video_stem}.mp4"
        metrics_json = output_dir / f"eval_{video_stem}_metrics.json"
        raw_video_tmp = output_dir / f"tmp_{video_stem}_annotated.mp4"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        limit_frames = total_source_frames
        if max_frames and 0 < max_frames < total_source_frames:
            limit_frames = int(max_frames)

        out_w = int(orig_w * self.output_scale)
        out_h = int(orig_h * self.output_scale)
        # Ensure dimensions are even numbers for H.264
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_video_tmp), fourcc, fps, (out_w, out_h))

        print(f"\nProcessing {video_path.name}:")
        print(f"  Source: {orig_w}x{orig_h} @ {fps:.2f}fps, total {limit_frames} frames ({limit_frames/fps:.1f}s)")
        print(f"  Output: {out_w}x{out_h} (scale {self.output_scale})")
        print(f"  Target: {final_mp4}")

        # Shuttlecock trajectory state
        trajectory = deque(maxlen=25)  # store (x, y, conf, frame_idx) in output resolution
        max_jump_px = int(220 * self.output_scale * 2.0)  # max motion jump allowed
        missing_ball_frames = 0

        # Metrics accumulators
        frame_metrics = []
        persons_count_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, "5+": 0}
        total_ball_detected_frames = 0
        consecutive_ball_streak = 0
        max_ball_streak = 0
        ball_confidences = []
        person_confidences = []

        t_start = time.perf_counter()
        frame_idx = 0

        # Skeletons connections for COCO17
        skeleton_pairs = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        # Colors for keypoints and bones
        kp_color = (0, 255, 255)  # yellow
        bone_color = (255, 128, 0) # cyan/blue

        while cap.isOpened() and frame_idx < limit_frames:
            ret, frame = cap.read()
            if not ret:
                break

            t_frame_start = time.perf_counter()

            # 1. YOLO Pose Inference (with tracking persistence)
            pose_res = self.pose_model.track(
                frame,
                persist=True,
                device=self.device,
                imgsz=self.pose_imgsz,
                conf=self.pose_conf,
                verbose=False
            )[0]

            # 2. YOLO Ball Inference
            ball_res = self.ball_model(
                frame,
                device=self.device,
                imgsz=self.ball_imgsz,
                conf=self.ball_conf,
                verbose=False
            )[0]

            # Downsample frame for output drawing
            if self.output_scale != 1.0:
                draw_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                draw_frame = frame.copy()

            # Scale factor for drawing
            scale_x = out_w / orig_w
            scale_y = out_h / orig_h

            # --- Process & Draw Pose ---
            detected_persons = 0
            frame_person_confs = []
            if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                detected_persons = len(pose_res.boxes)
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
                    frame_person_confs.append(p_conf)
                    person_confidences.append(p_conf)
                    tid = track_ids[p_idx]

                    # Scale bbox to draw_frame
                    bx1 = int(box[0] * scale_x)
                    by1 = int(box[1] * scale_y)
                    bx2 = int(box[2] * scale_x)
                    by2 = int(box[3] * scale_y)

                    # Player box
                    box_color = (0, 200, 0) if tid is not None else (0, 165, 255)
                    cv2.rectangle(draw_frame, (bx1, by1), (bx2, by2), box_color, 2)

                    # Label
                    label = f"#{tid} {p_conf:.2f}" if tid is not None else f"P {p_conf:.2f}"
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(draw_frame, (bx1, max(0, by1 - lh - 6)), (bx1 + lw + 4, max(lh + 6, by1)), box_color, -1)
                    cv2.putText(draw_frame, label, (bx1 + 2, max(lh + 2, by1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    # Draw keypoints and skeleton
                    if kpts is not None and p_idx < len(kpts):
                        pts = kpts[p_idx]
                        pts_scaled = []
                        for i in range(17):
                            px = int(pts[i][0] * scale_x)
                            py = int(pts[i][1] * scale_y)
                            p_visible = (pts[i][0] > 0 and pts[i][1] > 0)
                            if kpts_conf is not None and i < len(kpts_conf[p_idx]):
                                if kpts_conf[p_idx][i] < 0.2:
                                    p_visible = False
                            pts_scaled.append((px, py, p_visible))

                        # Draw bones
                        for p1_i, p2_i in skeleton_pairs:
                            x1, y1, v1 = pts_scaled[p1_i]
                            x2, y2, v2 = pts_scaled[p2_i]
                            if v1 and v2:
                                cv2.line(draw_frame, (x1, y1), (x2, y2), bone_color, 2, cv2.LINE_AA)

                        # Draw joint points
                        for px, py, v in pts_scaled:
                            if v:
                                cv2.circle(draw_frame, (px, py), 3, kp_color, -1, cv2.LINE_AA)

            # Update persons histogram
            if detected_persons >= 5:
                persons_count_hist["5+"] += 1
            else:
                persons_count_hist[detected_persons] += 1

            # --- Process & Draw Ball ---
            best_ball = None
            if ball_res.boxes is not None and len(ball_res.boxes) > 0:
                b_boxes = ball_res.boxes.xyxy.cpu().numpy()
                b_confs = ball_res.boxes.conf.cpu().numpy()

                candidates = []
                for b_idx in range(len(b_boxes)):
                    bbox = b_boxes[b_idx]
                    b_conf = float(b_confs[b_idx])
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    # Filter out impossibly large boxes
                    if bw > orig_w * 0.05 or bh > orig_h * 0.05:
                        continue
                    cx = (bbox[0] + bbox[2]) / 2.0 * scale_x
                    cy = (bbox[1] + bbox[3]) / 2.0 * scale_y
                    candidates.append({"cx": cx, "cy": cy, "conf": b_conf, "bbox": bbox})

                if candidates:
                    # Pick candidate with highest confidence or nearest to last trajectory
                    if trajectory:
                        last_x, last_y = trajectory[-1][0], trajectory[-1][1]
                        # Filter by jump distance if recent
                        valid_cands = [
                            c for c in candidates
                            if np.hypot(c["cx"] - last_x, c["cy"] - last_y) < max_jump_px or missing_ball_frames > 5
                        ]
                        if valid_cands:
                            best_ball = max(valid_cands, key=lambda c: c["conf"])
                    else:
                        best_ball = max(candidates, key=lambda c: c["conf"])

            ball_detected = False
            if best_ball is not None:
                bx = int(best_ball["cx"])
                by = int(best_ball["cy"])
                bconf = best_ball["conf"]
                trajectory.append((bx, by, bconf, frame_idx))
                total_ball_detected_frames += 1
                consecutive_ball_streak += 1
                max_ball_streak = max(max_ball_streak, consecutive_ball_streak)
                ball_confidences.append(bconf)
                ball_detected = True
                missing_ball_frames = 0
            else:
                missing_ball_frames += 1
                consecutive_ball_streak = 0
                if missing_ball_frames > 8:
                    trajectory.clear()

            # Draw trajectory trail
            if len(trajectory) > 1:
                t_points = list(trajectory)
                for t_i in range(1, len(t_points)):
                    p_start = (int(t_points[t_i - 1][0]), int(t_points[t_i - 1][1]))
                    p_end = (int(t_points[t_i][0]), int(t_points[t_i][1]))
                    # Fade color from yellow (old) to cyan/green (new)
                    alpha = t_i / len(t_points)
                    color = (
                        int(255 * alpha),
                        int(255 * (1 - alpha * 0.5)),
                        int(50 + 200 * alpha)
                    )
                    thickness = max(1, int(1 + alpha * 3))
                    cv2.line(draw_frame, p_start, p_end, color, thickness, cv2.LINE_AA)

            # Draw latest ball marker
            if ball_detected and trajectory:
                bx, by, bconf, _ = trajectory[-1]
                # Outer ring
                cv2.circle(draw_frame, (bx, by), 7, (0, 69, 255), 2, cv2.LINE_AA)
                # Inner dot
                cv2.circle(draw_frame, (bx, by), 3, (0, 255, 255), -1, cv2.LINE_AA)
                # Text
                b_text = f"Ball {bconf:.2f}"
                cv2.putText(draw_frame, b_text, (bx + 10, by - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # --- HUD Overlay ---
            hud_h = 36
            overlay = draw_frame.copy()
            cv2.rectangle(overlay, (0, 0), (out_w, hud_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, draw_frame, 0.3, 0, draw_frame)

            time_sec = frame_idx / fps
            m = int(time_sec // 60)
            s = int(time_sec % 60)
            ms = int((time_sec - int(time_sec)) * 100)

            hud_left = f"YOLO Pose & Ball Eval | Apple Silicon GPU (MPS)"
            hud_mid = f"Frame: {frame_idx:04d}/{limit_frames:04d} ({m:02d}:{s:02d}.{ms:02d})"
            hud_right = f"Players: {detected_persons} | Ball: {'DETECTED' if ball_detected else 'SEARCHING'}"

            cv2.putText(draw_frame, hud_left, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(draw_frame, hud_mid, (out_w // 2 - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(draw_frame, hud_right, (out_w - 280, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (50, 255, 50) if ball_detected else (200, 200, 200), 1, cv2.LINE_AA)

            # Write frame to temp video
            writer.write(draw_frame)

            # Frame metrics
            frame_metrics.append({
                "frame": frame_idx,
                "time_sec": round(time_sec, 3),
                "persons_detected": detected_persons,
                "persons_avg_conf": round(float(np.mean(frame_person_confs)), 3) if frame_person_confs else None,
                "ball_detected": ball_detected,
                "ball_conf": round(float(trajectory[-1][2]), 3) if ball_detected else None,
                "ball_pos": [int(trajectory[-1][0]), int(trajectory[-1][1])] if ball_detected else None,
            })

            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == limit_frames:
                elapsed_so_far = time.perf_counter() - t_start
                cur_fps = frame_idx / elapsed_so_far
                print(f"  Processed {frame_idx}/{limit_frames} frames ({frame_idx/limit_frames*100:.1f}%) "
                      f"at {cur_fps:.1f} FPS (elapsed: {elapsed_so_far:.1f}s)")

        cap.release()
        writer.release()

        total_elapsed = time.perf_counter() - t_start
        effective_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0

        print(f"  Inference & Annotation finished in {total_elapsed:.2f}s ({effective_fps:.1f} FPS)")

        # Mux original audio using ffmpeg
        print(f"  Muxing original audio into final MP4: {final_mp4.name}...")
        self._mux_audio_and_finalize(str(video_path), str(raw_video_tmp), str(final_mp4))
        if raw_video_tmp.exists():
            raw_video_tmp.unlink()

        # Compile final summary
        summary = {
            "video_name": video_path.name,
            "duration_seconds": round(frame_idx / fps, 2),
            "processed_frames": frame_idx,
            "source_fps": round(fps, 2),
            "source_resolution": [orig_w, orig_h],
            "output_resolution": [out_w, out_h],
            "inference_device": str(self.device),
            "inference_elapsed_seconds": round(total_elapsed, 2),
            "inference_fps": round(effective_fps, 2),
            "pose_evaluation": {
                "model": "yolo11n-pose.pt",
                "persons_histogram": persons_count_hist,
                "full_doubles_4p_ratio": round(persons_count_hist[4] / frame_idx, 4) if frame_idx else 0,
                "at_least_3p_ratio": round((persons_count_hist[3] + persons_count_hist[4] + persons_count_hist["5+"]) / frame_idx, 4) if frame_idx else 0,
                "avg_person_confidence": round(float(np.mean(person_confidences)), 3) if person_confidences else 0,
            },
            "ball_evaluation": {
                "model": "yolo11s-ball.pt",
                "detected_frames": total_ball_detected_frames,
                "detection_ratio": round(total_ball_detected_frames / frame_idx, 4) if frame_idx else 0,
                "max_continuous_streak": max_ball_streak,
                "avg_ball_confidence": round(float(np.mean(ball_confidences)), 3) if ball_confidences else 0,
                "high_conf_gt50_count": sum(1 for c in ball_confidences if c >= 0.50),
            },
            "frames": frame_metrics,
        }

        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"  Saved evaluation metrics to {metrics_json.name}")
        return summary

    def _mux_audio_and_finalize(self, orig_video, annotated_video, final_video):
        cmd = [
            "ffmpeg", "-y",
            "-i", annotated_video,
            "-i", orig_video,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "21",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            final_video
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"Warning: ffmpeg audio muxing failed, copying raw video as fallback. Error: {res.stderr.decode('utf-8', errors='ignore')}")
            shutil.copyfile(annotated_video, final_video)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Pose and Ball on real badminton videos.")
    parser.add_argument("--videos", nargs="+", required=True, help="List of video files to evaluate.")
    parser.add_argument("--output-dir", default="outputs/doubles_yolo_eval", help="Output directory for videos and metrics.")
    parser.add_argument("--pose-model", default="weights/yolo11n-pose.pt", help="Path to YOLO Pose model.")
    parser.add_argument("--ball-model", default="weights/yolo11s-ball.pt", help="Path to YOLO Ball model.")
    parser.add_argument("--device", default="auto", help="Inference device: auto/mps/cuda/cpu.")
    parser.add_argument("--pose-imgsz", type=int, default=960, help="Inference image size for pose.")
    parser.add_argument("--ball-imgsz", type=int, default=1280, help="Inference image size for ball.")
    parser.add_argument("--pose-conf", type=float, default=0.20, help="Pose detection confidence threshold.")
    parser.add_argument("--ball-conf", type=float, default=0.15, help="Ball detection confidence threshold.")
    parser.add_argument("--output-scale", type=float, default=0.5, help="Output resolution scale relative to 4K (0.5 = 1080p).")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (useful for testing).")

    args = parser.parse_args()

    evaluator = DoublesEvaluator(
        pose_model_path=args.pose_model,
        ball_model_path=args.ball_model,
        device=args.device,
        pose_imgsz=args.pose_imgsz,
        ball_imgsz=args.ball_imgsz,
        pose_conf=args.pose_conf,
        ball_conf=args.ball_conf,
        output_scale=args.output_scale,
    )

    all_summaries = []
    for v in args.videos:
        summary = evaluator.process_video(v, args.output_dir, max_frames=args.max_frames)
        all_summaries.append(summary)

    # Write aggregate report
    report_path = Path(args.output_dir) / "all_videos_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"ALL EVALUATIONS COMPLETE! Results saved in {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
