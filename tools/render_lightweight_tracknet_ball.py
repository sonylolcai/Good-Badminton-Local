#!/usr/bin/env python3
"""Render ball tracking and evidence-based rally-ending signals.

Requirements:
- Strict focus on shuttlecock (NO player bounding boxes or skeletons).
- Sub-pixel smooth neon target marker and dynamic fading ribbon trajectory.
- Show observed motion and confirmed endings; retain unknown action and player labels.
- Do not infer metric speed, a legal serve, or a score from image motion.
- High-efficiency encoding muxing original audio via ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from badminton_analysis.detection.lightweight_tracknet import LightweightTrackNetDetector
from badminton_analysis.sports.badminton import BadmintonRuleEngine
from badminton_analysis.sports.physics import PhysicalTrajectoryAnalyzer
from badminton_analysis.sports.observation import ObservationConfig
from badminton_analysis.sports.tactics import HandSide

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    current_det: Dict,
    recent_visible: bool,
    scale: float = 1.0,
):
    """Draw professional broadcast-style HUD overlay on the top-left corner."""
    t_sec = frame_idx / fps
    mins = int(t_sec // 60)
    secs = int(t_sec % 60)
    millis = int((t_sec - int(t_sec)) * 100)

    speed = current_det.get("speed_kmh") if current_det["visible"] else None
    status_text = "TRACKING" if current_det["visible"] else ("SEARCHING" if not recent_visible else "COASTING")
    status_color = (0, 255, 120) if current_det["visible"] else ((0, 200, 255) if recent_visible else (160, 160, 160))

    # HUD Box Dimensions (expanded for tactical insights)
    x1, y1 = int(24 * scale), int(24 * scale)
    w_box, h_box = int(420 * scale), int(172 * scale)
    x2, y2 = x1 + w_box, y1 + h_box

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 22, 28), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 68, 82), int(1 * scale))
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_mono = cv2.FONT_HERSHEY_DUPLEX
    f_scale = 0.52 * scale
    f_scale_sm = 0.40 * scale

    # Row 1: Model title + Status pill
    cv2.putText(frame, "TRACKNET V3 FAST | TACTICAL", (x1 + int(12 * scale), y1 + int(24 * scale)), font, f_scale_sm, (180, 200, 220), 1, cv2.LINE_AA)
    pill_w = int(90 * scale)
    pill_x1 = x2 - pill_w - int(12 * scale)
    pill_y1 = y1 + int(10 * scale)
    pill_y2 = pill_y1 + int(20 * scale)
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x1 + pill_w, pill_y2), (35, 40, 48), -1)
    cv2.circle(frame, (pill_x1 + int(10 * scale), (pill_y1 + pill_y2) // 2), int(4 * scale), status_color, -1)
    cv2.putText(frame, status_text, (pill_x1 + int(20 * scale), pill_y2 - int(5 * scale)), font, 0.38 * scale, status_color, 1, cv2.LINE_AA)

    # Row 2: Frame & Timestamp
    time_str = f"{mins:02d}:{secs:02d}.{millis:02d}"
    cv2.putText(frame, f"F: {frame_idx:04d} / {total_frames}", (x1 + int(12 * scale), y1 + int(56 * scale)), font_mono, f_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"TIME: {time_str}", (x1 + int(205 * scale), y1 + int(56 * scale)), font_mono, f_scale, (220, 220, 220), 1, cv2.LINE_AA)

    # Row 3: Ball Speed Display
    cv2.putText(frame, "BALL SPEED:", (x1 + int(12 * scale), y1 + int(90 * scale)), font, f_scale_sm, (160, 175, 190), 1, cv2.LINE_AA)
    speed_str = f"{speed:5.1f} km/h" if speed is not None else "UNKNOWN"
    speed_color = (0, 240, 255) if speed is not None and speed > 100 else (255, 255, 255)
    cv2.putText(frame, speed_str, (x1 + int(120 * scale), y1 + int(92 * scale)), font_mono, 0.65 * scale, speed_color, 2, cv2.LINE_AA)

    # Row 4: Rally & Shot Info
    rally_str = current_det.get("rally_display", "REST / INTER-RALLY")
    rally_color = (0, 255, 120) if "RALLY" in rally_str else ((0, 220, 255) if "END" in rally_str else (160, 175, 190))
    cv2.putText(frame, rally_str, (x1 + int(12 * scale), y1 + int(124 * scale)), font_mono, 0.48 * scale, rally_color, 1, cv2.LINE_AA)

    # Row 5: Tactical Details / Direction / Shot
    tactical_str = current_det.get("tactical_info", "STATUS: WAITING / READY")
    cv2.putText(frame, tactical_str, (x1 + int(12 * scale), y1 + int(152 * scale)), font, 0.40 * scale, (180, 210, 230), 1, cv2.LINE_AA)


def draw_pure_ball(
    frame: np.ndarray,
    history_pts: deque,
    scale: float = 1.0,
):
    """Draw sub-pixel neon ball marker with fading gradient trail."""
    pts = list(history_pts)
    if not pts:
        return

    n = len(pts)
    # 1. Draw glowing ribbon trail
    for i in range(1, n):
        pt1 = pts[i - 1]
        pt2 = pts[i]
        if pt1 is None or pt2 is None:
            continue

        progress = i / float(n)  # 0.0 (oldest) -> 1.0 (newest)
        # Color gradient: Cyan (255, 220, 0) fading to Neon Green/Yellow (0, 255, 255)
        b = int(255 * (1.0 - progress * 0.7))
        g = int(220 + 35 * progress)
        r = int(255 * progress)
        color = (b, g, r)

        thickness = max(1, int((1.0 + 3.5 * progress) * scale))
        p1 = (int(round(pt1[0] * scale)), int(round(pt1[1] * scale)))
        p2 = (int(round(pt2[0] * scale)), int(round(pt2[1] * scale)))
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    # 2. Draw current target marker if newest point is visible
    last_pt = pts[-1]
    if last_pt is not None:
        cx = int(round(last_pt[0] * scale))
        cy = int(round(last_pt[1] * scale))

        # Neon outer glow
        cv2.circle(frame, (cx, cy), int(7 * scale), (0, 255, 255), 1, cv2.LINE_AA)
        # Bright solid core
        cv2.circle(frame, (cx, cy), int(3 * scale), (255, 255, 255), -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-file", required=True, type=Path)
    parser.add_argument("--weights", default="weights/tracknetv3/ckpts/TrackNet_best.pt", type=Path)
    parser.add_argument("--output-dir", default="outputs/tracknet_fast_eval", type=Path)
    parser.add_argument("--output-scale", type=float, default=0.5, help="0.5 scales 4K down to 1080p")
    parser.add_argument("--batch-chunks", type=int, default=2)
    parser.add_argument("--court-roi-x", type=float, nargs=2, default=None, help="Optional detection ROI; not a court boundary")
    parser.add_argument("--trail-length", type=int, default=26)
    parser.add_argument("--dominant-hand", default="unknown", choices=["unknown", "left", "right"], help="Player dominant hand setting")
    parser.add_argument("--cached-csv", type=Path, default=None, help="Use existing detections CSV to speed up rendering")
    parser.add_argument("--analysis-config", type=Path, help="Observation threshold JSON")
    parser.add_argument("--court-calibration", type=Path, help="JSON with image_size and ordered court_polygon")
    args = parser.parse_args()

    video_path = args.video_file.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read video properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or orig_w <= 0 or orig_h <= 0:
        raise ValueError("Valid video dimensions and FPS are required")
    config = ObservationConfig.load(args.analysis_config)
    court_polygon = None
    if args.court_calibration:
        calibration = json.loads(args.court_calibration.read_text(encoding="utf-8"))
        if calibration.get("image_size") != [orig_w, orig_h]:
            raise ValueError("Court calibration image_size must match the source video")
        court_polygon = calibration["court_polygon"]
    court_x_min, court_x_max = args.court_roi_x or (0.0, float(orig_w))

    # Check if cached CSV exists and should be used
    use_cache = False
    cached_path = args.cached_csv
    if cached_path and cached_path.is_file():
        use_cache = True
        logger.info(f"Loading cached ball detections from: {cached_path}")
        detections = []
        with open(cached_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vis = int(row["visible"]) == 1
                detections.append({
                    "frame_index": int(row["frame_index"]),
                    "timestamp_s": float(row["timestamp_s"]),
                    "visible": vis,
                    "x": float(row["x"]) if vis and row["x"] else None,
                    "y": float(row["y"]) if vis and row["y"] else None,
                    "confidence": float(row["confidence"]) if row.get("confidence") else 0.0,
                    "speed_kmh": None,
                    "ground_contact": row.get("ground_contact", "").lower() in ("1", "true"),
                    "status": row.get("status") or ("detected" if vis else "missing"),
                })
        det_fps = None
    else:
        if cached_path is not None:
            raise FileNotFoundError(cached_path)
        logger.info("Initializing LightweightTrackNetDetector...")
        detector = LightweightTrackNetDetector(weights_path=args.weights)
        logger.info(f"Using device: {detector.device}")

        # Pass 1: Run fast inference
        t0 = time.time()
        logger.info("Running Pass 1: High-Speed Non-Overlap TrackNet Inference...")
        detections = detector.predict_video(video_path, batch_chunks=args.batch_chunks)
        det_time = time.time() - t0
        det_fps = len(detections) / max(0.001, det_time)
        logger.info(f"Inference completed in {det_time:.2f}s ({det_fps:.1f} FPS, {len(detections)} frames)")

    total_frames = len(detections)
    if total_frames != total_video_frames or any(d["frame_index"] != i for i, d in enumerate(detections)):
        raise ValueError("Detections must cover the source video in frame order")

    analyzer = PhysicalTrajectoryAnalyzer(fps=fps, image_size=(orig_w, orig_h), config=config)
    points = analyzer.extract_trajectory_points(detections)
    # The ROI may limit where motion candidates start. Preserve observations
    # outside it for terminal evaluation, especially out-of-court landings.
    motion_points = [replace(p, visible=p.visible and court_x_min <= p.x <= court_x_max) for p in points]
    arcs = analyzer.segment_flight_arcs(motion_points)
    for detection, point in zip(detections, points):
        detection["speed_kmh"] = point.speed_kmh

    rule_engine = BadmintonRuleEngine(
        court_polygon=court_polygon, image_size=(orig_w, orig_h), config=config,
        dominant_hand=args.dominant_hand,
    )
    rallies = rule_engine.segment_rallies(points, arcs)
    valid_rallies = [r for r in rallies if r.terminal is not None]
    logger.info("Observed %s motion groups, %s confirmed endings", len(rallies), len(valid_rallies))

    # Populate HUD annotations
    for d in detections:
        d["rally_display"] = "REST / INTER-RALLY"
        d["tactical_info"] = "STATUS: WAITING / READY"

    for r in rallies:
        for s_idx, shot in enumerate(r.shots):
            ts = r.tactical_shots[s_idx] if s_idx < len(r.tactical_shots) else None
            hitter_code = {"near_team": "NEAR", "far_team": "FAR"}.get(ts.hitter if ts else None, "UNKNOWN")
            hand_prefix = {"right": "R", "left": "L"}.get(args.dominant_hand, "?")
            hand_code = f"{hand_prefix}-" + {HandSide.BACKHAND: "BH", HandSide.FOREHAND: "FH"}.get(ts.hand_side if ts else None, "UNKNOWN")
            shot_code = ts.shot_type.replace('_', ' ').upper() if ts else shot.flight_direction.replace('_', ' ').upper()

            for f in range(shot.start_frame, shot.end_frame + 1):
                if 0 <= f < total_frames:
                    detections[f]["rally_display"] = f"MOTION #{r.rally_id} SEG {s_idx + 1}/{len(r.shots)}: {hitter_code} [{hand_code}]"
                    detections[f]["tactical_info"] = f"TYPE: {shot_code} | START: UNCONFIRMED"

        if r.terminal:
            term_end_f = min(total_frames, r.end_frame + int(1.8 * fps))
            term_reason = r.terminal.terminal_type.replace('_', ' ').upper()
            for f in range(r.end_frame, term_end_f):
                detections[f]["rally_display"] = f"END: {term_reason}"
                detections[f]["tactical_info"] = f"REASON: {term_reason}"

    # Export structured rallies and shots JSON
    rallies_export = []
    for r in rallies:
        rallies_export.append({
            "rally_id": r.rally_id,
            "start_time_s": r.start_time_s,
            "end_time_s": r.end_time_s,
            "duration_s": r.duration_s,
            "trajectory_segment_count": len(r.shots),
            "shot_count": None,
            "start_signal": r.start_signal,
            "start_confirmed": r.start_confirmed,
            "end_confirmed": r.terminal is not None,
            "interruption_reason": r.interruption_reason,
            "narrative": r.narrative,
            "dominant_hand": args.dominant_hand,
            "terminal": {
                "terminal_type": r.terminal.terminal_type,
                "frame_index": r.terminal.frame_index,
                "timestamp_s": r.terminal.timestamp_s,
                "evidence": r.terminal.evidence,
                "landing_xy": r.terminal.landing_xy,
                "reason": r.terminal.reason,
            } if r.terminal else None,
            "shots": [
                {
                    "shot_index": ts.shot_index,
                    "hitter": ts.hitter,
                    "hand_side": ts.hand_side.value,
                    "shot_type": ts.shot_type,
                    "shot_type_cn": ts.shot_type_cn,
                    "start_time_s": round(ts.start_time_s, 2),
                    "end_time_s": round(ts.end_time_s, 2),
                    "duration_s": ts.duration_s,
                    "peak_speed_kmh": ts.peak_speed_kmh,
                    "flight_direction": ts.flight_direction,
                    "tactical_line": ts.tactical_line,
                    "start_zone": ts.start_zone.name,
                    "target_zone": ts.target_zone.name,
                    "description": ts.description,
                }
                for ts in r.tactical_shots
            ] if r.tactical_shots else [
                {
                    "shot_index": s_idx + 1,
                    "start_time_s": round(s.start_time_s, 2),
                    "end_time_s": round(s.end_time_s, 2),
                    "flight_direction": s.flight_direction,
                    "peak_speed_kmh": s.peak_speed_kmh,
                    "tactical_line": s.tactical_line,
                }
                for s_idx, s in enumerate(r.shots)
            ],
        })

    rallies_json_path = output_dir / f"{video_path.stem}_rallies_and_shots.json"
    with open(rallies_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": str(video_path),
            "sport_id": "badminton",
            "dominant_hand": args.dominant_hand,
            "confirmed_endings": len(valid_rallies),
            "total_rallies": None,
            "analysis_status": "motion_candidates",
            "rallies": rallies_export,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved rally and shot analytics to {rallies_json_path}")

    # Export CSV if not using cache
    csv_path = output_dir / f"{video_path.stem}_ball_tracknet_fast.csv"
    if not use_cache or not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_index", "timestamp_s", "visible", "x", "y", "confidence", "speed_kmh", "status", "ground_contact"])
            writer.writeheader()
            for d in detections:
                writer.writerow({
                    "frame_index": d["frame_index"],
                    "timestamp_s": round(d["frame_index"] / fps, 3),
                    "visible": 1 if d["visible"] else 0,
                    "x": round(d["x"], 1) if d["x"] is not None else "",
                    "y": round(d["y"], 1) if d["y"] is not None else "",
                    "confidence": round(d["confidence"], 3) if d.get("confidence") is not None else "",
                    "speed_kmh": d.get("speed_kmh", 0.0),
                    "status": d.get("status", "missing"),
                    "ground_contact": 1 if d.get("ground_contact") is True else 0,
                })
        logger.info(f"Exported raw CSV to: {csv_path}")

    # Pass 2: High-Quality Video Rendering
    logger.info("Running Pass 2: Pure-Ball Video Rendering & Audio Muxing...")
    target_w = int(orig_w * args.output_scale)
    target_h = int(orig_h * args.output_scale)

    temp_video = output_dir / f"temp_{video_path.stem}_rendered.mp4"
    final_video = output_dir / f"eval_{video_path.stem}_tracknet_fast.mp4"

    # Use ffmpeg pipe for fast H.264 encoding
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{target_w}x{target_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        str(temp_video),
    ]

    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    trail_buffer = deque(maxlen=args.trail_length)
    recent_visible = False

    t_render_start = time.time()
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        if args.output_scale != 1.0:
            render_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            render_frame = frame

        cur_det = detections[frame_idx]
        if cur_det["visible"] and cur_det["x"] is not None:
            trail_buffer.append((cur_det["x"], cur_det["y"]))
            recent_visible = True
        else:
            trail_buffer.append(None)
            recent_visible = any(pt is not None for pt in list(trail_buffer)[-8:])

        # Render pure ball marker and trail
        draw_pure_ball(render_frame, trail_buffer, scale=args.output_scale)

        # Render professional HUD
        draw_hud(
            render_frame,
            frame_idx=frame_idx,
            total_frames=total_frames,
            fps=fps,
            current_det=cur_det,
            recent_visible=recent_visible,
            scale=args.output_scale,
        )

        pipe.stdin.write(render_frame.tobytes())

        if (frame_idx + 1) % 300 == 0 or frame_idx + 1 == total_frames:
            logger.info(f"Render progress: {frame_idx + 1}/{total_frames} frames ({ (frame_idx+1)*100.0/total_frames:.1f}%)")

    cap.release()
    pipe.stdin.close()
    if pipe.wait() != 0:
        raise RuntimeError("Video rendering failed; keeping the temporary output")

    render_time = time.time() - t_render_start
    logger.info(f"Video frames rendered in {render_time:.2f}s ({total_frames / max(0.001, render_time):.1f} FPS)")

    # Mux original audio
    logger.info("Muxing original audio...")
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", str(temp_video),
        "-i", str(video_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(final_video),
    ]
    subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    if temp_video.exists():
        temp_video.unlink()

    # Compute trajectory smoothness metric (2nd derivative jitter)
    vis_pts = [(d["x"], d["y"]) for d in detections if d["visible"] and d["x"] is not None]
    if len(vis_pts) > 2:
        xs = np.array([p[0] for p in vis_pts])
        ys = np.array([p[1] for p in vis_pts])
        d2x = np.diff(np.diff(xs))
        d2y = np.diff(np.diff(ys))
        jitter = float(np.sqrt(d2x**2 + d2y**2).mean())
    else:
        jitter = 0.0

    vis_count = sum(1 for d in detections if d["visible"])
    metrics = {
        "video": str(video_path),
        "total_frames": total_frames,
        "visible_frames": vis_count,
        "visibility_ratio": round(vis_count / max(1, total_frames), 4),
        "inference_fps": round(det_fps, 2) if det_fps is not None else None,
        "render_fps": round(total_frames / max(0.001, render_time), 2),
        "jitter_metric_px": round(jitter, 2),
        "output_video": str(final_video),
        "output_csv": str(csv_path),
        "confirmed_endings": len(valid_rallies),
        "total_rallies": None,
        "rallies_and_shots_json": str(rallies_json_path),
    }

    metrics_path = output_dir / f"{video_path.stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"SUCCESS: Final video generated at {final_video}")


if __name__ == "__main__":
    main()
