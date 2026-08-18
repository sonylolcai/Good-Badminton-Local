import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.court.mapper import (
    CourtMapper,
    auto_detect_preview,
    compute_expanded_roi,
    resolve_court_corners,
)
from badminton_analysis.court.detector import auto_detect_court_corners
from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies

_MAX_WEBUI_OUTPUTS = 10
_MAX_GENERATED_TEMPLATES = 20
_MAX_COURT_FRAME_SAMPLES = 24
_COURT_DETECTION_SIZE = (1080, 720)
_SAFE_OUTPUT_STEM_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')

_dependencies_loaded = False


def _prepare_tracknet_v3_raw(video_path, output_dir):
    """Run the configured GPU TrackNet runtime and return its immutable CSV.

    Good-Badminton intentionally does not package TrackNetV3 source code or
    checkpoints.  The API host owns those external files and exposes their
    locations through environment variables.  This fails loudly on a CPU-only
    workstation instead of silently substituting the legacy YOLO ball model.
    """
    tracknet_root = os.environ.get("GOOD_BADMINTON_TRACKNET_ROOT")
    checkpoint = os.environ.get("GOOD_BADMINTON_TRACKNET_CHECKPOINT")
    runtime_python = os.environ.get("GOOD_BADMINTON_TRACKNET_PYTHON", os.sys.executable)
    if not tracknet_root or not checkpoint:
        raise RuntimeError(
            "TrackNetV3 未配置：需要 GOOD_BADMINTON_TRACKNET_ROOT 和 "
            "GOOD_BADMINTON_TRACKNET_CHECKPOINT。当前本机不能自动回退到 YOLO。"
        )
    tracknet_root_path = Path(tracknet_root)
    checkpoint_path = Path(checkpoint)
    if not (tracknet_root_path / "predict.py").is_file() or not checkpoint_path.is_file():
        raise RuntimeError("TrackNetV3 源码或权重路径无效；请检查 GPU 服务环境变量。")

    project_root = Path(__file__).resolve().parents[1]
    runner = project_root / "evaluation" / "shuttle_tracknet_ab" / "run_tracknet_v3.py"
    fast_predictor = project_root / "evaluation" / "shuttle_tracknet_ab" / "fast_predict_tracknet_v3.py"
    if not runner.is_file() or not fast_predictor.is_file():
        raise RuntimeError("当前部署包缺少 TrackNetV3 主流程工具。")

    target_dir = Path(output_dir) / "tracknet_v3"
    batch_size = int(os.environ.get("GOOD_BADMINTON_TRACKNET_BATCH_SIZE", "16"))
    background_samples = int(os.environ.get("GOOD_BADMINTON_TRACKNET_BACKGROUND_SAMPLES", "120"))
    if batch_size <= 0 or background_samples <= 0:
        raise RuntimeError("TrackNetV3 批量大小和背景采样数必须为正整数。")
    command = [
        runtime_python,
        str(runner),
        "--video", str(Path(video_path).resolve()),
        "--tracknet-root", str(tracknet_root_path.resolve()),
        "--tracknet-python", runtime_python,
        "--tracknet-checkpoint", str(checkpoint_path.resolve()),
        "--fast-predictor", str(fast_predictor),
        "--output-dir", str(target_dir),
        "--batch-size", str(batch_size),
        "--background-sample-count", str(background_samples),
        "--overwrite",
    ]
    print("TrackNetV3 primary shuttle detector:", subprocess.list2cmdline(command))
    try:
        subprocess.run(command, cwd=str(project_root), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"TrackNetV3 推理失败（退出码 {exc.returncode}）。") from exc
    csv_path = target_dir / "tracknet_raw" / f"{Path(video_path).stem}_ball.csv"
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise RuntimeError("TrackNetV3 未生成原始球点 CSV。")
    return str(csv_path)


def imread_safe(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread with fallback for Unicode paths on Windows."""
    img = cv2.imread(path, flags)
    if img is None and os.path.isfile(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, flags)
        except Exception:
            pass
    return img


def _cleanup_old_outputs(base_dir="outputs", keep=_MAX_WEBUI_OUTPUTS):
    """Remove oldest webui output directories beyond *keep* count."""
    if not os.path.isdir(base_dir):
        return
    dirs = []
    for name in os.listdir(base_dir):
        if _is_webui_output_directory(name):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full):
                dirs.append((os.path.getmtime(full), full))
    dirs.sort(reverse=True)
    for _, path in dirs[keep:]:
        try:
            shutil.rmtree(path)
        except Exception:
            pass


def _cleanup_generated_templates(base_dir=os.path.join("outputs", "court_templates"),
                                 keep=_MAX_GENERATED_TEMPLATES):
    """Keep generated video-frame templates bounded without touching user files."""
    if not os.path.isdir(base_dir):
        return

    templates = []
    for name in os.listdir(base_dir):
        if not name.startswith("auto_court_"):
            continue
        full = os.path.join(base_dir, name)
        if os.path.isfile(full):
            templates.append((os.path.getmtime(full), full))

    templates.sort(reverse=True)
    for _, path in templates[keep:]:
        try:
            os.remove(path)
        except OSError:
            pass


def _is_webui_output_directory(name):
    """Recognise both legacy ``webui_*`` and timestamp-first result folders."""
    return name.startswith("webui_") or bool(re.match(r"^20\d{6}_\d{6}_webui_", name))


def _default_analysis_output_dir(video_path, timestamp):
    """Create a readable result folder whose first sortable field is time."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_name = _SAFE_OUTPUT_STEM_PATTERN.sub("_", video_name).strip(" ._") or "video"
    # Leave room for the output root and generated artifacts on Windows.
    return os.path.join("outputs", f"{timestamp}_webui_{safe_name[:80]}")


def _ensure_dependencies():
    global _dependencies_loaded
    if not _dependencies_loaded:
        load_runtime_dependencies()
        _dependencies_loaded = True


def prepare_court(template_path, manual_corners=None):
    """Detect (or apply manual) court corners and return results + preview.

    Returns:
        dict with keys: corners, roi_corners, mid_height, preview_bgr.
        On failure corners/roi_corners/mid_height are None.
    """
    _ensure_dependencies()
    template_color = imread_safe(template_path)
    if template_color is None:
        raise FileNotFoundError(f"Cannot read template image: {template_path}")

    if manual_corners and len(manual_corners) == 4:
        corners, roi_corners, mid_height = resolve_court_corners(
            template_color, manual_corners=manual_corners
        )
        preview = template_color.copy()
        if corners:
            pts = [list(c) for c in corners]
            cv2.polylines(preview, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 3)
            for idx, pt in enumerate(corners, 1):
                cv2.circle(preview, pt, 6, (0, 0, 255), -1)
                cv2.putText(preview, str(idx), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        corners_auto, preview = auto_detect_preview(template_color)
        if preview is not None:
            h, w = template_color.shape[:2]
            preview = cv2.resize(preview, (w, h))
        if corners_auto:
            corners, roi_corners, mid_height = resolve_court_corners(
                template_color, manual_corners=corners_auto
            )
        else:
            corners, roi_corners, mid_height = None, None, None

    return {
        "corners": corners,
        "roi_corners": roi_corners,
        "mid_height": mid_height,
        "preview_bgr": preview,
    }


def _sample_frame_indices(total_frames, max_samples=_MAX_COURT_FRAME_SAMPLES):
    """Return evenly distributed frame indices, avoiding likely intro/outro shots."""
    if total_frames <= 0:
        return []

    if total_frames == 1:
        return [0]

    margin = int(total_frames * 0.03)
    start = min(margin, total_frames - 1)
    end = max(start, total_frames - 1 - margin)
    sample_count = min(max_samples, end - start + 1)
    return sorted({int(index) for index in np.linspace(start, end, sample_count)})


def _court_frame_score(frame):
    """Score a frame using the existing court detector plus a clarity tie-breaker."""
    detection_frame = cv2.resize(frame, _COURT_DETECTION_SIZE)
    corners, _mask, debug = auto_detect_court_corners(detection_frame)
    if not corners:
        return None

    detector_score = float(debug.get("score") or 0.0)
    gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Court-line geometry decides the result; clarity only separates similar candidates.
    selection_score = detector_score + min(np.log1p(sharpness), 8.0)
    return {
        "detector_score": detector_score,
        "sharpness": sharpness,
        "selection_score": selection_score,
    }


def _fallback_frame_score(frame):
    """Rank clear court-like frames even when line geometry cannot find four corners."""
    detection_frame = cv2.resize(frame, _COURT_DETECTION_SIZE)
    hsv = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = (h >= 35) & (h <= 95) & (s >= 30) & (v >= 45)
    lower_court = green[int(green.shape[0] * 0.25):]
    green_ratio = float(np.count_nonzero(lower_court) / max(1, lower_court.size))
    gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "green_ratio": green_ratio,
        "sharpness": sharpness,
        "selection_score": green_ratio * 100.0 + min(np.log1p(sharpness), 8.0),
    }


def extract_best_court_template(video_path, output_dir=os.path.join("outputs", "court_templates"),
                                max_samples=_MAX_COURT_FRAME_SAMPLES):
    """Extract the most suitable court frame from a video using the native detector.

    The saved PNG is the exact video-resolution template used later for mapping;
    this avoids coordinate drift between the preview and analysis video.
    """
    _ensure_dependencies()
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError("Cannot read video for automatic court detection.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video metadata: {video_path}")

    best_detected = None
    best_fallback = None
    sampled_indices = _sample_frame_indices(total_frames, max_samples=max_samples)
    for frame_index in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        fallback_score = _fallback_frame_score(frame)
        fallback_candidate = {
            "frame": frame,
            "frame_index": frame_index,
            "detected": False,
            "detector_score": 0.0,
            **fallback_score,
        }
        if best_fallback is None or fallback_candidate["selection_score"] > best_fallback["selection_score"]:
            best_fallback = fallback_candidate

        score = _court_frame_score(frame)
        if score is not None:
            candidate = {
                "frame": frame,
                "frame_index": frame_index,
                "detected": True,
                "green_ratio": fallback_score["green_ratio"],
                **score,
            }
            if best_detected is None or candidate["selection_score"] > best_detected["selection_score"]:
                best_detected = candidate
    cap.release()

    best = best_detected or best_fallback
    if best is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    _cleanup_generated_templates(output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    template_path = os.path.join(output_dir, f"auto_court_{timestamp}_{best['frame_index']}.png")
    encoded, data = cv2.imencode(".png", best["frame"])
    if not encoded:
        raise RuntimeError("Failed to save the automatically selected court frame.")
    data.tofile(template_path)

    return {
        "template_path": template_path,
        "frame_index": best["frame_index"],
        "time_sec": best["frame_index"] / fps,
        "sampled_frames": len(sampled_indices),
        "detected": best["detected"],
        "detector_score": round(best["detector_score"], 2),
        "green_ratio": round(best["green_ratio"], 4),
        "sharpness": round(best["sharpness"], 2),
    }


def prepare_court_from_video(video_path):
    """Create a template from a video frame, then reuse the normal court workflow."""
    selected = extract_best_court_template(video_path)
    if selected is None:
        return {
            "corners": None,
            "roi_corners": None,
            "mid_height": None,
            "preview_bgr": None,
            "template_path": None,
            "selection": None,
        }

    prepared = prepare_court(selected["template_path"])
    prepared["template_path"] = selected["template_path"]
    prepared["selection"] = selected
    return prepared


def _find_ffmpeg():
    """Locate an ffmpeg binary — prefer imageio_ffmpeg (bundled with moviepy)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    path = shutil.which("ffmpeg")
    if path:
        return path
    return None


def _reencode_for_browser(video_path, output_dir):
    """Re-encode video to H.264 so browsers can play it.

    OpenCV's mp4v codec isn't browser-compatible.  This converts to H.264
    via ffmpeg.  Returns the path of the web-friendly file (or the original
    if ffmpeg is unavailable).
    """
    if not os.path.isfile(video_path):
        return video_path

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return video_path

    web_path = os.path.join(output_dir, "web_" + os.path.basename(video_path))
    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-movflags", "faststart",
                web_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
            check=True,
        )
        if os.path.isfile(web_path) and os.path.getsize(web_path) > 0:
            return web_path
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return video_path


def _scale_corners_to_video(corners, template_path, video_path):
    """Scale court corners from template resolution to video frame resolution."""
    template_img = imread_safe(template_path)
    if template_img is None:
        return corners

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return corners
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tmpl_h, tmpl_w = template_img.shape[:2]
    if tmpl_w == frame_w and tmpl_h == frame_h:
        return corners

    sx = frame_w / tmpl_w
    sy = frame_h / tmpl_h
    return [(int(x * sx), int(y * sy)) for x, y in corners]


def _max_template_match_score(video_path, template_path, max_samples=24):
    """Measure whether an uploaded template can pass the runtime court-view gate."""
    template = imread_safe(template_path, cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(video_path)
    if template is None or not cap.isOpened():
        if cap.isOpened():
            cap.release()
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    template = cv2.resize(template, (frame_w, frame_h))
    best_score = -1.0
    for frame_index in _sample_frame_indices(total_frames, max_samples=max_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max())
        best_score = max(best_score, score)
    cap.release()
    return None if best_score < 0 else best_score


def run_analysis(video_path, template_path, corners, options, progress_cb=None,
                 state_cb=None, output_dir=None, cleanup_outputs=True):
    """Run the full analysis pipeline headlessly.

    Args:
        video_path: Path to the input video file.
        template_path: Path to the court template image.
        corners: List of 4 (x, y) court corner tuples (template resolution).
        options: dict of analysis options (mirrors CLI flags).
        progress_cb: Optional callable(frame_count, total_frames).

    Returns:
        dict with output file paths.
    """
    _ensure_dependencies()
    if cleanup_outputs:
        _cleanup_old_outputs()

    match_score = _max_template_match_score(video_path, template_path)
    if match_score is not None and match_score < 0.75:
        raise RuntimeError(
            f"球场模板与当前视频不匹配（抽样最高匹配度 {match_score:.3f}，"
            "运行要求 0.750）。请清空已上传的模板图，再从当前视频自动选择球场帧。"
        )

    corners = _scale_corners_to_video(corners, template_path, video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    roi_corners = compute_expanded_roi(corners, (frame_h, frame_w, 3))
    mapper = CourtMapper(corners)
    mid_height = mapper.mid_height

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or _default_analysis_output_dir(video_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "court_annotations.txt"), "w") as f:
        f.write(f"corners={corners}\n")
        f.write(f"roi_corners={roi_corners}\n")
        f.write(f"mid_height={mid_height}\n")

    language = options.get("language", "zh")
    pose_family = options.get("pose_family", "yolo-pose")
    pose_mode = options.get("pose_mode", "balanced")
    yolo_pose_model = options.get("yolo_pose_model", "weights/yolo11n-pose.pt")
    ball_model = options.get("ball_model", "weights/yolo11s-ball.pt")
    keep_audio = options.get("audio", True)
    show_skeletons = options.get("show_skeletons", True)
    show_player_trajectories = options.get("show_player_trajectories", True)
    show_court_trajectory = options.get("show_court_trajectory", True)
    show_shuttlecock_trajectory = options.get("show_shuttlecock_trajectory", True)
    show_player_stats = options.get("show_player_stats", True)
    show_pose_roi = options.get("show_pose_roi", True)
    visualize_positions = options.get("visualize_positions", True)
    output_video_style = options.get("output_video_style", "annotated")
    pose_imgsz = int(options.get("pose_imgsz", 960))
    pose_sample_hz = float(options.get("pose_sample_hz", 10.0))
    pose_conf = float(options.get("pose_conf", 0.15))
    far_player_enhancement = bool(options.get("far_player_enhancement", False))
    far_pose_roi = options.get("far_pose_roi", (0.12, 0.30, 0.86, 0.82))
    match_mode = options.get("match_mode", "singles")
    tracker_backend = options.get("tracker_backend", "court_association")
    enable_bytetrack = bool(options.get("enable_bytetrack", False))
    lock_match_roster = bool(options.get("lock_match_roster", True))
    roster_stable_frames = int(options.get("roster_stable_frames", 2))
    shuttle_detector = options.get("shuttle_detector", "yolo")
    if shuttle_detector not in {"yolo", "tracknet_v3"}:
        raise ValueError("shuttle_detector must be 'yolo' or 'tracknet_v3'.")
    tracknet_measurements_path = None
    if shuttle_detector == "tracknet_v3":
        if state_cb is not None:
            state_cb({"phase": "tracknet_preprocessing"})
        tracknet_measurements_path = _prepare_tracknet_v3_raw(video_path, output_dir)
    if state_cb is not None:
        state_cb({"phase": "human_tracking"})

    system = BadmintonAnalysisSystem(
        video_path,
        show_display=False,
        show_skeletons=show_skeletons,
        show_player_trajectories=show_player_trajectories,
        show_court_trajectory=show_court_trajectory,
        show_shuttlecock_trajectory=show_shuttlecock_trajectory,
        show_player_stats=show_player_stats,
        show_performance_stats=False,
        save_images=False,
        language=language,
        output_dir=output_dir,
        ball_model_path=ball_model,
        template_path=template_path,
        pose_mode=pose_mode,
        pose_family=pose_family,
        yolo_pose_model=yolo_pose_model,
        show_pose_roi=show_pose_roi,
        output_video_style=output_video_style,
        pose_imgsz=pose_imgsz,
        pose_sample_hz=pose_sample_hz,
        pose_conf=pose_conf,
        far_player_enhancement=far_player_enhancement,
        far_pose_roi=far_pose_roi,
        match_mode=match_mode,
        tracker_backend=tracker_backend,
        enable_bytetrack=enable_bytetrack,
        lock_match_roster=lock_match_roster,
        roster_stable_frames=roster_stable_frames,
        shuttle_detector=shuttle_detector,
        tracknet_measurements_path=tracknet_measurements_path,
    )
    system.keep_audio = keep_audio
    system.process_video(progress_callback=progress_cb, state_callback=state_cb)

    # This is an opaque business correlation key, not an identity input for
    # visual tracking. Storing it in the result manifest lets the business
    # service attach post-match claims without exposing its participant list to
    # the GPU worker.
    match_session_ref = options.get("match_session_ref")
    if match_session_ref and os.path.isfile(system.metadata_path):
        import json
        with open(system.metadata_path, "r", encoding="utf-8") as source:
            metadata = json.load(source)
        metadata.setdefault("match_context", {})["match_session_ref"] = str(match_session_ref)
        from badminton_analysis.data.writer import write_json
        write_json(system.metadata_path, metadata)

    warnings = []
    has_detections = os.path.isfile(system.detections_path) and os.path.getsize(system.detections_path) > 0
    if not has_detections:
        warnings.append(
            "没有生成有效的球场检测数据，因此无法生成热力图和散点图。"
            "请使用当前视频中的模板帧，并检查四个球场角点。"
        )

    if visualize_positions and has_detections:
        if language == "en":
            from badminton_analysis.visualization.player_positions_en import analyze_player_positions
        else:
            from badminton_analysis.visualization.player_positions_zh import analyze_player_positions
        vis_dir = os.path.join(output_dir, "position_visualizations")
        visualization_ok = analyze_player_positions(system.detections_path, vis_dir, fps=system.fps)
        if not visualization_ok:
            warnings.append(
                "已经生成位置检测数据，但图表渲染失败。请打开右下角后台输出查看详情。"
            )

    web_video_path = _reencode_for_browser(system.output_video_path, output_dir)
    if not os.path.isfile(web_video_path) or os.path.getsize(web_video_path) == 0:
        raise RuntimeError("标注视频导出失败，未生成可播放文件。请打开右下角后台输出查看详情。")

    result = {
        "output_dir": output_dir,
        "video": web_video_path,
        "metadata": system.metadata_path,
        "detections": system.detections_path,
        "tracknet_raw_csv": tracknet_measurements_path,
        "performance_report": (getattr(system, "performance_report", None) or {}).get("report_path"),
        "derived": getattr(system, "offline_artifacts", None),
        "visualizations": [],
        "warnings": warnings,
    }

    vis_dir = os.path.join(output_dir, "position_visualizations")
    if os.path.isdir(vis_dir):
        for root, _dirs, files in os.walk(vis_dir):
            for fname in sorted(files):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    result["visualizations"].append(os.path.join(root, fname))

    return result
