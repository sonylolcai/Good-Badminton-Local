import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.cancellation import AnalysisCancelled, raise_if_cancelled
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
# Court geometry is deliberately an initialization step, not a full-video
# analysis pass. Eight evenly spread frames are sufficient to find a stable
# fixed camera shot and keep the WebUI responsive; the manual four-corner path
# remains available when none is suitable.
_MAX_COURT_FRAME_SAMPLES = 8
_COURT_DETECTION_SIZE = (1080, 720)
_SAFE_OUTPUT_STEM_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')

_dependencies_loaded = False
_TRACKNET_EVENT_PREFIX = "GOOD_BADMINTON_TRACKNET_EVENT="


def _emit_analysis_stage(callback, phase, stage, details=None):
    """Emit a small, durable stage update when a callback is configured."""
    if callback is not None:
        callback({
            "phase": phase,
            "stage": stage,
            "stage_detail": dict(details or {}),
        })


def _forward_tracknet_output(raw_line, state_cb):
    """Keep TrackNet logs visible and turn structured events into job state.

    The separate TrackNet process is deliberately left as an executable
    boundary.  A JSON-prefixed stdout event gives the API a way to persist
    real per-stage timestamps and inference progress without guessing from
    elapsed wall time or scraping human-oriented log text.
    """
    line = raw_line.rstrip()
    if not line:
        return
    print(line, flush=True)
    if not line.startswith(_TRACKNET_EVENT_PREFIX) or state_cb is None:
        return
    try:
        event = json.loads(line[len(_TRACKNET_EVENT_PREFIX):])
    except json.JSONDecodeError:
        print("TrackNet emitted an invalid structured progress event.", flush=True)
        return
    stage = str(event.pop("stage", "unknown"))
    _emit_analysis_stage(
        state_cb,
        "tracknet_preprocessing",
        f"tracknet.{stage}",
        event,
    )


def _prepare_tracknet_v3_raw(video_path, output_dir, cancel_cb=None, state_cb=None):
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
    chunk_frames = int(os.environ.get("GOOD_BADMINTON_TRACKNET_CHUNK_FRAMES", "96"))
    if batch_size <= 0 or background_samples <= 0 or chunk_frames <= 0:
        raise RuntimeError("TrackNetV3 批量大小、背景采样数和分块帧数必须为正整数。")
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
        "--chunk-frames", str(chunk_frames),
        "--overwrite",
    ]
    print("TrackNetV3 primary shuttle detector:", subprocess.list2cmdline(command))
    _emit_analysis_stage(
        state_cb,
        "tracknet_preprocessing",
        "tracknet.launch",
        {
            "batch_size": batch_size,
            "background_sample_count": background_samples,
            "chunk_frames": chunk_frames,
        },
    )
    runtime_env = os.environ.copy()
    runtime_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=runtime_env,
    )
    output_lines = queue.Queue()
    # The child process is intentionally a boundary, but a bare exit code is
    # not actionable after an API/WebUI restart.  Keep only a bounded tail so
    # a failed remote job can persist the immediate TrackNet diagnostic in its
    # durable job record without retaining an unbounded duplicate log.
    output_tail = deque(maxlen=80)

    def read_output():
        assert process.stdout is not None
        for raw_line in iter(process.stdout.readline, ""):
            output_lines.put(raw_line)
        process.stdout.close()

    output_reader = threading.Thread(target=read_output, name="tracknet-output", daemon=True)
    output_reader.start()
    try:
        while process.poll() is None or not output_lines.empty():
            try:
                raw_line = output_lines.get(timeout=0.2)
                output_tail.append(raw_line.rstrip())
                _forward_tracknet_output(raw_line, state_cb)
            except queue.Empty:
                pass
            if cancel_cb is not None and cancel_cb():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise AnalysisCancelled("TrackNetV3 推理已中断。")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        output_reader.join(timeout=2)
        while not output_lines.empty():
            raw_line = output_lines.get_nowait()
            output_tail.append(raw_line.rstrip())
            _forward_tracknet_output(raw_line, state_cb)
    if process.returncode:
        diagnostic = "\n".join(line for line in output_tail if line).strip()
        if diagnostic:
            # Keep the exception reasonably small: it is copied into both the
            # remote job JSON and the business-side task ledger.
            diagnostic = diagnostic[-6000:]
            raise RuntimeError(
                f"TrackNetV3 推理失败（退出码 {process.returncode}）。"
                f"\n子进程日志末尾：\n{diagnostic}"
            )
        raise RuntimeError(f"TrackNetV3 推理失败（退出码 {process.returncode}），且未输出诊断日志。")
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
                                max_samples=_MAX_COURT_FRAME_SAMPLES, progress_cb=None):
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
    for sample_number, frame_index in enumerate(sampled_indices, start=1):
        if progress_cb is not None:
            progress_cb(sample_number, len(sampled_indices), frame_index)
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


def prepare_court_from_video(video_path, progress_cb=None):
    """Create a template from a video frame, then reuse the normal court workflow."""
    selected = extract_best_court_template(video_path, progress_cb=progress_cb)
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


def _reencode_for_browser(video_path, output_dir, cancel_cb=None):
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
        process = subprocess.Popen(
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
        )
    except OSError:
        return video_path

    try:
        deadline = time.monotonic() + 300
        while process.poll() is None:
            if cancel_cb is not None and cancel_cb():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise AnalysisCancelled("浏览器视频转码已中断。")
            if time.monotonic() >= deadline:
                process.terminate()
                process.wait(timeout=10)
                return video_path
            time.sleep(0.2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    if process.returncode == 0 and os.path.isfile(web_path) and os.path.getsize(web_path) > 0:
        return web_path
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


def _max_template_match_score(video_path, template_path, max_samples=24, cancel_cb=None):
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
        raise_if_cancelled(cancel_cb)
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
                 state_cb=None, cancel_cb=None, output_dir=None, cleanup_outputs=True):
    """Run the full analysis pipeline headlessly.

    Args:
        video_path: Path to the input video file.
        template_path: Path to the court template image.
        corners: List of 4 (x, y) court corner tuples (template resolution).
        options: dict of analysis options (mirrors CLI flags).
        progress_cb: Optional callable(frame_count, total_frames).
        cancel_cb: Optional callable() returning True after an interrupt.

    Returns:
        dict with output file paths.
    """
    pipeline_started = time.perf_counter()
    pipeline_components = {}

    def record_pipeline_component(name, started):
        entry = pipeline_components.setdefault(name, {"calls": 0, "elapsed_seconds": 0.0})
        entry["calls"] += 1
        entry["elapsed_seconds"] += max(0.0, time.perf_counter() - started)

    raise_if_cancelled(cancel_cb)
    runtime_t0 = time.perf_counter()
    _emit_analysis_stage(state_cb, "preparing", "runtime_dependencies")
    _ensure_dependencies()
    record_pipeline_component("runtime_dependencies", runtime_t0)
    if cleanup_outputs:
        _emit_analysis_stage(state_cb, "preparing", "output_retention_cleanup")
        _cleanup_old_outputs()

    template_check_t0 = time.perf_counter()
    _emit_analysis_stage(state_cb, "preparing", "template_compatibility")
    match_score = _max_template_match_score(video_path, template_path, cancel_cb=cancel_cb)
    record_pipeline_component("template_compatibility_check", template_check_t0)
    raise_if_cancelled(cancel_cb)
    if match_score is not None and match_score < 0.75:
        raise RuntimeError(
            f"球场模板与当前视频不匹配（抽样最高匹配度 {match_score:.3f}，"
            "运行要求 0.750）。请清空已上传的模板图，再从当前视频自动选择球场帧。"
        )

    court_setup_t0 = time.perf_counter()
    _emit_analysis_stage(state_cb, "preparing", "court_configuration")
    corners = _scale_corners_to_video(corners, template_path, video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    roi_corners = compute_expanded_roi(corners, (frame_h, frame_w, 3))
    sport_id = "tennis" if options.get("sport_id") == "tennis" else "badminton"
    from api.vision_profiles import get_vision_profile
    vision_profile = get_vision_profile(sport_id)
    court_dimensions = tuple(float(value) for value in vision_profile.court_dimensions_m)
    calibration_world_points_m = [
        [0.0, 0.0], [court_dimensions[0], 0.0],
        [court_dimensions[0], court_dimensions[1]], [0.0, court_dimensions[1]],
    ]
    mapper = CourtMapper(
        corners,
        court_dimensions=court_dimensions,
        world_points_m=calibration_world_points_m,
    )
    mid_height = mapper.mid_height

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or _default_analysis_output_dir(video_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "court_annotations.txt"), "w") as f:
        f.write(f"corners={corners}\n")
        f.write(f"roi_corners={roi_corners}\n")
        f.write(f"mid_height={mid_height}\n")
    record_pipeline_component("court_configuration", court_setup_t0)

    language = options.get("language", "zh")
    pose_family = options.get("pose_family", "yolo-pose")
    pose_mode = options.get("pose_mode", "balanced")
    yolo_pose_model = options.get("yolo_pose_model", "weights/yolo11n-pose.pt")
    ball_model = options.get("ball_model", "weights/yolo11s-ball.pt")
    keep_audio = options.get("audio", True)
    generate_annotated_video = bool(options.get("generate_annotated_video", False))
    browser_video_reencode = bool(options.get("browser_video_reencode", False))
    # The second conversion has no source when primary video output is off.
    # Normalize it so a data-only task remains a valid low-latency request.
    if not generate_annotated_video:
        browser_video_reencode = False
    show_skeletons = options.get("show_skeletons", True)
    show_player_trajectories = options.get("show_player_trajectories", True)
    show_court_trajectory = options.get("show_court_trajectory", True)
    show_shuttlecock_trajectory = options.get("show_shuttlecock_trajectory", True)
    show_player_stats = options.get("show_player_stats", True)
    show_pose_roi = options.get("show_pose_roi", True)
    visualize_positions = options.get("visualize_positions", True)
    output_video_style = options.get("output_video_style", "annotated")
    pose_imgsz = int(options.get("pose_imgsz", 960))
    # All evidence-producing components share this cadence.  The legacy pose
    # option is retained as a fallback for saved tasks submitted before this
    # option was introduced.
    analysis_sample_hz = float(options.get("analysis_sample_hz", options.get("pose_sample_hz", 10.0)))
    pose_conf = float(options.get("pose_conf", 0.15))
    far_player_enhancement = bool(options.get("far_player_enhancement", False))
    far_pose_roi = options.get("far_pose_roi", (0.12, 0.30, 0.86, 0.82))
    match_mode = options.get("match_mode", "singles")
    # ByteTrack only receives already-sampled pose detections. The shared
    # timestamp cadence remains the sole detector/tracker update budget.
    tracker_backend = options.get("tracker_backend", "bytetrack")
    enable_bytetrack = bool(options.get("enable_bytetrack", True))
    lock_match_roster = bool(options.get("lock_match_roster", True))
    roster_stable_frames = int(options.get("roster_stable_frames", 2))
    shuttle_detector = options.get("shuttle_detector", "yolo")
    if shuttle_detector not in {"none", "yolo", "tracknet_v3"}:
        raise ValueError("shuttle_detector must be 'none', 'yolo', or 'tracknet_v3'.")
    movement_rally_settle_seconds = float(options.get("movement_rally_settle_seconds", 0.7))
    enable_huji_play_state = bool(options.get("enable_huji_play_state", True))
    tracknet_measurements_path = None
    if shuttle_detector == "tracknet_v3":
        tracknet_t0 = time.perf_counter()
        tracknet_measurements_path = _prepare_tracknet_v3_raw(
            video_path,
            output_dir,
            cancel_cb=cancel_cb,
            state_cb=state_cb,
        )
        record_pipeline_component("tracknet_v3_full_temporal", tracknet_t0)
    _emit_analysis_stage(state_cb, "human_tracking", "human_tracking_setup")

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
        pose_sample_hz=analysis_sample_hz,
        analysis_sample_hz=analysis_sample_hz,
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
        movement_rally_settle_seconds=movement_rally_settle_seconds,
        enable_huji_play_state=enable_huji_play_state,
        generate_annotated_video=generate_annotated_video,
        browser_video_reencode=browser_video_reencode,
        sport_id=sport_id,
        court_dimensions=court_dimensions,
        calibration_world_points_m=calibration_world_points_m,
        coordinate_system_id=vision_profile.coordinate_system_id,
    )
    system.keep_audio = keep_audio
    execution_metrics = system.process_video(
        progress_callback=progress_cb,
        state_callback=state_cb,
        cancel_callback=cancel_cb,
    )
    raise_if_cancelled(cancel_cb)

    # This is an opaque business correlation key, not an identity input for
    # visual tracking. Storing it in the result manifest lets the business
    # service attach post-match claims without exposing its participant list to
    # the GPU worker.
    match_session_ref = options.get("match_session_ref")
    if match_session_ref and os.path.isfile(system.metadata_path):
        _emit_analysis_stage(state_cb, "post_processing", "match_context_manifest")
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

    position_evidence_summary = None
    if visualize_positions and has_detections:
        visualizations_t0 = time.perf_counter()
        _emit_analysis_stage(state_cb, "post_processing", "position_visualizations")
        vis_dir = os.path.join(output_dir, "position_visualizations")
        raise_if_cancelled(cancel_cb)
        # v2 files keep a durable spatial track for every person.  Use that
        # contract first so doubles never collapse into legacy upper/lower
        # display slots.  Old result files still use their original renderer.
        from badminton_analysis.visualization.spatial_player_positions import analyze_spatial_track_positions
        position_result = analyze_spatial_track_positions(
            system.detections_path,
            vis_dir,
            fps=system.fps,
            language=language,
        )
        if position_result is None:
            if language == "en":
                from badminton_analysis.visualization.player_positions_en import analyze_player_positions
            else:
                from badminton_analysis.visualization.player_positions_zh import analyze_player_positions
            visualization_ok = analyze_player_positions(system.detections_path, vis_dir, fps=system.fps)
        else:
            visualization_ok = bool(position_result.get("success"))
            position_evidence_summary = position_result.get("summary_path")
            track_summaries = (position_result.get("summary") or {}).get("tracks") or {}
            if match_mode == "doubles" and len(track_summaries) < 4:
                warnings.append(
                    "双打名册未形成四条轨迹：本次只输出已实际追踪到的人员热力图，不能据此做四人能力对比。"
                )
            empty_tracks = [
                track_id for track_id, summary in track_summaries.items()
                if not summary.get("usable_measurements")
            ]
            if empty_tracks:
                warnings.append(
                    "以下轨迹没有满足置信度门槛的真实位置检测，已保留状态但未用于热力图或距离统计："
                    + "、".join(sorted(empty_tracks))
                )
        raise_if_cancelled(cancel_cb)
        if not visualization_ok:
            warnings.append(
                "已经生成位置检测数据，但图表渲染失败。请打开右下角后台输出查看详情。"
            )
        record_pipeline_component("position_visualizations", visualizations_t0)

    web_video_path = None
    if generate_annotated_video:
        if browser_video_reencode:
            browser_reencode_t0 = time.perf_counter()
            _emit_analysis_stage(state_cb, "post_processing", "browser_video_reencode")
            web_video_path = _reencode_for_browser(system.output_video_path, output_dir, cancel_cb=cancel_cb)
            record_pipeline_component("browser_video_reencode", browser_reencode_t0)
        else:
            _emit_analysis_stage(
                state_cb,
                "post_processing",
                "browser_video_reencode_skipped",
                {"reason": "browser_video_reencode=false"},
            )
            web_video_path = system.output_video_path
        if not os.path.isfile(web_video_path) or os.path.getsize(web_video_path) == 0:
            raise RuntimeError("标注视频导出失败，未生成可播放文件。请打开右下角后台输出查看详情。")
    else:
        _emit_analysis_stage(
            state_cb,
            "post_processing",
            "browser_video_reencode_skipped",
            {"reason": "generate_annotated_video=false"},
        )
        warnings.append(
            "本次按高级选项跳过标注视频绘制与编码；人物轨迹、速度、距离、JSONL 和统计结果已照常保存。"
        )

    execution_metrics = dict(execution_metrics or {})
    component_metrics = execution_metrics.setdefault("components", {})
    component_metrics.update(pipeline_components)
    execution_metrics["pipeline_total_elapsed_seconds"] = round(
        max(0.0, time.perf_counter() - pipeline_started), 6
    )

    result = {
        "output_dir": output_dir,
        "video": web_video_path,
        "metadata": system.metadata_path,
        "detections": system.detections_path,
        "tracknet_raw_csv": tracknet_measurements_path,
        "performance_report": (getattr(system, "performance_report", None) or {}).get("report_path"),
        "movement_metrics": (getattr(system, "movement_metrics", None) or {}).get("metrics_path"),
        "movement_rallies": (getattr(system, "offline_artifacts", None) or {}).get("rallies_path"),
        "movement_rally_window_sweep": (getattr(system, "offline_artifacts", None) or {}).get("rally_window_sweep_path"),
        "position_evidence_summary": position_evidence_summary,
        "derived": getattr(system, "offline_artifacts", None),
        "execution_metrics": execution_metrics,
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
