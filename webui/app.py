import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from webui.log_capture import get_backend_logs, install_backend_log_capture

install_backend_log_capture()

import cv2
import gradio as gr
import numpy as np

from webui.pipeline import (
    _max_template_match_score,
    imread_safe,
    prepare_court,
    prepare_court_from_video,
    run_analysis,
)
from webui.remote_gpu import RemoteAnalysisError, remote_gpu_config, run_remote_analysis
from webui.shot_review import (
    REVIEW_DECISIONS,
    SHOT_TYPES,
    add_manual_candidate,
    analysis_run_label,
    candidate_at_table_row,
    candidate_choices,
    candidate_table,
    candidate_view,
    create_or_load_review_session,
    default_analysis_run,
    find_analysis_runs,
    merge_review_candidates,
    review_summary,
    save_human_review,
    split_review_candidate,
    timeline_state,
)
from webui.task_ledger import BusinessTaskLedger

_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
_MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB


def _write_execution_metadata(result, execution):
    """Persist the execution origin so local fallback is never invisible."""
    metadata_path = result.get("metadata")
    if not metadata_path or not os.path.isfile(metadata_path):
        return
    with open(metadata_path, "r", encoding="utf-8") as source:
        metadata = json.load(source)
    metadata["execution"] = execution
    with open(metadata_path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _validate_file_size(path, max_bytes, label="File"):
    if path and os.path.isfile(path):
        size = os.path.getsize(path)
        if size > max_bytes:
            max_mb = max_bytes / (1024 * 1024)
            raise gr.Error(f"{label} exceeds {max_mb:.0f} MB limit.")


def _bgr_to_rgb(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_court(video_file, template_file, language="zh"):
    """Use an uploaded template when supplied; otherwise generate one from video."""
    text = _UI_TEXT.get(language, _UI_TEXT["zh"])
    if template_file:
        _validate_file_size(template_file, _MAX_IMAGE_BYTES, "Template image")
        result = prepare_court(template_file)
        template_path = template_file
        selection = None
    else:
        if video_file is None:
            raise gr.Error(text["need_video"])
        _validate_file_size(video_file, _MAX_VIDEO_BYTES, "Video")
        result = prepare_court_from_video(video_file)
        template_path = result["template_path"]
        selection = result["selection"]

    preview_rgb = _bgr_to_rgb(result["preview_bgr"])
    corners = result["corners"]
    if corners is None:
        if template_path:
            gr.Warning(text["auto_fail"])
            template_img = imread_safe(template_path)
            preview_rgb = _bgr_to_rgb(template_img)
            status = text["video_template_fail"] if selection else text["auto_fail"]
        else:
            status = text["video_extract_fail"]
    elif selection:
        status = text["video_template_ok"].format(
            selection["frame_index"], selection["time_sec"], len(corners)
        )
    else:
        status = text["template_ok"].format(len(corners))
    return preview_rgb, corners, template_path, status, bool(corners)


def on_court_image_select(corners_state, template_file, evt: gr.SelectData):
    """Accumulate clicked points and redraw markers on the template."""
    if not template_file:
        raise gr.Error("Please generate or upload a court template first.")
    if corners_state is None:
        corners_state = []

    if len(corners_state) >= 4:
        corners_state = []

    x, y = evt.index
    corners_state.append((x, y))
    if len(corners_state) == 4:
        corners_state = _normalize_court_corners(corners_state)

    template_img = imread_safe(template_file)
    preview = template_img.copy()
    for idx, pt in enumerate(corners_state, 1):
        cv2.circle(preview, pt, 6, (0, 0, 255), -1)
        cv2.putText(preview, str(idx), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    if len(corners_state) > 1:
        cv2.polylines(preview, [np.array(corners_state, dtype=np.int32)],
                      len(corners_state) == 4, (0, 255, 0), 2)

    status = f"Corners selected: {len(corners_state)}/4"
    if len(corners_state) == 4:
        status += " — corners locked. Click 'Apply Manual Corners' or re-click to restart."

    corners_out = corners_state if len(corners_state) == 4 else None
    return _bgr_to_rgb(preview), corners_state, corners_out, status


def _normalize_court_corners(points):
    """Normalize arbitrary clicks to TL, TR, BR, BL order."""
    top = sorted(sorted(points, key=lambda point: point[1])[:2], key=lambda point: point[0])
    bottom = sorted(sorted(points, key=lambda point: point[1])[2:], key=lambda point: point[0])
    return [top[0], top[1], bottom[1], bottom[0]]


def apply_manual_corners(template_file, corners_state):
    """Re-run court resolution with manually clicked corners."""
    if not template_file:
        raise gr.Error("Please generate or upload a court template first.")
    if not corners_state or len(corners_state) != 4:
        raise gr.Error("Please click exactly 4 corners on the court image first.")

    result = prepare_court(template_file, manual_corners=corners_state)
    preview_rgb = _bgr_to_rgb(result["preview_bgr"])
    corners = result["corners"]
    if corners is None:
        raise gr.Error("Failed to resolve corners. Please try again.")
    return preview_rgb, corners, True


def reset_court_selection(language="zh"):
    """Discard stale corners when either input file changes."""
    text = _UI_TEXT.get(language, _UI_TEXT["zh"])
    return None, None, [], None, text["corner_none"], False


def ensure_court_for_analysis(video_file, template_path, corners, click_corners, language="zh"):
    """Validate the selected template and fall back to a frame from the active video."""
    text = _UI_TEXT.get(language, _UI_TEXT["zh"])
    if video_file is None:
        raise gr.Error(text["need_video"])

    ready = bool(template_path and corners and len(corners) == 4)
    match_score = _max_template_match_score(video_file, template_path) if template_path else None
    if ready and match_score is not None and match_score >= 0.75:
        return gr.update(), corners, click_corners, template_path, gr.update(), True

    if match_score is not None:
        print(
            f"Court template mismatch ({match_score:.3f} < 0.750); "
            "switching to a frame from the active video."
        )
    result = prepare_court_from_video(video_file)
    generated_path = result.get("template_path")
    if not generated_path:
        raise gr.Error(text["video_extract_fail"])

    generated_corners = result.get("corners")
    preview = result.get("preview_bgr")
    if not generated_corners:
        template_img = imread_safe(generated_path)
        preview = template_img if template_img is not None else preview
        status = text["fallback_manual"]
        gr.Warning(status)
        return _bgr_to_rgb(preview), None, [], generated_path, status, False

    selection = result["selection"]
    status = text["fallback_auto"].format(selection["frame_index"], selection["time_sec"])
    gr.Info(status)
    return _bgr_to_rgb(preview), generated_corners, [], generated_path, status, True


def run_full_analysis(analysis_ready, video_file, template_path, corners,
                      pose_family, pose_mode, language, audio, match_mode,
                      output_video_style,
                      pose_imgsz, pose_conf, far_player_enhancement, far_pose_roi,
                      show_skeletons, show_player_trajectories,
                      show_court_trajectory, show_shuttlecock_trajectory,
                      show_player_stats, show_pose_roi, visualize_positions,
                      yolo_pose_model, ball_model,
                      progress=gr.Progress(track_tqdm=False)):
    if not analysis_ready:
        gr.Warning("已切换到当前视频帧，请在预览图中点击四个球场角点，然后应用手动角点。")
        return None, None, None, None
    if video_file is None:
        raise gr.Error("Please upload a video file.")
    if template_path is None:
        raise gr.Error("Please generate or upload a court template first.")
    if not corners or len(corners) != 4:
        raise gr.Error("Please detect or manually annotate court corners first.")

    _validate_file_size(video_file, _MAX_VIDEO_BYTES, "Video")

    try:
        parsed_far_roi = tuple(float(item.strip()) for item in far_pose_roi.split(','))
    except (AttributeError, ValueError) as exc:
        raise gr.Error("远端 ROI 必须是 x1,y1,x2,y2 四个归一化数值。") from exc
    if len(parsed_far_roi) != 4 or not (
        0 <= parsed_far_roi[0] < parsed_far_roi[2] <= 1
        and 0 <= parsed_far_roi[1] < parsed_far_roi[3] <= 1
    ):
        raise gr.Error("远端 ROI 必须满足 0 <= x1 < x2 <= 1 且 0 <= y1 < y2 <= 1。")

    options = {
        "pose_family": pose_family,
        "pose_mode": pose_mode,
        "language": language,
        "audio": audio,
        "match_mode": match_mode,
        "output_video_style": output_video_style,
        "pose_imgsz": int(pose_imgsz),
        "pose_conf": float(pose_conf),
        "far_player_enhancement": far_player_enhancement,
        "far_pose_roi": parsed_far_roi,
        "show_skeletons": show_skeletons,
        "show_player_trajectories": show_player_trajectories,
        "show_court_trajectory": show_court_trajectory,
        "show_shuttlecock_trajectory": show_shuttlecock_trajectory,
        "show_player_stats": show_player_stats,
        "show_pose_roi": show_pose_roi,
        "visualize_positions": visualize_positions,
        "yolo_pose_model": yolo_pose_model or "weights/yolo11n-pose.pt",
        "ball_model": ball_model or "weights/yolo11s-ball.pt",
    }

    events = queue.Queue()
    finished = threading.Event()
    outcome = {}
    started = time.monotonic()

    def publish(event):
        events.put(event)

    def worker():
        ledger = None
        business_task_id = None
        try:
            remote_output_dir = os.path.join(
                "outputs", "remote_jobs", datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            ledger = BusinessTaskLedger()
            business_task_id = ledger.start_task(
                output_dir=remote_output_dir,
                remote_base_url=remote_gpu_config()["base_url"],
            )

            def remote_status(event):
                ledger.record_remote_event(business_task_id, event)
                publish({**event, "business_task_id": business_task_id})

            try:
                result = run_remote_analysis(
                    video_path=video_file, template_path=template_path, corners=corners,
                    options=options, output_dir=remote_output_dir, status_cb=remote_status,
                    business_task_id=business_task_id,
                )
                outcome["result"] = result
                ledger.record_terminal(business_task_id, status="succeeded")
                publish({"mode": "remote_gpu", "phase": "succeeded", "business_task_id": business_task_id})
            except RemoteAnalysisError as remote_exc:
                fallback_reason = str(remote_exc)
                print(f"Remote GPU analysis failed; falling back locally: {fallback_reason}")
                ledger.record_terminal(
                    business_task_id, status="local_fallback", error={"message": fallback_reason}
                )
                publish({
                    "mode": "local_fallback", "phase": "local_analyzing",
                    "fallback_reason": fallback_reason, "business_task_id": business_task_id,
                })

                def local_progress(frame, total):
                    publish({
                        "mode": "local_fallback", "phase": "local_analyzing",
                        "processed_frames": frame, "total_frames": total,
                        "ratio": round(frame / total, 4) if total else 0.0,
                        "fallback_reason": fallback_reason,
                    })

                result = run_analysis(
                    video_path=video_file, template_path=template_path, corners=corners,
                    options=options, progress_cb=local_progress,
                )
                _write_execution_metadata(result, {
                    "mode": "local_fallback", "fallback_used": True,
                    "remote_failure": fallback_reason,
                })
                outcome["result"] = result
                ledger.record_terminal(business_task_id, status="succeeded")
                publish({
                    "mode": "local_fallback", "phase": "succeeded",
                    "fallback_reason": fallback_reason, "business_task_id": business_task_id,
                })
        except Exception as exc:
            traceback.print_exc()
            outcome["error"] = exc
            if ledger is not None and business_task_id is not None:
                ledger.record_terminal(business_task_id, status="failed", error={"message": str(exc)})
            publish({"phase": "failed", "error": str(exc), "business_task_id": business_task_id})
        finally:
            finished.set()

    threading.Thread(target=worker, name="webui-analysis-status", daemon=True).start()
    status = {"mode": "remote_gpu", "phase": "preparing", "elapsed_seconds": 0}
    while not finished.is_set() or not events.empty():
        updated = False
        while True:
            try:
                status.update(events.get_nowait())
                updated = True
            except queue.Empty:
                break
        status["elapsed_seconds"] = round(time.monotonic() - started, 1)
        if updated or status["phase"] == "preparing":
            yield None, None, None, None, None, None, status.copy()
        time.sleep(0.4)

    if "error" in outcome:
        raise gr.Error(f"分析失败：{outcome['error']}") from outcome["error"]
    result = outcome["result"]

    for warning in result.get("warnings", []):
        gr.Warning(warning)

    output_video = result["video"] if os.path.isfile(result["video"]) else None
    viz_images = [img for img in result["visualizations"] if os.path.isfile(img)]

    metadata_content = None
    if os.path.isfile(result["metadata"]):
        with open(result["metadata"], "r", encoding="utf-8") as f:
            metadata_content = json.load(f)

    detections_file = result["detections"] if os.path.isfile(result["detections"]) else None
    rally_summary, rally_rows = _rally_summary_from_result(result, metadata_content)

    status["phase"] = "succeeded"
    status["elapsed_seconds"] = round(time.monotonic() - started, 1)
    yield output_video, viz_images or None, metadata_content, detections_file, rally_summary, rally_rows, status.copy()


def _rally_summary_from_result(result, metadata):
    """Build a compact, review-only rally table from local derived artifacts.

    A remote result may carry server-side absolute artifact paths in its
    metadata.  The WebUI therefore first resolves the local download folder,
    and only rebuilds the small derived files from immutable detections when a
    legacy GPU service did not return them.
    """
    metadata = metadata or {}
    detections_path = Path(result.get("detections") or "")
    if not detections_path.is_file():
        return "### 回合与拍数\n未找到本地检测数据，暂时无法生成候选回合。", []
    derived = metadata.get("derived") or result.get("derived") or {}
    rally_path = Path(derived.get("rallies_path") or "")
    local_rally_path = detections_path.parent / "derived" / "rallies_v2.jsonl"
    if not rally_path.is_file() and local_rally_path.is_file():
        rally_path = local_rally_path
    if not rally_path.is_file():
        from badminton_analysis.analysis.offline_shot_reconstruction import generate_offline_artifacts

        derived = generate_offline_artifacts(detections_path)
        rally_path = Path(derived["rallies_path"])
        metadata["derived"] = derived
        metadata_path = Path(result.get("metadata") or "")
        if metadata_path.is_file():
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        with rally_path.open(encoding="utf-8") as source:
            rallies = [json.loads(line) for line in source if line.strip()]
    except (OSError, ValueError) as error:
        return f"### 回合与拍数\n读取候选回合失败：`{error}`", []

    rows = [
        [
            item.get("rally_id"),
            round(float(item.get("start_time_sec", 0.0)), 2),
            round(float(item.get("end_time_sec", 0.0)), 2),
            int(item.get("shot_count", 0)),
            int(item.get("observed_shot_count", 0)),
            int(item.get("motion_inferred_shot_count", 0)),
            item.get("end_reason"),
            round(float(item.get("confidence", 0.0)), 3),
        ]
        for item in rallies
    ]
    shot_count = sum(int(item.get("shot_count", 0)) for item in rallies)
    inferred_count = sum(int(item.get("motion_inferred_shot_count", 0)) for item in rallies)
    return (
        "### 回合与拍数（候选，待人工复核）\n"
        f"共 **{len(rallies)}** 个候选回合、**{shot_count}** 个候选拍；"
        f"其中 **{inferred_count}** 拍由缺球动作约束补出。"
        "回合边界、球种和二维球速都不会进入得分或能力统计，直到人工确认。",
        rows,
    )


_UI_TEXT = {
    "zh": {
        "title": "# Good Badminton — AI 羽毛球分析系统",
        "inputs": "### 输入",
        "video": "比赛视频",
        "template": "球场模板图像（可选）",
        "settings": "### 分析设置",
        "pose_family": "姿态模型",
        "pose_mode": "姿态模式",
        "language": "语言 / Language",
        "audio": "保留音频",
        "output_style": "输出视频样式",
        "pose_imgsz": "YOLO Pose 输入尺寸",
        "pose_conf": "远端人体置信阈值",
        "far_player_enhancement": "远端球员增强（全场640 + 远端ROI 640）",
        "far_pose_roi": "远端 ROI（相对姿态区域 x1,y1,x2,y2）",
        "advanced": "高级选项",
        "skeletons": "显示骨架",
        "player_traj": "显示球员轨迹",
        "court_traj": "显示球场轨迹",
        "shuttle_traj": "显示羽毛球轨迹",
        "player_stats": "显示球员统计",
        "pose_roi": "显示姿态 ROI",
        "viz_positions": "生成热力图和散点图",
        "yolo_pose_path": "YOLO 姿态模型路径",
        "ball_path": "羽毛球检测模型路径",
        "step1": "### 第一步 — 球场检测",
        "detect_btn": "自动选择视频帧并检测球场",
        "court_preview": "球场预览（点击标注角点）",
        "corner_status": "角点状态",
        "corner_none": "尚未检测到角点。",
        "apply_btn": "应用手动角点",
        "step2": "### 第二步 — 运行分析",
        "run_btn": "运行分析",
        "results": "### 结果",
        "out_video": "标注视频",
        "out_status": "分析进度（执行任务）",
        "out_gallery": "热力图和散点图",
        "out_metadata": "元数据",
        "out_detections": "检测数据 (JSONL)",
        "auto_ok": "自动检测到 {} 个角点。",
        "auto_fail": "自动检测失败 — 请手动点击 4 个角点。",
        "need_video": "请先上传比赛视频；也可以上传球场模板图进行检测。",
        "template_ok": "已使用上传的模板图，自动检测到 {} 个角点。",
        "video_template_ok": "已自动选择视频第 {} 帧（{:.1f} 秒）作为球场模板，检测到 {} 个角点。",
        "video_template_fail": "已从视频选择候选帧，但自动检测失败 — 请在预览图中手动点击 4 个角点。",
        "video_extract_fail": "未能从抽样视频帧中识别出完整球场。请上传模板图，或换一个球场完整且清晰的视频。",
        "fallback_auto": "旧模板与视频不匹配，已自动切换到当前视频第 {} 帧（{:.1f} 秒）并重新检测球场。",
        "fallback_manual": "旧模板与视频不匹配，已切换到当前视频候选帧。自动四角检测未成功，请在预览图上点击四个角点。",
        "manual_ok": "已应用手动角点（{} 个点）。",
        "manual_fail": "失败。",
    },
    "en": {
        "title": "# Good Badminton — AI Badminton Analysis",
        "inputs": "### Inputs",
        "video": "Match Video",
        "template": "Court Template Image (Optional)",
        "settings": "### Analysis Settings",
        "pose_family": "Pose Model Family",
        "pose_mode": "Pose Mode",
        "language": "Language / 语言",
        "audio": "Keep Audio",
        "output_style": "Output Video Style",
        "pose_imgsz": "YOLO Pose Input Size",
        "pose_conf": "Far-player confidence threshold",
        "far_player_enhancement": "Far-player enhancement (full 640 + far ROI 640)",
        "far_pose_roi": "Far ROI (relative pose region x1,y1,x2,y2)",
        "advanced": "Advanced Options",
        "skeletons": "Show Skeletons",
        "player_traj": "Show Player Trajectories",
        "court_traj": "Show Court Trajectory",
        "shuttle_traj": "Show Shuttlecock Trajectory",
        "player_stats": "Show Player Stats",
        "pose_roi": "Show Pose ROI",
        "viz_positions": "Generate Heatmaps & Scatter Plots",
        "yolo_pose_path": "YOLO Pose Model Path",
        "ball_path": "Shuttlecock Model Path",
        "step1": "### Step 1 — Court Detection",
        "detect_btn": "Auto-select Video Frame & Detect Court",
        "court_preview": "Court Preview (click to annotate corners)",
        "corner_status": "Corner Status",
        "corner_none": "No corners detected yet.",
        "apply_btn": "Apply Manual Corners",
        "step2": "### Step 2 — Run Analysis",
        "run_btn": "Run Analysis",
        "results": "### Results",
        "out_video": "Annotated Video",
        "out_status": "Analysis Progress (Execution Job)",
        "out_gallery": "Heatmaps & Scatter Plots",
        "out_metadata": "Metadata",
        "out_detections": "Detections (JSONL)",
        "auto_ok": "Auto-detected {} corners.",
        "auto_fail": "Auto-detection failed — click 4 corners manually.",
        "need_video": "Upload a match video first, or upload a court template image to detect.",
        "template_ok": "Using the uploaded template image; auto-detected {} corners.",
        "video_template_ok": "Auto-selected video frame {} ({:.1f}s) as the court template; detected {} corners.",
        "video_template_fail": "A video frame was selected, but auto-detection failed — click 4 corners on the preview manually.",
        "video_extract_fail": "No complete court was detected in the sampled video frames. Upload a template image or use a clearer full-court video.",
        "fallback_auto": "The old template did not match. Switched to frame {} ({:.1f}s) from the active video and detected the court again.",
        "fallback_manual": "The old template did not match. Switched to a frame from the active video; click four court corners on the preview.",
        "manual_ok": "Manual corners applied ({} points).",
        "manual_fail": "Failed.",
    },
}


_APP_CSS = """
#backend-console-trigger {
  position: fixed;
  right: 18px;
  bottom: 18px;
  z-index: 80;
  width: 44px;
  min-width: 44px;
  height: 44px;
}
#backend-console-trigger {
  width: 44px;
  min-width: 44px;
  height: 44px;
  padding: 0;
  border-radius: 999px;
  border: 0;
  background: #17171b;
  color: #f7f7f8;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.24);
  font-size: 0;
}
#backend-console-trigger::before {
  content: "›_";
  font: 700 14px/1 ui-monospace, SFMono-Regular, Consolas, monospace;
}
#backend-console-trigger:hover { background: #2a2a31; }
#backend-console-trigger:focus-visible { outline: 3px solid #7c83ff; outline-offset: 2px; }
#backend-console-panel {
  position: fixed;
  right: 18px;
  bottom: 72px;
  z-index: 70;
  width: min(720px, calc(100vw - 36px));
  max-height: min(68vh, 640px);
  padding: 12px;
  overflow: hidden;
  border-radius: 14px;
  background: #111114;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.34);
  transform-origin: bottom right;
  animation: console-in 180ms cubic-bezier(.22, 1, .36, 1);
}
#backend-console-panel .console-title,
#backend-console-panel .console-title p { color: #f5f5f7; font-weight: 650; margin: 0; }
#backend-console-panel textarea {
  min-height: min(52vh, 480px);
  resize: none;
  border: 0;
  background: #0a0a0c;
  color: #d7f9df;
  font: 12px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace;
}
#backend-console-close {
  flex: 0 0 auto;
  width: auto;
  min-width: 64px;
  max-width: 80px;
}
@keyframes console-in {
  from { opacity: 0; transform: translateY(8px) scale(.985); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
@media (prefers-reduced-motion: reduce) {
  #backend-console-panel { animation: none; }
}
#review-touch-record {
  margin-top: 8px;
}
#review-touch-record .table-wrap {
  max-height: 245px;
}
#review-editor-panel .block-label {
  margin-bottom: 2px;
}
#review-editor-panel .prose {
  margin: 2px 0;
}
#review-frame-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 6px 0 2px;
}
#review-frame-controls button {
  min-width: 96px;
}
#review-frame-controls .frame-rate-hint {
  color: #667085;
  font-size: 0.82rem;
}
"""


def toggle_backend_console(is_open):
    next_open = not bool(is_open)
    return next_open, gr.update(visible=next_open), get_backend_logs()


def close_backend_console():
    return False, gr.update(visible=False)


def _switch_language(lang):
    t = _UI_TEXT.get(lang, _UI_TEXT["zh"])
    return [
        gr.update(value=t["title"]),
        gr.update(value=t["inputs"]),
        gr.update(label=t["video"]),
        gr.update(label=t["template"]),
        gr.update(value=t["settings"]),
        gr.update(label=t["pose_family"]),
        gr.update(label=t["pose_mode"]),
        gr.update(label=t["audio"]),
        gr.update(label=t["pose_imgsz"]),
        gr.update(label=t["pose_conf"]),
        gr.update(label=t["far_player_enhancement"]),
        gr.update(label=t["far_pose_roi"]),
        gr.update(label=t["advanced"]),
        gr.update(label=t["skeletons"]),
        gr.update(label=t["player_traj"]),
        gr.update(label=t["court_traj"]),
        gr.update(label=t["shuttle_traj"]),
        gr.update(label=t["player_stats"]),
        gr.update(label=t["pose_roi"]),
        gr.update(label=t["viz_positions"]),
        gr.update(label=t["yolo_pose_path"]),
        gr.update(label=t["ball_path"]),
        gr.update(value=t["step1"]),
        gr.update(value=t["detect_btn"]),
        gr.update(label=t["court_preview"]),
        gr.update(label=t["corner_status"]),
        gr.update(value=t["apply_btn"]),
        gr.update(value=t["step2"]),
        gr.update(value=t["run_btn"]),
        gr.update(value=t["results"]),
        gr.update(label=t["out_status"]),
        gr.update(label=t["out_video"]),
        gr.update(label=t["out_gallery"]),
        gr.update(label=t["out_metadata"]),
        gr.update(label=t["out_detections"]),
    ]


def _review_run_choices():
    """Return newest-first analysis folders eligible for shot review."""
    return [(analysis_run_label(path), path) for path in find_analysis_runs("outputs")]


def _identity_claim_path(analysis_dir):
    return Path(analysis_dir).expanduser().resolve() / "match_identity_claims.json"


def _match_identity_table(analysis_dir):
    """Load post-match editable bindings without rewriting detection evidence."""
    if not analysis_dir:
        return [], "请先选择一场已完成的视频分析。"
    analysis_path = Path(analysis_dir).expanduser().resolve()
    summary_path = analysis_path / "spatial_match_summary.json"
    metadata_path = analysis_path / "metadata.json"
    if not summary_path.is_file():
        return [], "未找到 spatial_match_summary.json；请先完成视频分析。"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    except (OSError, ValueError) as error:
        return [], f"读取 Track ID 汇总失败：{error}"
    persisted = {}
    claim_path = _identity_claim_path(analysis_path)
    if claim_path.is_file():
        try:
            persisted = {
                item.get("track_id"): item
                for item in json.loads(claim_path.read_text(encoding="utf-8")).get("bindings", [])
                if item.get("track_id")
            }
        except (OSError, ValueError):
            persisted = {}
    rows = []
    for item in summary.get("player_style_inputs", []):
        track_id = item.get("track_id")
        binding = persisted.get(track_id, {})
        rows.append([
            track_id,
            binding.get("person_id", item.get("person_id") or ""),
            binding.get("team_id", item.get("team_id") or ""),
            item.get("detected_frames", 0),
            item.get("predicted_frames", 0),
            item.get("missing_frames", 0),
            item.get("distance_m", 0.0),
        ])
    mode = ((metadata.get("temporal_tracking") or {}).get("players") or {}).get("match_mode", "singles")
    return rows, (
        f"模式：**{mode}**。Track ID 是持续身份键；队伍不是由当前球场半区推断。"
        "可填 person_id 与 team_a/team_b；保存后写入单独的人工绑定文件，不改 detections.jsonl。"
    )


def _save_match_identity_table(analysis_dir, table_value):
    if not analysis_dir:
        raise gr.Error("请先选择一场已完成的视频分析。")
    analysis_path = Path(analysis_dir).expanduser().resolve()
    summary_path = analysis_path / "spatial_match_summary.json"
    metadata_path = analysis_path / "metadata.json"
    if not summary_path.is_file():
        raise gr.Error("未找到空间追踪汇总，无法保存身份绑定。")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    known = {item.get("track_id") for item in summary.get("player_style_inputs", [])}
    values = table_value.values.tolist() if hasattr(table_value, "values") else (table_value or [])
    bindings = []
    participants_by_team = {"team_a": set(), "team_b": set()}
    mode = ((metadata.get("temporal_tracking") or {}).get("players") or {}).get("match_mode", "singles")
    max_per_team = 1 if mode == "singles" else 2
    for row in values:
        if not row or not str(row[0] or "").strip():
            continue
        track_id = str(row[0]).strip()
        if track_id not in known:
            raise gr.Error(f"未知 Track ID：{track_id}")
        person_id = str(row[1] or "").strip() or None
        team_id = str(row[2] or "").strip() or None
        if team_id not in {None, "team_a", "team_b"}:
            raise gr.Error("team_id 只能留空、team_a 或 team_b。")
        if team_id:
            participants_by_team[team_id].add(person_id or track_id)
        bindings.append({
            "track_id": track_id,
            "person_id": person_id,
            "team_id": team_id,
            "binding_source": "post_match_human_review",
            "identity_alias": bool(person_id and sum(str(item[1] or "").strip() == person_id for item in values) > 1),
        })
    for team_id, participants in participants_by_team.items():
        if len(participants) > max_per_team:
            raise gr.Error(f"{mode} 模式下 {team_id} 最多确认 {max_per_team} 名不同球员。")
    payload = {
        "schema_version": "1.0",
        "match_mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "human post-match binding; immutable raw detections are not rewritten",
        "bindings": bindings,
    }
    path = _identity_claim_path(analysis_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return f"已保存 {len(bindings)} 条赛后身份/队伍绑定：`{path.name}`。原始检测和机器 Track ID 未被覆盖。"


def _review_summary_markdown(session, selected_id=None):
    summary = review_summary(session)
    active = candidate_choices(session)
    position = next((index + 1 for index, (_, value) in enumerate(active) if value == selected_id), 0)
    selected = f"当前：**{position} / {summary['candidate_count']}**" if position else f"共 **{summary['candidate_count']}** 个候选"
    completed = summary["confirmed"] + summary["corrected"] + summary["uncertain"] + summary["excluded"]
    return (
        f"**复核进度**　{selected}　已处理 **{completed}**　待复核 **{summary['pending']}**　"
        f"确认/修正 **{summary['confirmed'] + summary['corrected']}**　排除 **{summary['excluded']}**"
    )


def _review_details_markdown(view, session):
    details = view["details"]
    proposal = details["proposal"]
    evidence = details.get("evidence") or {}
    source = details.get("candidate_source", "unknown")
    source_display = {
        "spatial_hit_candidate": "人体/空间接近候选",
        "trajectory_turn": "羽毛球轨迹方向变化",
        "trajectory_gap_transition": "羽毛球短暂断检衔接",
        "manual_timeline": "人工补录时间点",
        "manual_merge": "人工合并候选",
    }.get(source, source)
    speed = evidence.get("outbound_speed_px_s")
    duration = evidence.get("flight_duration_sec")
    observations = evidence.get("observation_count", 0)
    limit_translations = {
        "Trajectory uses accepted 2D shuttle detections only.": "仅使用已接受的二维羽毛球轨迹。",
        "The current hit candidate is weak spatial evidence and requires human review.": "当前是弱证据候选，必须人工复核。",
        "Insufficient accepted shuttle detections after the candidate hit.": "候选击球后有效羽毛球观测不足。",
    }
    limits = "；".join(
        limit_translations.get(item, item)
        for item in (evidence.get("limitations") or ["二维轨迹证据有限，需要人工确认。"])
    )
    video = Path(view["clip"]).name if view.get("clip") else "无可播放片段"
    return (
        f"### {details['shot_id']} · {details['hit_time_sec']:.2f}s\n"
        f"自动建议：**{proposal['label_display']}**（建议置信度 {proposal['confidence']:.2f}，仅用于排序）  \n"
        f"候选来源：{source_display}；球观测数：{observations}；"
        f"最大二维速度：{speed if speed is not None else '未知'} px/s；"
        f"观测时长：{duration if duration is not None else '未知'} s。  \n"
        f"复核片段：`{video}`  \n"
        f"限制：{limits}"
    )


def _review_video_message(session, view):
    reference = Path((session.get("source") or {}).get("reference_video") or "").name
    if "skeleton" in reference.lower():
        return (
            "当前结果只有**骨架模式**视频。它不含真人画面；`850259` 当初以 Skeleton only 运行，"
            "不能仅凭 detections.jsonl 还原完整人体骨架。后续请在视频分析页选择“原视频标注”，"
            "即可在这里优先显示“人物 + 骨架 + 羽毛球”的复核片段。"
        )
    if "detect" in reference.lower() or "annotated" in reference.lower():
        return "当前正在播放**人物 + 骨架 + 羽毛球**标注视频的复核片段。"
    if view.get("clip"):
        return "当前正在播放外部复核视频片段；请以视频画面为准完成判断。"
    return "尚未生成可播放的复核片段。"


def _review_match_video_message(session):
    reference = Path((session.get("source") or {}).get("reference_video") or "")
    if not reference.is_file():
        return "未找到可播放的标注视频；可在视频分析页重新生成，或选择另一场分析结果。"
    if "skeleton" in reference.name.lower():
        return "当前是旧的**纯骨架**结果：整场可播放，但不含真人画面。新分析会使用“人物 + 骨架 + 羽毛球”标注视频。"
    return "正在播放整场**人物 + 骨架 + 羽毛球**标注视频；右侧判断会随播放时间切换。"


def _review_video_fps(video_path):
    """Read the displayed video's real frame rate for client-side frame stepping."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    # Broken/variable-rate container metadata must not create unusable seek steps.
    return round(fps, 3) if 1.0 <= fps <= 240.0 else 30.0


def _review_candidate_form(candidate):
    review = candidate.get("review") or {}
    proposal = candidate.get("proposal") or {}
    label = review.get("label") or proposal.get("label") or "unknown"
    return (
        label if label in SHOT_TYPES else "unknown",
        review.get("decision") or "pending",
        review.get("reviewer") or "",
        review.get("note") or "",
    )


def _review_editor_markdown(candidate):
    proposal = candidate.get("proposal") or {}
    review = candidate.get("review") or {}
    label = review.get("label") or proposal.get("label") or "unknown"
    display = SHOT_TYPES.get(label, label)
    source = {
        "spatial_hit_candidate": "人体/空间接近",
        "trajectory_turn": "羽毛球轨迹方向变化",
        "trajectory_gap_transition": "羽毛球短暂断检衔接",
        "manual_timeline": "人工添加时间点",
        "manual_merge": "人工合并候选",
    }.get(candidate.get("candidate_source"), candidate.get("candidate_source", "未知"))
    review_display = REVIEW_DECISIONS.get(review.get("decision", "pending"), "待复核")
    return (
        f"### 编辑触球 · {candidate['hit_time_sec']:.2f}s\n"
        f"候选：`{candidate['shot_id']}`；当前：**{display}**；状态：{review_display}。  \n"
        f"来源：{source}。修改球种或结果后点击“保存本次修改”。"
    )


def _review_live_markdown(session, playback_sec):
    state = timeline_state(session, playback_sec)
    current = state["current"]
    next_candidate = state["next"]
    time_text = f"{state['playback_sec']:.2f}s"
    if current is None:
        upcoming = f"下一条自动候选：{next_candidate['hit_time_sec']:.2f}s。" if next_candidate else "尚无自动候选，可在当前时间添加。"
        return f"## 当前球路：等待触球\n视频时间：**{time_text}**。{upcoming}"

    review = current.get("review") or {}
    proposal = current.get("proposal") or {}
    decision = review.get("decision", "pending")
    if decision in {"confirmed", "corrected"}:
        label = review.get("label") or proposal.get("label") or "unknown"
        provenance = "人工确认" if decision == "confirmed" else "人工修正"
    elif decision == "excluded":
        label = "unknown"
        provenance = "已排除，等待下一次触球"
    elif decision == "uncertain":
        label = review.get("label") or "unknown"
        provenance = "人工标记为不确定"
    else:
        label = proposal.get("label") or "unknown"
        provenance = f"自动建议，置信度 {float(proposal.get('confidence') or 0.0):.2f}，待人工复核"
    display = SHOT_TYPES.get(label, label)
    upcoming = f"下一条候选：{next_candidate['hit_time_sec']:.2f}s。" if next_candidate else "已到当前候选列表末尾。"
    return (
        f"## 当前球路：**{display}**\n"
        f"触球时间：**{current['hit_time_sec']:.2f}s**　视频时间：{time_text}  \n"
        f"{provenance}。{upcoming}"
    )


def _review_open_timeline(analysis_dir, reference_video=None):
    if not analysis_dir:
        raise gr.Error("请先选择一个含 detections.jsonl 的分析结果。")
    try:
        session = create_or_load_review_session(analysis_dir, reference_video)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    video_path = (session.get("source") or {}).get("reference_video")
    if not video_path or not Path(video_path).is_file():
        raise gr.Error("当前结果没有可播放的标注视频。请重新分析或选择可用的视频结果。")
    choices = candidate_choices(session)
    selected_id = choices[0][1] if choices else None
    if selected_id:
        candidate = next(item for item in session["candidates"] if item["shot_id"] == selected_id)
        label, decision, reviewer, note = _review_candidate_form(candidate)
        editor_details = _review_editor_markdown(candidate)
    else:
        label, decision, reviewer, note = "unknown", "pending", "", ""
        editor_details = "尚无自动候选。播放到触球处后点击“在当前时间添加触球”。"
    return (
        gr.update(choices=choices, value=selected_id),
        _review_summary_markdown(session, selected_id),
        candidate_table(session),
        video_path,
        _review_video_fps(video_path),
        _review_match_video_message(session),
        "0.00s",
        _review_live_markdown(session, 0.0),
        editor_details,
        label,
        decision,
        reviewer,
        note,
        "已打开整场时间轴。播放过程中右侧会显示最近一次触球的判断；自动候选不正确时可修改或在当前时间补加。",
    )


def _review_follow_playback(analysis_dir, playback_sec, reference_video=None):
    if not analysis_dir:
        return "—", "请先打开一场分析结果。"
    session = create_or_load_review_session(analysis_dir, reference_video)
    state = timeline_state(session, playback_sec)
    return f"{state['playback_sec']:.2f}s", _review_live_markdown(session, state["playback_sec"])


def _review_select_editor(analysis_dir, shot_id, reference_video=None):
    if not analysis_dir or not shot_id:
        raise gr.Error("请先从触球记录中选择一条，或在当前时间添加触球。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    candidate = next((item for item in session["candidates"] if item["shot_id"] == shot_id), None)
    if candidate is None or not candidate.get("active", True):
        raise gr.Error("未找到该有效触球记录，请刷新复核队列。")
    label, decision, reviewer, note = _review_candidate_form(candidate)
    return _review_editor_markdown(candidate), label, decision, reviewer, note


def _review_jump_to_table_row(analysis_dir, reference_video, evt: gr.SelectData):
    """Seek the match player by the selected shot's time in seconds, never frame index."""
    if not analysis_dir:
        raise gr.Error("请先打开一场整场复核。")
    index = getattr(evt, "index", None)
    row_index = index[0] if isinstance(index, (tuple, list)) else index
    session = create_or_load_review_session(analysis_dir, reference_video)
    try:
        candidate = candidate_at_table_row(session, row_index)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    hit_time = float(candidate["hit_time_sec"])
    label, decision, reviewer, note = _review_candidate_form(candidate)
    return (
        gr.update(playback_position=hit_time),
        hit_time,
        f"{hit_time:.2f}s",
        _review_live_markdown(session, hit_time),
        gr.update(value=candidate["shot_id"]),
        _review_editor_markdown(candidate),
        label,
        decision,
        reviewer,
        note,
        f"已跳转到 **{hit_time:.2f}s** 的触球记录。",
    )


def _review_edit_playback_candidate(analysis_dir, playback_sec, reference_video=None):
    if not analysis_dir:
        raise gr.Error("请先打开一场分析结果。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    candidate = timeline_state(session, playback_sec)["current"]
    if candidate is None:
        raise gr.Error("当前时间前没有自动候选。请使用“在当前时间添加触球”记录这一次击球。")
    label, decision, reviewer, note = _review_candidate_form(candidate)
    return gr.update(value=candidate["shot_id"]), _review_editor_markdown(candidate), label, decision, reviewer, note


def _review_add_at_playback(analysis_dir, playback_sec, reviewer, reference_video=None):
    if not analysis_dir:
        raise gr.Error("请先打开一场分析结果。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    state = timeline_state(session, playback_sec)
    try:
        session, candidate = add_manual_candidate(session, state["playback_sec"], reviewer)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    choices = candidate_choices(session)
    label, decision, reviewer, note = _review_candidate_form(candidate)
    return (
        gr.update(choices=choices, value=candidate["shot_id"]),
        _review_summary_markdown(session, candidate["shot_id"]),
        candidate_table(session),
        _review_editor_markdown(candidate),
        label,
        decision,
        reviewer,
        note,
        _review_live_markdown(session, state["playback_sec"]),
        f"已在 **{state['playback_sec']:.2f}s** 添加人工触球。现在选择球种和复核结果，再保存本次修改。",
    )


def _review_save_timeline_candidate(analysis_dir, shot_id, label, decision, reviewer, note, reference_video=None):
    if not analysis_dir or not shot_id:
        raise gr.Error("请先选择要修改的触球记录。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    try:
        session = save_human_review(session, shot_id, label, decision, reviewer, note)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    candidate = next(item for item in session["candidates"] if item["shot_id"] == shot_id)
    return (
        _review_summary_markdown(session, shot_id),
        candidate_table(session),
        _review_editor_markdown(candidate),
        "已保存：原始 detections.jsonl 未修改；人工结论已追加到 shot_review/annotations.jsonl。",
    )


def _review_open_session(analysis_dir, reference_video=None):
    if not analysis_dir:
        raise gr.Error("请先选择一个含 detections.jsonl 的分析结果。")
    try:
        session = create_or_load_review_session(analysis_dir, reference_video)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    choices = candidate_choices(session)
    if not choices:
        raise gr.Error("当前结果没有可复核的击球候选。请先确认羽毛球检测和击球候选是否产生。")
    first_id = choices[0][1]
    view = candidate_view(session, first_id)
    return (
        gr.update(choices=choices, value=first_id),
        gr.update(choices=choices, value=[]),
        _review_summary_markdown(session, first_id),
        candidate_table(session),
        view["clip"],
        _review_details_markdown(view, session),
        _review_video_message(session, view),
        view["label"],
        view["decision"],
        view["reviewer"],
        view["note"],
    )


def _review_refresh_runs(current_value):
    choices = _review_run_choices()
    allowed = {value for _, value in choices}
    return gr.update(choices=choices, value=current_value if current_value in allowed else (choices[0][1] if choices else None))


def _review_candidate_updates(session, selected_id=None):
    choices = candidate_choices(session)
    available = {value for _, value in choices}
    selected_id = selected_id if selected_id in available else (choices[0][1] if choices else None)
    return (
        gr.update(choices=choices, value=selected_id),
        gr.update(choices=choices, value=[]),
        selected_id,
    )


def _review_rebuild_session(analysis_dir, reference_video=None):
    if not analysis_dir:
        raise gr.Error("请先选择一个含 detections.jsonl 的分析结果。")
    session = create_or_load_review_session(analysis_dir, reference_video, regenerate=True)
    shot_update, merge_update, selected_id = _review_candidate_updates(session)
    if not selected_id:
        raise gr.Error("当前结果没有可复核的候选。")
    view = candidate_view(session, selected_id)
    return (
        shot_update, merge_update, _review_summary_markdown(session, selected_id), candidate_table(session),
        view["clip"], _review_details_markdown(view, session), _review_video_message(session, view), view["label"], view["decision"],
        view["reviewer"], view["note"],
        "已重新生成自动候选；已有非待定人工复核和手动候选已保留。",
    )


def _review_select_candidate(analysis_dir, shot_id, reference_video=None):
    if not analysis_dir or not shot_id:
        raise gr.Error("请先生成复核队列并选择一拍。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    try:
        view = candidate_view(session, shot_id)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return (
        view["clip"], _review_details_markdown(view, session), _review_video_message(session, view), view["label"], view["decision"],
        view["reviewer"], view["note"],
    )


def _review_move_candidate(analysis_dir, shot_id, offset, reference_video=None):
    if not analysis_dir or not shot_id:
        raise gr.Error("请先打开复核队列。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    choices = candidate_choices(session)
    ids = [value for _, value in choices]
    try:
        current_index = ids.index(shot_id)
    except ValueError as error:
        raise gr.Error("当前候选不存在，请重新打开复核队列。") from error
    selected_id = ids[max(0, min(len(ids) - 1, current_index + int(offset)))]
    view = candidate_view(session, selected_id)
    return (
        gr.update(value=selected_id), _review_summary_markdown(session, selected_id),
        view["clip"], _review_details_markdown(view, session), _review_video_message(session, view),
        view["label"], view["decision"], view["reviewer"], view["note"],
    )


def _review_save_candidate(analysis_dir, shot_id, label, decision, reviewer, note, reference_video=None):
    if not analysis_dir or not shot_id:
        raise gr.Error("请先选择需要保存的球路候选。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    try:
        session = save_human_review(session, shot_id, label, decision, reviewer, note)
        view = candidate_view(session, shot_id)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return (
        _review_summary_markdown(session, shot_id), candidate_table(session),
        gr.update(choices=candidate_choices(session), value=shot_id),
        gr.update(choices=candidate_choices(session), value=[]),
        _review_details_markdown(view, session),
        "已保存：原始 detections.jsonl 未被修改；本次人工判断已追加到 shot_review/annotations.jsonl。",
    )


def _review_add_manual_candidate(analysis_dir, hit_time_sec, reviewer, reference_video=None):
    if not analysis_dir or hit_time_sec is None:
        raise gr.Error("请输入要补录的击球时间（秒）。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    try:
        session, candidate = add_manual_candidate(session, hit_time_sec, reviewer)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    shot_update, merge_update, _ = _review_candidate_updates(session, candidate["shot_id"])
    view = candidate_view(session, candidate["shot_id"])
    return (
        shot_update, merge_update, _review_summary_markdown(session, candidate["shot_id"]), candidate_table(session),
        view["clip"], _review_details_markdown(view, session), _review_video_message(session, view), view["label"], view["decision"],
        view["reviewer"], view["note"],
        "已新增人工补拍。请在片段中确认球种后再保存复核结论。",
    )


def _review_merge_selected(analysis_dir, shot_ids, reviewer, reference_video=None):
    if not analysis_dir:
        raise gr.Error("请先选择分析结果。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    try:
        session, candidate = merge_review_candidates(session, shot_ids, reviewer)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    shot_update, merge_update, _ = _review_candidate_updates(session, candidate["shot_id"])
    view = candidate_view(session, candidate["shot_id"])
    return (
        shot_update, merge_update, _review_summary_markdown(session, candidate["shot_id"]), candidate_table(session),
        view["clip"], _review_details_markdown(view, session), _review_video_message(session, view), view["label"], view["decision"],
        view["reviewer"], view["note"],
        "已合并候选；被合并的原候选保留在审计记录中，但不再出现在有效队列。",
    )


def _review_split_selected(analysis_dir, shot_id, split_offset_sec, reviewer, reference_video=None):
    if not analysis_dir or not shot_id:
        raise gr.Error("请先选择要拆分的候选。")
    session = create_or_load_review_session(analysis_dir, reference_video)
    try:
        session, first, second = split_review_candidate(session, shot_id, split_offset_sec, reviewer)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    shot_update, merge_update, _ = _review_candidate_updates(session, first["shot_id"])
    view = candidate_view(session, first["shot_id"])
    return (
        shot_update, merge_update, _review_summary_markdown(session, first["shot_id"]), candidate_table(session),
        view["clip"], _review_details_markdown(view, session), _review_video_message(session, view), view["label"], view["decision"],
        view["reviewer"], view["note"],
        f"已拆分为两拍：{first['hit_time_sec']:.2f}s 与 {second['hit_time_sec']:.2f}s。可继续补录或分别复核。",
    )


def build_ui():
    t = _UI_TEXT["zh"]

    with gr.Blocks(
        title="Good Badminton — AI Badminton Analysis",
    ) as demo:
        md_title = gr.Markdown(t["title"])

        corners_state = gr.State(value=None)
        click_corners_state = gr.State(value=[])
        template_path_state = gr.State(value=None)
        analysis_ready_state = gr.State(value=False)

        with gr.Tabs():
            analysis_tab = gr.Tab("视频分析")
            review_tab = gr.Tab("球路复核")

        with analysis_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    md_inputs = gr.Markdown(t["inputs"])
                    video_input = gr.File(label=t["video"], file_types=["video"])
                    template_input = gr.File(label=t["template"], file_types=["image"])

                    md_settings = gr.Markdown(t["settings"])
                    pose_family = gr.Dropdown(
                        choices=["yolo-pose", "rtmpose", "rtmo"],
                        value="yolo-pose", label=t["pose_family"],
                    )
                    pose_mode = gr.Dropdown(
                        choices=["lightweight", "balanced", "performance"],
                        value="balanced", label=t["pose_mode"],
                    )
                    language = gr.Radio(
                        choices=[("中文", "zh"), ("English", "en")],
                        value="zh", label=t["language"],
                    )
                    audio = gr.Checkbox(value=True, label=t["audio"])
                    match_mode = gr.Radio(
                        choices=[
                            ("Singles (one player per team)", "singles"),
                            ("Doubles (up to two players per team)", "doubles"),
                        ],
                        value="singles",
                        label="比赛模式 / Match mode",
                        info="New tracking uses spatial.tracks. Doubles keeps same-side players instead of upper/lower filtering.",
                    )
                    gr.Markdown("**输出视频：** 原视频人物 + 人体骨架 + 羽毛球标注")
                    # Preserve the analysis callback contract while preventing
                    # accidental Skeleton-only output in the normal workflow.
                    output_video_style = gr.State(value="annotated")
                    pose_imgsz = gr.Dropdown(
                        choices=[640, 960, 1280], value=1280, label=t["pose_imgsz"],
                    )
                    pose_conf = gr.Slider(
                        minimum=0.10, maximum=0.50, step=0.01, value=0.15,
                        label=t["pose_conf"],
                    )
                    far_player_enhancement = gr.Checkbox(
                        value=False, label=t["far_player_enhancement"],
                    )
                    far_pose_roi = gr.Textbox(
                        value="0.12,0.30,0.86,0.82", label=t["far_pose_roi"],
                    )

                    with gr.Accordion(t["advanced"], open=False) as adv_accordion:
                        show_skeletons = gr.Checkbox(value=True, label=t["skeletons"])
                        show_player_trajectories = gr.Checkbox(value=True, label=t["player_traj"])
                        show_court_trajectory = gr.Checkbox(value=True, label=t["court_traj"])
                        show_shuttlecock_trajectory = gr.Checkbox(value=True, label=t["shuttle_traj"])
                        show_player_stats = gr.Checkbox(value=True, label=t["player_stats"])
                        show_pose_roi = gr.Checkbox(value=True, label=t["pose_roi"])
                        visualize_positions = gr.Checkbox(value=True, label=t["viz_positions"])
                        yolo_pose_model = gr.Textbox(value="weights/yolo11n-pose.pt", label=t["yolo_pose_path"])
                        ball_model = gr.Textbox(value="weights/yolo11s-ball.pt", label=t["ball_path"])

                with gr.Column(scale=2):
                    md_step1 = gr.Markdown(t["step1"])
                    detect_btn = gr.Button(t["detect_btn"], variant="primary")
                    court_image = gr.Image(label=t["court_preview"], interactive=False, type="numpy")
                    corner_status = gr.Textbox(label=t["corner_status"], interactive=False, value=t["corner_none"])
                    apply_btn = gr.Button(t["apply_btn"], variant="secondary")

                    md_step2 = gr.Markdown(t["step2"])
                    run_btn = gr.Button(t["run_btn"], variant="primary")

                    md_results = gr.Markdown(t["results"])
                    output_status = gr.JSON(
                        label=t["out_status"],
                        value={"phase": "idle", "hint": "点击运行分析后显示上传、排队、帧进度与执行来源。"},
                    )
                    output_video = gr.Video(label=t["out_video"])
                    output_gallery = gr.Gallery(label=t["out_gallery"], columns=2, height="auto")
                    output_metadata = gr.JSON(label=t["out_metadata"])
                    output_detections = gr.File(label=t["out_detections"])
                    output_rally_summary = gr.Markdown("### 回合与拍数\n完成分析后显示候选回合与每回合拍数。")
                    output_rallies = gr.Dataframe(
                        headers=["回合", "开始(s)", "结束(s)", "候选拍数", "可见球候选", "缺球补拍", "结束依据", "置信度"],
                        datatype=["str", "number", "number", "number", "number", "number", "str", "number"],
                        interactive=False,
                        label="候选回合明细（所有结果均待人工复核）",
                        max_height=300,
                    )

        with review_tab:
            gr.Markdown(
                "## 球路复核\n"
                "播放**整场标注视频**。右侧会随播放时间显示最近一次触球的球路判断；"
                "自动建议不正确时，直接修改；漏拍时，在当前播放时间添加。"
            )
            with gr.Row():
                review_analysis_dir = gr.Dropdown(
                    choices=_review_run_choices(),
                    value=default_analysis_run("outputs") or None,
                    label="分析结果目录（含 detections.jsonl）",
                    scale=4,
                )
                refresh_review_runs = gr.Button("刷新分析结果", scale=1)
            with gr.Row():
                open_review_btn = gr.Button("打开整场复核", variant="primary")
                rebuild_review_btn = gr.Button("重新加载时间轴", size="sm")
            review_source_video = gr.File(
                label="可选：更换为已有的“人物 + 骨架”标注视频",
                file_types=["video"],
                visible=False,
            )
            review_summary_output = gr.Markdown("请先打开复核队列。")
            review_notice = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=3):
                    review_video_message = gr.Markdown()
                    review_match_video = gr.Video(
                        label="整场标注视频（可拖动进度条）",
                        height=440,
                        include_audio=True,
                        elem_id="review-match-video",
                    )
                    review_playback_clock = gr.HTML(
                        value=0.0,
                        elem_id="review-playback-clock",
                        html_template="<span class='review-playback-clock' aria-hidden='true'></span>",
                        css_template=".review-playback-clock { display: none; }",
                        js_on_load="""
                            let attachedVideo = null;
                            let lastSent = -1;
                            const emitTime = (force = false) => {
                                if (!attachedVideo) return;
                                const seconds = Number(attachedVideo.currentTime || 0);
                                if (!Number.isFinite(seconds) || (!force && Math.abs(seconds - lastSent) < 0.45)) return;
                                lastSent = seconds;
                                props.value = seconds;
                                trigger('input');
                            };
                            const bindVideo = () => {
                                const video = document.querySelector('#review-match-video video');
                                if (!video || video === attachedVideo) return;
                                attachedVideo = video;
                                lastSent = -1;
                                video.addEventListener('loadedmetadata', emitTime);
                                video.addEventListener('timeupdate', emitTime);
                                video.addEventListener('seeking', () => emitTime(true));
                                video.addEventListener('seeked', () => emitTime(true));
                            };
                            bindVideo();
                            window.setInterval(bindVideo, 500);
                        """,
                    )
                    review_frame_controls = gr.HTML(
                        value=30.0,
                        elem_id="review-frame-controls",
                        html_template="""
                            <button type="button" data-frame-direction="-1" aria-label="上一帧">◀ 上一帧</button>
                            <button type="button" data-frame-direction="1" aria-label="下一帧">下一帧 ▶</button>
                            <span class="frame-rate-hint">按视频帧率逐帧（${value} FPS）</span>
                        """,
                        js_on_load="""
                            const getVideo = () => document.querySelector('#review-match-video video');
                            const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(value, maximum));
                            const stepFrame = (direction) => {
                                const video = getVideo();
                                if (!video) return;
                                const fps = Number(props.value);
                                const safeFps = Number.isFinite(fps) && fps >= 1 && fps <= 240 ? fps : 30;
                                const duration = Number(video.duration);
                                if (!Number.isFinite(duration)) return;
                                video.pause();
                                video.currentTime = clamp(video.currentTime + direction / safeFps, 0, duration);
                            };
                            element.querySelectorAll('[data-frame-direction]').forEach((button) => {
                                button.addEventListener('click', () => stepFrame(Number(button.dataset.frameDirection)));
                            });
                        """,
                    )
                    review_seek_request = gr.HTML(
                        value=-1,
                        elem_id="review-seek-request",
                        html_template="<span class='review-seek-request' aria-hidden='true'></span>",
                        css_template=".review-seek-request { display: none; }",
                        js_on_load="""
                            const targetSeconds = Number(props.value);
                            if (!Number.isFinite(targetSeconds) || targetSeconds < 0) return;
                            let attempts = 0;
                            const seekBySeconds = () => {
                                const video = document.querySelector('#review-match-video video');
                                if (!video) return false;
                                const duration = Number(video.duration);
                                if (!Number.isFinite(duration)) return false;
                                video.currentTime = Math.max(0, Math.min(targetSeconds, duration));
                                return true;
                            };
                            if (seekBySeconds()) return;
                            const timer = window.setInterval(() => {
                                attempts += 1;
                                if (seekBySeconds() || attempts >= 30) window.clearInterval(timer);
                            }, 100);
                        """,
                    )
                    review_candidates_table = gr.Dataframe(
                        headers=["候选", "击球时间(s)", "自动建议", "建议置信度", "候选来源", "复核结果"],
                        datatype=["str", "number", "str", "number", "str", "str"],
                        interactive=False,
                        label="触球记录（点击任一行跳转到该击球时间）",
                        max_height=245,
                        elem_id="review-touch-record",
                    )
                with gr.Column(scale=2, elem_id="review-editor-panel"):
                    review_current_time = gr.Textbox(label="当前视频时间", value="—", interactive=False)
                    review_live_judgement = gr.Markdown("### 当前球路：等待打开视频")
                    with gr.Row():
                        edit_playback_candidate_btn = gr.Button("编辑当前触球", size="sm")
                        add_playback_candidate_btn = gr.Button("在当前时间添加触球", size="sm")
                    gr.Markdown("### 人工修改")
                    review_shot_id = gr.Dropdown(label="编辑对象（可手动切换）")
                    review_details = gr.Markdown("播放到需处理的触球处，选择“编辑当前触球”或“在当前时间添加触球”。")
                    review_label = gr.Radio(
                        choices=[(display, key) for key, display in SHOT_TYPES.items()],
                        value="unknown",
                        label="球种",
                    )
                    review_decision = gr.Radio(
                        choices=[
                            ("确认自动建议", "confirmed"),
                            ("人工修正", "corrected"),
                            ("仍不确定", "uncertain"),
                            ("排除：不是有效击球", "excluded"),
                        ],
                        value="pending",
                        label="复核结果",
                    )
                    with gr.Row():
                        review_reviewer = gr.Textbox(label="复核人（可选）", scale=1)
                        review_note = gr.Textbox(label="备注（可选）", lines=1, scale=2)
                    save_review_btn = gr.Button("保存本次修改", variant="primary", size="sm")

            with gr.Accordion("赛后身份与队伍确认 / Post-match identity", open=False):
                gr.Markdown(
                    "仅用于赛后人工确认或纠错：可把同一人的分段 Track ID 标为同一个 person_id。"
                    "team_id 仅可为 team_a/team_b，不能从当前球场半区自动推断。"
                )
                with gr.Row():
                    identity_reload_btn = gr.Button("读取 Track ID", size="sm")
                    identity_save_btn = gr.Button("保存身份/队伍绑定", variant="primary", size="sm")
                identity_table = gr.Dataframe(
                    headers=["track_id", "person_id（可选）", "team_id（team_a/team_b）", "detected", "predicted", "missing", "distance_m"],
                    datatype=["str", "str", "str", "number", "number", "number", "number"],
                    interactive=True,
                    label="空间追踪身份绑定（不改原始 detections.jsonl）",
                    max_height=260,
                )
                identity_notice = gr.Markdown("先选择分析结果后，点击“读取 Track ID”。")

        console_open_state = gr.State(value=False)
        console_trigger = gr.Button(
            "打开后台输出", size="sm", elem_id="backend-console-trigger",
        )
        with gr.Column(visible=False, elem_id="backend-console-panel") as console_panel:
            with gr.Row():
                gr.Markdown("后台输出 · 实时", elem_classes="console-title")
                console_close = gr.Button("关闭", size="sm", elem_id="backend-console-close")
            console_output = gr.Textbox(
                value=get_backend_logs(), lines=22, max_lines=22,
                show_label=False, interactive=False, autoscroll=True,
            )
        console_timer = gr.Timer(value=1.0, active=True)

        console_trigger.click(
            fn=toggle_backend_console,
            inputs=[console_open_state],
            outputs=[console_open_state, console_panel, console_output],
            show_progress="hidden",
        )
        console_close.click(
            fn=close_backend_console,
            outputs=[console_open_state, console_panel],
            show_progress="hidden",
        )
        console_timer.tick(
            fn=get_backend_logs,
            outputs=[console_output],
            show_progress="hidden",
        )

        identity_reload_btn.click(
            fn=_match_identity_table,
            inputs=[review_analysis_dir],
            outputs=[identity_table, identity_notice],
            show_progress="hidden",
        )
        identity_save_btn.click(
            fn=_save_match_identity_table,
            inputs=[review_analysis_dir, identity_table],
            outputs=[identity_notice],
        )

        refresh_review_runs.click(
            fn=_review_refresh_runs,
            inputs=[review_analysis_dir],
            outputs=[review_analysis_dir],
            show_progress="hidden",
        )
        # A job can finish while the analysis tab remains open. Refresh choices
        # when the reviewer enters this tab so last night's result is visible
        # without restarting the WebUI; keep an existing selection intact.
        review_tab.select(
            fn=_review_refresh_runs,
            inputs=[review_analysis_dir],
            outputs=[review_analysis_dir],
            show_progress="hidden",
        )
        open_review_btn.click(
            fn=_review_open_timeline,
            inputs=[review_analysis_dir, review_source_video],
            outputs=[
                review_shot_id, review_summary_output, review_candidates_table, review_match_video,
                review_frame_controls, review_video_message, review_current_time, review_live_judgement, review_details,
                review_label, review_decision, review_reviewer, review_note, review_notice,
            ],
        )
        rebuild_review_btn.click(
            fn=_review_open_timeline,
            inputs=[review_analysis_dir, review_source_video],
            outputs=[
                review_shot_id, review_summary_output, review_candidates_table, review_match_video,
                review_frame_controls, review_video_message, review_current_time, review_live_judgement, review_details,
                review_label, review_decision, review_reviewer, review_note, review_notice,
            ],
        )
        review_playback_clock.input(
            fn=_review_follow_playback,
            inputs=[review_analysis_dir, review_playback_clock, review_source_video],
            outputs=[review_current_time, review_live_judgement],
            show_progress="hidden",
            queue=False,
        )
        review_shot_id.change(
            fn=_review_select_editor,
            inputs=[review_analysis_dir, review_shot_id, review_source_video],
            outputs=[review_details, review_label, review_decision, review_reviewer, review_note],
            show_progress="hidden",
        )
        review_candidates_table.select(
            fn=_review_jump_to_table_row,
            inputs=[review_analysis_dir, review_source_video],
            outputs=[
                review_match_video, review_seek_request, review_current_time, review_live_judgement,
                review_shot_id, review_details, review_label, review_decision,
                review_reviewer, review_note, review_notice,
            ],
            show_progress="hidden",
        )
        edit_playback_candidate_btn.click(
            fn=_review_edit_playback_candidate,
            inputs=[review_analysis_dir, review_playback_clock, review_source_video],
            outputs=[review_shot_id, review_details, review_label, review_decision, review_reviewer, review_note],
            show_progress="hidden",
        )
        add_playback_candidate_btn.click(
            fn=_review_add_at_playback,
            inputs=[review_analysis_dir, review_playback_clock, review_reviewer, review_source_video],
            outputs=[
                review_shot_id, review_summary_output, review_candidates_table, review_details,
                review_label, review_decision, review_reviewer, review_note, review_live_judgement,
                review_notice,
            ],
            show_progress="hidden",
        )
        save_review_btn.click(
            fn=_review_save_timeline_candidate,
            inputs=[
                review_analysis_dir, review_shot_id, review_label, review_decision,
                review_reviewer, review_note, review_source_video,
            ],
            outputs=[
                review_summary_output, review_candidates_table,
                review_details, review_notice,
            ],
        )

        lang_outputs = [
            md_title, md_inputs, video_input, template_input,
            md_settings, pose_family, pose_mode, audio,
            pose_imgsz, pose_conf, far_player_enhancement, far_pose_roi, adv_accordion,
            show_skeletons, show_player_trajectories, show_court_trajectory,
            show_shuttlecock_trajectory, show_player_stats, show_pose_roi,
            visualize_positions, yolo_pose_model, ball_model,
            md_step1, detect_btn, court_image, corner_status, apply_btn,
            md_step2, run_btn, md_results,
            output_status, output_video, output_gallery, output_metadata, output_detections,
        ]
        language.change(fn=_switch_language, inputs=[language], outputs=lang_outputs)

        video_input.change(
            fn=reset_court_selection,
            inputs=[language],
            outputs=[court_image, corners_state, click_corners_state, template_path_state, corner_status, analysis_ready_state],
        )
        template_input.change(
            fn=reset_court_selection,
            inputs=[language],
            outputs=[court_image, corners_state, click_corners_state, template_path_state, corner_status, analysis_ready_state],
        )

        detect_btn.click(
            fn=detect_court,
            inputs=[video_input, template_input, language],
            outputs=[court_image, corners_state, template_path_state, corner_status, analysis_ready_state],
        )

        court_image.select(
            fn=on_court_image_select,
            inputs=[click_corners_state, template_path_state],
            outputs=[court_image, click_corners_state, corners_state, corner_status],
        )

        apply_btn.click(
            fn=apply_manual_corners,
            inputs=[template_path_state, click_corners_state],
            outputs=[court_image, corners_state, analysis_ready_state],
        ).then(
            fn=lambda c, lang: _UI_TEXT.get(lang, _UI_TEXT["zh"])["manual_ok"].format(len(c)) if c
               else _UI_TEXT.get(lang, _UI_TEXT["zh"])["manual_fail"],
            inputs=[corners_state, language],
            outputs=[corner_status],
        )

        run_preflight = run_btn.click(
            fn=ensure_court_for_analysis,
            inputs=[video_input, template_path_state, corners_state, click_corners_state, language],
            outputs=[
                court_image, corners_state, click_corners_state,
                template_path_state, corner_status, analysis_ready_state,
            ],
        )
        run_preflight.then(
            fn=run_full_analysis,
            inputs=[
                analysis_ready_state, video_input, template_path_state, corners_state,
                pose_family, pose_mode, language, audio, match_mode, output_video_style,
                pose_imgsz, pose_conf, far_player_enhancement, far_pose_roi,
                show_skeletons, show_player_trajectories,
                show_court_trajectory, show_shuttlecock_trajectory,
                show_player_stats, show_pose_roi, visualize_positions,
                yolo_pose_model, ball_model,
            ],
            outputs=[
                output_video, output_gallery, output_metadata, output_detections,
                output_rally_summary, output_rallies, output_status,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7861,
        theme=gr.themes.Soft(),
        css=_APP_CSS,
    )
