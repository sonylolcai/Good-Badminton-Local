import json
import os
import traceback

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

_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
_MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB


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
                      pose_family, pose_mode, language, audio,
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

    options = {
        "pose_family": pose_family,
        "pose_mode": pose_mode,
        "language": language,
        "audio": audio,
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

    def progress_cb(frame, total):
        progress(frame / total, desc=f"Processing frame {frame}/{total}")

    try:
        result = run_analysis(
            video_path=video_file,
            template_path=template_path,
            corners=corners,
            options=options,
            progress_cb=progress_cb,
        )
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"分析失败：{exc}") from exc

    for warning in result.get("warnings", []):
        gr.Warning(warning)

    output_video = result["video"] if os.path.isfile(result["video"]) else None
    viz_images = [img for img in result["visualizations"] if os.path.isfile(img)]

    metadata_content = None
    if os.path.isfile(result["metadata"]):
        with open(result["metadata"], "r", encoding="utf-8") as f:
            metadata_content = json.load(f)

    detections_file = result["detections"] if os.path.isfile(result["detections"]) else None

    return output_video, viz_images or None, metadata_content, detections_file


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
        gr.update(label=t["out_video"]),
        gr.update(label=t["out_gallery"]),
        gr.update(label=t["out_metadata"]),
        gr.update(label=t["out_detections"]),
    ]


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
                output_video = gr.Video(label=t["out_video"])
                output_gallery = gr.Gallery(label=t["out_gallery"], columns=2, height="auto")
                output_metadata = gr.JSON(label=t["out_metadata"])
                output_detections = gr.File(label=t["out_detections"])

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

        lang_outputs = [
            md_title, md_inputs, video_input, template_input,
            md_settings, pose_family, pose_mode, audio, adv_accordion,
            show_skeletons, show_player_trajectories, show_court_trajectory,
            show_shuttlecock_trajectory, show_player_stats, show_pose_roi,
            visualize_positions, yolo_pose_model, ball_model,
            md_step1, detect_btn, court_image, corner_status, apply_btn,
            md_step2, run_btn, md_results,
            output_video, output_gallery, output_metadata, output_detections,
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
                pose_family, pose_mode, language, audio,
                show_skeletons, show_player_trajectories,
                show_court_trajectory, show_shuttlecock_trajectory,
                show_player_stats, show_pose_roi, visualize_positions,
                yolo_pose_model, ball_model,
            ],
            outputs=[output_video, output_gallery, output_metadata, output_detections],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(theme=gr.themes.Soft(), css=_APP_CSS)
