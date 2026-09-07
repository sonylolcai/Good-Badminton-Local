import json
import os
import queue
import re
import threading
import time
import traceback
from datetime import datetime
from html import escape
from pathlib import Path

from runtime_config import load_runtime_environment, webui_listener

load_runtime_environment()

from webui.log_capture import get_backend_logs, install_backend_log_capture
from webui.operator_backoffice import OPERATOR_BACKOFFICE_CSS, render_backoffice_tabs
from webui.player_results import (
    PLAYER_RESULT_HEADERS,
    TENNIS_PLAYER_RESULT_HEADERS,
    build_player_result_display,
    extract_full_video_candidate_photos,
)

install_backend_log_capture()

import cv2
import gradio as gr
import numpy as np

from badminton_analysis.cancellation import AnalysisCancelled
from business_gateway.streaming.client import StreamAPIError
from webui.pipeline import (
    _max_template_match_score,
    imread_safe,
    prepare_court,
    prepare_court_from_video,
    run_analysis,
)
from webui.remote_gpu import (
    RemoteAnalysisError,
    configured_gpu_base_url,
    iter_remote_two_second_stream,
    recover_remote_two_second_stream,
    remote_gpu_config,
    run_remote_analysis,
    verify_remote_gpu_sport,
)
from webui.stream_replay import (
    StreamReplayError,
    iter_local_stream_replay,
    poll_local_stream_replay,
    start_local_stream_replay,
)
from webui.reconcile_remote_tasks import reconcile_once
from webui.shot_review import (
    REVIEW_DECISIONS,
    RALLY_TERMINAL_OUTCOMES,
    SHOT_TYPES,
    add_manual_candidate,
    add_manual_rally_terminal,
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
    rally_playback_state,
    reviewed_rally_table,
    save_human_review,
    split_review_candidate,
    timeline_state,
)
from webui.task_ledger import BusinessTaskLedger
from webui.task_control import AnalysisTaskController

_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
_MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB
_ANALYSIS_TASKS = AnalysisTaskController()

TENNIS_MOVEMENT_METRIC_HEADERS = [
    "匿名视觉 Track ID",
    "距离(m)",
    "平均速度(m/s)",
    "峰值速度(m/s)",
    "有效移动(s)",
    "可用覆盖率(%)",
    "数据质量",
]


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


def _local_fallback_policy(ledger, business_task_id, options):
    """Return whether a remote error may safely start a new local analysis.

    A local run is a useful continuity fallback only when a GPU submission
    never received a durable receipt.  Once the GPU has accepted a job, it may
    have already spent substantial time processing (or have partial output).
    Starting a second full analysis on the workstation after that job fails
    hides the real error and can waste an entire second match-length run.
    """
    if options.get("shuttle_detector") == "tracknet_v3":
        return False, (
            "远程 TrackNetV3 主流程失败；为避免悄悄改用本地 YOLO，"
            "本次不会自动本地回退。"
        )

    task = ledger.get(business_task_id) if ledger is not None and business_task_id else None
    remote = (task or {}).get("remote") or {}
    if remote.get("accepted") and remote.get("job_id"):
        return False, (
            "远端 GPU 已确认接收并执行过该任务；为避免重复计算，"
            "本次不会自动切换为本地分析。"
        )
    return True, None


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


def _extract_tennis_calibration_frame(video_file):
    """Save one readable frame for manual tennis calibration without line AI."""
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise gr.Error("无法读取网球视频；请上传可播放的视频或清晰的球场截图。")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Avoid the very first frame, which is often a fade-in. This is only a
        # manual-calibration canvas; no badminton court detector is invoked.
        candidates = [max(0, total // 8), total // 2, 0] if total else [0]
        frame = None
        for index in candidates:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
                break
    finally:
        capture.release()
    if frame is None:
        raise gr.Error("无法从网球视频提取标定帧；请上传一张清晰的球场截图。")
    target = Path("outputs") / "court_templates"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"tennis_manual_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise gr.Error("保存网球标定帧失败。")
    return str(path)


def detect_court(
    video_file,
    template_file,
    language="zh",
    sport_id="badminton",
    progress=gr.Progress(track_tqdm=False),
):
    """Use an uploaded template when supplied; otherwise generate one from video."""
    text = _UI_TEXT.get(language, _UI_TEXT["zh"])
    if sport_id == "tennis":
        if template_file:
            _validate_file_size(template_file, _MAX_IMAGE_BYTES, "Template image")
            template_path = template_file
        else:
            if video_file is None:
                raise gr.Error(text["need_video"])
            _validate_file_size(video_file, _MAX_VIDEO_BYTES, "Video")
            template_path = _extract_tennis_calibration_frame(video_file)
        template_img = imread_safe(template_path)
        if template_img is None:
            raise gr.Error("无法读取网球标定图。")
        return (
            _bgr_to_rgb(template_img),
            None,
            template_path,
            "网球模式不使用羽毛球自动线检测。请按左上、右上、右下、左下依次点击单打场地四角，再确认手动角点。",
            False,
        )
    if template_file:
        _validate_file_size(template_file, _MAX_IMAGE_BYTES, "Template image")
        result = prepare_court(template_file)
        template_path = template_file
        selection = None
    else:
        if video_file is None:
            raise gr.Error(text["need_video"])
        _validate_file_size(video_file, _MAX_VIDEO_BYTES, "Video")
        result = prepare_court_from_video(
            video_file,
            progress_cb=lambda current, total, frame: progress(
                (current, total),
                desc=f"正在检查球场候选帧 {current}/{total}（源帧 {frame}）",
            ),
        )
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


def on_court_image_select(corners_state, template_file, sport_id="badminton", evt: gr.SelectData = None):
    """Accumulate clicked points and redraw markers on the template."""
    if not template_file:
        raise gr.Error("Please generate or upload a court template first.")
    if corners_state is None:
        corners_state = []

    if len(corners_state) >= 4:
        corners_state = []

    if evt is None:
        raise gr.Error("未收到球场点击坐标。")
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

    status = f"已选择角点：{len(corners_state)}/4"
    if len(corners_state) == 4:
        if sport_id == "tennis":
            status += " — 已按单打场地四角排序。请确认手动角点。"
        else:
            status += " — 角点已锁定。请确认手动角点，或继续点击后重新选择。"

    corners_out = corners_state if len(corners_state) == 4 else None
    return _bgr_to_rgb(preview), corners_state, corners_out, status


def _normalize_court_corners(points):
    """Normalize arbitrary clicks to TL, TR, BR, BL order."""
    top = sorted(sorted(points, key=lambda point: point[1])[:2], key=lambda point: point[0])
    bottom = sorted(sorted(points, key=lambda point: point[1])[2:], key=lambda point: point[0])
    return [top[0], top[1], bottom[1], bottom[0]]


def apply_manual_corners(template_file, corners_state, sport_id="badminton"):
    """Confirm the four detected or manually selected court corners.

    ``corners_state`` is the canonical resolved state: automatic detection
    writes its four points there, and the manual click handler only writes it
    once the fourth point has been selected.  The separate click accumulator
    is intentionally not accepted here because it remains empty after a
    successful automatic detection.
    """
    if not template_file:
        raise gr.Error("Please generate or upload a court template first.")
    if not corners_state or len(corners_state) != 4:
        raise gr.Error("Please click exactly 4 corners on the court image first.")

    if sport_id == "tennis":
        template_img = imread_safe(template_file)
        if template_img is None:
            raise gr.Error("无法读取网球标定图。")
        preview = template_img.copy()
        normalized = _normalize_court_corners(corners_state)
        cv2.polylines(preview, [np.array(normalized, dtype=np.int32)], True, (0, 255, 0), 3)
        for index, point in enumerate(normalized, 1):
            cv2.circle(preview, point, 6, (0, 0, 255), -1)
            cv2.putText(preview, str(index), (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return _bgr_to_rgb(preview), normalized, True

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


def configure_sport_mode(sport_id):
    """Update only presentation defaults; server-side validation remains final."""
    sport_id = "tennis" if sport_id == "tennis" else "badminton"
    if sport_id == "tennis":
        return (
            "## 网球单打视觉分析\n"
            "只上传单打对打视频。服务端固定追踪两名匿名运动员，并输出每人的场地平面速度；"
            "当前不检测网球、不判分、不生成回合或训练结论。",
            "### 网球标定\n"
            "请使用完整单打场地画面，手动点击左上、右上、右下、左下四个角点。"
            "不要使用羽毛球自动线检测结果。",
            gr.update(
                value=configured_gpu_base_url("tennis"),
                label="网球 GPU 服务地址",
                info="必须指向 health 返回 sport_id=tennis 的纯视觉流式 GPU 服务。",
            ),
            gr.update(value="none", visible=False),
            gr.update(value=True, visible=False, interactive=False),
            gr.update(
                choices=[("2 人（网球单打，固定）", 2)],
                value=2,
                interactive=False,
                label="网球视觉 roster",
                info="网球对打模式固定为两名运动员。",
            ),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=False, visible=False),
        )
    return (
        "## 羽毛球视觉分析\n"
        "保留原有完整视频与 2 秒直连分片两种上传方式；运动模式由当前羽毛球 GPU 服务固定。",
        "### 羽毛球标定\n"
        "可自动检测或手动确认球场四角；自动结果不可靠时请手动复核。",
        gr.update(
            value=configured_gpu_base_url("badminton"),
            label="羽毛球 GPU 服务地址（开发用）",
            info="完整上传与模拟流式分析均使用此地址；API Key 仍只从 WebUI 服务器配置读取。",
        ),
        gr.update(value="yolo", visible=True),
        gr.update(value=False, visible=True, interactive=True),
        gr.update(
            choices=[("2 人（单打）", 2), ("4 人（双打）", 4)],
            value=2,
            interactive=True,
            label="分片流式场上人数",
            info="仅在勾选 2 秒分片推送时生效。",
        ),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def configure_sport_presentation(sport_id):
    """Keep labels and result columns aligned with the selected visual mode."""
    if sport_id == "tennis":
        return (
            gr.update(value="提取网球标定帧（再手动选四角）"),
            gr.update(value="### 第一步 — 网球标定图与手动角点"),
            gr.update(
                value=(
                    "网球会固定使用 2 秒直连分片上传到网球 GPU；"
                    "不调用本地业务网关，也不会分析球、比分、回合或身体参数。"
                )
            ),
            gr.update(
                headers=TENNIS_PLAYER_RESULT_HEADERS,
                datatype=["str", "str", "str", "number", "number", "number", "number", "number", "number", "str"],
                label="网球逐人视觉速度与证据",
            ),
            gr.update(
                headers=TENNIS_MOVEMENT_METRIC_HEADERS,
                datatype=["str", "number", "number", "number", "number", "number", "str"],
                label="网球逐人速度汇总（仅使用真实高置信检测）",
            ),
            gr.update(label="网球视觉速度原始数据（覆盖率、排除原因与标定限制）"),
        )
    return (
        gr.update(value="自动选择视频帧并检测球场"),
        gr.update(value="### 第一步 — 球场检测"),
        gr.update(
            value=(
                "勾选左侧的 `每 2 秒直接分片推送到 GPU` 后，点击 `运行分析` 会直接向上方 GPU 服务地址创建流会话并上传分片。"
                "下方旧按钮仍保留给本地业务网关联调，不使用上方地址作为第一跳。"
            )
        ),
        gr.update(
            headers=PLAYER_RESULT_HEADERS,
            datatype=["str", "str", "str", "number", "number", "number", "number", "number",
                      "number", "number", "number", "str", "number", "str"],
            label="逐人运动与证据数据",
        ),
        gr.update(
            headers=[
                "track_id", "距离(m)", "平均速度(m/s)", "峰值速度(m/s)",
                "有效移动(s)", "高强度(s)", "加速事件", "减速事件", "敏捷移动（整场/30秒峰值）",
                "可用覆盖率(%)", "运动消耗估算(kcal)", "数据质量",
            ],
            datatype=["str", "number", "number", "number", "number", "number",
                      "number", "number", "str", "number", "str", "str"],
            label="运动员运动数据汇总（仅使用真实高置信检测）",
        ),
        gr.update(label="运动员完整分析数据（覆盖率、排除原因、场地区域与能量估算）"),
    )


def verify_selected_gpu(sport_id, gpu_base_url):
    """Expose the same fail-closed health check used immediately before upload."""
    try:
        health = verify_remote_gpu_sport(sport_id, gpu_base_url)
    except RemoteAnalysisError as exc:
        return f"⚠️ GPU 未通过校验：{escape(str(exc))}"
    return (
        f"✅ 已连接 **{escape(str(health.get('service') or 'GPU service'))}**"
        f"（sport_id=`{escape(str(health.get('sport_id')))}`，"
        f"支持模式：{', '.join(escape(str(item)) for item in health.get('supported_session_modes') or []) or '未声明'}）。"
    )


def reset_analysis_results_for_sport(sport_id):
    """Clear stale artifacts before presenting another fixed sport mode.

    Files remain on disk for later recovery, but an old badminton report must
    never look like the result of the currently selected tennis session.
    """
    sport_id = "tennis" if sport_id == "tennis" else "badminton"
    status = {
        "phase": "idle",
        "sport_id": sport_id,
        "hint": "已切换运动模式；请上传视频、确认该运动的球场角点后重新分析。",
    }
    return (
        None, None, None, None, None, None, None, [],
        None, [], {"status": "waiting_for_analysis", "sport_id": sport_id},
        None, [], None, status, status,
        None, [], {
            "schema_version": "webui-player-result.v1",
            "sport_id": sport_id,
            "players": [],
            "limitations": ["切换运动模式后不复用上一模式的可视化结果。"],
        },
    )


def ensure_court_for_analysis(
    video_file,
    template_path,
    corners,
    click_corners,
    language="zh",
    sport_id="badminton",
):
    """Validate the selected template and fall back to a frame from the active video."""
    text = _UI_TEXT.get(language, _UI_TEXT["zh"])
    if video_file is None:
        raise gr.Error(text["need_video"])

    ready = bool(template_path and corners and len(corners) == 4)
    if sport_id == "tennis":
        if not ready:
            raise gr.Error("网球对打需先在标定图上手动确认完整单打场地的四个角点。")
        return gr.update(), corners, click_corners, template_path, gr.update(), True
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
                       output_video_style, shuttle_detector, tracker_backend,
                       movement_rally_settle_seconds, enable_huji_play_state,
                       pose_imgsz, analysis_sample_hz, pose_conf, far_player_enhancement, far_pose_roi,
                       generate_annotated_video, browser_video_reencode,
                       show_skeletons, show_player_trajectories,
                      show_court_trajectory, show_shuttlecock_trajectory,
                       show_player_stats, show_pose_roi, visualize_positions,
                      yolo_pose_model, ball_model, gpu_base_url,
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
        # The upload UI intentionally does not ask for singles/doubles.  The
        # legacy full-file pipeline therefore stays in open anonymous-track
        # mode: detected people are retained instead of withholding the entire
        # result until a predeclared 2- or 4-person roster happens to appear.
        # A later business flow may bind teams and identities to these tracks.
        "lock_match_roster": False,
        "roster_stable_frames": 2,
        "tracker_backend": tracker_backend,
        "enable_bytetrack": tracker_backend == "bytetrack",
        "output_video_style": output_video_style,
        "generate_annotated_video": generate_annotated_video,
        "browser_video_reencode": browser_video_reencode,
        # This is an execution mode rather than a display toggle. ``none``
        # does not invoke a shuttle detector or create ball evidence.
        "shuttle_detector": shuttle_detector,
        "movement_rally_settle_seconds": float(movement_rally_settle_seconds),
        "enable_huji_play_state": bool(enable_huji_play_state),
        "pose_imgsz": int(pose_imgsz),
        # One selection controls every primary evidence-producing component:
        # pose, YOLO shuttle, track/roster updates, derived rallies and JSONL.
        "analysis_sample_hz": float(analysis_sample_hz),
        "pose_sample_hz": float(analysis_sample_hz),
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

    task_handle = _ANALYSIS_TASKS.start()
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
                remote_base_url=remote_gpu_config(gpu_base_url)["base_url"],
            )

            def remote_status(event):
                ledger.record_remote_event(business_task_id, event)
                publish({**event, "business_task_id": business_task_id})

            try:
                result = run_remote_analysis(
                    video_path=video_file, template_path=template_path, corners=corners,
                    options=options, output_dir=remote_output_dir, status_cb=remote_status,
                    business_task_id=business_task_id,
                    cancel_cb=task_handle.is_cancelled,
                    gpu_base_url=gpu_base_url,
                )
                _attach_business_interpretation(result, remote_status)
                outcome["result"] = result
                trace_record = ledger.archive_performance_trace(
                    business_task_id,
                    result.get("performance_trace"),
                )
                ledger.record_terminal(
                    business_task_id,
                    status="succeeded",
                    details={"performance_trace": trace_record} if trace_record else None,
                )
                publish({"mode": "remote_gpu", "phase": "succeeded", "business_task_id": business_task_id})
            except RemoteAnalysisError as remote_exc:
                remote_trace_record = ledger.archive_performance_trace(
                    business_task_id,
                    getattr(remote_exc, "performance_trace_path", None),
                )
                if task_handle.is_cancelled():
                    message = f"本地已停止等待，但远端 GPU 中断未确认：{remote_exc}"
                    outcome["interrupt_unconfirmed"] = message
                    if ledger is not None and business_task_id is not None:
                        ledger.record_terminal(
                            business_task_id,
                            status="interrupted_unconfirmed",
                            error={"type": "RemoteCancellationUnconfirmed", "message": message},
                            details={"performance_trace": remote_trace_record} if remote_trace_record else None,
                        )
                    publish({
                        "phase": "interrupted_unconfirmed",
                        "message": message,
                        "business_task_id": business_task_id,
                    })
                    return
                fallback_reason = str(remote_exc)
                allow_fallback, block_reason = _local_fallback_policy(
                    ledger, business_task_id, options,
                )
                if not allow_fallback:
                    raise RuntimeError(f"{block_reason} 原因：{fallback_reason}") from remote_exc
                print(f"Remote GPU analysis failed; falling back locally: {fallback_reason}")
                ledger.record_terminal(
                    business_task_id,
                    status="local_fallback",
                    error={"message": fallback_reason},
                    details={"performance_trace": remote_trace_record} if remote_trace_record else None,
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
                    cancel_cb=task_handle.is_cancelled,
                )
                _attach_business_interpretation(result, remote_status)
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
        except AnalysisCancelled as exc:
            outcome["cancelled"] = str(exc)
            if ledger is not None and business_task_id is not None:
                trace_record = ledger.archive_performance_trace(
                    business_task_id,
                    getattr(exc, "performance_trace_path", None),
                )
                ledger.record_terminal(
                    business_task_id,
                    status="cancelled",
                    error={"type": "TaskCancelled", "message": str(exc)},
                    details={"performance_trace": trace_record} if trace_record else None,
                )
            publish({"phase": "cancelled", "message": str(exc), "business_task_id": business_task_id})
        except Exception as exc:
            traceback.print_exc()
            outcome["error"] = exc
            if ledger is not None and business_task_id is not None:
                ledger.record_terminal(business_task_id, status="failed", error={"message": str(exc)})
            publish({"phase": "failed", "error": str(exc), "business_task_id": business_task_id})
        finally:
            # Finalize only after every success/failure/cancellation path has
            # written its durable terminal ledger event.  This is deliberately
            # outside the remote client so a single document can cover client
            # upload, GPU timing, polling, result transfer and local storage.
            if ledger is not None and business_task_id is not None:
                try:
                    task = ledger.get(business_task_id) or {}
                    if task.get("status") in {"succeeded", "failed", "cancelled", "interrupted_unconfirmed"}:
                        end_to_end_trace = ledger.finalize_end_to_end_trace(business_task_id)
                        if outcome.get("result") is not None and end_to_end_trace:
                            outcome["result"]["end_to_end_trace"] = end_to_end_trace.get("result_copy_path")
                        publish({
                            "mode": "remote_gpu",
                            "phase": task.get("status"),
                            "business_task_id": business_task_id,
                            "end_to_end_trace": end_to_end_trace,
                        })
                except Exception:
                    # Observability must never conceal the original analysis
                    # outcome.  The traceback still makes a broken trace
                    # implementation diagnosable from the business process.
                    traceback.print_exc()
            finished.set()
            _ANALYSIS_TASKS.finish(task_handle)

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
        if task_handle.is_cancelled() and status.get("phase") not in {"cancelled", "succeeded", "failed"}:
            status["phase"] = "cancelling"
            status["cancellation_requested"] = True
        if updated or status["phase"] == "preparing":
            yield None, None, None, None, None, None, None, [], None, [], {}, None, [], None, status.copy(), None, [], {}
        time.sleep(0.4)

    if "cancelled" in outcome:
        status.update({"phase": "cancelled", "message": outcome["cancelled"]})
        status["elapsed_seconds"] = round(time.monotonic() - started, 1)
        yield None, None, None, None, None, None, None, [], None, [], {}, None, [], None, status.copy(), None, [], {}
        return
    if "interrupt_unconfirmed" in outcome:
        status.update({
            "phase": "interrupted_unconfirmed",
            "message": outcome["interrupt_unconfirmed"],
        })
        status["elapsed_seconds"] = round(time.monotonic() - started, 1)
        yield None, None, None, None, None, None, None, [], None, [], {}, None, [], None, status.copy(), None, [], {}
        return
    if "error" in outcome:
        # Preserve the failure in the visible progress panel.  Raising a
        # Gradio exception immediately can replace the last streamed JSON with
        # a short toast, which made transport failures impossible to diagnose.
        status.update({
            "phase": "failed",
            "error": str(outcome["error"]),
            "error_type": type(outcome["error"]).__name__,
            "action": "请查看 error 字段；若为上传/连接失败，请先恢复 127.0.0.1:8080 SSH 隧道。",
        })
        status["elapsed_seconds"] = round(time.monotonic() - started, 1)
        yield None, None, None, None, None, None, None, [], None, [], {}, None, [], None, status.copy(), None, [], {}
        return
    result = outcome["result"]

    for warning in result.get("warnings", []):
        gr.Warning(warning)

    video_candidate = result.get("video")
    output_video = video_candidate if video_candidate and os.path.isfile(video_candidate) else None
    viz_images = [img for img in result["visualizations"] if os.path.isfile(img)]

    metadata_content = None
    if os.path.isfile(result["metadata"]):
        with open(result["metadata"], "r", encoding="utf-8") as f:
            metadata_content = json.load(f)

    detections_file = result["detections"] if os.path.isfile(result["detections"]) else None
    tracknet_raw_file = result.get("tracknet_raw_csv")
    if tracknet_raw_file and not os.path.isfile(tracknet_raw_file):
        tracknet_raw_file = None
    performance_report_file = result.get("performance_report")
    if performance_report_file and not os.path.isfile(performance_report_file):
        performance_report_file = None
    rally_summary, rally_rows = _rally_summary_from_result(result, metadata_content)
    movement_metrics_file = result.get("movement_metrics")
    if movement_metrics_file and not os.path.isfile(movement_metrics_file):
        movement_metrics_file = None
    movement_rally_window_sweep_file = result.get("movement_rally_window_sweep")
    if movement_rally_window_sweep_file and not os.path.isfile(movement_rally_window_sweep_file):
        movement_rally_window_sweep_file = None
    movement_metrics_detail = _read_json_mapping(
        movement_metrics_file,
        {"status": "not_generated", "message": "本次分析未生成运动数据。"},
    )
    movement_metric_rows = _movement_metric_summary_rows(movement_metrics_detail)
    body_profile_rows = _body_profile_rows_from_metrics(movement_metrics_file)
    player_photos = extract_full_video_candidate_photos(
        video_file,
        detections_file,
        result.get("output_dir") or Path(detections_file).parent if detections_file else "",
    )
    player_gallery, player_rows, player_detail = build_player_result_display(
        movement_metrics_detail,
        photo_records=player_photos,
    )

    status["phase"] = "succeeded"
    status["elapsed_seconds"] = round(time.monotonic() - started, 1)
    roster = ((metadata_content or {}).get("temporal_tracking") or {}).get("players", {}).get("match_roster")
    if roster is not None:
        status["match_roster"] = roster
    yield (
        output_video, viz_images or None, metadata_content, detections_file,
        tracknet_raw_file, performance_report_file, rally_summary, rally_rows,
        movement_metrics_file, movement_metric_rows, movement_metrics_detail,
        movement_rally_window_sweep_file,
        body_profile_rows, result.get("output_dir"), status.copy(),
        player_gallery or None, player_rows, player_detail,
    )


def run_local_stream_replay(
    analysis_ready,
    video_file,
    corners,
    shuttle_detector,
    tracker_backend,
    pose_imgsz,
    analysis_sample_hz,
    generate_annotated_video,
    far_player_enhancement,
    gpu_base_url,
    sport_id="badminton",
):
    """Submit a recording and surface the business-derived movement metrics."""

    if sport_id == "tennis":
        raise gr.Error("网球模式只允许直连网球 GPU 流式视觉服务，不进入羽毛球业务网关。")
    if not analysis_ready:
        raise gr.Error("请先自动检测或手动确认四个球场角点，再启动模拟流式分析。")
    if video_file is None:
        raise gr.Error("请先上传视频。")
    try:
        initial = start_local_stream_replay(
            video_file,
            corners,
            shuttle_detector=shuttle_detector,
            tracker_backend=tracker_backend,
            pose_imgsz=int(pose_imgsz),
            analysis_sample_hz=int(analysis_sample_hz),
            generate_annotated_video=bool(generate_annotated_video),
            far_player_enhancement=bool(far_player_enhancement),
            gpu_base_url=gpu_base_url,
        )
        for status in iter_local_stream_replay(initial):
            derivation = ((status.get("result") or {}).get("business_derivation") or {})
            metrics_path = derivation.get("movement_metrics_path")
            if metrics_path and not os.path.isfile(metrics_path):
                metrics_path = None
            metrics = _read_json_mapping(
                metrics_path,
                {
                    "status": "processing",
                    "message": "GPU 已收到流分片；等待业务侧汇总真实高置信人物观测。",
                },
            )
            analysis_dir = str(Path(metrics_path).parents[1]) if metrics_path else None
            yield (
                metrics_path,
                _movement_metric_summary_rows(metrics),
                metrics,
                _body_profile_rows_from_metrics(metrics_path),
                analysis_dir,
                status,
            )
    except StreamReplayError as exc:
        yield (
            None,
            [],
            {"status": "failed", "message": str(exc)},
            [],
            None,
            {
                "mode": "local_business_to_gpu_segment_replay",
                "status": "failed",
                "error": str(exc),
                "action": "确认业务网关和 GPU API 已按当前环境变量启动；开发默认业务网关为 127.0.0.1:8080。",
            },
        )


def _full_video_upload_update(update):
    """Keep the stream-status panel explicit for a direct full-file upload."""

    return (*update, {
        "mode": "full_video_direct_gpu",
        "status": "idle",
        "hint": "当前任务按完整视频直接提交到 GPU。",
    })


def _two_second_segment_upload_update(stream_status):
    """Display direct-GPU segment progress in the primary analysis outputs."""

    status = {
        **(stream_status or {}),
        "upload_mode": "two_second_segments",
        "segment_seconds": 2.0,
    }
    stream_result = status.get("webui_result") or {}
    metrics_path = stream_result.get("movement_metrics_path")
    if metrics_path and not os.path.isfile(metrics_path):
        metrics_path = None
    metrics = _read_json_mapping(
        metrics_path,
        {"status": "streaming", "message": "等待远端会话生成可验证的人物运动数据。"},
    )
    gallery, player_rows, player_detail = build_player_result_display(
        metrics,
        track_candidates=status.get("track_candidates"),
        photo_records=stream_result.get("candidate_photos"),
    )
    derivation = stream_result.get("derivation") or {}
    return (
        None, None, None, None, None, None, None, [],
        metrics_path, _movement_metric_summary_rows(metrics), metrics,
        None, _body_profile_rows_from_metrics(metrics_path), stream_result.get("analysis_output_dir"), status, status,
        gallery or None, player_rows, {
            **player_detail,
            "stream_derivation": derivation,
            "stream_status": status,
        },
    )


def _find_remote_stream_session_workdir(analysis_session_id, stream_root=None):
    """Find the local ledger for one direct-GPU stream session safely."""

    session_id = str(analysis_session_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
        return None
    root = Path(stream_root or Path("outputs") / "remote_stream_sessions")
    if not root.is_dir():
        return None
    for ledger_path in sorted(root.glob("*/delivery-ledger.json"), reverse=True):
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(ledger.get("analysis_session_id") or "") == session_id:
            return ledger_path.parent
    return None


def load_remote_two_second_stream_result(analysis_session_id, gpu_base_url, sport_id="badminton"):
    """Restore screenshots and all per-track evidence for an existing GPU session."""

    session_id = str(analysis_session_id or "").strip()
    if not session_id:
        raise gr.Error("请输入直连 GPU 会话 ID，例如 ssn_xxx。")
    workdir = _find_remote_stream_session_workdir(session_id)
    if workdir is None:
        raise gr.Error("本机未找到此会话的上传记录；请在原来的 WebUI 工作目录中恢复。")
    try:
        status = recover_remote_two_second_stream(
            session_id,
            workdir,
            gpu_base_url=gpu_base_url,
            sport_id=sport_id,
        )
    except (RemoteAnalysisError, StreamAPIError) as exc:
        raise gr.Error(str(exc)) from exc
    terminal_state = str(status.get("status") or "")
    if terminal_state not in {"finalized", "partial"}:
        detail = status.get("error") or "会话尚未生成可展示的完成结果。"
        raise gr.Error(f"会话 {session_id} 当前状态为 {terminal_state!r}：{detail}")
    return _two_second_segment_upload_update(status)


def run_analysis_with_upload_mode(
    analysis_ready, video_file, template_path, corners,
    pose_family, pose_mode, language, audio, match_mode,
    output_video_style, shuttle_detector, tracker_backend,
    movement_rally_settle_seconds, enable_huji_play_state,
    pose_imgsz, analysis_sample_hz, pose_conf, far_player_enhancement, far_pose_roi,
    generate_annotated_video, browser_video_reencode,
    show_skeletons, show_player_trajectories,
    show_court_trajectory, show_shuttlecock_trajectory,
    show_player_stats, show_pose_roi, visualize_positions,
    yolo_pose_model, ball_model, gpu_base_url, two_second_segment_push, expected_player_count,
    sport_id="badminton",
    progress=gr.Progress(track_tqdm=False),
):
    """Run the selected GPU transport without changing analysis settings.

    When selected, the browser sends the recording to the local business
    gateway, which produces independently decodable two-second MP4 segments
    and pushes them to the GPU in sequence.  Otherwise the existing complete
    file upload to ``/api/v1/jobs`` remains unchanged.
    """

    sport_id = "tennis" if sport_id == "tennis" else "badminton"
    # Tennis has no legacy full-file or local fallback path. Its fixed sport
    # service accepts only stream sessions, which keeps video interpretation
    # separate from the legacy badminton whole-video business facade.
    if sport_id == "tennis" or bool(two_second_segment_push):
        if not analysis_ready:
            raise gr.Error("请先自动检测或手动确认四个球场角点，再启动 GPU 分片推送。")
        if video_file is None:
            raise gr.Error("请先上传视频。")
        if not corners or len(corners) != 4:
            raise gr.Error("请先确认四个球场角点。")
        _validate_file_size(video_file, _MAX_VIDEO_BYTES, "Video")
        stream_options = {
            "analysis_sample_hz": int(analysis_sample_hz),
            "pose_imgsz": int(pose_imgsz),
            "shuttle_detector": "none" if sport_id == "tennis" else shuttle_detector,
            "tracker_backend": tracker_backend,
            "far_player_enhancement": bool(far_player_enhancement),
            "far_pose_roi": tuple(float(item.strip()) for item in far_pose_roi.split(',')),
            "expected_player_count": 2 if sport_id == "tennis" else int(expected_player_count),
        }
        stream_output_dir = os.path.join(
            "outputs", "remote_stream_sessions", datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        try:
            for stream_status in iter_remote_two_second_stream(
                video_file,
                corners,
                stream_options,
                stream_output_dir,
                gpu_base_url=gpu_base_url,
                sport_id=sport_id,
                session_mode="singles_match" if sport_id == "tennis" else None,
            ):
                yield _two_second_segment_upload_update(stream_status)
        except RemoteAnalysisError as exc:
            yield _two_second_segment_upload_update({
                "mode": "remote_gpu_two_second_stream",
                "phase": "failed",
                "remote_base_url": str(gpu_base_url or "").strip(),
                "error": str(exc),
            })
        return

    for update in run_full_analysis(
        analysis_ready,
        video_file,
        template_path,
        corners,
        pose_family,
        pose_mode,
        language,
        audio,
        match_mode,
        output_video_style,
        shuttle_detector,
        tracker_backend,
        movement_rally_settle_seconds,
        enable_huji_play_state,
        pose_imgsz,
        analysis_sample_hz,
        pose_conf,
        far_player_enhancement,
        far_pose_roi,
        generate_annotated_video,
        browser_video_reencode,
        show_skeletons,
        show_player_trajectories,
        show_court_trajectory,
        show_shuttlecock_trajectory,
        show_player_stats,
        show_pose_roi,
        visualize_positions,
        yolo_pose_model,
        ball_model,
        gpu_base_url,
        progress=progress,
    ):
        yield _full_video_upload_update(update)


def load_local_stream_replay_result(business_task_id: str, sport_id="badminton"):
    """Reload one durable stream-replay result into the visible WebUI.

    A stream replay runs asynchronously in the separate business gateway.  A
    browser refresh must not make its already-derived metrics invisible or
    force the user to replay the source video.  This is intentionally a read
    operation: it never creates a GPU session, reuploads a segment, or edits
    the received evidence.
    """

    if sport_id == "tennis":
        raise gr.Error("网球模式不接入羽毛球业务网关的模拟流式与赛后指标。")
    task_id = str(business_task_id or "").strip()
    if not task_id:
        raise gr.Error("请输入业务任务 ID，例如 bstr_xxx。")
    try:
        status = poll_local_stream_replay(task_id)
    except StreamReplayError as exc:
        raise gr.Error(str(exc)) from exc
    derivation = ((status.get("result") or {}).get("business_derivation") or {})
    metrics_path = derivation.get("movement_metrics_path")
    if status.get("status") != "completed" or not metrics_path or not os.path.isfile(metrics_path):
        detail = status.get("error") or derivation.get("message") or "任务尚未生成可读取的运动数据。"
        raise gr.Error(f"任务 {task_id} 当前状态为 {status.get('status')!r}：{detail}")
    metrics = _read_json_mapping(metrics_path, {"status": "missing"})
    status = dict(status)
    status["display_source"] = "loaded_existing_stream_replay"
    status["movement_metrics_path"] = metrics_path
    return (
        metrics_path,
        _movement_metric_summary_rows(metrics),
        metrics,
        _body_profile_rows_from_metrics(metrics_path),
        str(Path(metrics_path).parents[1]),
        status,
    )


def _attach_business_interpretation(result, publish=None):
    """Derive lightweight outputs after artifacts reach the business side.

    Metric/report failure never invalidates anonymous GPU evidence. The status
    remains explicit and retryable without another video upload or inference.
    """
    output_dir = (result or {}).get("output_dir")
    if not output_dir:
        return result
    started = time.monotonic()
    if publish is not None:
        publish({"mode": "business_gateway", "phase": "business_interpretation_running"})
    try:
        from business_gateway.post_match import generate_business_interpretation

        interpretation = generate_business_interpretation(output_dir)
    except (OSError, ValueError) as exc:
        status = {
            "status": "failed",
            "owner": "business_gateway",
            "reason": str(exc),
            "retryable_without_video_analysis": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        result["business_interpretation"] = status
        result.setdefault("warnings", []).append(
            f"匿名视频数据已保存，但业务指标生成失败，可直接重试，无需重新分析视频：{exc}"
        )
        if publish is not None:
            publish({"mode": "business_gateway", "phase": "business_interpretation_failed", **status})
        return result

    result["movement_metrics"] = interpretation.get("movement_metrics_path")
    result["performance_report"] = interpretation.get("performance_report_path")
    result["business_interpretation"] = interpretation
    if publish is not None:
        publish({
            "mode": "business_gateway",
            "phase": "business_interpretation_succeeded",
            "elapsed_seconds": interpretation.get("elapsed_seconds"),
            "movement_metrics": interpretation.get("movement_metrics_path"),
            "performance_report": interpretation.get("performance_report_path"),
        })
    return result


def interrupt_active_analysis(language="zh"):
    """Signal the active WebUI task; its worker exits at the next safe checkpoint."""
    active = _ANALYSIS_TASKS.request_cancel()
    if active is None:
        return {
            "phase": "idle",
            "message": "当前没有可中断的分析任务。" if language == "zh" else "No analysis task is running.",
        }
    return {
        "phase": "cancelling",
        "message": "已请求中断；正在等待当前帧或子进程安全退出。" if language == "zh"
        else "Interrupt requested; waiting for the current frame or child process to exit safely.",
        **active,
    }


def _rally_summary_from_result(result, metadata):
    """Build a compact, review-only rally table from local derived artifacts.

    A remote result may carry server-side absolute artifact paths in its
    metadata.  The WebUI therefore first resolves the local download folder,
    and only rebuilds the small derived files from immutable detections when a
    legacy GPU service did not return them.
    """
    metadata = metadata or {}
    derived = metadata.get("derived") or result.get("derived") or {}
    shuttle_source = (
        (metadata.get("models") or {})
        .get("shuttlecock_detection", {})
        .get("primary_source")
    )
    if shuttle_source == "none" or derived.get("status") == "movement_only":
        movement_rallies_path = Path(result.get("movement_rallies") or derived.get("rallies_path") or "")
        if movement_rallies_path.is_file():
            try:
                movement_payload = json.loads(movement_rallies_path.read_text(encoding="utf-8"))
                rallies = list(movement_payload.get("rallies") or [])
            except (OSError, ValueError):
                rallies = []
            rows = [
                [
                    item.get("rally_id"),
                    round(float(item.get("start_frame", 0)) / _result_fps(metadata), 2),
                    round(float(item.get("end_frame", 0)) / _result_fps(metadata), 2),
                    0, 0, 0, item.get("end_reason"),
                    round(float(item.get("confidence", 0.0)), 3),
                ]
                for item in rallies
            ]
            return (
                "### 人体稳定候选回合（待人工复核）\n"
                f"共 **{len(rallies)}** 个候选回合；未检测羽毛球，因此**不生成拍数、球速、球种或得分**。"
                "边界仅由所有预期球员连续稳定的窗口产生，任何检测缺失都会使窗口失效。",
                rows,
            )
        return (
            "### 回合与拍数\n"
            "本次选择了**不检测羽毛球**：只输出人物跑位、姿态和持续追踪数据，"
            "不会生成球轨迹、候选击球或候选回合。",
            [],
        )
    detections_path = Path(result.get("detections") or "")
    if not detections_path.is_file():
        return "### 回合与拍数\n未找到本地检测数据，暂时无法生成候选回合。", []
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


def _result_fps(metadata):
    fps = float(((metadata or {}).get("video") or {}).get("fps") or 0.0)
    return fps if fps > 0 else 30.0


def _body_profile_rows_from_metrics(metrics_path):
    """Seed a reviewable body-profile table from visual tracks only."""
    if not metrics_path or not os.path.isfile(metrics_path):
        return []
    try:
        payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    profile_by_track = {}
    profile_path = Path(metrics_path).parent / "player_body_profiles_v1.json"
    try:
        profile_by_track = {
            str(item.get("track_id")): item
            for item in json.loads(profile_path.read_text(encoding="utf-8")).get("profiles") or []
            if isinstance(item, dict) and item.get("track_id")
        }
    except (OSError, ValueError):
        pass
    return [
        [
            item.get("track_id"),
            (profile_by_track.get(str(item.get("track_id"))) or {}).get("weight_kg"),
            (profile_by_track.get(str(item.get("track_id"))) or {}).get("height_cm"),
        ]
        for item in payload.get("players") or []
        if item.get("track_id")
    ]


def _read_json_mapping(path, fallback=None):
    """Read a derived JSON artifact for direct WebUI display.

    Files remain the durable/exportable contract, while this helper gives the
    operator the same evidence in-page immediately after an analysis or a
    body-profile refresh.
    """
    if not path or not os.path.isfile(path):
        return fallback if fallback is not None else {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback if fallback is not None else {}
    return payload if isinstance(payload, dict) else (fallback if fallback is not None else {})


def _movement_metric_summary_rows(metrics):
    """Flatten per-track movement evidence into a readable result table.

    The complete object is shown in a JSON panel as well. This table is only
    a navigation aid: it retains data coverage and quality so a large distance
    or speed cannot be read without its evidence limitations.
    """
    rows = []
    tennis_visual_only = (metrics or {}).get("sport_id") == "tennis"
    for item in (metrics or {}).get("players") or []:
        movement = item.get("movement") or {}
        coverage = item.get("measurement_coverage") or {}
        energy = item.get("energy_estimate") or {}
        energy_value = energy.get("estimated_kcal_rounded")
        if energy_value is None and energy.get("estimated_kcal") is not None:
            energy_value = int(round(float(energy["estimated_kcal"])))
        # Existing result folders can still contain the previous range-only
        # schema. Display its midpoint as one migration-safe product value;
        # a subsequent save/refresh writes the new single-value schema.
        energy_range = energy.get("estimated_kcal_range") or []
        if energy_value is None and len(energy_range) >= 2:
            energy_value = int(round((float(energy_range[0]) + float(energy_range[1])) / 2.0))
        if energy_value is not None:
            energy_display = str(energy_value)
        else:
            energy_display = "待填写体重"
        if tennis_visual_only:
            rows.append([
                item.get("track_id"),
                movement.get("distance_m"),
                movement.get("mean_speed_mps"),
                movement.get("peak_speed_mps"),
                movement.get("moving_time_sec"),
                round(float(coverage.get("usable_measurement_ratio") or 0.0) * 100.0, 1),
                (item.get("quality") or {}).get("status"),
            ])
        else:
            rows.append([
                item.get("track_id"),
                movement.get("distance_m"),
                movement.get("mean_speed_mps"),
                movement.get("peak_speed_mps"),
                movement.get("moving_time_sec"),
                movement.get("high_intensity_movement_time_sec"),
                movement.get("acceleration_event_count"),
                movement.get("deceleration_event_count"),
                f"{movement.get('direction_change_count') or 0} / {movement.get('peak_direction_changes_30s') or 0}",
                round(float(coverage.get("usable_measurement_ratio") or 0.0) * 100.0, 1),
                energy_display,
                (item.get("quality") or {}).get("status"),
            ])
    return rows


def _load_latest_movement_display():
    """Hydrate the analysis tab from the newest completed local result.

    Restarting the WebUI must not force an operator to upload and analyse the
    same video again merely to inspect or amend already-generated movement
    data. This reads only durable local artifacts and never reruns inference.
    """
    run_dir = default_analysis_run("outputs")
    if not run_dir:
        return None, [], {"status": "waiting_for_analysis"}, [], None, {
            "status": "waiting_for_analysis",
            "message": "尚未找到可展示的本地运动数据。",
        }
    metrics_path = Path(run_dir) / "derived" / "player_movement_metrics_v1.json"
    metrics = _read_json_mapping(
        metrics_path,
        {"status": "not_generated", "message": "该结果尚未生成运动数据。"},
    )
    return (
        str(metrics_path) if metrics_path.is_file() else None,
        _movement_metric_summary_rows(metrics),
        metrics,
        _body_profile_rows_from_metrics(metrics_path),
        str(run_dir),
        {
            "status": "loaded_existing_result",
            "analysis_dir": str(run_dir),
            "movement_metrics": str(metrics_path) if metrics_path.is_file() else None,
        },
    )


def save_body_profiles_and_refresh_metrics(analysis_dir, profile_rows, consent, sport_id="badminton"):
    """Save explicit user inputs, recompute lightweight metrics, then refresh one report."""
    if sport_id == "tennis":
        raise gr.Error("网球模式当前仅输出匿名视觉速度，不能写入羽毛球赛后身体参数或报告。")
    run_dir = Path(str(analysis_dir or ""))
    if not run_dir.is_dir():
        raise gr.Error("请先完成一次视频分析，再填写体重和身高。")
    if not consent:
        raise gr.Error("请确认同意仅将身高体重用于本次赛后运动消耗估算。")
    rows = _dataframe_records(profile_rows)
    try:
        from business_gateway.metrics.movement import write_body_profiles
        from business_gateway.post_match import generate_business_interpretation

        body_path = write_body_profiles(run_dir, rows, consent=True)
        interpretation = generate_business_interpretation(
            run_dir,
            body_profiles_path=body_path,
        )
        metrics = interpretation["movement_metrics"]
        report = interpretation["performance_report"] or {"status": "not_requested"}
    except (OSError, ValueError) as exc:
        raise gr.Error(f"保存运动数据失败：{exc}") from exc
    return (
        metrics.get("metrics_path"),
        _movement_metric_summary_rows(metrics),
        metrics,
        _body_profile_rows_from_metrics(metrics.get("metrics_path")),
        report.get("report_path"),
        {
            "status": "succeeded",
            "body_profiles": body_path,
            "movement_metrics": metrics.get("metrics_path"),
            "performance_report_status": report.get("status"),
            "performance_report": report.get("report_path"),
        },
    )


def _dataframe_records(value):
    if hasattr(value, "to_dict"):
        value = value.to_dict("records")
    if not isinstance(value, list):
        return []
    records = []
    for row in value:
        if isinstance(row, dict):
            records.append(row)
        elif isinstance(row, (list, tuple)):
            records.append({
                "track_id": row[0] if len(row) > 0 else None,
                "weight_kg": row[1] if len(row) > 1 else None,
                "height_cm": row[2] if len(row) > 2 else None,
            })
    return records


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
        "analysis_sample_hz": "统一分析频率",
        "pose_conf": "远端人体置信阈值",
        "far_player_enhancement": "远端球员增强（全场 + 远端 ROI，使用当前 Pose 尺寸）",
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
        "interrupt_btn": "中断当前任务",
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
        "analysis_sample_hz": "Unified Analysis Sampling Rate",
        "pose_conf": "Far-player confidence threshold",
        "far_player_enhancement": "Far-player enhancement (full frame + far ROI at the selected Pose size)",
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
        "interrupt_btn": "Interrupt Current Task",
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
/* Gradio 6 applies value/label updates reliably, but component visibility
 * updates are not consistently reflected after a queued callback.  Keep the
 * sport boundary explicit in the browser as well: tennis never presents a
 * badminton detector, legacy business-stream restore, or post-match body
 * workflow.  The Python callbacks still enforce the same boundary server-side.
 */
html[data-good-sport-mode="tennis"] #badminton-shuttle-detector,
html[data-good-sport-mode="tennis"] #badminton-annotated-video,
html[data-good-sport-mode="tennis"] #badminton-legacy-stream-button,
html[data-good-sport-mode="tennis"] #badminton-legacy-stream-restore,
html[data-good-sport-mode="tennis"] #badminton-tracknet-raw,
html[data-good-sport-mode="tennis"] #badminton-performance-report,
html[data-good-sport-mode="tennis"] #badminton-business-results,
html[data-good-sport-mode="tennis"] #badminton-rally-review-tab-button,
html[data-good-sport-mode="tennis"] #badminton-analysis-history-tab-button {
  display: none !important;
}

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


_SPORT_MODE_CLIENT_SYNC = """
(sportId) => {
  document.documentElement.dataset.goodSportMode = sportId === "tennis" ? "tennis" : "badminton";
  return sportId;
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
        gr.update(label=t["analysis_sample_hz"]),
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
        gr.update(value=t["interrupt_btn"]),
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
    reviewed_count = len(session.get("reviewed_rallies") or [])
    reviewed = f"　人工确认回合 **{reviewed_count}**" if reviewed_count else ""
    return (
        f"**复核进度**　{selected}　已处理 **{completed}**　待复核 **{summary['pending']}**　"
        f"确认/修正 **{summary['confirmed'] + summary['corrected']}**　排除 **{summary['excluded']}**{reviewed}"
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


def _review_empty_editor_markdown(playback_sec=0.0):
    return (
        "### 编辑触球\n"
        f"当前视频时间：**{float(playback_sec or 0.0):.2f}s**。"
        "当前时间前没有可编辑的触球；播放到触球处后会自动切换，"
        "也可以在这里添加人工触球。"
    )


def _review_rally_overlay(session, playback_sec):
    """Render the non-authoritative rally counter placed over the match video."""
    state = rally_playback_state(session, playback_sec)
    if state["status"] == "human_reviewed":
        rally = f"人工确认回合 {state['rally_number']} / {state['rally_count']}"
        if state["shot_count"]:
            shots = f"本回合已到第 {state['shot_index']} / {state['shot_count']} 拍"
        else:
            shots = "本回合尚未检测到触球"
        outcome = RALLY_TERMINAL_OUTCOMES.get(state.get("terminal_outcome"), "人工确认终止")
        detail = f"终止：{outcome} · {state['terminal_time_sec']:.2f}s"
    elif state["status"] == "after_last_human_terminal":
        rally = f"人工确认回合 {state['rally_number']} / {state['rally_count']} 已结束"
        shots = f"本回合 {state['shot_count']} 拍"
        outcome = RALLY_TERMINAL_OUTCOMES.get(state.get("terminal_outcome"), "人工确认终止")
        detail = f"终止：{outcome} · {state['terminal_time_sec']:.2f}s"
    elif state["status"] == "before_first_human_terminal":
        rally = f"人工确认回合 1 / {state['rally_count']}"
        shots = "等待第一拍"
        detail = f"首个终止点：{state['terminal_time_sec']:.2f}s"
    elif state["status"] == "assigned":
        rally = f"候选回合 {state['rally_number']} / {state['rally_count']}"
        shots = f"本回合第 {state['shot_index']} / {state['shot_count']} 拍"
        detail = "仅统计当前离线候选；待人工复核"
    elif state["status"] == "unassigned_touch":
        rally = "候选回合：待重新分段"
        shots = "当前为人工补拍，尚未归属回合"
        detail = "不会自动并入相邻回合"
    else:
        rally = "候选回合：等待首拍"
        shots = "当前回合 0 拍"
        detail = "播放至第一条触球记录后显示"
    return (
        "<div class='review-rally-overlay-card'>"
        f"<strong>{escape(rally)}</strong>"
        f"<span>{escape(shots)}</span>"
        f"<small>{escape(detail)}</small>"
        "</div>"
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
    initial_timeline = timeline_state(session, 0.0)
    initial_candidate = initial_timeline["current"]
    selected_id = initial_candidate["shot_id"] if initial_candidate else None
    if selected_id:
        candidate = next(item for item in session["candidates"] if item["shot_id"] == selected_id)
        label, decision, reviewer, note = _review_candidate_form(candidate)
        editor_details = _review_editor_markdown(candidate)
    else:
        label, decision, reviewer, note = "unknown", "pending", "", ""
        editor_details = _review_empty_editor_markdown(0.0)
    return (
        gr.update(choices=choices, value=selected_id),
        _review_summary_markdown(session, selected_id),
        candidate_table(session),
        reviewed_rally_table(session),
        video_path,
        _review_video_fps(video_path),
        _review_match_video_message(session),
        "0.00s",
        _review_live_markdown(session, 0.0),
        _review_rally_overlay(session, 0.0),
        editor_details,
        label,
        decision,
        reviewer,
        note,
        "已打开整场时间轴。播放过程中右侧会显示最近一次触球的判断；自动候选不正确时可修改或在当前时间补加。",
    )


def _review_follow_playback(analysis_dir, playback_sec, selected_shot_id=None, reference_video=None):
    """Keep the editor on the touch at the displayed video time.

    The form is deliberately left alone while the player remains within the
    same touch candidate, so a reviewer does not lose an unsaved selection.
    Only a transition to another touch replaces the editor values.
    """
    if not analysis_dir:
        return (
            "—", "请先打开一场分析结果。", _review_rally_overlay({"candidates": []}, 0.0),
            gr.update(value=None), _review_empty_editor_markdown(), "unknown", "pending", "", "",
        )
    session = create_or_load_review_session(analysis_dir, reference_video)
    state = timeline_state(session, playback_sec)
    current = state["current"]
    next_shot_id = current["shot_id"] if current else None
    updates = [gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()]
    if next_shot_id != selected_shot_id:
        if current is None:
            updates = [
                gr.update(value=None), _review_empty_editor_markdown(state["playback_sec"]),
                "unknown", "pending", "", "",
            ]
        else:
            label, decision, reviewer, note = _review_candidate_form(current)
            updates = [
                gr.update(value=next_shot_id), _review_editor_markdown(current),
                label, decision, reviewer, note,
            ]
    return (
        f"{state['playback_sec']:.2f}s",
        _review_live_markdown(session, state["playback_sec"]),
        _review_rally_overlay(session, state["playback_sec"]),
        *updates,
    )


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
        _review_rally_overlay(session, hit_time),
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
        _review_rally_overlay(session, state["playback_sec"]),
        f"已在 **{state['playback_sec']:.2f}s** 添加人工触球。现在选择球种和复核结果，再保存本次修改。",
    )


def _review_add_terminal_at_playback(analysis_dir, playback_sec, outcome, reviewer, note, reference_video=None):
    if not analysis_dir:
        raise gr.Error("请先打开一场分析结果。")
    try:
        record, _ = add_manual_rally_terminal(analysis_dir, playback_sec, outcome, reviewer, note)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    session = create_or_load_review_session(analysis_dir, reference_video)
    outcome_display = RALLY_TERMINAL_OUTCOMES.get(record["outcome"], record["outcome"])
    return (
        reviewed_rally_table(session),
        _review_summary_markdown(session),
        _review_rally_overlay(session, playback_sec),
        f"已在 **{record['time_sec']:.2f}s** 标记人工回合结束：**{outcome_display}**。"
        "该记录写入 `shot_review/rally_terminals.jsonl`，原始检测未修改。",
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


_TASK_STATUS_TEXT = {
    "submitting": "正在提交",
    "accepted": "已接收",
    "queued": "远端排队",
    "running": "远端运行中",
    "downloading": "正在拉取结果",
    "downloaded": "已下载，等待整理",
    "succeeded": "已完成",
    "failed": "失败",
    "cancelled": "已中断",
    "interrupted_unconfirmed": "本地中断，远端状态未确认",
    "local_fallback": "本地回退处理中",
    "submission_unconfirmed": "未获得远端接收回执",
}


def _task_history_last_remote_details(task):
    """Return the latest remote status payload from an append-only ledger."""
    for entry in reversed(task.get("history") or []):
        if entry.get("event") == "remote_status_polled":
            return dict(entry.get("details") or {})
    return {}


def _task_history_error(task):
    error = task.get("error") or {}
    if isinstance(error, dict):
        return str(error.get("message") or error.get("type") or "")
    return str(error or "")


def _task_history_rows(ledger=None):
    """Build a compact, restart-safe task index without making remote calls."""
    ledger = ledger or BusinessTaskLedger()
    rows = []
    for task in ledger.list_tasks():
        remote = task.get("remote") or {}
        details = _task_history_last_remote_details(task)
        progress = details.get("ratio")
        progress_text = "—" if progress is None else f"{float(progress) * 100:.1f}%"
        processed = details.get("processed_frames")
        total = details.get("total_frames")
        if processed is not None:
            progress_text += f" ({processed}/{total if total is not None else '?'})"
        tracking = details.get("tracking") or {}
        stage = details.get("stage") or tracking.get("stage") or tracking.get("phase") or "—"
        output_dir_text = task.get("output_dir")
        output_dir = Path(output_dir_text) if output_dir_text else None
        has_result = bool(output_dir and (output_dir / "metadata.json").is_file())
        rows.append([
            task.get("task_id"),
            task.get("created_at"),
            _TASK_STATUS_TEXT.get(task.get("status"), task.get("status")),
            remote.get("job_id") or "—",
            progress_text,
            stage,
            "已落地" if has_result else "未落地",
            _task_history_error(task)[:180] or "—",
        ])
    return rows


def _task_history_choices(ledger=None):
    ledger = ledger or BusinessTaskLedger()
    choices = []
    for task in ledger.list_tasks():
        remote = task.get("remote") or {}
        label = (
            f"{str(task.get('created_at') or '')[:19]} · "
            f"{_TASK_STATUS_TEXT.get(task.get('status'), task.get('status'))} · "
            f"{str(remote.get('job_id') or task.get('task_id'))[:12]}"
        )
        choices.append((label, task["task_id"]))
    return choices


def _task_history_detail(task_id, ledger=None):
    if not task_id:
        return {"hint": "选择一条任务查看完整业务事件、远端 Job ID、错误和本地结果目录。"}
    ledger = ledger or BusinessTaskLedger()
    task = ledger.get(task_id)
    if task is None:
        return {"error": "任务记录不存在或已被手动删除。", "task_id": task_id}
    # Keep the on-screen payload legible. The complete audit trail remains in
    # outputs/business_tasks/<task_id>.json for export and forensic debugging.
    return {
        "task_id": task.get("task_id"),
        "status": task.get("status"),
        "status_text": _TASK_STATUS_TEXT.get(task.get("status"), task.get("status")),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "output_dir": task.get("output_dir"),
        "output_metadata_exists": bool(
            task.get("output_dir")
            and (Path(task["output_dir"]) / "metadata.json").is_file()
        ),
        "remote": task.get("remote"),
        "last_remote_status": _task_history_last_remote_details(task),
        "error": task.get("error"),
        "performance_trace": task.get("performance_trace"),
        "end_to_end_trace": task.get("end_to_end_trace"),
        "event_count": len(task.get("history") or []),
        "recent_events": (task.get("history") or [])[-30:],
        "full_ledger_path": str(ledger._path(task_id)),
    }


def _task_history_refresh(selected_task_id=None):
    """Run one bounded reconciliation pass, then refresh the durable index."""
    try:
        reconciliation = reconcile_once()
        note = f"已完成一次远端对账：检查 {len(reconciliation)} 条未结束任务。"
    except Exception as exc:
        reconciliation = []
        note = f"远端对账未完成：{exc}。已保留原有本地任务记录，可稍后重试。"
    ledger = BusinessTaskLedger()
    choices = _task_history_choices(ledger)
    allowed = {value for _, value in choices}
    selected = selected_task_id if selected_task_id in allowed else (choices[0][1] if choices else None)
    detail = _task_history_detail(selected, ledger)
    detail["last_reconciliation"] = reconciliation
    return _task_history_rows(ledger), gr.update(choices=choices, value=selected), detail, note


def _task_history_select(task_id):
    return _task_history_detail(task_id)


def _start_task_reconciliation_worker():
    """Reconcile after a WebUI restart without delaying the first page render."""
    def worker():
        try:
            summary = reconcile_once()
            if summary:
                print(f"Restart-safe remote task reconciliation: {len(summary)} task(s) checked")
        except Exception:
            # The history tab exposes the persistent task state and allows a
            # manual retry. A temporarily unreachable remote API must never
            # prevent Gradio from starting.
            traceback.print_exc()

    threading.Thread(target=worker, name="remote-task-reconciliation", daemon=True).start()


def build_ui():
    t = _UI_TEXT["zh"]

    with gr.Blocks(
        title="Good Badminton — AI Badminton Analysis",
    ) as demo:
        md_title = gr.Markdown(t["title"])
        sport_mode = gr.Radio(
            choices=[("羽毛球模式", "badminton"), ("网球模式（单打对打）", "tennis")],
            value="badminton",
            label="运动模式",
            info="每次分析固定一种运动，并在上传前校验对应 GPU 实例的 sport_id。",
        )
        sport_mode_banner = gr.Markdown(
            "## 羽毛球视觉分析\n"
            "保留原有完整视频与 2 秒直连分片两种上传方式；运动模式由当前羽毛球 GPU 服务固定。"
        )

        corners_state = gr.State(value=None)
        click_corners_state = gr.State(value=[])
        template_path_state = gr.State(value=None)
        analysis_ready_state = gr.State(value=False)

        with gr.Tabs():
            render_backoffice_tabs(_ANALYSIS_TASKS)
            with gr.Tab("分析工作台"):
                with gr.Tabs():
                    analysis_tab = gr.Tab("视频分析")
                    review_tab = gr.Tab("球路复核", elem_id="badminton-rally-review-tab")
                    history_tab = gr.Tab(
                        "分析任务历史",
                        elem_id="badminton-analysis-history-tab",
                    )

        with analysis_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    md_inputs = gr.Markdown(t["inputs"])
                    video_input = gr.File(label=t["video"], file_types=["video"])
                    template_input = gr.File(label=t["template"], file_types=["image"])

                    # The analysis service is anonymous and mode-free: it
                    # keeps every visible person as a track, without asking
                    # whether this is singles, doubles or an informal game.
                    # The legacy file pipeline gets its most permissive
                    # compatible setting internally; teams are post-analysis
                    # business data, not video-analysis input.
                    md_settings = gr.Markdown(
                        "### 开发分析\n"
                        "不需要选择单打或双打；系统采集匿名轨迹，赛后再由业务侧认领。"
                    )
                    pose_family = gr.State(value="yolo-pose")
                    pose_mode = gr.State(value="balanced")
                    language = gr.State(value="zh")
                    audio = gr.State(value=False)
                    match_mode = gr.State(value="doubles")
                    output_video_style = gr.State(value="annotated")
                    shuttle_detector = gr.Dropdown(
                        choices=[
                            ("不检测羽毛球（最快，仅人物跑位/姿态）", "none"),
                            ("YOLO 羽毛球检测（采集球点与球速候选）", "yolo"),
                            ("TrackNetV3 羽毛球轨迹增强（较慢）", "tracknet_v3"),
                        ],
                        value="yolo",
                        label="羽毛球数据",
                        info=(
                            "不检测仅保留人物数据；YOLO 采集球点与球速候选；"
                            "TrackNetV3 更适合试验球轨迹，但耗时明显更高。"
                        ),
                        elem_id="badminton-shuttle-detector",
                    )
                    with gr.Row(visible=True) as legacy_business_stream_controls:
                        pose_imgsz = gr.Dropdown(
                            choices=[640, 960, 1280], value=960,
                            label="Pose 尺寸", scale=1,
                        )
                        analysis_sample_hz = gr.Dropdown(
                            choices=[10, 15, 30], value=10,
                            label="分析频率 (Hz)", scale=1,
                        )
                    tracker_backend = gr.Dropdown(
                        choices=[
                            ("ByteTrack（持续 ID）", "bytetrack"),
                            ("球场关联（可恢复）", "court_association"),
                        ],
                        value="bytetrack",
                        label="人物追踪方式",
                        info="只影响同一人跨帧关联；不会要求选择单打或双打。",
                    )
                    generate_annotated_video = gr.Checkbox(
                        value=False,
                        label="生成标注视频（较慢，可用于肉眼复核）",
                        elem_id="badminton-annotated-video",
                    )
                    two_second_segment_push = gr.Checkbox(
                        value=False,
                        label="每 2 秒直接分片推送到 GPU（开发测试）",
                        info=(
                            "勾选：WebUI 在本机将录像切成连续的 2 秒 MP4 片段，直接按顺序推送到上方 GPU 服务地址；"
                            "不勾选：整段视频直接提交 GPU。此模式不经过本地业务网关或 127.0.0.1:8080。"
                        ),
                    )
                    expected_player_count = gr.Radio(
                        choices=[("2 人（单打）", 2), ("4 人（双打）", 4)],
                        value=2,
                        label="分片流式场上人数",
                        info="仅在勾选 2 秒分片推送时生效。GPU 连续识别稳定人数后锁定名单，避免临时轨迹变成额外球员。",
                    )
                    # Keep remaining implementation controls in the callback
                    # contract, but make them deployment defaults rather than
                    # routine end-user choices.
                    movement_rally_settle_seconds = gr.State(value=0.7)
                    enable_huji_play_state = gr.State(value=False)
                    pose_conf = gr.State(value=0.15)
                    far_player_enhancement = gr.Checkbox(
                        value=False,
                        label=t["far_player_enhancement"],
                        info="默认关闭。开启后会额外检测远端球场区域；全场和 ROI 均使用上方选择的 Pose 尺寸。",
                    )
                    far_pose_roi = gr.State(value="0.12,0.30,0.86,0.82")
                    gpu_base_url = gr.Textbox(
                        label="GPU 服务地址（开发用）",
                        value=configured_gpu_base_url("badminton"),
                        placeholder="例如 http://xn-g.suanjiayun.com:55606",
                        info="完整上传与模拟流式分析均使用此地址；API Key 仍只从本机配置读取。生产环境应由业务服务固定配置。",
                    )
                    with gr.Row():
                        verify_gpu_btn = gr.Button("校验当前 GPU 运动身份", size="sm")
                    gpu_service_status = gr.Markdown(
                        "尚未校验 GPU。开始分析前会再次自动校验，错误运动实例不会接收视频。"
                    )
                    browser_video_reencode = gr.State(value=False)
                    show_skeletons = gr.State(value=True)
                    show_player_trajectories = gr.State(value=True)
                    show_court_trajectory = gr.State(value=True)
                    show_shuttlecock_trajectory = gr.State(value=True)
                    show_player_stats = gr.State(value=True)
                    show_pose_roi = gr.State(value=True)
                    visualize_positions = gr.State(value=True)
                    yolo_pose_model = gr.State(value="weights/yolo11n-pose.pt")
                    ball_model = gr.State(value="weights/yolo11s-ball.pt")

                with gr.Column(scale=2):
                    sport_calibration_help = gr.Markdown(
                        "### 羽毛球标定\n"
                        "可自动检测或手动确认球场四角；自动结果不可靠时请手动复核。"
                    )
                    md_step1 = gr.Markdown(t["step1"])
                    detect_btn = gr.Button(t["detect_btn"], variant="primary")
                    court_image = gr.Image(label=t["court_preview"], interactive=False, type="numpy")
                    corner_status = gr.Textbox(label=t["corner_status"], interactive=False, value=t["corner_none"])
                    apply_btn = gr.Button(t["apply_btn"], variant="secondary")

                    md_step2 = gr.Markdown(t["step2"])
                    with gr.Row():
                        run_btn = gr.Button(t["run_btn"], variant="primary")
                        stream_replay_btn = gr.Button(
                            "旧：经本地网关模拟流式（需 127.0.0.1:8080）",
                            variant="secondary",
                            elem_id="badminton-legacy-stream-button",
                        )
                        interrupt_btn = gr.Button(t["interrupt_btn"], variant="stop")
                    upload_transport_hint = gr.Markdown(
                        "勾选左侧的 `每 2 秒直接分片推送到 GPU` 后，点击 `运行分析` 会直接向上方 GPU 服务地址创建流会话并上传分片。"
                        "下方旧按钮仍保留给本地业务网关联调，不使用上方地址作为第一跳。"
                    )
                    with gr.Row(elem_id="badminton-legacy-stream-restore"):
                        stream_replay_task_id = gr.Textbox(
                            label="加载已完成流式任务",
                            placeholder="bstr_xxx（刷新页面后可重新查看数据，不重跑视频）",
                            scale=5,
                        )
                        load_stream_replay_btn = gr.Button("加载流式结果", variant="secondary", scale=1)
                    with gr.Row():
                        remote_stream_session_id = gr.Textbox(
                            label="加载已完成直连 GPU 会话",
                            placeholder="ssn_xxx（恢复截图和所有逐人数据，不重传视频）",
                            scale=5,
                        )
                        load_remote_stream_btn = gr.Button("恢复 GPU 结果", variant="primary", scale=1)

                    md_results = gr.Markdown(t["results"])
                    output_status = gr.JSON(
                        label=t["out_status"],
                        value={"phase": "idle", "hint": "点击运行分析后显示上传、排队、帧进度与执行来源。"},
                    )
                    stream_replay_status = gr.JSON(
                        label="分片任务状态（直连 GPU；旧按钮才经本地网关）",
                        value={
                            "status": "idle",
                            "hint": "勾选直连选项后显示远端 GPU session、每段上传和处理状态；旧按钮才会显示本地业务任务。",
                        },
                    )
                    output_video = gr.Video(label=t["out_video"])
                    output_gallery = gr.Gallery(label=t["out_gallery"], columns=2, height="auto")
                    gr.Markdown(
                        "### 人物截图与逐人数据\n"
                        "每张截图对应匿名视觉 `track_id`。完整视频从同一份高置信检测框生成截图；"
                        "2 秒分片会话使用 GPU 返回的候选截图。截图不是人脸识别，不能自动推断姓名或队伍。"
                    )
                    output_player_gallery = gr.Gallery(
                        label="人物截图（按视觉 Track ID）", columns=4, height="auto",
                    )
                    output_player_result_table = gr.Dataframe(
                        headers=PLAYER_RESULT_HEADERS,
                        datatype=["str", "str", "str", "number", "number", "number", "number", "number",
                                  "number", "number", "number", "str", "number", "str"],
                        interactive=False,
                        label="逐人运动与证据数据",
                        max_height=360,
                    )
                    output_player_result_detail = gr.JSON(
                        label="逐人完整证据数据（截图来源、覆盖率、速度、区域与排除原因）",
                        value={"status": "waiting_for_analysis"},
                    )
                    output_metadata = gr.JSON(label=t["out_metadata"])
                    output_detections = gr.File(label=t["out_detections"])
                    output_tracknet_raw = gr.File(
                        label="TrackNetV3 原始球点 CSV（可下载复核）",
                        elem_id="badminton-tracknet-raw",
                    )
                    output_performance_report = gr.File(
                        label="运动表现报告（含大模型状态）",
                        elem_id="badminton-performance-report",
                    )
                    output_movement_metrics = gr.File(label="运动数据（按视觉 Track ID）")
                    output_movement_metric_summary = gr.Dataframe(
                        headers=[
                            "track_id", "距离(m)", "平均速度(m/s)", "峰值速度(m/s)",
                            "有效移动(s)", "高强度(s)", "加速事件", "减速事件", "敏捷移动（整场/30秒峰值）",
                            "可用覆盖率(%)", "运动消耗估算(kcal)", "数据质量",
                        ],
                        datatype=["str", "number", "number", "number", "number", "number",
                                  "number", "number", "str", "number", "str", "str"],
                        interactive=False,
                        label="运动员运动数据汇总（仅使用真实高置信检测）",
                    )
                    output_movement_metric_detail = gr.JSON(
                        label="运动员完整分析数据（覆盖率、排除原因、场地区域与能量估算）",
                        value={"status": "waiting_for_analysis"},
                    )
                    analysis_output_dir_state = gr.State(value=None)
                    with gr.Group(
                        visible=True,
                        elem_id="badminton-business-results",
                    ) as badminton_business_results:
                        output_movement_rally_window_sweep = gr.File(label="无球回合稳定窗口对照（0.5/0.7/1.0 秒）")
                        gr.Markdown(
                            "### 赛后运动数据与身体参数\n"
                            "分析完成后会按每个视觉 `track_id` 生成距离、速度、加减速、变向和场地区域数据。"
                            "填写体重/身高并确认后，系统只计算**已测移动时段**的能量消耗区间；不会从视频猜测身体数据。"
                        )
                        body_profile_table = gr.Dataframe(
                            headers=["track_id", "weight_kg", "height_cm"],
                            datatype=["str", "number", "number"],
                            interactive=True,
                            label="赛后填写（按 Track ID；单打通常两行）",
                            max_height=180,
                        )
                        body_profile_consent = gr.Checkbox(
                            value=False,
                            label="我同意仅将上述身高体重用于本次赛后能量消耗估算",
                        )
                        save_body_profile_btn = gr.Button("保存身体参数并刷新运动数据/赛后报告")
                        body_profile_status = gr.JSON(label="身体参数与运动数据刷新状态", value={"status": "waiting_for_analysis"})
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
                    with gr.Group(elem_id="review-video-stage"):
                        review_match_video = gr.Video(
                            label="整场标注视频（可拖动进度条）",
                            height=440,
                            include_audio=True,
                            elem_id="review-match-video",
                        )
                        review_rally_overlay = gr.HTML(
                            value=(
                                "<div class='review-rally-overlay-card'>"
                                "<strong>候选回合：等待打开视频</strong>"
                                "<span>当前回合 0 拍</span>"
                                "</div>"
                            ),
                            elem_id="review-rally-overlay",
                            html_template="${value}",
                            css_template="""
                                #review-video-stage {
                                    position: relative !important;
                                    overflow: hidden;
                                }
                                #review-rally-overlay {
                                    position: absolute !important;
                                    right: 16px;
                                    bottom: 18px;
                                    z-index: 10;
                                    width: auto !important;
                                    margin: 0 !important;
                                    pointer-events: none;
                                }
                                #review-rally-overlay .review-rally-overlay-card {
                                    display: flex;
                                    flex-direction: column;
                                    gap: 3px;
                                    min-width: 180px;
                                    padding: 9px 11px;
                                    border: 1px solid rgba(144, 163, 255, 0.8);
                                    border-radius: 8px;
                                    background: rgba(8, 13, 30, 0.82);
                                    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.28);
                                    color: #fff;
                                    text-align: right;
                                }
                                #review-rally-overlay strong { font-size: 14px; }
                                #review-rally-overlay span { color: #dce5ff; font-size: 13px; }
                                #review-rally-overlay small { color: #aebce7; font-size: 11px; }
                            """,
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
                    review_rallies_table = gr.Dataframe(
                        headers=["回合", "开始(s)", "结束(s)", "候选拍数", "人工终止证据", "来源"],
                        datatype=["str", "number", "number", "number", "str", "str"],
                        interactive=False,
                        label="人工确认回合（优先于自动静止候选）",
                        max_height=220,
                    )
                with gr.Column(scale=2, elem_id="review-editor-panel"):
                    review_current_time = gr.Textbox(label="当前视频时间", value="—", interactive=False)
                    review_live_judgement = gr.Markdown("### 当前球路：等待打开视频")
                    with gr.Row():
                        edit_playback_candidate_btn = gr.Button("编辑当前触球", size="sm")
                        add_playback_candidate_btn = gr.Button("在当前时间添加触球", size="sm")
                    with gr.Accordion("人工确认回合结束", open=False):
                        review_terminal_outcome = gr.Dropdown(
                            choices=[(display, key) for key, display in RALLY_TERMINAL_OUTCOMES.items()],
                            value="landed_unknown",
                            label="当前时间的回合终止类型",
                        )
                        add_terminal_btn = gr.Button("在当前时间标记回合结束", size="sm")
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

        with history_tab:
            gr.Markdown(
                "## 分析任务历史\n"
                "每次提交、远端接收回执、轮询状态、结果下载和失败信息都会持久化到本机。"
                "WebUI 重启后不会重复上传；打开本页或点击对账会从远端恢复已确认任务的最新状态。"
            )
            initial_task_choices = _task_history_choices()
            initial_task_id = initial_task_choices[0][1] if initial_task_choices else None
            with gr.Row():
                refresh_task_history_btn = gr.Button("刷新并对账远端", variant="primary")
                task_history_selector = gr.Dropdown(
                    choices=initial_task_choices,
                    value=initial_task_id,
                    label="查看任务详情",
                    scale=3,
                )
            task_history_notice = gr.Markdown("本页未自动重复提交视频；只读取既有业务任务记录并轮询已确认的远端 Job。")
            task_history_table = gr.Dataframe(
                value=_task_history_rows(),
                headers=["业务任务", "发起时间", "当前状态", "远端 Job", "最后进度", "最后阶段", "本地结果", "错误摘要"],
                datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                interactive=False,
                label="全部已发起分析任务（最新在上）",
                max_height=360,
            )
            task_history_detail = gr.JSON(
                value=_task_history_detail(initial_task_id),
                label="任务详情与最近事件（完整账本路径在字段中）",
            )

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
        refresh_task_history_btn.click(
            fn=_task_history_refresh,
            inputs=[task_history_selector],
            outputs=[task_history_table, task_history_selector, task_history_detail, task_history_notice],
            show_progress="hidden",
        )
        history_tab.select(
            fn=_task_history_refresh,
            inputs=[task_history_selector],
            outputs=[task_history_table, task_history_selector, task_history_detail, task_history_notice],
            show_progress="hidden",
        )
        task_history_selector.change(
            fn=_task_history_select,
            inputs=[task_history_selector],
            outputs=[task_history_detail],
            show_progress="hidden",
        )
        open_review_btn.click(
            fn=_review_open_timeline,
            inputs=[review_analysis_dir, review_source_video],
            outputs=[
                review_shot_id, review_summary_output, review_candidates_table, review_rallies_table, review_match_video,
                review_frame_controls, review_video_message, review_current_time, review_live_judgement, review_rally_overlay,
                review_details, review_label, review_decision, review_reviewer, review_note, review_notice,
            ],
        )
        rebuild_review_btn.click(
            fn=_review_open_timeline,
            inputs=[review_analysis_dir, review_source_video],
            outputs=[
                review_shot_id, review_summary_output, review_candidates_table, review_rallies_table, review_match_video,
                review_frame_controls, review_video_message, review_current_time, review_live_judgement, review_rally_overlay,
                review_details, review_label, review_decision, review_reviewer, review_note, review_notice,
            ],
        )
        review_playback_clock.input(
            fn=_review_follow_playback,
            inputs=[review_analysis_dir, review_playback_clock, review_shot_id, review_source_video],
            outputs=[
                review_current_time, review_live_judgement, review_rally_overlay,
                review_shot_id, review_details, review_label, review_decision, review_reviewer, review_note,
            ],
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
                review_rally_overlay, review_shot_id, review_details, review_label, review_decision,
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
                review_rally_overlay, review_notice,
            ],
            show_progress="hidden",
        )
        add_terminal_btn.click(
            fn=_review_add_terminal_at_playback,
            inputs=[
                review_analysis_dir, review_playback_clock, review_terminal_outcome,
                review_reviewer, review_note, review_source_video,
            ],
            outputs=[review_rallies_table, review_summary_output, review_rally_overlay, review_notice],
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

        sport_mode_change = sport_mode.change(
            fn=configure_sport_mode,
            inputs=[sport_mode],
            outputs=[
                sport_mode_banner,
                sport_calibration_help,
                gpu_base_url,
                shuttle_detector,
                two_second_segment_push,
                expected_player_count,
                stream_replay_btn,
                stream_replay_task_id,
                load_stream_replay_btn,
                output_tracknet_raw,
                output_performance_report,
                badminton_business_results,
                generate_annotated_video,
            ],
            js=_SPORT_MODE_CLIENT_SYNC,
            show_progress="hidden",
        )
        sport_mode_change.then(
            fn=reset_court_selection,
            inputs=[language],
            outputs=[
                court_image,
                corners_state,
                click_corners_state,
                template_path_state,
                corner_status,
                analysis_ready_state,
            ],
            show_progress="hidden",
        )
        sport_mode_change.then(
            fn=reset_analysis_results_for_sport,
            inputs=[sport_mode],
            outputs=[
                output_video, output_gallery, output_metadata, output_detections, output_tracknet_raw,
                output_performance_report,
                output_rally_summary, output_rallies,
                output_movement_metrics, output_movement_metric_summary, output_movement_metric_detail,
                output_movement_rally_window_sweep,
                body_profile_table, analysis_output_dir_state, output_status,
                stream_replay_status,
                output_player_gallery, output_player_result_table, output_player_result_detail,
            ],
            show_progress="hidden",
        )
        sport_mode_change.then(
            fn=configure_sport_presentation,
            inputs=[sport_mode],
            outputs=[
                detect_btn,
                md_step1,
                upload_transport_hint,
                output_player_result_table,
                output_movement_metric_summary,
                output_movement_metric_detail,
            ],
            show_progress="hidden",
        )
        verify_gpu_btn.click(
            fn=verify_selected_gpu,
            inputs=[sport_mode, gpu_base_url],
            outputs=[gpu_service_status],
            show_progress="hidden",
        )

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
            inputs=[video_input, template_input, language, sport_mode],
            outputs=[court_image, corners_state, template_path_state, corner_status, analysis_ready_state],
        )

        court_image.select(
            fn=on_court_image_select,
            inputs=[click_corners_state, template_path_state, sport_mode],
            outputs=[court_image, click_corners_state, corners_state, corner_status],
        )

        apply_btn.click(
            fn=apply_manual_corners,
            # Automatic detection produces ``corners_state`` directly. Manual
            # clicks promote their completed four-point set into the same
            # state, so both paths use one canonical input at confirmation.
            inputs=[template_path_state, corners_state, sport_mode],
            outputs=[court_image, corners_state, analysis_ready_state],
        ).then(
            fn=lambda c, lang: _UI_TEXT.get(lang, _UI_TEXT["zh"])["manual_ok"].format(len(c)) if c
               else _UI_TEXT.get(lang, _UI_TEXT["zh"])["manual_fail"],
            inputs=[corners_state, language],
            outputs=[corner_status],
        )

        run_preflight = run_btn.click(
            fn=ensure_court_for_analysis,
            inputs=[video_input, template_path_state, corners_state, click_corners_state, language, sport_mode],
            outputs=[
                court_image, corners_state, click_corners_state,
                template_path_state, corner_status, analysis_ready_state,
            ],
        )
        run_preflight.then(
            fn=run_analysis_with_upload_mode,
            inputs=[
                 analysis_ready_state, video_input, template_path_state, corners_state,
                 pose_family, pose_mode, language, audio, match_mode, output_video_style, shuttle_detector, tracker_backend,
                 movement_rally_settle_seconds, enable_huji_play_state,
                 pose_imgsz, analysis_sample_hz, pose_conf, far_player_enhancement, far_pose_roi,
                 generate_annotated_video, browser_video_reencode,
                 show_skeletons, show_player_trajectories,
                show_court_trajectory, show_shuttlecock_trajectory,
                show_player_stats, show_pose_roi, visualize_positions,
                yolo_pose_model, ball_model, gpu_base_url, two_second_segment_push, expected_player_count,
                sport_mode,
            ],
            outputs=[
                output_video, output_gallery, output_metadata, output_detections, output_tracknet_raw,
                output_performance_report,
                output_rally_summary, output_rallies,
                output_movement_metrics, output_movement_metric_summary, output_movement_metric_detail,
                output_movement_rally_window_sweep,
                body_profile_table, analysis_output_dir_state, output_status,
                stream_replay_status,
                output_player_gallery, output_player_result_table, output_player_result_detail,
            ],
        )
        stream_preflight = stream_replay_btn.click(
            fn=ensure_court_for_analysis,
            inputs=[video_input, template_path_state, corners_state, click_corners_state, language, sport_mode],
            outputs=[
                court_image, corners_state, click_corners_state,
                template_path_state, corner_status, analysis_ready_state,
            ],
        )
        stream_preflight.then(
            fn=run_local_stream_replay,
            inputs=[
                analysis_ready_state, video_input, corners_state,
                shuttle_detector, tracker_backend, pose_imgsz, analysis_sample_hz,
                generate_annotated_video, far_player_enhancement,
                gpu_base_url, sport_mode,
            ],
            outputs=[
                output_movement_metrics,
                output_movement_metric_summary,
                output_movement_metric_detail,
                body_profile_table,
                analysis_output_dir_state,
                stream_replay_status,
            ],
        )
        load_stream_replay_btn.click(
            fn=load_local_stream_replay_result,
            inputs=[stream_replay_task_id, sport_mode],
            outputs=[
                output_movement_metrics,
                output_movement_metric_summary,
                output_movement_metric_detail,
                body_profile_table,
                analysis_output_dir_state,
                stream_replay_status,
            ],
        )
        load_remote_stream_btn.click(
            fn=load_remote_two_second_stream_result,
            inputs=[remote_stream_session_id, gpu_base_url, sport_mode],
            outputs=[
                output_video, output_gallery, output_metadata, output_detections, output_tracknet_raw,
                output_performance_report,
                output_rally_summary, output_rallies,
                output_movement_metrics, output_movement_metric_summary, output_movement_metric_detail,
                output_movement_rally_window_sweep,
                body_profile_table, analysis_output_dir_state, output_status,
                stream_replay_status,
                output_player_gallery, output_player_result_table, output_player_result_detail,
            ],
        )
        save_body_profile_btn.click(
            fn=save_body_profiles_and_refresh_metrics,
            inputs=[analysis_output_dir_state, body_profile_table, body_profile_consent, sport_mode],
            outputs=[
                output_movement_metrics, output_movement_metric_summary, output_movement_metric_detail,
                body_profile_table,
                output_performance_report, body_profile_status,
            ],
        )
        demo.load(
            fn=_load_latest_movement_display,
            outputs=[
                output_movement_metrics, output_movement_metric_summary, output_movement_metric_detail,
                body_profile_table, analysis_output_dir_state, body_profile_status,
            ],
            show_progress="hidden",
        )
        interrupt_btn.click(
            fn=interrupt_active_analysis,
            inputs=[language],
            outputs=[output_status],
            queue=False,
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    _start_task_reconciliation_worker()
    demo = build_ui()
    host, port = webui_listener()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        theme=gr.themes.Soft(),
        css=_APP_CSS + OPERATOR_BACKOFFICE_CSS,
    )
