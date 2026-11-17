"""Business-side renderer for a time-synchronised AI promotion video.

The GPU worker remains responsible only for visual evidence.  This module runs
*after* the existing annotated video and JSON artifacts are available.  It
never calls a detector, never changes ``detections.jsonl`` and never sends the
source video to an LLM.  Its only inputs are the rendered evidence video and
the already persisted anonymous tracking measurements.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from base64 import b64encode
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib import error, request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROMOTION_SCHEMA_VERSION = "promotion-video.v1"
_PANEL_WIDTH = 360
_CANVAS_WIDTH = 1920
_CANVAS_HEIGHT = 1080
_MAX_LLM_SEGMENTS = 4
_MAX_CURRENT_SPEED_GAP_SECONDS = 0.5
_MAX_PLAUSIBLE_SPEED_MPS = 10.0
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


class PromotionVideoError(RuntimeError):
    """Raised when the evidence video cannot be assembled into a promotion video."""


def generate_promotion_video(
    output_dir: str | Path,
    annotated_video_path: str | Path,
    *,
    sport_id: str = "badminton",
    cancel_cb: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Create the promotional layout from completed analysis artifacts.

    ``annotated_video_path`` must already contain the skeleton/ball overlay.
    This keeps the export independent from inference and makes a failed export
    safely retryable from the output directory without re-uploading video.
    """
    root = Path(output_dir).expanduser().resolve()
    source_video = Path(annotated_video_path).expanduser().resolve()
    if not source_video.is_file() or source_video.stat().st_size == 0:
        raise PromotionVideoError("宣传视频需要已生成的标注视频作为中央画面。")
    if not root.is_dir():
        raise PromotionVideoError("未找到本次视频分析的结果目录。")

    _raise_if_cancelled(cancel_cb)
    metadata = _read_json(root / "metadata.json")
    metrics = _read_json(root / "derived" / "player_movement_metrics_v1.json")
    observations = _load_track_observations(root / "detections.jsonl")
    players = _build_player_profiles(metrics, observations)
    if not players:
        raise PromotionVideoError("本次分析没有可用于宣传展示的匿名运动员轨迹。")

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise PromotionVideoError("无法读取已生成的标注视频。")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if not (1.0 <= fps <= 240.0) or frame_count <= 0 or source_width <= 0 or source_height <= 0:
            raise PromotionVideoError("标注视频缺少可用的帧率、尺寸或帧数信息。")

        duration_sec = frame_count / fps
        promotion_dir = root / "promotion"
        promotion_dir.mkdir(parents=True, exist_ok=True)
        commentary = _build_commentary_timeline(
            players,
            duration_sec=duration_sec,
            sport_id=sport_id,
        )
        _extract_annotated_moment_frames(source_video, commentary, promotion_dir)
        llm_result = _enrich_commentary_with_llm(commentary, players, sport_id=sport_id)
        commentary = llm_result["segments"]

        timeline_path = promotion_dir / "promotion_video_timeline_v1.json"
        timeline = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "kind": "ai_promotion_video_timeline",
            "source": {
                "annotated_video": str(source_video),
                "metadata": str(root / "metadata.json"),
                "movement_metrics": str(root / "derived" / "player_movement_metrics_v1.json"),
                "detections": str(root / "detections.jsonl"),
            },
            "sport_id": sport_id,
            "duration_sec": round(duration_sec, 3),
            "players": players,
            "commentary": commentary,
            "llm": {key: value for key, value in llm_result.items() if key != "segments"},
            "policy": {
                "video_analysis_reused": True,
                "gpu_inference_rerun": False,
                "raw_video_sent_to_llm": False,
                "annotated_moment_frames_sent_to_llm": bool(llm_result.get("uses_annotated_frames")),
                "identity_policy": "track_id is an anonymous visual identifier, not a confirmed person identity",
                "technical_commentary_policy": "LLM text is shown only as an evidence-bounded observation, not a score, diagnosis or factual ruling",
            },
        }
        _write_json(timeline_path, timeline)

        temporary_video = promotion_dir / "ai_promotion_video.rendering.mp4"
        final_video = promotion_dir / "ai_promotion_video.mp4"
        _render_layout(
            capture,
            temporary_video,
            fps=fps,
            frame_count=frame_count,
            players=players,
            commentary=commentary,
            sport_id=sport_id,
            cancel_cb=cancel_cb,
            progress_cb=progress_cb,
        )
    finally:
        capture.release()

    _raise_if_cancelled(cancel_cb)
    _transcode_with_source_audio(temporary_video, final_video, source_video)
    if temporary_video.is_file():
        temporary_video.unlink()
    return {
        "status": "succeeded",
        "video_path": str(final_video),
        "timeline_path": str(timeline_path),
        "llm_status": llm_result["status"],
        "llm_reason": llm_result.get("reason"),
        "player_count": len(players),
        "commentary_segment_count": len(commentary),
    }


def _load_track_observations(detections_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read only fresh detected court positions from immutable JSONL evidence."""
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not detections_path.is_file():
        return samples
    with detections_path.open(encoding="utf-8") as source:
        for raw_line in source:
            try:
                row = json.loads(raw_line)
                time_sec = float(row["time_sec"])
            except (ValueError, KeyError, TypeError):
                continue
            tracks = ((row.get("spatial") or {}).get("tracks") or [])
            for track in tracks:
                if not isinstance(track, Mapping) or track.get("status") != "detected":
                    continue
                track_id = str(track.get("track_id") or "")
                xy = track.get("court_xy_m")
                try:
                    x, y = float(xy[0]), float(xy[1])
                except (TypeError, ValueError, IndexError):
                    continue
                samples[track_id].append({
                    "time_sec": time_sec,
                    "x": x,
                    "y": y,
                    "zone_id": str(track.get("zone_id") or "未知区域"),
                    "court_end": str(track.get("court_end") or "unknown"),
                })
    for values in samples.values():
        values.sort(key=lambda item: item["time_sec"])
    return samples


def _build_player_profiles(metrics: Mapping[str, Any], observations: Mapping[str, list[dict[str, Any]]]):
    by_track = {
        str(item.get("track_id")): item
        for item in (metrics.get("players") or [])
        if isinstance(item, Mapping) and item.get("track_id")
    }
    track_ids = sorted(set(by_track).union(observations))
    profiles = []
    for track_id in track_ids:
        observed = list(observations.get(track_id) or [])
        metric = by_track.get(track_id) or {}
        movement = metric.get("movement") if isinstance(metric.get("movement"), Mapping) else {}
        coverage = metric.get("measurement_coverage") if isinstance(metric.get("measurement_coverage"), Mapping) else {}
        if not observed and not movement:
            continue
        average_y = sum(item["y"] for item in observed) / len(observed) if observed else math.inf
        profiles.append({
            "track_id": track_id,
            "panel_label": "",
            "average_court_y": average_y,
            "static_metrics": {
                "distance_m": _number_or_none(movement.get("distance_m")),
                "mean_speed_mps": _number_or_none(movement.get("mean_speed_mps")),
                "peak_speed_mps": _number_or_none(movement.get("peak_speed_mps")),
                "coverage_ratio": _number_or_none(coverage.get("usable_measurement_ratio")),
            },
            "observations": observed,
        })
    profiles.sort(key=lambda item: (item["average_court_y"], item["track_id"]))
    for index, profile in enumerate(profiles):
        profile["panel_label"] = "上半场运动员" if index == 0 else "下半场运动员" if index == 1 else f"运动员 {index + 1}"
    return profiles


def _build_commentary_timeline(players, *, duration_sec: float, sport_id: str):
    """Create deterministic evidence windows before optionally asking the LLM."""
    count = min(_MAX_LLM_SEGMENTS, max(1, int(math.ceil(duration_sec / 28.0))))
    span = duration_sec / count
    segments = []
    for index in range(count):
        start = round(index * span, 3)
        end = round(duration_sec if index == count - 1 else (index + 1) * span, 3)
        midpoint = (start + end) / 2
        observations = []
        for player in players[:2]:
            observation = _nearest_observation(player["observations"], midpoint)
            observations.append({
                "track_id": player["track_id"],
                "zone_id": observation.get("zone_id") if observation else "暂无有效位置",
                "current_speed_mps": _current_speed(player["observations"], midpoint),
            })
        segments.append({
            "segment_id": f"segment_{index + 1:02d}",
            "start_time_sec": start,
            "end_time_sec": end,
            "evidence": observations,
            "left_positive": _fallback_positive(
                observations[0] if observations else {}, sport_id
            ),
            "left_improvement": _fallback_improvement(sport_id),
            "right_positive": _fallback_positive(
                observations[1] if len(observations) > 1 else {}, sport_id
            ),
            "right_improvement": _fallback_improvement(sport_id),
            "commentary_source": "deterministic_evidence_placeholder",
            "annotated_frame_path": None,
        })
    return segments


def _extract_annotated_moment_frames(source_video: Path, segments, promotion_dir: Path):
    """Persist one evidence frame per commentary interval for vision-capable LLMs.

    The frame is sampled from the *already annotated* video, so the model sees
    the same skeleton and ball evidence as the final promotion video.  It is
    deliberately not a source-video upload and remains a reviewable artifact
    next to the rendered result.
    """
    frame_dir = promotion_dir / "moment_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        return
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        for segment in segments:
            if fps <= 0:
                break
            midpoint = (float(segment["start_time_sec"]) + float(segment["end_time_sec"])) / 2
            capture.set(cv2.CAP_PROP_POS_MSEC, midpoint * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            path = frame_dir / f"{segment['segment_id']}.jpg"
            if cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                segment["annotated_frame_path"] = str(path)
    finally:
        capture.release()


def _enrich_commentary_with_llm(segments, players, *, sport_id: str):
    """Use one bounded, OpenAI-compatible request when the user configures it."""
    config = _llm_config_from_environment()
    if config is None:
        return {
            "status": "not_configured", "reason": "LLM service is not configured",
            "uses_annotated_frames": False, "segments": segments,
        }
    prompt = _build_llm_prompt(segments, players, sport_id)
    try:
        image_paths = [segment.get("annotated_frame_path") for segment in segments]
        content = _request_openai_compatible(config, prompt, image_paths=image_paths)
        parsed = _parse_llm_segments(content, segments)
    except (OSError, ValueError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"status": "failed", "reason": str(exc), "uses_annotated_frames": False, "segments": segments}
    return {
        "status": "succeeded", "model": config["model"],
        "uses_annotated_frames": any(image_paths), "segments": parsed,
    }


def _build_llm_prompt(segments, players, sport_id):
    evidence = {
        "sport_id": sport_id,
        "players": [
            {
                "track_id": player["track_id"],
                "static_metrics": player["static_metrics"],
                "identity_policy": "anonymous visual track only",
            }
            for player in players[:2]
        ],
        "segments": [
            {
                "segment_id": segment["segment_id"],
                "start_time_sec": segment["start_time_sec"],
                "end_time_sec": segment["end_time_sec"],
                "evidence": segment["evidence"],
            }
            for segment in segments
        ],
    }
    return (
        "你是体育 AI 宣传视频的技战术解说编辑。只能根据 JSON 中已经测得的匿名跑位、区域和速度证据，及附带的标注抽帧，"
        "为每位运动员写一条正向观察和一条待改进观察。不要识别真人身份，不要判断比分、胜负、伤病或事实性对错。"
        "没有连续姿态、球体或球路直接证据时，不能评价击球质量；待改进项必须写成“证据不足，建议结合完整回放复核”。\n"
        "你还会收到按 segment_id 顺序排列的标注抽帧；只描述图中可见的骨架、球体标注和跑位。"
        "请只返回严格 JSON：{\"segments\":[{\"segment_id\":string,\"left_positive\":string,\"left_improvement\":string,\"right_positive\":string,\"right_improvement\":string}]}。"
        "每条不超过 42 个中文字符；正向观察和待改进观察都必须带“可观察到”或“建议结合完整回放复核”等证据限定语。\n证据：\n"
        + json.dumps(evidence, ensure_ascii=False)
    )


def _parse_llm_segments(content: str, baseline):
    parsed = json.loads(content)
    entries = {
        str(item.get("segment_id")): item
        for item in (parsed.get("segments") or [])
        if isinstance(item, Mapping) and item.get("segment_id")
    }
    enriched = []
    for segment in baseline:
        item = entries.get(segment["segment_id"])
        if item is None:
            raise ValueError("LLM response omitted a requested commentary segment")
        left_positive = _safe_comment(item.get("left_positive"))
        left_improvement = _safe_comment(item.get("left_improvement"))
        right_positive = _safe_comment(item.get("right_positive"))
        right_improvement = _safe_comment(item.get("right_improvement"))
        if not all((left_positive, left_improvement, right_positive, right_improvement)):
            raise ValueError("LLM commentary must contain positive and improvement observations for both players")
        enriched.append({
            **segment,
            "left_positive": left_positive,
            "left_improvement": left_improvement,
            "right_positive": right_positive,
            "right_improvement": right_improvement,
            "commentary_source": "llm_evidence_bounded",
        })
    return enriched


def _render_layout(capture, output_path, *, fps, frame_count, players, commentary, sport_id, cancel_cb, progress_cb):
    """Render all presentation elements in one pass over the annotated source."""
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (_CANVAS_WIDTH, _CANVAS_HEIGHT),
    )
    if not writer.isOpened():
        raise PromotionVideoError("无法创建宣传视频输出文件。")
    try:
        frame_index = 0
        while True:
            _raise_if_cancelled(cancel_cb)
            ok, frame = capture.read()
            if not ok:
                break
            time_sec = frame_index / fps
            canvas = _promotion_canvas(frame)
            active = _active_segment(commentary, time_sec)
            _draw_player_panel(canvas, 0, players[0] if players else None, active, time_sec, side="left")
            _draw_player_panel(canvas, _CANVAS_WIDTH - _PANEL_WIDTH, players[1] if len(players) > 1 else None, active, time_sec, side="right")
            _draw_header(canvas, sport_id, time_sec)
            writer.write(canvas)
            frame_index += 1
            if progress_cb and (frame_index == 1 or frame_index % max(1, int(fps)) == 0 or frame_index == frame_count):
                progress_cb(frame_index, frame_count)
    finally:
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise PromotionVideoError("宣传视频渲染没有产生有效文件。")


def _promotion_canvas(frame):
    canvas = np.zeros((_CANVAS_HEIGHT, _CANVAS_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (15, 19, 29)
    center_width = _CANVAS_WIDTH - (_PANEL_WIDTH * 2)
    source_h, source_w = frame.shape[:2]
    scale = min(center_width / source_w, _CANVAS_HEIGHT / source_h)
    width, height = max(1, int(source_w * scale)), max(1, int(source_h * scale))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    x = _PANEL_WIDTH + (center_width - width) // 2
    y = (_CANVAS_HEIGHT - height) // 2
    canvas[y:y + height, x:x + width] = resized
    cv2.rectangle(canvas, (_PANEL_WIDTH, 0), (_PANEL_WIDTH + center_width - 1, _CANVAS_HEIGHT - 1), (54, 93, 156), 2)
    return canvas


def _draw_header(canvas, sport_id, time_sec):
    title = "AI MATCH INSIGHT" if sport_id == "tennis" else "AI BADMINTON INSIGHT"
    cv2.putText(canvas, title, (_PANEL_WIDTH + 26, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (235, 243, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{time_sec:05.1f}s", (_CANVAS_WIDTH - _PANEL_WIDTH - 126, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (123, 211, 255), 2, cv2.LINE_AA)


def _draw_player_panel(canvas, x, player, segment, time_sec, *, side):
    panel = canvas[:, x:x + _PANEL_WIDTH]
    panel[:] = (22, 29, 45)
    accent = (255, 186, 73) if side == "left" else (103, 210, 255)
    cv2.rectangle(panel, (0, 0), (_PANEL_WIDTH - 1, _CANVAS_HEIGHT - 1), accent, 2)
    if player is None:
        _draw_text(panel, "等待第二名运动员", 28, 80, 24, (210, 220, 235))
        return
    _draw_text(panel, player["panel_label"], 28, 72, 27, accent)
    _draw_text(panel, player["track_id"], 28, 106, 20, (205, 215, 232))
    static = player["static_metrics"]
    _draw_metric(panel, "跑动距离", _format_metric(static.get("distance_m"), "m"), 28, 178, accent)
    _draw_metric(panel, "当前速度", _format_metric(_current_speed(player["observations"], time_sec), "m/s"), 28, 272, accent)
    _draw_metric(panel, "最大速度", _format_metric(static.get("peak_speed_mps"), "m/s"), 28, 366, accent)
    coverage = static.get("coverage_ratio")
    _draw_text(panel, f"检测覆盖率  {coverage * 100:.0f}%" if coverage is not None else "检测覆盖率  数据不足", 28, 426, 20, (160, 176, 199))
    cv2.line(panel, (28, 492), (_PANEL_WIDTH - 28, 492), (67, 82, 108), 1)
    heading = "AI 技战术观察" if segment and segment.get("commentary_source") == "llm_evidence_bounded" else "视觉运动观察"
    _draw_text(panel, heading, 28, 540, 23, (230, 238, 252))
    positive_key = "left_positive" if side == "left" else "right_positive"
    improvement_key = "left_improvement" if side == "left" else "right_improvement"
    _draw_text(panel, "积极观察", 28, 584, 19, (123, 219, 161))
    _draw_wrapped_text(
        panel,
        (segment or {}).get(positive_key, "等待可用运动数据。"),
        28,
        612,
        _PANEL_WIDTH - 56,
        24,
        (222, 230, 241),
        max_lines=2,
    )
    _draw_text(panel, "待改进", 28, 684, 19, (255, 190, 103))
    _draw_wrapped_text(
        panel,
        (segment or {}).get(improvement_key, "证据不足，建议结合完整回放复核。"),
        28,
        712,
        _PANEL_WIDTH - 56,
        24,
        (222, 230, 241),
        max_lines=2,
    )
    _draw_text(panel, "仅基于匿名视觉证据", 28, _CANVAS_HEIGHT - 38, 16, (137, 151, 175))


def _draw_metric(image, label, value, x, y, accent):
    _draw_text(image, label, x, y, 20, (179, 192, 214))
    _draw_text(image, value, x, y + 40, 32, accent)


def _draw_text(image, text, x, y, size, color):
    """Draw Unicode text with the repository's bundled Chinese font."""
    font = _font(size)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    ImageDraw.Draw(canvas).text((x, y - size), str(text), font=font, fill=tuple(reversed(color)))
    image[:] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _draw_wrapped_text(image, text, x, y, width, line_height, color, *, max_lines=None):
    font = _font(20)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    line = ""
    cursor_y = y
    for word in str(text):
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] > width and line:
            draw.text((x, cursor_y), line, font=font, fill=tuple(reversed(color)))
            cursor_y += line_height
            if max_lines is not None and (cursor_y - y) // line_height >= max_lines:
                break
            line = word
        else:
            line = candidate
    if line and (max_lines is None or (cursor_y - y) // line_height < max_lines):
        draw.text((x, cursor_y), line, font=font, fill=tuple(reversed(color)))
    image[:] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _font(size):
    """Reuse the project font rather than relying on platform font discovery."""
    if size not in _FONT_CACHE:
        font_path = Path(__file__).resolve().parents[1] / "simhei.ttf"
        if not font_path.is_file():
            raise PromotionVideoError("宣传视频缺少 simhei.ttf 中文字体文件。")
        _FONT_CACHE[size] = ImageFont.truetype(str(font_path), size)
    return _FONT_CACHE[size]


def _active_segment(segments, time_sec):
    for segment in segments:
        if float(segment["start_time_sec"]) <= time_sec <= float(segment["end_time_sec"]):
            return segment
    return segments[-1] if segments else None


def _nearest_observation(observations, time_sec):
    if not observations:
        return None
    return min(observations, key=lambda item: abs(item["time_sec"] - time_sec))


def _current_speed(observations, time_sec):
    previous = None
    for observation in observations:
        if observation["time_sec"] > time_sec:
            break
        previous = observation
    if previous is None:
        return None
    index = observations.index(previous)
    if index == 0:
        return None
    earlier = observations[index - 1]
    elapsed = previous["time_sec"] - earlier["time_sec"]
    if elapsed <= 0 or elapsed > _MAX_CURRENT_SPEED_GAP_SECONDS:
        return None
    speed = math.hypot(previous["x"] - earlier["x"], previous["y"] - earlier["y"]) / elapsed
    return round(speed, 3) if speed <= _MAX_PLAUSIBLE_SPEED_MPS else None


def _fallback_positive(evidence, sport_id):
    """Describe only the measured motion, never infer hitting quality."""
    zone = evidence.get("zone_id") or "暂无有效位置"
    speed = _format_metric(evidence.get("current_speed_mps"), "m/s")
    noun = "网球" if sport_id == "tennis" else "羽毛球"
    return f"可观察到该运动员位于 {zone}，当前视觉速度 {speed}；建议结合完整 {noun} 回放复核。"


def _fallback_improvement(sport_id):
    """Keep negative claims out when only a single measured moment exists."""
    noun = "网球" if sport_id == "tennis" else "羽毛球"
    return f"当前证据不足以判定击球质量；建议结合完整 {noun} 回放复核。"


def _format_metric(value, unit):
    return f"{float(value):.2f} {unit}" if value is not None else "数据不足"


def _safe_comment(value):
    text = " ".join(str(value or "").split())
    # Two fixed-height panel cards need a predictable maximum.  The prompt
    # asks for the same limit, but this protects the rendered layout if an LLM
    # ignores it.
    return text[:42] if text else ""


def _number_or_none(value):
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _raise_if_cancelled(cancel_cb):
    if cancel_cb is not None and cancel_cb():
        raise PromotionVideoError("宣传视频渲染已中断。")


def _llm_config_from_environment():
    base_url = os.environ.get("GOOD_BADMINTON_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("GOOD_BADMINTON_LLM_API_KEY", "").strip()
    model = os.environ.get("GOOD_BADMINTON_LLM_MODEL", "").strip()
    if not (base_url and api_key and model):
        return None
    timeout = int(os.environ.get("GOOD_BADMINTON_LLM_TIMEOUT_SECONDS", "75"))
    if timeout <= 0 or timeout > 180:
        raise ValueError("GOOD_BADMINTON_LLM_TIMEOUT_SECONDS must be between 1 and 180")
    return {"base_url": base_url.rstrip("/"), "api_key": api_key, "model": model, "timeout": timeout}


def _request_openai_compatible(config, prompt, *, image_paths=None):
    endpoint = config["base_url"]
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    content = [{"type": "text", "text": prompt}]
    for path in image_paths or []:
        if not path or not Path(path).is_file():
            continue
        encoded = b64encode(Path(path).read_bytes()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"},
        })
    payload = json.dumps({
        "model": config["model"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. Never invent sports evidence."},
            {"role": "user", "content": content},
        ],
    }, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        endpoint, data=payload, method="POST",
        headers={"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"},
    )
    with request.urlopen(http_request, timeout=config["timeout"]) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response does not contain choices[0].message.content") from exc


def _transcode_with_source_audio(temporary_video: Path, final_video: Path, audio_source: Path):
    """Use FFmpeg directly with a fast preset; export failure remains retryable."""
    executable = _find_ffmpeg()
    command = [
        executable, "-y", "-i", str(temporary_video), "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0?", "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(final_video),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=360)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PromotionVideoError(f"宣传视频编码失败：{exc}") from exc
    if result.returncode != 0 or not final_video.is_file() or final_video.stat().st_size == 0:
        detail = (result.stderr or "unknown ffmpeg error").strip()[-1000:]
        raise PromotionVideoError(f"宣传视频编码失败：{detail}")


def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"
