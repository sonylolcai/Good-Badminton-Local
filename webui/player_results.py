"""Anonymous player-result presentation helpers for the Gradio WebUI.

The GPU keeps track IDs anonymous.  This module only turns already-authorised
visual evidence into local result cards: one best available crop and one row of
movement evidence per track.  It deliberately does not infer names, teams,
scores, or medical / ability conclusions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PLAYER_RESULT_HEADERS = [
    "视觉 Track ID",
    "追踪状态",
    "截图证据",
    "截图时刻(s)",
    "检测覆盖率(%)",
    "追踪置信度",
    "距离(m)",
    "平均速度(m/s)",
    "峰值速度(m/s)",
    "有效移动(s)",
    "高强度移动(s)",
    "加速/减速次数",
    "变向次数",
    "数据质量",
]

_MIN_PHOTO_DETECTION_CONFIDENCE = 0.80
_MIN_PHOTO_LOCATION_CONFIDENCE = 0.80
_MIN_PHOTO_IDENTITY_CONFIDENCE = 0.80

TENNIS_PLAYER_RESULT_HEADERS = [
    "匿名视觉 Track ID",
    "追踪状态",
    "截图证据",
    "截图时刻(s)",
    "检测覆盖率(%)",
    "距离(m)",
    "平均速度(m/s)",
    "峰值速度(m/s)",
    "有效移动(s)",
    "数据质量",
]


def build_player_result_display(
    metrics: Mapping[str, Any] | None,
    *,
    track_candidates: Iterable[Mapping[str, Any]] | None = None,
    photo_records: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[list[tuple[str, str]], list[list[Any]], dict[str, Any]]:
    """Return gallery items, table rows and a complete evidence object.

    ``track_candidates`` is the stream-session status representation; complete
    video jobs may omit it.  ``photo_records`` contains only local paths that
    the WebUI can safely render without exposing a GPU API key to the browser.
    """

    metrics = dict(metrics or {})
    tennis_visual_only = metrics.get("sport_id") == "tennis"
    candidates = {
        str(item.get("track_id")): dict(item)
        for item in (track_candidates or [])
        if isinstance(item, Mapping) and item.get("track_id")
    }
    photos = {
        str(item.get("track_id")): dict(item)
        for item in (photo_records or [])
        if isinstance(item, Mapping) and item.get("track_id")
    }
    players = [
        dict(item) for item in metrics.get("players") or []
        if isinstance(item, Mapping) and item.get("track_id")
    ]
    seen = {str(player["track_id"]) for player in players}
    # Stream candidates with too little usable evidence still deserve an
    # explicit row: hiding them made the operator think the GPU had lost data.
    players.extend({"track_id": track_id} for track_id in sorted(candidates) if track_id not in seen)

    gallery: list[tuple[str, str]] = []
    rows: list[list[Any]] = []
    details: list[dict[str, Any]] = []
    for player in sorted(players, key=lambda item: str(item["track_id"])):
        track_id = str(player["track_id"])
        candidate = candidates.get(track_id, {})
        photo = photos.get(track_id, {})
        movement = dict(player.get("movement") or {})
        coverage = dict(player.get("measurement_coverage") or {})
        quality = dict(player.get("quality") or {})
        candidate_photo = dict(candidate.get("candidate_photo") or {})
        photo_path = str(photo.get("path") or "")
        if photo_path and Path(photo_path).is_file():
            view = str(photo.get("view_label") or candidate_photo.get("view_label") or "未评估")
            capture_time = _number(photo.get("source_time_sec", candidate_photo.get("source_time_sec")))
            caption = f"{track_id} · {view}" + (f" · {capture_time:.2f}s" if capture_time is not None else "")
            gallery.append((photo_path, caption))
            photo_state = "已返回"
        else:
            capture_time = _number(candidate_photo.get("source_time_sec"))
            photo_state = str(photo.get("status") or "未提供")
        usable_ratio = _number(coverage.get("usable_measurement_ratio"))
        detected_ratio = _number(candidate.get("detected_coverage"))
        coverage_percent = usable_ratio if usable_ratio is not None else detected_ratio
        unconfirmed_roster = candidate.get("state") == "unconfirmed_roster"
        state = "名单待确认" if unconfirmed_roster else candidate.get("state") or "已完成"
        quality_status = (
            "名单未确认；不生成速度、距离等正式指标"
            if unconfirmed_roster
            else quality.get("status") or "证据待汇总"
        )
        if tennis_visual_only:
            rows.append([
                track_id,
                state,
                photo_state,
                _round(capture_time),
                _percent(coverage_percent),
                _round(movement.get("distance_m")),
                _round(movement.get("mean_speed_mps")),
                _round(movement.get("peak_speed_mps")),
                _round(movement.get("moving_time_sec")),
                quality_status,
            ])
        else:
            rows.append([
                track_id,
                state,
                photo_state,
                _round(capture_time),
                _percent(coverage_percent),
                _round(candidate.get("confidence")),
                _round(movement.get("distance_m")),
                _round(movement.get("mean_speed_mps")),
                _round(movement.get("peak_speed_mps")),
                _round(movement.get("moving_time_sec")),
                _round(movement.get("high_intensity_movement_time_sec")),
                f"{movement.get('acceleration_event_count') or 0} / {movement.get('deceleration_event_count') or 0}",
                movement.get("direction_change_count") or 0,
                quality_status,
            ])
        details.append({
            "track_id": track_id,
            "candidate": candidate,
            "photo": photo,
            "movement": movement,
            "measurement_coverage": coverage,
            "quality": quality,
        })
    return gallery, rows, {
        "schema_version": "webui-player-result.v1",
        "metrics": metrics,
        "players": details,
        "limitations": list(metrics.get("limitations") or []),
    }


def extract_full_video_candidate_photos(
    video_path: str | Path,
    detections_path: str | Path,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Make one local evidence crop for each complete-video visual track.

    Stream sessions receive their crops from the GPU's candidate-photo API.
    Full-file jobs already have the source video plus detector bounding boxes,
    so generating a local crop avoids a second GPU inference and keeps both
    upload modes equally reviewable.
    """

    source = Path(video_path)
    detections = Path(detections_path)
    target = Path(output_dir) / "candidate_photos"
    if not source.is_file() or not detections.is_file():
        return []
    best = _best_detected_observations(detections)
    if not best:
        return []
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return []
    output: list[dict[str, Any]] = []
    try:
        target.mkdir(parents=True, exist_ok=True)
        for track_id, observation in sorted(best.items()):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(observation["frame"]))
            ok, frame = capture.read()
            crop = _crop(frame, observation["bbox"]) if ok else None
            if crop is None:
                continue
            path = target / f"{track_id}.jpg"
            if not cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                continue
            output.append({
                "track_id": track_id,
                "path": str(path),
                "status": "generated_from_full_video_detection",
                "source_time_sec": observation["source_time_sec"],
                "capture_quality": observation["score"],
                "view_label": "未评估",
                "selection_policy": "first_high_confidence_then_higher_score_v1",
            })
    finally:
        capture.release()
    return output


def _best_detected_observations(path: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    try:
        source = path.open("r", encoding="utf-8")
    except OSError:
        return best
    with source:
        for raw in source:
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            frame = record.get("frame")
            source_time = _number(record.get("time_sec"))
            if not isinstance(frame, int) or source_time is None:
                continue
            tracks = ((record.get("spatial") or {}).get("tracks") or [])
            for track in tracks:
                if not isinstance(track, Mapping) or str(track.get("status")) != "detected":
                    continue
                track_id = str(track.get("track_id") or "")
                evidence = dict(track.get("location_evidence") or {})
                bbox = evidence.get("bbox_xyxy")
                if not track_id or not _valid_bbox(bbox):
                    continue
                detection_confidence = _number(track.get("confidence")) or 0.0
                location_confidence = _number(evidence.get("confidence")) or 0.0
                association = dict(track.get("association") or {})
                identity_confidence = _number(association.get("identity_confidence")) or 0.0
                if (
                    detection_confidence < _MIN_PHOTO_DETECTION_CONFIDENCE
                    or location_confidence < _MIN_PHOTO_LOCATION_CONFIDENCE
                    or identity_confidence < _MIN_PHOTO_IDENTITY_CONFIDENCE
                ):
                    continue
                score = (
                    0.45 * detection_confidence
                    + 0.35 * location_confidence
                    + 0.20 * identity_confidence
                )
                if score <= float(best.get(track_id, {}).get("score", -1.0)):
                    continue
                best[track_id] = {
                    "frame": frame,
                    "source_time_sec": source_time,
                    "bbox": [float(value) for value in bbox],
                    "score": round(score, 4),
                }
    return best


def _crop(frame: Any, bbox: list[float]):
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = max(4, int(round((x2 - x1) * 0.12)))
    pad_y = max(4, int(round((y2 - y1) * 0.08)))
    left, top = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
    right, bottom = min(width, int(x2) + pad_x), min(height, int(y2) + pad_y)
    if right - left < 24 or bottom - top < 48:
        return None
    return frame[top:bottom, left:right]


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _round(value: Any) -> float | None:
    number = _number(value)
    return round(number, 3) if number is not None else None


def _percent(value: Any) -> float | None:
    number = _number(value)
    return round(number * 100.0, 1) if number is not None else None
