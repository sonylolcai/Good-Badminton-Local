"""Evidence-aware per-track court-position visualizations.

The legacy position plots read ``players.upper`` and ``players.lower``.  Those
compatibility slots are useful for old singles files, but they are not player
identities and therefore cannot represent doubles.  This module only consumes
the v2 ``spatial.tracks`` contract and creates a separate report for every
durable ``track_id``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

from good_badminton_contracts.detections_reader import collect_track_position_evidence


COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.50
DEFAULT_MIN_LOCATION_CONFIDENCE = 0.50
DEFAULT_MIN_IDENTITY_CONFIDENCE = 0.70
PORTRAIT_MIN_CONFIDENCE = 0.80
# Tracking is allowed to use a conservative same-side roster rebind so a
# doubles player is not lost after an occlusion.  That rebind is useful for
# coverage but can join two locations that were not observed as one continuous
# physical motion.  Do not let it manufacture a player-speed spike.
SPEED_METRIC_ASSOCIATION_SOURCES = frozenset({
    "bytetrack", "court_association", "roster_bootstrap", "legacy_direct_measurement",
})
MAX_REPORTED_PLAYER_SPEED_MPS = 6.5


def _configure_chinese_font():
    """Use the bundled font when available so exported labels are not boxes."""
    font_path = Path(__file__).resolve().parents[2] / "simhei.ttf"
    if not font_path.is_file():
        return False
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.family"] = [family, "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    return True


_CHINESE_FONT_AVAILABLE = _configure_chinese_font()


def analyze_spatial_track_positions(detections_path, output_dir=None, *, fps=30, language="zh"):
    """Write one heatmap and scatter plot per track plus an auditable summary.

    Returns ``None`` for an old detections file with no spatial contract so the
    caller can preserve the legacy upper/lower visualizer for compatibility.
    """
    evidence = collect_track_position_evidence(detections_path)
    # A minimal Linux GPU image may not carry a CJK font.  Prefer legible
    # English labels to silent tofu glyphs; the JSON evidence remains Chinese
    # and machine-readable either way.
    if language != "en" and not _CHINESE_FONT_AVAILABLE:
        language = "en"
    if not evidence["has_spatial_tracks"]:
        return None

    target = Path(output_dir or Path(detections_path).resolve().parent / "position_visualizations")
    target.mkdir(parents=True, exist_ok=True)
    image_paths = []
    track_summaries = {}
    for track_id in sorted(evidence["tracks"]):
        entry = evidence["tracks"][track_id]
        track_dir = target / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        summary = _track_summary(entry, source_frames=evidence["source_frames"], fps=fps)
        track_summaries[track_id] = summary
        heatmap_path = track_dir / "heatmap.png"
        scatter_path = track_dir / "scatter.png"
        _render_track_heatmap(entry["usable_points"], heatmap_path, track_id, summary, language)
        _render_track_scatter(entry["usable_points"], scatter_path, track_id, summary, language)
        image_paths.extend([str(heatmap_path), str(scatter_path)])

    summary_path = target / "position_evidence_summary.json"
    payload = {
        "schema_version": "1.0",
        "data_source": "detections.jsonl spatial.tracks",
        "match_mode": evidence["match_mode"],
        "source_frame_count": evidence["source_frames"],
        "thresholds": evidence["thresholds"],
        "policy": (
            "Only detected, high-confidence pose measurements are used for heatmaps and movement. "
            "Predicted, missing, low foot-point-confidence, and low identity-confidence rows remain in the summary only."
        ),
        "tracks": track_summaries,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "success": True,
        "image_paths": image_paths,
        "summary_path": str(summary_path),
        "summary": payload,
    }


def extract_high_confidence_player_portraits(video_path, detections_path, output_dir, *, minimum_confidence=PORTRAIT_MIN_CONFIDENCE):
    """Export one anonymous full-body crop per stable visual track.

    A portrait is produced only from a real detected pose whose person,
    location and association confidences all satisfy the UI's 0.80 threshold.
    This deliberately avoids using predicted tracks, identities, or faces.
    """
    candidates = _portrait_candidates(detections_path, minimum_confidence)
    if not candidates:
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}
    target = Path(output_dir) / "player_portraits"
    target.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for track_id, candidate in sorted(candidates.items()):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, candidate["frame"] - 1))
            readable, frame = capture.read()
            if not readable or frame is None:
                continue
            crop = _portrait_crop(frame, candidate["bbox_xyxy"])
            if crop is None:
                continue
            safe_track_id = "".join(char if char.isalnum() or char in "_-" else "_" for char in track_id)
            portrait_path = target / f"{safe_track_id}.jpg"
            if cv2.imwrite(str(portrait_path), crop):
                results[track_id] = str(portrait_path)
    finally:
        capture.release()
    return results


def _portrait_candidates(detections_path, minimum_confidence):
    selected = {}
    try:
        source = Path(detections_path).open(encoding="utf-8")
    except OSError:
        return selected
    with source:
        for raw in source:
            try:
                row = json.loads(raw)
                frame = int(row["frame"])
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
            for track in ((row.get("spatial") or {}).get("tracks") or []):
                if not isinstance(track, dict) or track.get("status") != "detected":
                    continue
                location = track.get("location_evidence") or {}
                association = track.get("association") or {}
                bbox = location.get("bbox_xyxy") or []
                try:
                    track_id = str(track["track_id"])
                    confidence = min(
                        float(track["confidence"]),
                        float(location["confidence"]),
                        float(association["identity_confidence"]),
                    )
                    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
                except (KeyError, TypeError, ValueError):
                    continue
                if confidence < float(minimum_confidence) or x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                rank = (confidence, area)
                if rank > selected.get(track_id, {}).get("rank", (-1.0, -1.0)):
                    selected[track_id] = {"frame": frame, "bbox_xyxy": [x1, y1, x2, y2], "rank": rank}
    return selected


def _portrait_crop(frame, bbox_xyxy):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    pad_x = max(4, int((x2 - x1) * 0.12))
    pad_y = max(4, int((y2 - y1) * 0.08))
    left, top = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
    right, bottom = min(width, int(x2) + pad_x), min(height, int(y2) + pad_y)
    if right - left < 16 or bottom - top < 24:
        return None
    return frame[top:bottom, left:right].copy()


def _track_summary(entry, *, source_frames, fps):
    points = sorted(
        entry["usable_points"],
        key=lambda item: (item["frame"] is None, item["frame"] if item["frame"] is not None else 0),
    )
    (
        distance_m,
        accepted_segments,
        excluded_segments,
        excluded_by_reason,
        moving_time_sec,
        peak_speed_mps,
    ) = _distance_from_measurements(points, fps)
    confidence_values = [point["detection_confidence"] for point in points]
    location_values = [point["location_confidence"] for point in points]
    identity_values = [point["identity_confidence"] for point in points]
    return {
        "track_rows": entry["track_rows"],
        "source_coverage": round(entry["track_rows"] / source_frames, 4) if source_frames else 0.0,
        "state_counts": dict(sorted(entry["state_counts"].items())),
        "usable_measurements": len(points),
        "excluded": dict(sorted(entry["excluded"].items())),
        "usable_measurement_coverage": round(len(points) / source_frames, 4) if source_frames else 0.0,
        "mean_detection_confidence": _mean_or_none(confidence_values),
        "mean_location_confidence": _mean_or_none(location_values),
        "mean_identity_confidence": _mean_or_none(identity_values),
        "movement_distance_m": round(distance_m, 3),
        "movement_segment_count": accepted_segments,
        "movement_segments_excluded": excluded_segments,
        "movement_segments_excluded_by_reason": excluded_by_reason,
        "movement_mean_speed_mps": (
            round(distance_m / moving_time_sec, 3) if moving_time_sec else None
        ),
        "movement_peak_speed_mps": (
            round(peak_speed_mps, 3) if accepted_segments else None
        ),
        "movement_time_sec": round(moving_time_sec, 3),
        "movement_policy": (
            "Speed joins only contiguous high-confidence detected measurements with a "
            "direct tracker association; roster rebind transitions, long gaps, and "
            "speeds above 6.5 m/s are excluded from reported movement."
        ),
    }


def _distance_from_measurements(
    points,
    fps,
    max_gap_seconds=0.5,
    max_speed_mps=MAX_REPORTED_PLAYER_SPEED_MPS,
):
    distance_m = 0.0
    accepted = 0
    excluded_by_reason = {}
    moving_time_sec = 0.0
    peak_speed_mps = 0.0

    def exclude(reason):
        excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1

    rate = max(float(fps or 0), 1.0)
    for previous, current in zip(points, points[1:]):
        if previous["frame"] is None or current["frame"] is None:
            exclude("invalid_frame")
            continue
        delta_frames = current["frame"] - previous["frame"]
        if delta_frames <= 0 or delta_frames / rate > max_gap_seconds:
            exclude("non_contiguous_measurement")
            continue
        if (
            previous.get("association_source") not in SPEED_METRIC_ASSOCIATION_SOURCES
            or current.get("association_source") not in SPEED_METRIC_ASSOCIATION_SOURCES
        ):
            exclude("non_direct_tracker_association")
            continue
        left = previous["court_xy_m"]
        right = current["court_xy_m"]
        segment = math.hypot(right[0] - left[0], right[1] - left[1])
        seconds = delta_frames / rate
        speed_mps = segment / seconds
        if speed_mps > max_speed_mps:
            exclude("reported_speed_guardrail")
            continue
        distance_m += segment
        accepted += 1
        moving_time_sec += seconds
        peak_speed_mps = max(peak_speed_mps, speed_mps)
    return (
        distance_m,
        accepted,
        sum(excluded_by_reason.values()),
        dict(sorted(excluded_by_reason.items())),
        moving_time_sec,
        peak_speed_mps,
    )


def _render_track_heatmap(points, path, track_id, summary, language):
    figure, axis = _court_figure(track_id, summary, language, _label(language, "heatmap"))
    if points:
        x_values = [point["court_xy_m"][0] for point in points]
        y_values = [point["court_xy_m"][1] for point in points]
        histogram, x_edges, y_edges = np.histogram2d(
            x_values,
            y_values,
            bins=(16, 32),
            range=((0, COURT_WIDTH_M), (0, COURT_LENGTH_M)),
        )
        image = axis.imshow(
            histogram.T,
            origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            cmap="YlOrRd",
            alpha=0.82,
            aspect="auto",
        )
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03, label=_label(language, "measurement_count"))
    else:
        _no_measurement_annotation(axis, language)
    _save_figure(figure, path)


def _render_track_scatter(points, path, track_id, summary, language):
    figure, axis = _court_figure(track_id, summary, language, _label(language, "scatter"))
    if points:
        x_values = [point["court_xy_m"][0] for point in points]
        y_values = [point["court_xy_m"][1] for point in points]
        color_values = [point["detection_confidence"] for point in points]
        scatter = axis.scatter(x_values, y_values, c=color_values, cmap="viridis", s=16, alpha=0.75, vmin=0.5, vmax=1.0)
        figure.colorbar(scatter, ax=axis, fraction=0.045, pad=0.03, label=_label(language, "detection_confidence"))
    else:
        _no_measurement_annotation(axis, language)
    _save_figure(figure, path)


def _court_figure(track_id, summary, language, report_kind):
    figure, axis = plt.subplots(figsize=(5.2, 9.2))
    axis.set_facecolor("#153d2a")
    figure.patch.set_facecolor("#101820")
    _draw_standard_doubles_court(axis)
    axis.set_xlim(-0.35, COURT_WIDTH_M + 0.35)
    axis.set_ylim(-0.35, COURT_LENGTH_M + 0.35)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"{track_id} · {report_kind}\n"
        f"{_label(language, 'usable')}: {summary['usable_measurements']} | "
        f"{_label(language, 'coverage')}: {summary['usable_measurement_coverage']:.1%}",
        color="white",
        fontsize=11,
    )
    return figure, axis


def _draw_standard_doubles_court(axis):
    line = {"color": "#f6f3e8", "linewidth": 1.2, "alpha": 0.95}
    axis.plot([0, COURT_WIDTH_M, COURT_WIDTH_M, 0, 0], [0, 0, COURT_LENGTH_M, COURT_LENGTH_M, 0], **line)
    # Singles sidelines, net, long/short service guides.  They provide a
    # familiar court reference without treating image orientation as identity.
    singles_inset = 0.46
    axis.plot([singles_inset, singles_inset], [0, COURT_LENGTH_M], **line)
    axis.plot([COURT_WIDTH_M - singles_inset, COURT_WIDTH_M - singles_inset], [0, COURT_LENGTH_M], **line)
    axis.plot([0, COURT_WIDTH_M], [COURT_LENGTH_M / 2, COURT_LENGTH_M / 2], color="#ffffff", linewidth=1.6)
    for y_value in (1.98, COURT_LENGTH_M - 1.98, 0.76, COURT_LENGTH_M - 0.76):
        axis.plot([0, COURT_WIDTH_M], [y_value, y_value], **line)
    axis.plot([COURT_WIDTH_M / 2, COURT_WIDTH_M / 2], [0, 1.98], **line)
    axis.plot([COURT_WIDTH_M / 2, COURT_WIDTH_M / 2], [COURT_LENGTH_M - 1.98, COURT_LENGTH_M], **line)


def _no_measurement_annotation(axis, language):
    axis.text(
        COURT_WIDTH_M / 2,
        COURT_LENGTH_M / 2,
        _label(language, "no_measurements"),
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#263238", "alpha": 0.86, "edgecolor": "none"},
    )


def _save_figure(figure, path):
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _mean_or_none(values):
    return round(sum(values) / len(values), 4) if values else None


def _label(language, key):
    chinese = language != "en"
    labels = {
        "heatmap": "跑位热力图" if chinese else "Position heatmap",
        "scatter": "高置信落点" if chinese else "High-confidence positions",
        "measurement_count": "有效检测次数" if chinese else "Usable measurements",
        "detection_confidence": "人体检测置信度" if chinese else "Pose detection confidence",
        "usable": "有效点" if chinese else "Usable points",
        "coverage": "有效覆盖" if chinese else "Usable coverage",
        "no_measurements": "没有满足置信度门槛的真实检测" if chinese else "No high-confidence detected measurements",
    }
    return labels[key]
