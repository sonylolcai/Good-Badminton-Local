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


def _track_summary(entry, *, source_frames, fps):
    points = sorted(
        entry["usable_points"],
        key=lambda item: (item["frame"] is None, item["frame"] if item["frame"] is not None else 0),
    )
    distance_m, accepted_segments, excluded_segments, moving_time_sec, speeds = _distance_from_measurements(points, fps)
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
        "movement_mean_speed_mps": round(distance_m / moving_time_sec, 3) if moving_time_sec else None,
        "movement_peak_speed_mps": round(max(speeds), 3) if speeds else None,
        "movement_time_sec": round(moving_time_sec, 3),
        "movement_segment_count": accepted_segments,
        "movement_segments_excluded": excluded_segments,
        "movement_policy": (
            "Distance joins only contiguous high-confidence detected measurements; "
            "long gaps and implausibly fast jumps are excluded."
        ),
    }


def _distance_from_measurements(points, fps, max_gap_seconds=0.5, max_speed_mps=10.0):
    distance_m = 0.0
    moving_time_sec = 0.0
    speeds = []
    accepted = 0
    excluded = 0
    rate = max(float(fps or 0), 1.0)
    for previous, current in zip(points, points[1:]):
        if previous["frame"] is None or current["frame"] is None:
            excluded += 1
            continue
        delta_frames = current["frame"] - previous["frame"]
        if delta_frames <= 0 or delta_frames / rate > max_gap_seconds:
            excluded += 1
            continue
        left = previous["court_xy_m"]
        right = current["court_xy_m"]
        segment = math.hypot(right[0] - left[0], right[1] - left[1])
        duration_sec = delta_frames / rate
        speed_mps = segment / duration_sec
        if speed_mps > max_speed_mps:
            excluded += 1
            continue
        distance_m += segment
        moving_time_sec += duration_sec
        speeds.append(speed_mps)
        accepted += 1
    return distance_m, accepted, excluded, moving_time_sec, speeds


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
