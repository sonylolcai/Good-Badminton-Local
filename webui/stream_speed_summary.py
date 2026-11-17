"""Presentation-side speed summary for anonymous GPU stream observations.

This is deliberately not a sport business-rule module.  It reads only
``person_observation`` events already emitted by the GPU and reports distance
and speed for each anonymous visual ``track_id``.  No player identity, score,
rally, shot, energy, ability or training conclusion is inferred here.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


MAX_JOIN_GAP_SECONDS = 0.5
MAX_PLAUSIBLE_SPEED_MPS = 10.0
SPEED_WINDOW_SECONDS = 0.5


def summarize_stream_player_speeds(
    output_dir: str | Path,
    *,
    client,
    terminal_status: Mapping[str, Any],
    create_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialise GPU events and write a reviewable per-track speed summary.

    Predicted or missing positions are never interpolated.  A speed segment is
    omitted if its source-time gap is too large or its implied speed is outside
    the conservative plausibility guard.  This makes the displayed metres per
    second traceable to the fixed-profile court calibration rather than a
    business-side estimate.
    """
    output = Path(output_dir)
    evidence_dir = output / "evidence"
    derived_dir = output / "derived"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    events = _read_all_events(client)
    raw_events_path = evidence_dir / "stream_events.jsonl"
    _write_jsonl(raw_events_path, events)

    configuration = dict(create_request.get("configuration") or {})
    samples, observation_counts = _detected_samples(events)
    ball_detection = _ball_detection_summary(events)
    players = [
        _player_speed_summary(track_id, values, observation_counts.get(track_id, 0))
        for track_id, values in sorted(samples.items())
    ]
    for track_id in sorted(set(observation_counts).difference(samples)):
        players.append(_player_speed_summary(track_id, [], observation_counts[track_id]))

    payload = {
        "schema_version": "webui-stream-speed-summary.v1",
        "kind": "anonymous_visual_speed_summary",
        "analysis_session_id": terminal_status.get("analysis_session_id"),
        "sport_id": configuration.get("sport_id") or "unknown",
        "session_mode": configuration.get("session_mode") or "unknown",
        "coordinate_system_id": configuration.get("coordinate_system_id"),
        "source": "GPU stream-session.v1 person_observation events",
        "measurement_policy": {
            "detected_position_only": True,
            "max_join_gap_seconds": MAX_JOIN_GAP_SECONDS,
            "max_plausible_speed_mps": MAX_PLAUSIBLE_SPEED_MPS,
            "speed_window_seconds": SPEED_WINDOW_SECONDS,
        },
        "players": players,
        "ball_detection": ball_detection,
        "limitations": [
            "track_id is an anonymous visual identifier, not a confirmed person identity.",
            "Only adjacent detected court positions are used; predicted, missing and gapped observations are excluded.",
            "Speed is court-plane metres per second and is only as accurate as the approved four-point calibration.",
            "Ball detection rate is not accuracy. Precision/recall requires manually labelled tennis-ball ground truth for this same video.",
        ],
    }
    speed_path = derived_dir / "player_speed_summary_v1.json"
    speed_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "succeeded",
        "summary_path": str(speed_path),
        "movement_metrics_path": str(speed_path),
        "raw_events_path": str(raw_events_path),
        "player_count": len(players),
        "policy": payload["measurement_policy"],
    }


def _read_all_events(client) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor = None
    seen_cursors = set()
    while True:
        page = client.read_events(cursor=cursor, limit=500)
        events.extend(item for item in (page.get("events") or []) if isinstance(item, dict))
        next_cursor = page.get("next_cursor")
        if not next_cursor:
            return events
        if next_cursor in seen_cursors:
            raise RuntimeError("stream event pagination cursor repeated")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _detected_samples(events: list[dict[str, Any]]):
    samples: dict[str, list[dict[str, float]]] = defaultdict(list)
    observation_counts: dict[str, int] = defaultdict(int)
    series_by_track: dict[str, int] = defaultdict(int)
    seen = set()
    for event in sorted(events, key=lambda item: float(item.get("source_time_sec") or 0.0)):
        if event.get("event_type") != "person_observation":
            continue
        track = (event.get("data") or {}).get("track")
        if not isinstance(track, Mapping):
            continue
        track_id = str(track.get("track_id") or "")
        if not track_id:
            continue
        observation_counts[track_id] += 1
        if str(track.get("status") or "") != "detected":
            # A predicted or missing point is evidence of an unknown physical
            # path. It breaks the next speed series instead of being skipped
            # and accidentally bridged by two detected neighbours.
            series_by_track[track_id] += 1
            continue
        xy = track.get("court_xy_m")
        try:
            x, y = float(xy[0]), float(xy[1])
            source_time = float(event.get("source_time_sec"))
        except (TypeError, ValueError, IndexError):
            continue
        bucket = track.get("measurement_bucket")
        key = (track_id, bucket if isinstance(bucket, int) else round(source_time, 6))
        if key in seen:
            continue
        seen.add(key)
        samples[track_id].append(
            {
                "time_sec": source_time,
                "x": x,
                "y": y,
                "series": series_by_track[track_id],
            }
        )
    for values in samples.values():
        values.sort(key=lambda item: item["time_sec"])
    return samples, observation_counts


def _ball_detection_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose review counts without turning unlabelled detections into accuracy."""
    ball_events = [
        event for event in events if event.get("event_type") == "ball_observation"
    ]
    detected = [
        event for event in ball_events if event.get("evidence_state") == "detected"
    ]
    metadata = [
        (event.get("data") or {}) for event in ball_events
        if isinstance(event.get("data"), Mapping)
    ]
    return {
        "requested": bool(ball_events),
        "event_count": len(ball_events),
        "detected_event_count": len(detected),
        "missing_event_count": sum(
            1 for event in ball_events if event.get("evidence_state") == "missing"
        ),
        "detection_rate": _round(len(detected) / len(ball_events)) if ball_events else None,
        "detector_modes": sorted({
            str(item.get("detector_mode") or "unknown") for item in metadata
        }),
        "experimental": any(bool(item.get("experimental")) for item in metadata),
        "accuracy": None,
        "accuracy_status": (
            "requires_same_video_ground_truth"
            if ball_events else "not_requested"
        ),
    }


def _player_speed_summary(track_id: str, samples: list[dict[str, float]], opportunity_count: int):
    segments = []
    excluded = 0
    for left, right in zip(samples, samples[1:]):
        if left["series"] != right["series"]:
            excluded += 1
            continue
        elapsed = right["time_sec"] - left["time_sec"]
        if elapsed <= 0 or elapsed > MAX_JOIN_GAP_SECONDS:
            excluded += 1
            continue
        distance = math.hypot(right["x"] - left["x"], right["y"] - left["y"])
        speed = distance / elapsed
        if speed > MAX_PLAUSIBLE_SPEED_MPS:
            excluded += 1
            continue
        segments.append({
            "start_time_sec": left["time_sec"],
            "end_time_sec": right["time_sec"],
            "seconds": elapsed,
            "distance_m": distance,
            "vector": (right["x"] - left["x"], right["y"] - left["y"]),
        })

    total_distance = sum(item["distance_m"] for item in segments)
    measured_seconds = sum(item["seconds"] for item in segments)
    speed_windows = _speed_windows(segments)
    coverage = len(samples) / max(1, int(opportunity_count))
    has_speed = bool(speed_windows)
    return {
        "track_id": track_id,
        "movement": {
            "distance_m": _round(total_distance),
            "mean_speed_mps": _round(total_distance / measured_seconds) if measured_seconds else None,
            "peak_speed_mps": _round(max(item["speed_mps"] for item in speed_windows)) if has_speed else None,
            "moving_time_sec": _round(measured_seconds),
            "high_intensity_movement_time_sec": None,
            "acceleration_event_count": None,
            "deceleration_event_count": None,
            "direction_change_count": None,
            "peak_direction_changes_30s": None,
        },
        "measurement_coverage": {
            "usable_measurement_ratio": _round(coverage),
            "detected_position_count": len(samples),
            "observation_count": int(opportunity_count),
            "accepted_segment_count": len(segments),
            "excluded_segment_count": excluded,
            "speed_window_count": len(speed_windows),
        },
        "quality": {
            "status": "measured" if has_speed else "insufficient_speed_evidence",
            "analytics_eligible": bool(has_speed),
        },
    }


def _speed_windows(segments: list[dict[str, Any]]) -> list[dict[str, float]]:
    windows = []
    window = None
    previous_end = None
    for segment in segments:
        if previous_end is not None and abs(segment["start_time_sec"] - previous_end) > 1e-6:
            window = None
        previous_end = segment["end_time_sec"]
        if window is None:
            window = {
                "start_time_sec": segment["start_time_sec"],
                "end_time_sec": segment["end_time_sec"],
                "dx": segment["vector"][0],
                "dy": segment["vector"][1],
            }
        else:
            window["end_time_sec"] = segment["end_time_sec"]
            window["dx"] += segment["vector"][0]
            window["dy"] += segment["vector"][1]
        elapsed = window["end_time_sec"] - window["start_time_sec"]
        if elapsed + 1e-9 < SPEED_WINDOW_SECONDS:
            continue
        windows.append({
            "speed_mps": math.hypot(window["dx"], window["dy"]) / elapsed,
        })
        window = None
    return windows


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in values),
        encoding="utf-8",
    )


def _round(value: float | None):
    return round(float(value), 4) if value is not None else None
