"""Auditable per-track movement metrics and optional calorie estimates.

This module is intentionally a post-processing step.  It consumes only fresh,
high-confidence ``spatial.tracks`` measurements and never reconstructs gaps as
movement.  A height/weight profile is user supplied after analysis; it is not
guessed from video and it can be omitted without losing the visual metrics.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from business_gateway.metrics.detections_reader import (
    collect_track_position_evidence,
)


SCHEMA_VERSION = "1.0"
MAX_JOIN_GAP_SECONDS = 0.5
# This is a reporting guardrail, not a tracker gate.  ByteTrack and roster
# recovery keep their broader motion tolerance so identities survive an
# occlusion; movement metrics only accept direct observations below this bound.
MAX_REPORTED_SPEED_MPS = 6.5
SPEED_METRIC_ASSOCIATION_SOURCES = frozenset({
    "bytetrack", "court_association", "roster_bootstrap", "legacy_direct_measurement",
})
MOVING_SPEED_MPS = 0.25
HIGH_INTENSITY_SPEED_MPS = 2.0
ACCELERATION_EVENT_MPS2 = 1.5
ACCELERATION_SPRINT_MIN_DISTANCE_M = 2.0
ACCELERATION_SPRINT_WINDOW_SECONDS = 0.5
SPEED_ACCELERATION_STATISTICS_INTERVAL_SECONDS = 0.5
TURN_ANGLE_DEGREES = 90.0
DIRECTION_CHANGE_MIN_LEG_DISTANCE_M = 1.0
DIRECTION_CHANGE_MAX_LEG_SECONDS = 0.5
DIRECTION_CHANGE_PEAK_WINDOW_SECONDS = 30.0
# 2024 Adult Compendium reference values for badminton: social/general 5.5,
# competitive 7.0, and competitive match play 9.0 MET.  Video-only movement
# cannot determine physiology, so these are reference anchors for a simple
# product estimate rather than a medical calorie measurement.
BADMINTON_SOCIAL_MET = 5.5
BADMINTON_COMPETITIVE_MET = 7.0
BADMINTON_MATCH_PLAY_MET = 9.0
ABILITY_SCORE_VERSION = "movement-ability-score.v1"
COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4


def generate_movement_metrics(
    output_dir,
    detections_path,
    spatial_summary_path,
    metadata_path=None,
    body_profiles_path=None,
):
    """Persist per-track movement evidence and a simple energy estimate.

    The output is generated even without profiles so the two visual tracks can
    be reviewed first.  Calories appear only once the user provides a valid
    body mass and explicit consent through ``write_body_profiles``.
    """
    output_dir = Path(output_dir)
    derived_dir = output_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_json(metadata_path)
    spatial_summary = _read_json(spatial_summary_path)
    profiles = _read_json(body_profiles_path or derived_dir / "player_body_profiles_v1.json")
    profile_by_track = {
        str(item.get("track_id")): item
        for item in profiles.get("profiles") or []
        if isinstance(item, dict) and item.get("track_id")
    }
    fps = float(((metadata.get("video") or {}).get("fps")) or 0.0)
    evidence = collect_track_position_evidence(detections_path)
    style_by_track = {
        str(item.get("track_id")): item
        for item in spatial_summary.get("player_style_inputs") or []
        if isinstance(item, dict) and item.get("track_id")
    }
    players = []
    for track_id, entry in sorted(evidence.get("tracks", {}).items()):
        player = _build_player_metrics(
            track_id,
            entry,
            fps=fps,
            source_frames=int(evidence.get("source_frames") or 0),
            style=style_by_track.get(track_id) or {},
            profile=profile_by_track.get(track_id),
        )
        players.append(player)

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "kind": "visual_track_movement_metrics",
        "generated_at": _utc_timestamp(),
        "data_source": "detections.jsonl spatial.tracks (fresh high-confidence detections only)",
        "measurement_policy": {
            "max_join_gap_seconds": MAX_JOIN_GAP_SECONDS,
            "max_reported_speed_mps": MAX_REPORTED_SPEED_MPS,
            "speed_metric_association_sources": sorted(SPEED_METRIC_ASSOCIATION_SOURCES),
            "moving_speed_mps": MOVING_SPEED_MPS,
            "high_intensity_speed_mps": HIGH_INTENSITY_SPEED_MPS,
            "acceleration_event_mps2": ACCELERATION_EVENT_MPS2,
            "minimum_acceleration_sprint_distance_m": ACCELERATION_SPRINT_MIN_DISTANCE_M,
            "acceleration_sprint_window_seconds": ACCELERATION_SPRINT_WINDOW_SECONDS,
            "speed_and_acceleration_statistics_interval_seconds": (
                SPEED_ACCELERATION_STATISTICS_INTERVAL_SECONDS
            ),
            "turn_angle_degrees": TURN_ANGLE_DEGREES,
            "direction_change_max_leg_seconds": DIRECTION_CHANGE_MAX_LEG_SECONDS,
            "missing_and_predicted_policy": "excluded from all movement and calorie calculations",
        },
        "match": {
            "mode": evidence.get("match_mode"),
            "video_duration_sec": ((metadata.get("video") or {}).get("duration_sec")),
            "video_fps": fps or None,
        },
        "players": players,
        "limitations": [
            "track_id is a visual identity until a user claims it after the match.",
            "Distance, speed and change counts require contiguous detected court positions; gaps are excluded rather than filled.",
            "Energy is an optional single heuristic for measured movement time, not a medical measurement or a total-session calorie claim.",
        ],
    }
    metrics_path = derived_dir / "player_movement_metrics_v1.json"
    _write_json(metrics_path, metrics)
    return {
        "status": "succeeded",
        "metrics_path": str(metrics_path),
        "players": players,
        "track_count": len(players),
    }


def write_body_profiles(output_dir, rows, *, consent=False):
    """Store user-entered body dimensions without inferring them from video."""
    if not consent:
        raise ValueError("Explicit consent is required before storing height or weight.")
    profiles = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        track_id = str(row.get("track_id") or "").strip()
        if not track_id:
            continue
        weight_kg = _positive_float_or_none(row.get("weight_kg"))
        height_cm = _positive_float_or_none(row.get("height_cm"))
        if weight_kg is not None and not 20.0 <= weight_kg <= 300.0:
            raise ValueError(f"weight_kg for {track_id} must be between 20 and 300")
        if height_cm is not None and not 80.0 <= height_cm <= 260.0:
            raise ValueError(f"height_cm for {track_id} must be between 80 and 260")
        profiles.append(
            {
                "track_id": track_id,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "source": "user_post_match_entry",
            }
        )
    destination = Path(output_dir) / "derived" / "player_body_profiles_v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "user_provided_body_profiles",
            "consent": True,
            "purpose": "post-match movement energy estimate only",
            "updated_at": _utc_timestamp(),
            "profiles": profiles,
        },
    )
    return str(destination)


def _build_player_metrics(track_id, entry, *, fps, source_frames, style, profile):
    points = sorted(
        entry.get("usable_points") or [],
        key=lambda item: (item.get("frame") is None, item.get("frame") or 0),
    )
    segments, excluded_segments = _valid_segments(points, fps)
    statistic_segments = _speed_statistic_segments(segments)
    speeds = [segment["speed_mps"] for segment in statistic_segments]
    moving = [segment for segment in statistic_segments if segment["speed_mps"] >= MOVING_SPEED_MPS]
    high = [segment for segment in statistic_segments if segment["speed_mps"] >= HIGH_INTENSITY_SPEED_MPS]
    active_seconds = sum(segment["seconds"] for segment in moving)
    high_intensity_seconds = sum(segment["seconds"] for segment in high)
    mean_speed_mps = _mean(speeds) or 0.0
    acceleration_event_count, deceleration_event_count = _acceleration_event_counts(statistic_segments)
    direction_change_events = _direction_change_events(statistic_segments)
    direction_change_count = len(direction_change_events)
    peak_direction_changes_30s = _peak_direction_changes_in_window(
        direction_change_events,
        window_seconds=DIRECTION_CHANGE_PEAK_WINDOW_SECONDS,
    )
    track_rows = int(entry.get("track_rows") or 0)
    coverage_ratio = len(points) / source_frames if source_frames else 0.0
    energy = _energy_estimate(
        profile,
        active_seconds=active_seconds,
        high_intensity_seconds=high_intensity_seconds,
        coverage_ratio=coverage_ratio,
        mean_speed_mps=mean_speed_mps,
        direction_change_count=direction_change_count,
    )
    duration_sec = float(source_frames) / float(fps) if fps else 0.0
    ability_scores = _ability_scores(
        points=points,
        statistic_segments=statistic_segments,
        fps=fps,
        duration_sec=duration_sec,
        moving_time_sec=active_seconds,
        high_intensity_time_sec=high_intensity_seconds,
        mean_speed_mps=mean_speed_mps,
        peak_speed_mps=max(speeds) if speeds else 0.0,
        direction_change_count=direction_change_count,
        zone_measurements=dict(style.get("zone_frames") or {}),
    )
    return {
        "track_id": track_id,
        "measurement_coverage": {
            "track_rows": track_rows,
            "usable_measurements": len(points),
            "usable_measurement_ratio": round(coverage_ratio, 4),
            "excluded_rows": dict(entry.get("excluded") or {}),
            "excluded_segments": excluded_segments,
        },
        "movement": {
            "distance_m": round(sum(segment["distance_m"] for segment in segments), 3),
            "accepted_segment_count": len(segments),
            "speed_statistic_segment_count": len(statistic_segments),
            "speed_and_acceleration_statistics_interval_seconds": (
                SPEED_ACCELERATION_STATISTICS_INTERVAL_SECONDS
            ),
            "mean_speed_mps": _round_or_none(mean_speed_mps),
            "peak_speed_mps": _round_or_none(max(speeds) if speeds else None),
            "moving_time_sec": round(sum(segment["seconds"] for segment in moving), 3),
            "high_intensity_movement_time_sec": round(sum(segment["seconds"] for segment in high), 3),
            "acceleration_event_count": acceleration_event_count,
            "deceleration_event_count": deceleration_event_count,
            "minimum_acceleration_sprint_distance_m": ACCELERATION_SPRINT_MIN_DISTANCE_M,
            "acceleration_sprint_window_seconds": ACCELERATION_SPRINT_WINDOW_SECONDS,
            "direction_change_count": direction_change_count,
            "peak_direction_changes_30s": peak_direction_changes_30s,
            "agility_movement": {
                "whole_match_confirmed_direction_changes": direction_change_count,
                "peak_window_seconds": DIRECTION_CHANGE_PEAK_WINDOW_SECONDS,
                "peak_window_confirmed_direction_changes": peak_direction_changes_30s,
                "turn_angle_degrees": TURN_ANGLE_DEGREES,
                "minimum_distance_each_leg_m": DIRECTION_CHANGE_MIN_LEG_DISTANCE_M,
                "maximum_seconds_each_leg": DIRECTION_CHANGE_MAX_LEG_SECONDS,
                "speed_and_acceleration_statistics_interval_seconds": (
                    SPEED_ACCELERATION_STATISTICS_INTERVAL_SECONDS
                ),
                "scope": "movement agility evidence only; does not measure stimulus-to-movement reaction time",
            },
            "zone_measurements": dict(style.get("zone_frames") or {}),
        },
        "ability_scores": ability_scores,
        "match_load": _match_load(
            duration_sec=duration_sec,
            moving_time_sec=active_seconds,
            high_intensity_time_sec=high_intensity_seconds,
            mean_speed_mps=mean_speed_mps,
            direction_change_count=direction_change_count,
        ),
        "energy_estimate": energy,
        "quality": {
            "status": "reviewable" if points else "insufficient_measurements",
            "policy": "Only fresh high-confidence detected positions are used; no predicted position or missing gap is converted into movement.",
        },
    }


def _ability_scores(
    *,
    points,
    statistic_segments,
    fps,
    duration_sec,
    moving_time_sec,
    high_intensity_time_sec,
    mean_speed_mps,
    peak_speed_mps,
    direction_change_count,
    zone_measurements,
):
    """Calculate reviewable MVP ability scores from measured movement only."""

    duration = max(0.0, float(duration_sec))
    active = max(0.0, float(moving_time_sec))
    high = min(active, max(0.0, float(high_intensity_time_sec)))
    effective_ratio = active / duration if duration else 0.0
    coverage_count = len([zone for zone, count in (zone_measurements or {}).items() if count])
    return {
        "schema_version": ABILITY_SCORE_VERSION,
        "endurance": _endurance_score(statistic_segments, duration),
        "agility": _agility_score(
            peak_speed_mps=peak_speed_mps,
            direction_change_count=direction_change_count,
            moving_time_sec=active,
        ),
        "coverage": {
            "status": "measured",
            "score": _score(coverage_count / 9.0),
            "covered_zone_count": coverage_count,
            "total_zone_count": 9,
            "formula": "covered_zones / 9",
        },
        "returning": _returning_score(points, fps=fps),
        "effective_running": {
            "status": "measured" if duration > 0 else "insufficient_measurements",
            "score": _score(effective_ratio) if duration > 0 else None,
            "moving_time_sec": round(active, 3),
            "match_duration_sec": round(duration, 3),
            "formula": "measured_moving_time_sec / parsed_match_duration_sec",
        },
        "evidence": {
            "mean_speed_mps": _round_or_none(mean_speed_mps),
            "high_intensity_movement_time_sec": round(high, 3),
            "direction_change_count": int(direction_change_count),
        },
    }


def _endurance_score(statistic_segments, duration_sec):
    duration = max(0.0, float(duration_sec))
    if duration <= 0:
        return {"status": "insufficient_measurements", "score": None}
    first_cutoff = duration / 3.0
    final_cutoff = duration * 2.0 / 3.0
    early = [segment["speed_mps"] for segment in statistic_segments if float(segment["end_time_sec"]) <= first_cutoff]
    late = [segment["speed_mps"] for segment in statistic_segments if float(segment["start_time_sec"]) >= final_cutoff]
    if len(early) < 3 or len(late) < 3:
        return {
            "status": "insufficient_measurements",
            "score": None,
            "early_sample_count": len(early),
            "late_sample_count": len(late),
            "formula": "first_third_vs_final_third_mean_speed_decline",
        }
    early_mean = _mean(early) or 0.0
    late_mean = _mean(late) or 0.0
    decline_ratio = max(0.0, (early_mean - late_mean) / early_mean) if early_mean else 0.0
    return {
        "status": "measured",
        "score": max(0, min(100, int(round(100 - decline_ratio * 200)))),
        "early_mean_speed_mps": _round_or_none(early_mean),
        "late_mean_speed_mps": _round_or_none(late_mean),
        "speed_decline_ratio": round(decline_ratio, 4),
        "formula": "score = clamp(100 - 200 * max(0, early_speed - late_speed) / early_speed)",
    }


def _agility_score(*, peak_speed_mps, direction_change_count, moving_time_sec):
    active_minutes = max(float(moving_time_sec) / 60.0, 0.25)
    turns_per_minute = max(0.0, float(direction_change_count)) / active_minutes
    speed_component = min(1.0, max(0.0, float(peak_speed_mps) / 6.0))
    turn_component = min(1.0, turns_per_minute / 12.0)
    return {
        "status": "measured" if moving_time_sec > 0 else "insufficient_measurements",
        "score": _score(0.6 * speed_component + 0.4 * turn_component) if moving_time_sec > 0 else None,
        "peak_speed_mps": _round_or_none(peak_speed_mps),
        "direction_changes_per_min": round(turns_per_minute, 3),
        "formula": "0.60 * clamp(peak_speed_mps / 6) + 0.40 * clamp(direction_changes_per_min / 12)",
    }


def _returning_score(points, *, fps):
    ordered = [point for point in points if point.get("frame") is not None and point.get("court_xy_m")]
    if len(ordered) < 3:
        return {"status": "insufficient_measurements", "score": None, "formula": "return_to_midcourt_waiting_zone"}
    ys = sorted(float(point["court_xy_m"][1]) for point in ordered)
    median_y = ys[len(ys) // 2]
    is_lower_half = median_y >= COURT_LENGTH_M / 2.0

    def in_waiting_zone(point):
        x, y = (float(value) for value in point["court_xy_m"])
        in_center_width = abs(x - COURT_WIDTH_M / 2.0) <= 1.2
        if is_lower_half:
            return in_center_width and COURT_LENGTH_M * 0.57 <= y <= COURT_LENGTH_M * 0.86
        return in_center_width and COURT_LENGTH_M * 0.14 <= y <= COURT_LENGTH_M * 0.43

    episodes = []
    was_home = in_waiting_zone(ordered[0])
    departure_time = None
    for point in ordered[1:]:
        time_sec = float(point["frame"])
        home = in_waiting_zone(point)
        if was_home and not home:
            departure_time = time_sec
        elif departure_time is not None and home:
            # ``frame`` timestamps are converted below by their observed
            # sampling period; the caller supplies no synthetic positions.
            episodes.append((departure_time, time_sec))
            departure_time = None
        was_home = home
    # ``frame`` is the materialized analysis-sample frame index. Its rate is
    # the configured analysis frequency (10 Hz for the current MVP), not the
    # original camera frame rate.
    seconds_per_frame = 1.0 / max(float(fps or 0.0), 1.0)
    return_times = [
        (end - start) * seconds_per_frame
        for start, end in episodes
        if 0 < (end - start) * seconds_per_frame <= 8.0
    ]
    if len(return_times) < 2:
        return {
            "status": "insufficient_measurements",
            "score": None,
            "return_event_count": len(return_times),
            "formula": "time_from_leaving_own_midcourt_waiting_zone_to_returning_to_it",
        }
    mean_return = _mean(return_times) or 0.0
    return {
        "status": "measured",
        "score": _score((5.0 - mean_return) / 4.0),
        "mean_return_time_sec": round(mean_return, 3),
        "return_event_count": len(return_times),
        "waiting_zone": "own_half_center_band",
        "formula": "score = clamp((5 - mean_return_time_sec) / 4)",
    }


def _match_load(*, duration_sec, moving_time_sec, high_intensity_time_sec, mean_speed_mps, direction_change_count):
    duration = max(0.0, float(duration_sec))
    active = min(duration, max(0.0, float(moving_time_sec)))
    high = min(active, max(0.0, float(high_intensity_time_sec)))
    active_ratio = active / duration if duration else 0.0
    high_ratio = high / duration if duration else 0.0
    speed_component = min(1.0, max(0.0, float(mean_speed_mps) / 3.0))
    active_minutes = max(active / 60.0, 0.25)
    turn_component = min(1.0, max(0.0, float(direction_change_count)) / active_minutes / 12.0)
    duration_factor = min(1.0, duration / 1800.0)
    intensity = 0.35 * active_ratio + 0.25 * high_ratio + 0.25 * speed_component + 0.15 * turn_component
    return {
        "status": "measured" if duration else "insufficient_measurements",
        "score": _score(duration_factor * intensity) if duration else None,
        "duration_factor": round(duration_factor, 4),
        "components": {
            "effective_running_ratio": round(active_ratio, 4),
            "high_intensity_ratio": round(high_ratio, 4),
            "mean_speed_component": round(speed_component, 4),
            "direction_change_component": round(turn_component, 4),
        },
        "formula": "duration_factor(min(duration/1800,1)) * (0.35*effective_running + 0.25*high_intensity + 0.25*mean_speed + 0.15*direction_change)",
    }


def _score(value):
    return max(0, min(100, int(round(max(0.0, min(1.0, float(value))) * 100))))


def _valid_segments(points, fps):
    rate = max(float(fps or 0.0), 1.0)
    segments = []
    excluded_by_reason = {}

    def exclude(reason):
        excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
    for left, right in zip(points, points[1:]):
        frame_left = left.get("frame")
        frame_right = right.get("frame")
        if frame_left is None or frame_right is None:
            exclude("invalid_frame")
            continue
        delta_frames = int(frame_right) - int(frame_left)
        seconds = delta_frames / rate
        if delta_frames <= 0 or seconds > MAX_JOIN_GAP_SECONDS:
            exclude("non_contiguous_measurement")
            continue
        if (
            left.get("association_source") not in SPEED_METRIC_ASSOCIATION_SOURCES
            or right.get("association_source") not in SPEED_METRIC_ASSOCIATION_SOURCES
        ):
            exclude("non_direct_tracker_association")
            continue
        point_left = left["court_xy_m"]
        point_right = right["court_xy_m"]
        dx = float(point_right[0]) - float(point_left[0])
        dy = float(point_right[1]) - float(point_left[1])
        distance = math.hypot(dx, dy)
        speed = distance / seconds
        if speed > MAX_REPORTED_SPEED_MPS:
            exclude("reported_speed_guardrail")
            continue
        segments.append(
            {
                "start_time_sec": int(frame_left) / rate,
                "seconds": seconds,
                "distance_m": distance,
                "speed_mps": speed,
                "vector": (dx, dy),
                "end_time_sec": int(frame_right) / rate,
            }
        )
    return segments, dict(sorted(excluded_by_reason.items()))


def _speed_statistic_segments(segments, *, interval_seconds=SPEED_ACCELERATION_STATISTICS_INTERVAL_SECONDS):
    """Return non-overlapping speed windows from fresh measured trajectory segments.

    User-facing speed, acceleration and turning metrics deliberately use a
    roughly half-second window instead of individual 10/15Hz pose steps.
    This suppresses short localisation jitter without fabricating positions:
    an incomplete window or a detector gap is discarded, and no interpolation
    is introduced to force a window to end at exactly 0.5 seconds.
    """
    interval_seconds = max(0.001, float(interval_seconds))
    samples = []
    window = None
    previous_end_time = None
    series_id = 0
    for segment in segments:
        start_time = segment.get("start_time_sec")
        end_time = segment.get("end_time_sec")
        try:
            start_time = float(start_time)
            end_time = float(end_time)
        except (TypeError, ValueError):
            continue
        if end_time <= start_time:
            continue
        if previous_end_time is not None and abs(start_time - previous_end_time) > 1e-6:
            # Do not connect a statistic window across a rejected or missing
            # observation.  Its physical movement is unknown.
            window = None
            series_id += 1
        previous_end_time = end_time
        vector = tuple(float(value) for value in (segment.get("vector") or ()))
        if len(vector) < 2:
            continue
        if window is None:
            window = {
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "vector": [vector[0], vector[1]],
                "series_id": series_id,
            }
        else:
            window["end_time_sec"] = end_time
            window["vector"][0] += vector[0]
            window["vector"][1] += vector[1]
        elapsed = window["end_time_sec"] - window["start_time_sec"]
        if elapsed + 1e-9 < interval_seconds:
            continue
        dx, dy = window["vector"]
        net_distance = math.hypot(dx, dy)
        samples.append(
            {
                "start_time_sec": window["start_time_sec"],
                "end_time_sec": window["end_time_sec"],
                "seconds": elapsed,
                "distance_m": net_distance,
                "speed_mps": net_distance / elapsed,
                "vector": (dx, dy),
                "series_id": window["series_id"],
            }
        )
        window = None
    return samples


def _accelerations(segments):
    values = []
    for left, right in zip(segments, segments[1:]):
        if left.get("series_id") != right.get("series_id"):
            continue
        seconds = max((left["seconds"] + right["seconds"]) / 2.0, 1e-6)
        values.append((right["speed_mps"] - left["speed_mps"]) / seconds)
    return values


def _acceleration_event_counts(segments):
    """Count acceleration runs with a two-metre burst inside a half-second.

    A high acceleration sample alone is often a pose/court-localisation wobble.
    The player must sustain the acceleration signal and have at least one
    qualifying speed window that covers two metres in no more than 0.5 seconds.
    Deceleration remains in the raw payload for compatibility, but it is not a
    player-facing card; the UI displays the already confirmed turn count.
    """
    acceleration_count = 0
    deceleration_count = 0
    active_direction = 0
    has_qualifying_sprint_window = False
    previous = None

    def finish_active():
        nonlocal acceleration_count, has_qualifying_sprint_window
        if (
            active_direction == 1
            and has_qualifying_sprint_window
        ):
            acceleration_count += 1
        has_qualifying_sprint_window = False

    for current in segments:
        if previous is None or current.get("series_id") != previous.get("series_id"):
            finish_active()
            active_direction = 0
            previous = current
            continue
        seconds = max((float(previous["seconds"]) + float(current["seconds"])) / 2.0, 1e-6)
        value = (float(current["speed_mps"]) - float(previous["speed_mps"])) / seconds
        if value >= ACCELERATION_EVENT_MPS2:
            direction = 1
        elif value <= -ACCELERATION_EVENT_MPS2:
            direction = -1
        else:
            direction = 0
        if direction != active_direction:
            finish_active()
            if direction < 0:
                deceleration_count += 1
            has_qualifying_sprint_window = direction == 1 and _is_qualifying_burst_window(current)
        elif direction == 1:
            has_qualifying_sprint_window = (
                has_qualifying_sprint_window or _is_qualifying_burst_window(current)
            )
        active_direction = direction
        previous = current
    finish_active()
    return acceleration_count, deceleration_count


def _direction_change_events(segments):
    """Confirm a sharp turn with one metre on each side within 0.5 seconds."""
    confirmed = []
    for previous, current in zip(segments, segments[1:]):
        if previous.get("series_id") != current.get("series_id"):
            continue
        if not _is_qualifying_turn_leg(previous) or not _is_qualifying_turn_leg(current):
            continue
        if _angle_degrees(previous["vector"], current["vector"]) < TURN_ANGLE_DEGREES:
            continue
        confirmed.append({
            "time_sec": float(current.get("end_time_sec") or 0.0),
            "from_distance_m": float(previous["distance_m"]),
            "to_distance_m": float(current["distance_m"]),
            "from_seconds": float(previous["seconds"]),
            "to_seconds": float(current["seconds"]),
        })
    return confirmed


def _is_qualifying_burst_window(segment):
    return (
        float(segment.get("seconds") or 0.0) <= ACCELERATION_SPRINT_WINDOW_SECONDS
        and float(segment.get("distance_m") or 0.0) >= ACCELERATION_SPRINT_MIN_DISTANCE_M
    )


def _is_qualifying_turn_leg(segment):
    vector = tuple(float(value) for value in (segment.get("vector") or ()))
    return (
        float(segment.get("speed_mps") or 0.0) >= MOVING_SPEED_MPS
        and float(segment.get("seconds") or 0.0) <= DIRECTION_CHANGE_MAX_LEG_SECONDS
        and float(segment.get("distance_m") or 0.0) >= DIRECTION_CHANGE_MIN_LEG_DISTANCE_M
        and len(vector) >= 2
        and math.hypot(*vector) > 1e-9
    )


def _motion_leg(segment):
    vector = tuple(float(value) for value in (segment.get("vector") or ()))
    if len(vector) < 2:
        return None
    distance_m = float(segment.get("distance_m") or 0.0)
    if distance_m <= 0.0 or math.hypot(*vector) <= 1e-9:
        return None
    return {
        "vector": vector,
        "distance_m": distance_m,
        "end_time_sec": float(segment.get("end_time_sec") or 0.0),
    }


def _extend_motion_leg(target, incoming):
    target["vector"] = (
        target["vector"][0] + incoming["vector"][0],
        target["vector"][1] + incoming["vector"][1],
    )
    target["distance_m"] += incoming["distance_m"]
    target["end_time_sec"] = incoming["end_time_sec"]


def _angle_degrees(left_vector, right_vector):
    left_norm = math.hypot(*left_vector)
    right_norm = math.hypot(*right_vector)
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, (
        left_vector[0] * right_vector[0] + left_vector[1] * right_vector[1]
    ) / (left_norm * right_norm)))
    return math.degrees(math.acos(cosine))


def _peak_direction_changes_in_window(events, *, window_seconds):
    """Return the maximum confirmed events in any complete moving-time window."""
    times = sorted(float(event["time_sec"]) for event in events if event.get("time_sec") is not None)
    if not times:
        return 0
    window_seconds = max(0.001, float(window_seconds))
    peak = 0
    start = 0
    for end, current_time in enumerate(times):
        while current_time - times[start] > window_seconds:
            start += 1
        peak = max(peak, end - start + 1)
    return peak


def _energy_estimate(
    profile,
    *,
    active_seconds,
    high_intensity_seconds,
    coverage_ratio,
    mean_speed_mps,
    direction_change_count,
):
    """Estimate energy from measured movement intensity, with explicit limits.

    This is deliberately a single product estimate rather than a medical
    calorie range.  It combines measured mean speed, high-intensity movement
    share, and direction-change rate into one bounded intensity proxy between
    the published social and match-play badminton anchors.  The inputs and
    confidence remain in the JSON so the heuristic can be calibrated later.
    """
    profile = profile or {}
    weight_kg = _positive_float_or_none(profile.get("weight_kg"))
    height_cm = _positive_float_or_none(profile.get("height_cm"))
    if weight_kg is None:
        return {
            "status": "requires_weight",
            "height_cm": height_cm,
            "reason": "Weight is required for a movement-time calorie estimate; it is never estimated from video.",
        }
    active_seconds = max(0.0, float(active_seconds))
    high_intensity_seconds = min(active_seconds, max(0.0, float(high_intensity_seconds)))
    high_intensity_fraction = high_intensity_seconds / active_seconds if active_seconds else 0.0
    mean_speed_score = max(0.0, min(1.0, float(mean_speed_mps) / 3.0))
    direction_change_rate = float(direction_change_count) / active_seconds if active_seconds else 0.0
    direction_change_score = max(0.0, min(1.0, direction_change_rate / 0.8))
    intensity_score = (
        0.55 * high_intensity_fraction
        + 0.30 * mean_speed_score
        + 0.15 * direction_change_score
    )
    met_estimate = BADMINTON_SOCIAL_MET + (
        BADMINTON_MATCH_PLAY_MET - BADMINTON_SOCIAL_MET
    ) * intensity_score
    hours = active_seconds / 3600.0
    coverage_ratio = max(0.0, min(1.0, float(coverage_ratio)))
    if coverage_ratio >= 0.85 and active_seconds >= 120.0:
        confidence = "high"
    elif coverage_ratio >= 0.60 and active_seconds >= 60.0:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "status": "estimated",
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "measured_active_movement_time_sec": round(active_seconds, 3),
        "high_intensity_movement_fraction": round(high_intensity_fraction, 4),
        "measurement_coverage_ratio": round(coverage_ratio, 4),
        "reference_met_values": {
            "social_general": BADMINTON_SOCIAL_MET,
            "competitive": BADMINTON_COMPETITIVE_MET,
            "competitive_match_play": BADMINTON_MATCH_PLAY_MET,
        },
        "met_estimate": round(met_estimate, 3),
        "movement_intensity_inputs": {
            "mean_speed_mps": round(float(mean_speed_mps), 3),
            "high_intensity_movement_fraction": round(high_intensity_fraction, 4),
            "direction_change_rate_per_sec": round(direction_change_rate, 4),
            "intensity_score": round(intensity_score, 4),
        },
        "estimated_kcal": round(weight_kg * met_estimate * hours, 3),
        "estimated_kcal_rounded": int(round(weight_kg * met_estimate * hours)),
        "confidence": confidence,
        "method": "weight_kg × speed-and-direction-weighted MET proxy × measured_active_movement_hours",
        "scope": "measured movement segments only; not a complete-session or medical calorie measurement",
    }


def _read_json(path):
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _positive_float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _mean(values):
    return sum(values) / len(values) if values else None


def _round_or_none(value):
    return round(value, 3) if value is not None else None


def _utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
