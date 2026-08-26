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
MAX_PLAUSIBLE_SPEED_MPS = 10.0
MOVING_SPEED_MPS = 0.25
HIGH_INTENSITY_SPEED_MPS = 2.0
ACCELERATION_EVENT_MPS2 = 1.5
SPEED_ACCELERATION_STATISTICS_INTERVAL_SECONDS = 0.5
TURN_ANGLE_DEGREES = 90.0
DIRECTION_CHANGE_MIN_LEG_DISTANCE_M = 1.0
DIRECTION_CHANGE_PEAK_WINDOW_SECONDS = 30.0
# 2024 Adult Compendium reference values for badminton: social/general 5.5,
# competitive 7.0, and competitive match play 9.0 MET.  Video-only movement
# cannot determine physiology, so these are reference anchors for a simple
# product estimate rather than a medical calorie measurement.
BADMINTON_SOCIAL_MET = 5.5
BADMINTON_COMPETITIVE_MET = 7.0
BADMINTON_MATCH_PLAY_MET = 9.0


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
            "max_plausible_speed_mps": MAX_PLAUSIBLE_SPEED_MPS,
            "moving_speed_mps": MOVING_SPEED_MPS,
            "high_intensity_speed_mps": HIGH_INTENSITY_SPEED_MPS,
            "acceleration_event_mps2": ACCELERATION_EVENT_MPS2,
            "speed_and_acceleration_statistics_interval_seconds": (
                SPEED_ACCELERATION_STATISTICS_INTERVAL_SECONDS
            ),
            "turn_angle_degrees": TURN_ANGLE_DEGREES,
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
    accelerations = _accelerations(statistic_segments)
    acceleration_event_count, deceleration_event_count = _acceleration_event_counts(accelerations)
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
            "direction_change_count": direction_change_count,
            "peak_direction_changes_30s": peak_direction_changes_30s,
            "agility_movement": {
                "whole_match_confirmed_direction_changes": direction_change_count,
                "peak_window_seconds": DIRECTION_CHANGE_PEAK_WINDOW_SECONDS,
                "peak_window_confirmed_direction_changes": peak_direction_changes_30s,
                "turn_angle_degrees": TURN_ANGLE_DEGREES,
                "minimum_distance_each_leg_m": DIRECTION_CHANGE_MIN_LEG_DISTANCE_M,
                "speed_and_acceleration_statistics_interval_seconds": (
                    SPEED_ACCELERATION_STATISTICS_INTERVAL_SECONDS
                ),
                "scope": "movement agility evidence only; does not measure stimulus-to-movement reaction time",
            },
            "zone_measurements": dict(style.get("zone_frames") or {}),
        },
        "energy_estimate": energy,
        "quality": {
            "status": "reviewable" if points else "insufficient_measurements",
            "policy": "Only fresh high-confidence detected positions are used; no predicted position or missing gap is converted into movement.",
        },
    }


def _valid_segments(points, fps):
    rate = max(float(fps or 0.0), 1.0)
    segments = []
    excluded = 0
    for left, right in zip(points, points[1:]):
        frame_left = left.get("frame")
        frame_right = right.get("frame")
        if frame_left is None or frame_right is None:
            excluded += 1
            continue
        delta_frames = int(frame_right) - int(frame_left)
        seconds = delta_frames / rate
        if delta_frames <= 0 or seconds > MAX_JOIN_GAP_SECONDS:
            excluded += 1
            continue
        point_left = left["court_xy_m"]
        point_right = right["court_xy_m"]
        dx = float(point_right[0]) - float(point_left[0])
        dy = float(point_right[1]) - float(point_left[1])
        distance = math.hypot(dx, dy)
        speed = distance / seconds
        if speed > MAX_PLAUSIBLE_SPEED_MPS:
            excluded += 1
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
    return segments, excluded


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


def _acceleration_event_counts(accelerations):
    """Merge contiguous threshold samples into one acceleration/deceleration event.

    The former implementation counted every adjacent pair above the threshold,
    which could turn one sustained run-up or braking action into several user
    visible "events".  A new event starts only after the signal has returned
    inside the neutral band or crossed into the opposite direction.
    """
    acceleration_count = 0
    deceleration_count = 0
    active_direction = 0
    for value in accelerations:
        if value >= ACCELERATION_EVENT_MPS2:
            direction = 1
        elif value <= -ACCELERATION_EVENT_MPS2:
            direction = -1
        else:
            direction = 0
        if direction and direction != active_direction:
            if direction > 0:
                acceleration_count += 1
            else:
                deceleration_count += 1
        active_direction = direction
    return acceleration_count, deceleration_count


def _direction_change_events(segments):
    """Confirm substantial direction changes without counting body sway.

    A turn is emitted only after the player has moved at least one metre in
    the original leg, changed direction by at least 90 degrees, and then
    travelled one metre in the new leg.  The latter confirmation prevents a
    short corrective step from becoming a movement-agility event.
    """
    confirmed = []
    established_leg = None
    candidate_leg = None
    current_series_id = object()
    for segment in segments:
        if segment.get("series_id") != current_series_id:
            established_leg = None
            candidate_leg = None
            current_series_id = segment.get("series_id")
        if float(segment.get("speed_mps") or 0.0) < MOVING_SPEED_MPS:
            continue
        leg = _motion_leg(segment)
        if leg is None:
            continue
        if established_leg is None:
            established_leg = leg
            continue

        if candidate_leg is not None:
            if _angle_degrees(candidate_leg["vector"], leg["vector"]) < TURN_ANGLE_DEGREES:
                _extend_motion_leg(candidate_leg, leg)
                if candidate_leg["distance_m"] >= DIRECTION_CHANGE_MIN_LEG_DISTANCE_M:
                    confirmed.append({
                        "time_sec": candidate_leg["end_time_sec"],
                        "from_distance_m": established_leg["distance_m"],
                        "to_distance_m": candidate_leg["distance_m"],
                    })
                    established_leg = candidate_leg
                    candidate_leg = None
                continue

            # The new segment did not sustain the candidate direction. If it
            # aligns with the established path it was a brief sway; otherwise
            # begin a fresh candidate leg from this new direction.
            if _angle_degrees(established_leg["vector"], leg["vector"]) < TURN_ANGLE_DEGREES:
                _extend_motion_leg(established_leg, leg)
                candidate_leg = None
            else:
                candidate_leg = leg
            continue

        if (
            established_leg["distance_m"] >= DIRECTION_CHANGE_MIN_LEG_DISTANCE_M
            and _angle_degrees(established_leg["vector"], leg["vector"]) >= TURN_ANGLE_DEGREES
        ):
            candidate_leg = leg
        else:
            _extend_motion_leg(established_leg, leg)
    return confirmed


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
