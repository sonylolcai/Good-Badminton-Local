"""Offline shuttle reconstruction and conservative shot-event proposals.

This module runs *after* a full video analysis.  It never mutates
``detections.jsonl``: raw detector evidence stays immutable, while every
interpolated point and every inferred hit is written as a separate derived
artifact with an explicit confidence and provenance.

The result is intentionally suitable for a human-review queue, not automatic
scoring.  In particular, a long or physically inconsistent shuttle gap remains
``unknown_gap`` instead of being fabricated into a trajectory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable

from .huji_play_state import HUJI_PLAY_STATE_FILENAME, load_huji_play_state, terminal_play_context


DERIVED_DIRNAME = "derived"
SHUTTLE_TRACK_FILENAME = "shuttle_tracks_v2.jsonl"
SHOT_EVENT_FILENAME = "shot_events_v2.jsonl"
RALLY_FILENAME = "rallies_v2.jsonl"
TERMINAL_CANDIDATE_FILENAME = "terminal_candidates_v1.jsonl"
RALLY_BOUNDARY_REFERENCE_FILENAME = "rally_boundary_reference_user_review.jsonl"
DERIVATION_VERSION = "2.4"


def generate_offline_artifacts(detections_path, output_dir=None, fps=None, play_state_path=None):
    """Create versioned derived artifacts next to immutable detector output."""
    detections_path = Path(detections_path)
    rows = load_jsonl(detections_path)
    if not rows:
        raise ValueError("detections.jsonl is empty; offline reconstruction has no evidence")
    output_dir = Path(output_dir) if output_dir else detections_path.parent / DERIVED_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    configured_play_state_path = Path(play_state_path) if play_state_path else output_dir / HUJI_PLAY_STATE_FILENAME
    play_state_rows = load_huji_play_state(configured_play_state_path)
    fps = float(fps or infer_fps(rows) or 30.0)
    width, height = video_dimensions(detections_path.parent)
    track_rows = reconstruct_shuttle_track(rows, fps=fps, width=width, height=height)
    events = build_shot_events(rows, track_rows, fps=fps)
    events = infer_missing_shuttle_events(rows, events)
    events = refresh_event_evidence(events, track_rows)
    court_polygon = _court_polygon_from_metadata(detections_path.parent)
    terminal_candidates = detect_rally_terminal_candidates(
        track_rows,
        raw_rows=rows,
        court_polygon=court_polygon,
    )
    terminal_candidates = apply_huji_play_state_context(terminal_candidates, play_state_rows)
    # A static visual false-positive is especially harmful here: it turns one
    # rally into two.  Every accepted boundary retains its evidence source and
    # confidence; manual references are never an input to this calculation.
    rallies = build_rallies(
        track_rows,
        events,
        court_polygon=court_polygon,
        terminal_candidates=terminal_candidates,
    )
    track_path = output_dir / SHUTTLE_TRACK_FILENAME
    event_path = output_dir / SHOT_EVENT_FILENAME
    rally_path = output_dir / RALLY_FILENAME
    terminal_candidate_path = output_dir / TERMINAL_CANDIDATE_FILENAME
    write_jsonl(track_path, track_rows)
    write_jsonl(event_path, events)
    write_jsonl(rally_path, rallies)
    write_jsonl(terminal_candidate_path, terminal_candidates)
    return {
        "version": DERIVATION_VERSION,
        "tracks_path": str(track_path),
        "events_path": str(event_path),
        "rallies_path": str(rally_path),
        "terminal_candidates_path": str(terminal_candidate_path),
        "frame_count": len(track_rows),
        "event_count": len(events),
        "rally_count": len(rallies),
        "terminal_candidate_count": len(terminal_candidates),
        "play_state": {
            "path": str(configured_play_state_path) if configured_play_state_path.is_file() else None,
            "status": "available" if play_state_rows else "not_available",
            "sample_count": len(play_state_rows),
            "policy": (
                "Huji-compatible scene state can only reduce a direction-reversal-after-gap "
                "candidate when active play continues; it is never terminal or score proof."
            ),
        },
        "observed_shot_count": sum(event.get("event_origin") == "machine_candidate" for event in events),
        "inferred_shot_count": sum(event.get("event_origin") == "motion_constraint_candidate" for event in events),
        "fps": fps,
        "policy": (
            "raw detections are immutable; only short, bounded gaps may be reconstructed; "
            "rallies and missing-shuttle hits are review candidates, never score evidence"
        ),
    }


def reconstruct_shuttle_track(rows, fps=30.0, width=None, height=None, max_gap_sec=0.35, max_speed_px_s=3200.0):
    """Build one per-analysis-frame shuttle sequence with honest gap labels."""
    candidates = [_raw_shuttle_candidate(row) for row in rows]
    rejected_indices = edge_static_artifacts(candidates, width=width, height=height)
    track_rows = []
    for index, (row, candidate) in enumerate(zip(rows, candidates)):
        base = _base_track_record(row, index)
        if candidate and index not in rejected_indices:
            base.update(
                {
                    "status": "detected",
                    "image_xy": candidate["image_xy"],
                    "confidence": candidate["confidence"],
                    "provenance": {
                        "kind": "raw_detection",
                        "raw_status": candidate["raw_status"],
                        "accepted": True,
                    },
                }
            )
        elif candidate:
            base.update(
                {
                    "status": "rejected_artifact",
                    "image_xy": None,
                    "confidence": 0.0,
                    "provenance": {
                        "kind": "raw_detection_rejected",
                        "raw_status": candidate["raw_status"],
                        "accepted": False,
                        "reason": "repeated_low_confidence_edge_point",
                    },
                }
            )
        else:
            base.update(
                {
                    "status": "missing",
                    "image_xy": None,
                    "confidence": 0.0,
                    "provenance": {"kind": "no_accepted_raw_detection", "accepted": False},
                }
            )
        track_rows.append(base)

    detected = [index for index, item in enumerate(track_rows) if item["status"] == "detected"]
    max_gap_frames = max(1, int(round(float(fps) * max_gap_sec)))
    for before_index, after_index in zip(detected, detected[1:]):
        interior = list(range(before_index + 1, after_index))
        if not interior:
            continue
        before, after = track_rows[before_index], track_rows[after_index]
        elapsed = float(after["time_sec"]) - float(before["time_sec"])
        distance = _distance(before["image_xy"], after["image_xy"])
        speed = distance / elapsed if elapsed > 0 else math.inf
        can_reconstruct = len(interior) <= max_gap_frames and elapsed > 0 and speed <= max_speed_px_s
        for index in interior:
            record = track_rows[index]
            if record["status"] == "rejected_artifact":
                continue
            if can_reconstruct:
                ratio = (float(record["time_sec"]) - float(before["time_sec"])) / elapsed
                x = before["image_xy"][0] + (after["image_xy"][0] - before["image_xy"][0]) * ratio
                y = before["image_xy"][1] + (after["image_xy"][1] - before["image_xy"][1]) * ratio
                record.update(
                    {
                        "status": "reconstructed",
                        "image_xy": [round(x, 3), round(y, 3)],
                        "confidence": round(min(before["confidence"], after["confidence"]) * 0.45, 4),
                        "provenance": {
                            "kind": "bidirectional_linear_interpolation",
                            "accepted": False,
                            "before_frame": before["frame"],
                            "after_frame": after["frame"],
                            "gap_frames": len(interior),
                            "speed_px_s": round(speed, 2),
                        },
                    }
                )
            else:
                record.update(
                    {
                        "status": "unknown_gap",
                        "image_xy": None,
                        "confidence": 0.0,
                        "provenance": {
                            "kind": "unfilled_gap",
                            "accepted": False,
                            "before_frame": before["frame"],
                            "after_frame": after["frame"],
                            "gap_frames": len(interior),
                            "reason": "gap_too_long_or_speed_inconsistent",
                        },
                    }
                )
    _assign_trajectory_ids(track_rows)
    return track_rows


def build_shot_events(raw_rows, track_rows, fps=30.0):
    """Create conservative hit/receiver/shot-type candidates from future context."""
    by_frame = {int(item["frame"]): item for item in track_rows}
    seeds = []
    for raw in raw_rows:
        for event in ((raw.get("spatial") or {}).get("hit_events") or []):
            if event.get("status") != "candidate":
                continue
            seeds.append(
                {
                    "frame": int(raw.get("frame", 0)),
                    "time_sec": float(raw.get("time_sec", 0.0)),
                    "source": "spatial_hit_candidate",
                    "confidence": float(event.get("confidence") or 0.0),
                    "hitter_track_id": event.get("hitter_track_id"),
                    "reason": event.get("reason"),
                }
            )
    seeds.extend(_trajectory_turn_seeds(track_rows))
    seeds.extend(_trajectory_heading_change_near_hand_seeds(raw_rows, track_rows))
    seeds.extend(_reconstructed_gap_hand_proximity_seeds(raw_rows, track_rows))
    seeds = _deduplicate_seeds(seeds)
    events = []
    for index, seed in enumerate(seeds, start=1):
        raw = _row_at_frame(raw_rows, seed["frame"])
        track = by_frame.get(seed["frame"])
        hitter = _hitter_evidence(raw, track, seed.get("hitter_track_id"))
        event_origin = seed.get("event_origin", "machine_candidate")
        events.append(
            {
                "schema_version": DERIVATION_VERSION,
                "event_id": f"shot_{index:04d}",
                "frame": seed["frame"],
                "hit_time_sec": round(seed["time_sec"], 6),
                "status": "candidate",
                "candidate_source": seed["source"],
                "candidate_reason": seed.get("reason"),
                "event_origin": event_origin,
                "time_confidence": round(min(0.55, float(seed["confidence"])), 4),
                "trajectory": (
                    _missing_shuttle_trajectory()
                    if event_origin == "motion_constraint_candidate"
                    else _event_trajectory_evidence(track_rows, seed["time_sec"])
                ),
                "hitter": hitter,
                "receiver": {"track_id": None, "confidence": 0.0, "source": "not_yet_inferred"},
                "proposal": {"label": "unknown", "confidence": 0.0, "status": "needs_human_review"},
                "decision": {"status": "pending", "label_source": "machine", "eligible_for_statistics": False},
            }
        )

    return refresh_event_evidence(events, track_rows)


def refresh_event_evidence(events, track_rows):
    """Recompute per-shot evidence after candidates have been inserted.

    A shot's flight window ends at the next candidate hit, rather than at an
    arbitrary fixed duration.  This avoids accidentally using the next shot's
    shuttle points to describe the current shot.
    """
    ordered = sorted(events, key=lambda item: (float(item["hit_time_sec"]), int(item.get("frame", 0))))
    for index, current in enumerate(ordered):
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        if current.get("event_origin") != "motion_constraint_candidate":
            current["trajectory"] = _event_trajectory_evidence(
                track_rows,
                current["hit_time_sec"],
                until_time_sec=following["hit_time_sec"] if following else None,
            )
        current["receiver"] = {"track_id": None, "confidence": 0.0, "source": "not_yet_inferred"}
        if following and following.get("event_origin") != "motion_constraint_candidate":
            next_hitter = following.get("hitter") or {}
            if next_hitter.get("track_id") and next_hitter["track_id"] != (current.get("hitter") or {}).get("track_id"):
                current["receiver"] = {
                    "track_id": next_hitter["track_id"],
                    "confidence": round(min(0.45, next_hitter.get("confidence", 0.0)), 4),
                    "source": "next_hit_candidate",
                }
        current["proposal"] = _propose_from_space(current, following)
    return _renumber_events(ordered)


def infer_missing_shuttle_events(raw_rows, events, min_gap_sec=0.45, max_gap_sec=4.0):
    """Insert a low-confidence singles-only *candidate* for a missing return.

    The rule is deliberately narrow: two supported hits by the same player,
    a unique other visible track, and an explicit singles mode are required.
    It does not invent a shuttle point, a speed, a score, or a doubles hitter.
    A real pose-action classifier can later provide stronger evidence.
    """
    if not events or _match_mode(raw_rows) != "singles":
        return list(events)
    ordered = sorted(events, key=lambda item: float(item["hit_time_sec"]))
    inferred = []
    for before, after in zip(ordered, ordered[1:]):
        before_hitter = (before.get("hitter") or {}).get("track_id")
        after_hitter = (after.get("hitter") or {}).get("track_id")
        elapsed = float(after["hit_time_sec"]) - float(before["hit_time_sec"])
        if not before_hitter or before_hitter != after_hitter or not min_gap_sec <= elapsed <= max_gap_sec:
            continue
        visible_tracks = _visible_detected_track_ids(raw_rows, before["hit_time_sec"], after["hit_time_sec"])
        opponents = sorted(visible_tracks.difference({before_hitter}))
        if len(opponents) != 1:
            continue
        hit_time = round((float(before["hit_time_sec"]) + float(after["hit_time_sec"])) / 2.0, 6)
        inferred.append(
            {
                "schema_version": DERIVATION_VERSION,
                "event_id": "pending_inferred",
                "frame": _nearest_frame(raw_rows, hit_time),
                "hit_time_sec": hit_time,
                "status": "candidate",
                "candidate_source": "motion_inferred_missing_shuttle",
                "candidate_reason": (
                    "same supported singles hitter appears before and after an unobserved interval; "
                    "the unique opponent is a return candidate, not a detector measurement"
                ),
                "event_origin": "motion_constraint_candidate",
                "time_confidence": 0.15,
                "trajectory": _missing_shuttle_trajectory(),
                "hitter": {
                    "track_id": opponents[0],
                    "confidence": 0.15,
                    "source": "singles_alternation_constraint",
                    "measurement_status": "inferred",
                },
                "receiver": {"track_id": None, "confidence": 0.0, "source": "not_yet_inferred"},
                "proposal": _proposal("unknown", 0.0, "Missing shuttle evidence; human review required."),
                "decision": {"status": "pending", "label_source": "machine", "eligible_for_statistics": False},
            }
        )
    return _renumber_events(sorted([*ordered, *inferred], key=lambda item: float(item["hit_time_sec"])))


def build_rallies(
    track_rows,
    events,
    stationary_speed_px_s=45.0,
    stationary_min_duration_sec=0.75,
    max_unobserved_between_rallies_sec=4.0,
    court_polygon=None,
    terminal_candidates=None,
    min_terminal_confidence=0.65,
):
    """Segment a reviewable rally sequence and annotate every shot candidate.

    A long run of slow, continuous shuttle observations is the preferred
    boundary.  Missing shuttle evidence is not a boundary: it remains unknown
    until a later terminal observation or a human review is available.
    """
    ordered = sorted(events, key=lambda item: float(item["hit_time_sec"]))
    if not ordered:
        return []
    stationary_windows = _stationary_windows(
        track_rows,
        speed_limit_px_s=stationary_speed_px_s,
        min_duration_sec=stationary_min_duration_sec,
        court_polygon=court_polygon,
    )
    accepted_terminals = [
        item for item in terminal_candidates or []
        if float(item.get("confidence") or 0.0) >= float(min_terminal_confidence)
    ]
    groups = []
    current = [ordered[0]]
    boundary_reasons = []
    for event in ordered[1:]:
        previous = current[-1]
        boundary = _boundary_between_events(
            track_rows,
            stationary_windows,
            previous_time=float(previous["hit_time_sec"]),
            next_time=float(event["hit_time_sec"]),
            max_unobserved_sec=max_unobserved_between_rallies_sec,
            terminal_candidates=accepted_terminals,
        )
        if boundary:
            groups.append(current)
            boundary_reasons.append(boundary)
            current = [event]
        else:
            current.append(event)
    groups.append(current)

    video_end = max((float(item.get("time_sec", 0.0)) for item in track_rows), default=float(ordered[-1]["hit_time_sec"]))
    rallies = []
    for index, group in enumerate(groups, start=1):
        following_boundary = boundary_reasons[index - 1] if index - 1 < len(boundary_reasons) else None
        terminal_boundary = following_boundary or _terminal_boundary_after_event(
            stationary_windows,
            last_event_time=float(group[-1]["hit_time_sec"]),
            terminal_candidates=accepted_terminals,
        )
        end_time = terminal_boundary["end_time_sec"] if terminal_boundary else video_end
        end_reason = terminal_boundary["reason"] if terminal_boundary else "video_end_without_confirmed_terminal_event"
        observed = sum(item.get("event_origin") != "motion_constraint_candidate" for item in group)
        inferred = len(group) - observed
        rally_id = f"rally_{index:04d}"
        for shot_index, event in enumerate(group, start=1):
            event["rally_id"] = rally_id
            event["shot_index_in_rally"] = shot_index
        confidence = 0.25 + min(0.35, 0.07 * observed) - min(0.12, 0.04 * inferred)
        if terminal_boundary and terminal_boundary["reason"] == "shuttle_stationary_or_slow":
            confidence += 0.12
        rallies.append(
            {
                "schema_version": DERIVATION_VERSION,
                "rally_id": rally_id,
                "status": "candidate",
                "start_time_sec": round(float(group[0]["hit_time_sec"]), 6),
                "end_time_sec": round(float(end_time), 6),
                "end_reason": end_reason,
                "terminal_evidence": {
                    "status": "candidate" if terminal_boundary else "unknown",
                    "source": terminal_boundary.get("reason") if terminal_boundary else None,
                    "confidence": terminal_boundary.get("confidence") if terminal_boundary else 0.0,
                },
                "shot_count": len(group),
                "observed_shot_count": observed,
                "motion_inferred_shot_count": inferred,
                "shot_event_ids": [item["event_id"] for item in group],
                "confidence": round(max(0.0, min(0.75, confidence)), 4),
                "score": {
                    "status": "unknown",
                    "winner_track_id": None,
                    "included_in_player_statistics": False,
                    "reason": "Rally segmentation and shot candidates are not score evidence.",
                },
            }
        )
    return rallies


def build_rallies_from_manual_terminals(events, terminals, video_start_sec=0.0, video_end_sec=None):
    """Create a reviewed rally timeline from append-only human terminal facts.

    This is deliberately separate from ``build_rallies``.  A reviewer can
    confirm that a shuttle landed or went out even when the detector did not
    see it; that fact must not rewrite raw detections or be presented as a
    machine terminal decision.
    """
    ordered_events = sorted(events, key=lambda item: float(item.get("hit_time_sec", 0.0)))
    normalized = []
    for terminal in sorted(terminals, key=lambda item: float(item.get("time_sec", 0.0))):
        try:
            time_sec = float(terminal.get("time_sec"))
        except (TypeError, ValueError):
            continue
        if time_sec < float(video_start_sec):
            continue
        if normalized and time_sec <= normalized[-1]["time_sec"]:
            continue
        normalized.append({**terminal, "time_sec": round(time_sec, 6)})

    reviewed = []
    start_time = float(video_start_sec)
    for index, terminal in enumerate(normalized, start=1):
        end_time = terminal["time_sec"]
        event_group = [
            item for item in ordered_events
            if start_time < float(item.get("hit_time_sec", 0.0)) <= end_time
        ]
        observed = sum(item.get("event_origin") != "motion_constraint_candidate" for item in event_group)
        inferred = len(event_group) - observed
        outcome = str(terminal.get("outcome") or "unknown_terminal")
        reviewed.append(
            {
                "schema_version": "1.0",
                "rally_id": f"reviewed_rally_{index:04d}",
                "status": "human_terminal_reviewed",
                "start_time_sec": round(start_time, 6),
                "end_time_sec": end_time,
                "end_reason": f"human_confirmed_{outcome}",
                "terminal": {
                    "source": "human_review",
                    "terminal_id": terminal.get("terminal_id"),
                    "outcome": outcome,
                    "reviewer": terminal.get("reviewer") or "",
                    "note": terminal.get("note") or "",
                    "recorded_at": terminal.get("recorded_at"),
                },
                "shot_count": len(event_group),
                "observed_shot_count": observed,
                "motion_inferred_shot_count": inferred,
                "shot_event_ids": [item.get("event_id") for item in event_group if item.get("event_id")],
                "shot_times_sec": [round(float(item.get("hit_time_sec", 0.0)), 6) for item in event_group],
                "confidence": 1.0,
                "score": {
                    "status": "unknown",
                    "winner_track_id": None,
                    "included_in_player_statistics": False,
                    "reason": "Human terminal review confirms only a rally boundary, not score evidence.",
                },
            }
        )
        start_time = end_time
    return reviewed


def detect_rally_terminal_candidates(track_rows, raw_rows=None, court_polygon=None):
    """Find explainable terminal *candidates* from fixed-camera shuttle evidence.

    A badminton shuttle often disappears at landing or after an out call before
    a long low-speed run is visible.  This function therefore combines several
    independent visual signatures.  It never reads human boundary references,
    never assigns a score, and leaves lower-confidence evidence available for
    review rather than forcing a rally split.
    """
    if not court_polygon:
        return []
    observed = [
        item for item in track_rows
        if item.get("status") == "detected"
        and float(item.get("confidence") or 0.0) >= 0.20
        and _valid_point(item.get("image_xy"))
    ]
    if len(observed) < 3:
        return []

    candidates = []
    ground_edge = _ground_facing_court_edge(court_polygon)
    for index, (previous, current) in enumerate(zip(observed, observed[1:])):
        if index == 0:
            continue
        before = observed[index - 1]
        previous_time = float(previous["time_sec"])
        current_time = float(current["time_sec"])
        gap_sec = current_time - previous_time
        before_gap_sec = previous_time - float(before["time_sec"])
        if ground_edge is not None and _distance_to_segment(previous["image_xy"], *ground_edge) <= 85.0:
            strict_current_inside = _point_in_or_near_polygon(current["image_xy"], court_polygon, margin=0.0)
            transition_speed = _distance(previous["image_xy"], current["image_xy"]) / gap_sec
            if not strict_current_inside and gap_sec <= 0.12 and transition_speed <= 1600.0:
                candidates.append(
                    _terminal_candidate(
                        previous_time,
                        "ground_facing_boundary_exit",
                        0.74,
                        gap_sec=gap_sec,
                        transition_speed_px_s=transition_speed,
                    )
                )
            elif not strict_current_inside and gap_sec <= 0.50 and transition_speed > 3200.0:
                candidates.append(
                    _terminal_candidate(
                        previous_time,
                        "implausible_reacquisition_after_descent",
                        0.70,
                        gap_sec=gap_sec,
                        transition_speed_px_s=transition_speed,
                    )
                )

        if not (0 < before_gap_sec <= 0.16 and 0.12 <= gap_sec <= 2.50):
            continue
        inbound = _velocity(before["image_xy"], previous["image_xy"], before_gap_sec)
        resumed = _velocity(previous["image_xy"], current["image_xy"], gap_sec)
        inbound_speed = math.hypot(*inbound)
        direction_cosine = _cosine(inbound, resumed)
        inside_previous = _point_in_or_near_polygon(previous["image_xy"], court_polygon)
        inside_current = _point_in_or_near_polygon(current["image_xy"], court_polygon)
        hand_distance = _nearest_hand_distance(raw_rows, previous_time, previous["image_xy"])

        if (
            inside_previous
            and inside_current
            and 100.0 <= inbound_speed <= 1500.0
            and direction_cosine <= -0.50
            and not (hand_distance is not None and hand_distance <= 30.0)
        ):
            confidence = 0.48
            if gap_sec >= 0.40:
                confidence += 0.12
            if float(previous.get("confidence") or 0.0) >= 0.50:
                confidence += 0.08
            if direction_cosine <= -0.80:
                confidence += 0.10
            candidates.append(
                _terminal_candidate(
                    previous_time,
                    "direction_reversal_after_gap",
                    confidence,
                    gap_sec=gap_sec,
                    inbound_speed_px_s=inbound_speed,
                    direction_cosine=direction_cosine,
                    nearest_hand_distance_px=hand_distance,
                )
            )

    for window in _stationary_windows(
        track_rows,
        speed_limit_px_s=60.0,
        min_duration_sec=0.14,
        max_point_gap_sec=0.12,
        court_polygon=court_polygon,
    ):
        end_time = float(window["end_time_sec"])
        next_observation = next(
            (
                item for item in observed
                if end_time < float(item["time_sec"]) <= end_time + 0.50
                and not _point_in_or_near_polygon(item["image_xy"], court_polygon, margin=0.0)
            ),
            None,
        )
        if next_observation is None:
            continue
        gap_sec = float(next_observation["time_sec"]) - end_time
        if not 0.12 <= gap_sec <= 0.50:
            continue
        candidates.append(
            _terminal_candidate(
                (float(window["start_time_sec"]) + end_time) / 2.0,
                "short_stationary_then_evidence_break",
                0.72,
                stationary_duration_sec=end_time - float(window["start_time_sec"]),
                gap_sec=gap_sec,
            )
        )
    return _deduplicate_terminal_candidates(candidates)


def _terminal_candidate(time_sec, source, confidence, **evidence):
    return {
        "schema_version": DERIVATION_VERSION,
        "status": "candidate",
        "time_sec": round(float(time_sec), 6),
        "source": source,
        "confidence": round(max(0.0, min(0.85, float(confidence))), 4),
        "evidence": {key: round(float(value), 6) if isinstance(value, (int, float)) else value for key, value in evidence.items()},
        "eligible_for_scoring": False,
    }


def apply_huji_play_state_context(candidates, play_state_rows):
    """Preserve Huji-compatible scene evidence without treating it as a boundary.

    The only automatic change is deliberately narrow: a
    ``direction_reversal_after_gap`` candidate is a common source of false
    splits when the shuttle is lost briefly.  If a separate scene classifier
    sees active play *continue* beyond that candidate, its confidence is
    lowered below the default splitting threshold.  Landing, boundary-exit,
    stationary-shuttle, manual, and score data are never overridden.
    """
    adjusted = []
    for original in candidates or []:
        candidate = dict(original)
        evidence = dict(candidate.get("evidence") or {})
        context = terminal_play_context(play_state_rows, candidate.get("time_sec", 0.0))
        if context is not None:
            evidence["play_state_context"] = context
            candidate["evidence"] = evidence
            if candidate.get("source") == "direction_reversal_after_gap":
                candidate["confidence_before_play_state"] = candidate.get("confidence")
                candidate["confidence"] = round(min(float(candidate.get("confidence") or 0.0), 0.64), 4)
                candidate["status"] = "candidate_conflicted_by_active_play"
                candidate["review_reason"] = (
                    "Active-play scene evidence continues after a shuttle-gap direction reversal; "
                    "do not automatically split without additional terminal evidence."
                )
        adjusted.append(candidate)
    return adjusted


def _deduplicate_terminal_candidates(candidates, merge_window_sec=1.0):
    """Keep the strongest explanation in a local time cluster, preserving sources."""
    clusters = []
    for candidate in sorted(candidates, key=lambda item: float(item["time_sec"])):
        if clusters and float(candidate["time_sec"]) - float(clusters[-1][-1]["time_sec"]) <= merge_window_sec:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    merged = []
    for index, cluster in enumerate(clusters, start=1):
        best = max(cluster, key=lambda item: (float(item["confidence"]), -float(item["time_sec"])))
        merged.append(
            {
                **best,
                "terminal_candidate_id": f"terminal_candidate_{index:04d}",
                "evidence": {
                    **(best.get("evidence") or {}),
                    "merged_sources": [item["source"] for item in cluster],
                },
            }
        )
    return merged


def _ground_facing_court_edge(polygon):
    if len(polygon) < 3:
        return None
    edges = list(zip(polygon, [*polygon[1:], polygon[0]]))
    return max(edges, key=lambda edge: (float(edge[0][1]) + float(edge[1][1])) / 2.0)


def _nearest_hand_distance(raw_rows, time_sec, point):
    if not raw_rows or not _valid_point(point):
        return None
    raw = min(raw_rows, key=lambda item: abs(float(item.get("time_sec", 0.0)) - float(time_sec)))
    distances = []
    for player in (raw.get("players") or {}).values():
        for hand in (player.get("hands") or {}).values():
            if _valid_point(hand):
                distances.append(_distance(point, hand))
    return min(distances) if distances else None


def load_rally_boundary_reference(analysis_dir):
    """Load human terminal facts used only to evaluate automatic splitting.

    The reference is intentionally stored outside ``shot_review`` so opening a
    review session cannot accidentally turn it into an active manual override.
    """
    path = Path(analysis_dir) / "evaluation" / RALLY_BOUNDARY_REFERENCE_FILENAME
    return load_jsonl(path) if path.is_file() else []


def evaluate_rally_terminal_predictions(rallies, reference_terminals, tolerance_sec=0.60):
    """Measure candidate terminal boundaries without using references as input.

    Matching is one-to-one and time-tolerant.  A video-end marker is not a
    terminal prediction, and neither are human-reviewed segments; this keeps
    the report honest about what the automatic splitter actually found.
    """
    tolerance_sec = max(0.0, float(tolerance_sec))
    predictions = []
    for rally in rallies:
        reason = str(rally.get("end_reason") or "")
        if reason.startswith("video_end_") or reason.startswith("human_confirmed_"):
            continue
        try:
            predictions.append(
                {
                    "rally_id": rally.get("rally_id"),
                    "time_sec": round(float(rally["end_time_sec"]), 6),
                    "reason": reason or "unspecified_candidate_boundary",
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    references = []
    for terminal in reference_terminals:
        try:
            references.append(
                {
                    "terminal_id": terminal.get("terminal_id"),
                    "time_sec": round(float(terminal["time_sec"]), 6),
                    "outcome": terminal.get("outcome"),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    predictions.sort(key=lambda item: item["time_sec"])
    references.sort(key=lambda item: item["time_sec"])
    unmatched_reference_indexes = set(range(len(references)))
    matches = []
    false_positives = []
    for prediction in predictions:
        nearby = [
            index for index in unmatched_reference_indexes
            if abs(references[index]["time_sec"] - prediction["time_sec"]) <= tolerance_sec
        ]
        if not nearby:
            false_positives.append(prediction)
            continue
        index = min(nearby, key=lambda item: abs(references[item]["time_sec"] - prediction["time_sec"]))
        reference = references[index]
        unmatched_reference_indexes.remove(index)
        matches.append(
            {
                "prediction": prediction,
                "reference": reference,
                "absolute_error_sec": round(abs(reference["time_sec"] - prediction["time_sec"]), 6),
            }
        )

    missed_references = [references[index] for index in sorted(unmatched_reference_indexes)]
    precision = len(matches) / len(predictions) if predictions else None
    recall = len(matches) / len(references) if references else None
    return {
        "schema_version": "1.0",
        "tolerance_sec": tolerance_sec,
        "prediction_count": len(predictions),
        "reference_count": len(references),
        "matched_count": len(matches),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "mean_absolute_error_sec": round(
            sum(item["absolute_error_sec"] for item in matches) / len(matches), 6
        ) if matches else None,
        "matches": matches,
        "false_positives": false_positives,
        "missed_references": missed_references,
    }


def edge_static_artifacts(candidates, width=None, height=None, min_repeat=3, max_confidence=0.60):
    """Reject only the repeated, low-confidence border artifact pattern.

    A legitimate high clear can be near the top of an image, so proximity to a
    border alone is never a rejection condition.
    """
    rejected = set()
    run_start = 0
    while run_start < len(candidates):
        candidate = candidates[run_start]
        if candidate is None:
            run_start += 1
            continue
        run_end = run_start + 1
        while run_end < len(candidates) and _same_point(candidate, candidates[run_end]):
            run_end += 1
        run = candidates[run_start:run_end]
        if len(run) >= min_repeat and _is_near_edge(candidate["image_xy"], width, height):
            average_confidence = sum(item["confidence"] for item in run) / len(run)
            if average_confidence <= max_confidence:
                rejected.update(range(run_start, run_end))
        run_start = run_end
    return rejected


def _base_track_record(row, input_index):
    return {
        "schema_version": DERIVATION_VERSION,
        "input_index": int(input_index),
        "frame": int(row.get("frame", input_index)),
        "time_sec": round(float(row.get("time_sec", 0.0)), 6),
        "trajectory_id": None,
    }


def _raw_shuttle_candidate(row):
    shuttle = row.get("shuttlecock") or {}
    point = shuttle.get("image")
    if not (shuttle.get("accepted") and shuttle.get("status") == "detected" and _valid_point(point)):
        return None
    return {
        "image_xy": [float(point[0]), float(point[1])],
        "confidence": float(shuttle.get("confidence") or 0.0),
        "raw_status": shuttle.get("status"),
    }


def _assign_trajectory_ids(track_rows):
    trajectory_index = 0
    active = False
    for row in track_rows:
        if row["status"] in {"detected", "reconstructed"}:
            if not active:
                trajectory_index += 1
                active = True
            row["trajectory_id"] = f"trajectory_{trajectory_index:04d}"
        else:
            active = False


def _trajectory_turn_seeds(track_rows):
    observed = [item for item in track_rows if item["status"] in {"detected", "reconstructed"}]
    seeds = []
    for before, current, after in zip(observed, observed[1:], observed[2:]):
        if before.get("trajectory_id") != current.get("trajectory_id") or current.get("trajectory_id") != after.get("trajectory_id"):
            continue
        dt_before = current["time_sec"] - before["time_sec"]
        dt_after = after["time_sec"] - current["time_sec"]
        if not (0.02 <= dt_before <= 0.30 and 0.02 <= dt_after <= 0.30):
            continue
        incoming = _velocity(before["image_xy"], current["image_xy"], dt_before)
        outgoing = _velocity(current["image_xy"], after["image_xy"], dt_after)
        speed = min(math.hypot(*incoming), math.hypot(*outgoing))
        cosine = _cosine(incoming, outgoing)
        if speed < 120.0 or cosine > -0.50:
            continue
        confidence = min(0.42, 0.08 + 0.34 * min(current["confidence"], 1.0))
        seeds.append(
            {
                "frame": current["frame"],
                "time_sec": current["time_sec"],
                "source": "trajectory_turn",
                "confidence": confidence,
                "hitter_track_id": None,
                "reason": f"2d_direction_reversal cosine={cosine:.2f}; offline review required",
            }
        )
    return seeds


def _trajectory_heading_change_near_hand_seeds(raw_rows, track_rows):
    """Find a contact-shaped 2D turn only when a detected hand supports it.

    A return seen by a fixed camera often changes from a fast vertical descent
    into a horizontal or diagonal departure.  That is a sharp heading change,
    but not necessarily a 180-degree reversal, so ``_trajectory_turn_seeds``
    deliberately does not cover it.  A player hand close to the *raw detected*
    shuttle position is required to keep an ordinary curved flight from
    becoming a hit candidate.

    This emits a review candidate, never a confirmed contact or a score fact.
    """
    raw_by_frame = {int(item.get("frame", 0)): item for item in raw_rows}
    observed = [item for item in track_rows if item.get("status") == "detected"]
    seeds = []
    for before, current, after in zip(observed, observed[1:], observed[2:]):
        if before.get("trajectory_id") != current.get("trajectory_id") or current.get("trajectory_id") != after.get("trajectory_id"):
            continue
        dt_before = float(current["time_sec"]) - float(before["time_sec"])
        dt_after = float(after["time_sec"]) - float(current["time_sec"])
        if not (0.02 <= dt_before <= 0.30 and 0.02 <= dt_after <= 0.30):
            continue
        incoming = _velocity(before["image_xy"], current["image_xy"], dt_before)
        outgoing = _velocity(current["image_xy"], after["image_xy"], dt_after)
        incoming_speed = math.hypot(*incoming)
        outgoing_speed = math.hypot(*outgoing)
        cosine = _cosine(incoming, outgoing)
        # ``cosine <= 0.45`` is an approximately 63-degree turn.  The speed
        # and hand gates below make this stricter than a generic trajectory
        # curvature test while retaining down-to-side returns.
        if min(incoming_speed, outgoing_speed) < 400.0 or cosine > 0.45:
            continue
        hitter = _hitter_evidence(raw_by_frame.get(int(current["frame"])), current, None)
        if hitter.get("source") != "visible_hand_proximity" or float(hitter.get("distance_px") or math.inf) > 140.0:
            continue
        hand_support = max(0.0, 1.0 - float(hitter["distance_px"]) / 140.0)
        turn_strength = max(0.0, min(1.0, (0.45 - cosine) / 1.45))
        confidence = min(0.42, 0.16 + 0.14 * hand_support + 0.12 * turn_strength)
        seeds.append(
            {
                "frame": int(current["frame"]),
                "time_sec": float(current["time_sec"]),
                "source": "trajectory_heading_change_near_hand",
                "confidence": confidence,
                "hitter_track_id": None,
                "reason": (
                    "fast 2d heading change near visible hand "
                    f"cosine={cosine:.2f}; hand_distance_px={float(hitter['distance_px']):.1f}; "
                    "offline review required"
                ),
            }
        )
    return seeds


def _reconstructed_gap_hand_proximity_seeds(raw_rows, track_rows, max_hand_distance_px=100.0):
    """Turn a short TrackNet gap near a hand into an explicitly inferred cue.

    The shuttle coordinate within a reconstructed gap is not a detector
    measurement.  It is nevertheless useful review evidence when the bounded
    interpolation crosses a visible player's hand.  One strongest cue is kept
    per interpolation gap; it is marked ``motion_constraint_candidate`` so it
    cannot contribute a fabricated observed speed or a confirmed shot count.
    """
    raw_by_frame = {int(item.get("frame", 0)): item for item in raw_rows}
    grouped = {}
    for current in track_rows:
        if current.get("status") != "reconstructed":
            continue
        provenance = current.get("provenance") or {}
        if provenance.get("kind") != "bidirectional_linear_interpolation" or int(provenance.get("gap_frames") or 0) < 2:
            continue
        hitter = _hitter_evidence(raw_by_frame.get(int(current["frame"])), current, None)
        distance = float(hitter.get("distance_px") or math.inf)
        if hitter.get("source") != "visible_hand_proximity" or distance > float(max_hand_distance_px):
            continue
        key = (
            provenance.get("before_frame"),
            provenance.get("after_frame"),
            hitter.get("track_id"),
        )
        existing = grouped.get(key)
        if existing is None or distance < existing["distance"]:
            grouped[key] = {"track": current, "hitter": hitter, "distance": distance, "provenance": provenance}

    seeds = []
    for item in grouped.values():
        current = item["track"]
        support = max(0.0, 1.0 - item["distance"] / float(max_hand_distance_px))
        seeds.append(
            {
                "frame": int(current["frame"]),
                "time_sec": float(current["time_sec"]),
                "source": "reconstructed_gap_hand_proximity",
                "confidence": min(0.22, 0.10 + 0.12 * support),
                "hitter_track_id": None,
                "event_origin": "motion_constraint_candidate",
                "reason": (
                    "short TrackNet gap interpolates near a visible hand "
                    f"distance_px={item['distance']:.1f}; gap_frames={int(item['provenance']['gap_frames'])}; "
                    "inferred candidate, not a detector measurement"
                ),
            }
        )
    return seeds


def _deduplicate_seeds(seeds, min_gap_sec=0.35):
    priority = {
        "spatial_hit_candidate": 3,
        "trajectory_heading_change_near_hand": 2,
        "trajectory_turn": 1,
        "reconstructed_gap_hand_proximity": 0,
    }
    grouped = []
    for seed in sorted(seeds, key=lambda item: item["time_sec"]):
        if grouped and seed["time_sec"] - grouped[-1]["time_sec"] <= min_gap_sec:
            previous = grouped[-1]
            sources = {previous.get("source"), seed.get("source")}
            # A review-supported rapid exchange can be roughly 0.3 seconds
            # apart.  A short reconstructed gap is a distinct *inferred* cue
            # once it is more than 0.20 seconds after a hand-supported turn;
            # only merge it when it is temporally indistinguishable from the
            # observed contact it might be explaining.
            if (
                "reconstructed_gap_hand_proximity" in sources
                and abs(float(seed["time_sec"]) - float(previous["time_sec"])) > 0.20
            ):
                grouped.append(seed)
                continue
            if (priority.get(seed["source"], 0), seed["confidence"]) >= (priority.get(previous["source"], 0), previous["confidence"]):
                grouped[-1] = seed
            continue
        grouped.append(seed)
    return grouped


def _hitter_evidence(raw, track_row, preferred_track_id):
    tracks = ((raw or {}).get("spatial") or {}).get("tracks") or []
    if preferred_track_id:
        matching = next((item for item in tracks if item.get("track_id") == preferred_track_id), None)
        if matching and matching.get("status") == "detected":
            return {
                "track_id": preferred_track_id,
                "confidence": round(min(0.45, float(matching.get("confidence") or 0.0)), 4),
                "source": "spatial_proximity",
                "measurement_status": "detected",
                "zone_id": matching.get("zone_id"),
                "court_xy_m": matching.get("court_xy_m"),
            }
    shuttle_point = (track_row or {}).get("image_xy")
    if not shuttle_point:
        return {"track_id": None, "confidence": 0.0, "source": "no_shuttle_measurement"}
    best = (None, None, math.inf)
    for track in tracks:
        if track.get("status") != "detected":
            continue
        hands = ((track.get("location_evidence") or {}).get("hands_image") or {}).values()
        for hand in hands:
            if _valid_point(hand):
                distance = _distance(shuttle_point, hand)
                if distance < best[2]:
                    best = (track, hand, distance)
    if best[0] is None or best[2] > 150.0:
        return {"track_id": None, "confidence": 0.0, "source": "no_near_visible_hand"}
    return {
        "track_id": best[0].get("track_id"),
        "confidence": round(max(0.0, min(0.42, (1.0 - best[2] / 150.0) * float(best[0].get("confidence") or 0.0))), 4),
        "source": "visible_hand_proximity",
        "distance_px": round(best[2], 2),
        "measurement_status": "detected",
        "zone_id": best[0].get("zone_id"),
        "court_xy_m": best[0].get("court_xy_m"),
    }


def _event_trajectory_evidence(track_rows, hit_time_sec, until_time_sec=None):
    """Describe one shot using its first two post-contact shuttle points.

    ``outbound_speed_px_s`` is intentionally an image-plane value.  A fixed
    monocular camera cannot turn it into an authoritative shuttle speed in
    metres per second without a calibrated 3D reconstruction.
    """
    window_end = min(
        float(hit_time_sec) + 1.8,
        float(until_time_sec) if until_time_sec is not None else math.inf,
    )
    after = [
        item for item in track_rows
        if hit_time_sec <= item["time_sec"] < window_end and item["status"] in {"detected", "reconstructed"}
    ]
    if len(after) < 2:
        return {
            "observation_count": len(after),
            "detected_count": sum(item["status"] == "detected" for item in after),
            "reconstructed_count": sum(item["status"] == "reconstructed" for item in after),
            "outbound_speed_px_s": None,
            "outbound_speed": {
                "value": None,
                "unit": "px/s",
                "basis": "first_two_post_hit_trajectory_points",
                "confidence": 0.0,
                "point_statuses": [item["status"] for item in after],
            },
            "flight_duration_sec": None,
            "quality": "insufficient",
            "limitations": ["Insufficient post-hit shuttle evidence; leave the shot label unknown."],
        }
    first, second = after[0], after[1]
    elapsed = float(second["time_sec"]) - float(first["time_sec"])
    speed = _distance(first["image_xy"], second["image_xy"]) / elapsed if elapsed > 0 else None
    detected_count = sum(item["status"] == "detected" for item in after)
    reconstructed_count = sum(item["status"] == "reconstructed" for item in after)
    speed_confidence = min(float(first["confidence"]), float(second["confidence"]))
    if "reconstructed" in {first["status"], second["status"]}:
        speed_confidence *= 0.45
    return {
        "observation_count": len(after),
        "detected_count": detected_count,
        "reconstructed_count": reconstructed_count,
        "outbound_speed_px_s": round(speed, 2) if speed is not None else None,
        "outbound_speed": {
            "value": round(speed, 2) if speed is not None else None,
            "unit": "px/s",
            "basis": "first_two_post_hit_trajectory_points",
            "confidence": round(max(0.0, min(0.75, speed_confidence)), 4),
            "point_statuses": [first["status"], second["status"]],
            "point_frames": [first["frame"], second["frame"]],
            "point_times_sec": [first["time_sec"], second["time_sec"]],
        },
        "flight_duration_sec": round(after[-1]["time_sec"] - after[0]["time_sec"], 4),
        "image_displacement_px": round(_distance(after[0]["image_xy"], after[-1]["image_xy"]), 2),
        "quality": "moderate" if detected_count >= 3 else "weak",
        "limitations": [
            "2D image-plane motion only; no precise monocular 3D claim.",
            "Reconstructed points are context, not detector measurements.",
        ],
    }


def _missing_shuttle_trajectory():
    return {
        "observation_count": 0,
        "detected_count": 0,
        "reconstructed_count": 0,
        "outbound_speed_px_s": None,
        "outbound_speed": {
            "value": None,
            "unit": "px/s",
            "basis": "no_shuttle_measurement",
            "confidence": 0.0,
            "point_statuses": [],
        },
        "flight_duration_sec": None,
        "quality": "insufficient",
        "limitations": [
            "No accepted shuttle observations in this inferred return interval.",
            "This is a motion-constraint candidate, not a reconstructed ball trajectory.",
        ],
    }


def _match_mode(rows):
    modes = {
        ((row.get("spatial") or {}).get("match") or {}).get("mode")
        for row in rows
        if ((row.get("spatial") or {}).get("match") or {}).get("mode")
    }
    return next(iter(modes)) if len(modes) == 1 else None


def _visible_detected_track_ids(rows, start_time_sec, end_time_sec):
    visible = set()
    for row in rows:
        time_sec = float(row.get("time_sec", 0.0))
        if start_time_sec <= time_sec <= end_time_sec:
            visible.update(
                str(track["track_id"])
                for track in ((row.get("spatial") or {}).get("tracks") or [])
                if track.get("track_id") and track.get("status") == "detected"
            )
    return visible


def _nearest_frame(rows, time_sec):
    if not rows:
        return 0
    nearest = min(rows, key=lambda item: abs(float(item.get("time_sec", 0.0)) - float(time_sec)))
    return int(nearest.get("frame", 0))


def _renumber_events(events):
    for index, event in enumerate(events, start=1):
        event["event_id"] = f"shot_{index:04d}"
    return events


def _stationary_windows(
    track_rows,
    speed_limit_px_s,
    min_duration_sec,
    max_point_gap_sec=0.25,
    court_polygon=None,
    min_detection_confidence=0.35,
):
    """Find conservative, in-court, raw-detection stationary intervals.

    Reconstructed points and unknown gaps are useful trajectory context but
    cannot prove that a shuttle landed.  Requiring an available court polygon
    also prevents a fixed scoreboard/logo point outside the court from ending
    a rally.
    """
    if not court_polygon:
        return []
    observed = [
        item for item in track_rows
        if item.get("status") == "detected"
        and float(item.get("confidence") or 0.0) >= min_detection_confidence
        and _point_in_or_near_polygon(item.get("image_xy"), court_polygon)
    ]
    windows = []
    run_start = None
    previous = None
    for current in observed:
        if previous is None:
            previous = current
            continue
        elapsed = float(current["time_sec"]) - float(previous["time_sec"])
        speed = _distance(previous["image_xy"], current["image_xy"]) / elapsed if elapsed > 0 else math.inf
        slow_and_continuous = 0 < elapsed <= max_point_gap_sec and speed <= speed_limit_px_s
        if slow_and_continuous:
            run_start = previous if run_start is None else run_start
        elif run_start is not None:
            if float(previous["time_sec"]) - float(run_start["time_sec"]) >= min_duration_sec:
                windows.append({"start_time_sec": float(run_start["time_sec"]), "end_time_sec": float(previous["time_sec"])})
            run_start = None
        previous = current
    if run_start is not None and previous is not None:
        if float(previous["time_sec"]) - float(run_start["time_sec"]) >= min_duration_sec:
            windows.append({"start_time_sec": float(run_start["time_sec"]), "end_time_sec": float(previous["time_sec"])})
    return windows


def _boundary_between_events(
    track_rows,
    stationary_windows,
    previous_time,
    next_time,
    max_unobserved_sec,
    terminal_candidates=None,
):
    for candidate in terminal_candidates or []:
        if previous_time < float(candidate["time_sec"]) < next_time:
            return {
                "reason": f"terminal_evidence_{candidate.get('source', 'candidate')}",
                "end_time_sec": float(candidate["time_sec"]),
                "confidence": candidate.get("confidence"),
            }
    for window in stationary_windows:
        if previous_time < window["start_time_sec"] < next_time:
            return {
                "reason": "shuttle_stationary_or_slow",
                "end_time_sec": window["start_time_sec"],
            }
    # A gap is unknown, not evidence that one rally ended and another began.
    # Keep this parameter for callers on the previous public signature.
    _ = (track_rows, max_unobserved_sec)
    return None


def _terminal_boundary_after_event(stationary_windows, last_event_time, terminal_candidates=None):
    for candidate in terminal_candidates or []:
        if float(candidate["time_sec"]) > last_event_time:
            return {
                "reason": f"terminal_evidence_{candidate.get('source', 'candidate')}",
                "end_time_sec": float(candidate["time_sec"]),
                "confidence": candidate.get("confidence"),
            }
    for window in stationary_windows:
        if window["start_time_sec"] > last_event_time:
            return {
                "reason": "shuttle_stationary_or_slow",
                "end_time_sec": window["start_time_sec"],
            }
    return None


def _court_polygon_from_metadata(analysis_dir):
    metadata_path = Path(analysis_dir) / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        corners = json.loads(metadata_path.read_text(encoding="utf-8")).get("court", {}).get("corners")
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(corners, list) or len(corners) < 3:
        return None
    polygon = [[float(point[0]), float(point[1])] for point in corners if _valid_point(point)]
    return polygon if len(polygon) >= 3 else None


def _point_in_or_near_polygon(point, polygon, margin=24.0):
    if not _valid_point(point):
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    for left, right in zip(polygon, [*polygon[1:], polygon[0]]):
        x1, y1 = float(left[0]), float(left[1])
        x2, y2 = float(right[0]), float(right[1])
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        if _distance_to_segment((x, y), (x1, y1), (x2, y2)) <= margin:
            return True
    return inside


def _distance_to_segment(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return _distance(point, start)
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return _distance(point, (start[0] + ratio * dx, start[1] + ratio * dy))


def _propose_from_space(event, following):
    trajectory = event["trajectory"]
    hitter_id = event["hitter"].get("track_id")
    receiver_id = event["receiver"].get("track_id")
    if not hitter_id or not receiver_id or following is None:
        return _proposal("unknown", 0.0, "No separately supported hitter and receiver; human review required.")
    # Receiver is the next hit candidate. Its location evidence is used only as
    # a spatial cue, never as a forced event attribution.
    receiver_zone = _track_zone_from_event(following, receiver_id)
    hitter_zone = _track_zone_from_event(event, hitter_id)
    duration = trajectory.get("flight_duration_sec")
    speed = trajectory.get("outbound_speed_px_s")
    confidence = min(0.45, event["time_confidence"] * max(event["hitter"].get("confidence", 0.0), event["receiver"].get("confidence", 0.0)))
    if hitter_zone and receiver_zone and hitter_zone.startswith("front_") and receiver_zone.startswith("rear_"):
        return _proposal("lift", confidence, "Front-court hitter to next rear-court receiver is a lift/clear candidate.")
    if hitter_zone and receiver_zone and hitter_zone.startswith("rear_") and receiver_zone.startswith("front_"):
        if speed is not None and duration is not None and speed >= 1000 and duration <= 0.9:
            return _proposal("smash", confidence, "Rear-to-front, short fast flight is a smash candidate.")
        return _proposal("drop", confidence, "Rear-court hitter to next front-court receiver is a drop candidate.")
    if speed is not None and duration is not None and speed >= 700 and duration <= 0.9:
        return _proposal("drive", confidence * 0.8, "Short, fast 2D flight is a drive candidate.")
    return _proposal("unknown", confidence * 0.5, "Spatial evidence does not distinguish the shot type reliably.")


def _track_zone_from_event(event, track_id):
    # Current generated events store only compact attribution. Spatial zones are
    # intentionally not repeated as fact when the next hit is weak.
    return (event.get("hitter") or {}).get("zone_id") if event.get("hitter", {}).get("track_id") == track_id else None


def _proposal(label, confidence, rationale):
    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "status": "needs_human_review",
        "rationale": rationale,
    }


def _row_at_frame(rows, frame):
    return next((item for item in rows if int(item.get("frame", -1)) == int(frame)), None)


def _same_point(left, right, tolerance=1.0):
    if left is None or right is None:
        return False
    return _distance(left["image_xy"], right["image_xy"]) <= tolerance


def _is_near_edge(point, width, height, margin=18.0):
    if not _valid_point(point):
        return False
    x, y = point
    # Dimensions may not be available for historical runs.  Top-edge evidence
    # remains safe to inspect, while right/bottom edges require a known image size.
    if x <= margin or y <= margin:
        return True
    return bool((width and x >= width - margin) or (height and y >= height - margin))


def _valid_point(point):
    try:
        return point is not None and len(point) >= 2 and math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
    except (TypeError, ValueError):
        return False


def _distance(left, right):
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _velocity(left, right, seconds):
    return ((float(right[0]) - float(left[0])) / seconds, (float(right[1]) - float(left[1])) / seconds)


def _cosine(left, right):
    denominator = math.hypot(*left) * math.hypot(*right)
    return 1.0 if denominator == 0 else (left[0] * right[0] + left[1] * right[1]) / denominator


def infer_fps(rows):
    times = [float(row.get("time_sec", 0.0)) for row in rows]
    deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
    return 1.0 / median(deltas) if deltas else None


def video_dimensions(analysis_dir):
    metadata_path = Path(analysis_dir) / "metadata.json"
    if not metadata_path.is_file():
        return None, None
    try:
        video = json.loads(metadata_path.read_text(encoding="utf-8")).get("video", {})
        return video.get("width"), video.get("height")
    except (OSError, ValueError, TypeError):
        return None, None


def load_jsonl(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path, records: Iterable[dict]):
    path = Path(path)
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            destination.write("\n")
