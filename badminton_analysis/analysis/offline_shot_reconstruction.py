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


DERIVED_DIRNAME = "derived"
SHUTTLE_TRACK_FILENAME = "shuttle_tracks_v2.jsonl"
SHOT_EVENT_FILENAME = "shot_events_v2.jsonl"
DERIVATION_VERSION = "2.0"


def generate_offline_artifacts(detections_path, output_dir=None, fps=None):
    """Create versioned derived artifacts next to immutable detector output."""
    detections_path = Path(detections_path)
    rows = load_jsonl(detections_path)
    if not rows:
        raise ValueError("detections.jsonl is empty; offline reconstruction has no evidence")
    output_dir = Path(output_dir) if output_dir else detections_path.parent / DERIVED_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = float(fps or infer_fps(rows) or 30.0)
    width, height = video_dimensions(detections_path.parent)
    track_rows = reconstruct_shuttle_track(rows, fps=fps, width=width, height=height)
    events = build_shot_events(rows, track_rows, fps=fps)
    track_path = output_dir / SHUTTLE_TRACK_FILENAME
    event_path = output_dir / SHOT_EVENT_FILENAME
    write_jsonl(track_path, track_rows)
    write_jsonl(event_path, events)
    return {
        "version": DERIVATION_VERSION,
        "tracks_path": str(track_path),
        "events_path": str(event_path),
        "frame_count": len(track_rows),
        "event_count": len(events),
        "fps": fps,
        "policy": "raw detections are immutable; only short, bounded gaps may be reconstructed",
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
    seeds = _deduplicate_seeds(seeds)
    events = []
    for index, seed in enumerate(seeds, start=1):
        raw = _row_at_frame(raw_rows, seed["frame"])
        track = by_frame.get(seed["frame"])
        hitter = _hitter_evidence(raw, track, seed.get("hitter_track_id"))
        events.append(
            {
                "schema_version": DERIVATION_VERSION,
                "event_id": f"shot_{index:04d}",
                "frame": seed["frame"],
                "hit_time_sec": round(seed["time_sec"], 6),
                "status": "candidate",
                "candidate_source": seed["source"],
                "candidate_reason": seed.get("reason"),
                "event_origin": "machine_candidate",
                "time_confidence": round(min(0.55, float(seed["confidence"])), 4),
                "trajectory": _event_trajectory_evidence(track_rows, seed["time_sec"]),
                "hitter": hitter,
                "receiver": {"track_id": None, "confidence": 0.0, "source": "not_yet_inferred"},
                "proposal": {"label": "unknown", "confidence": 0.0, "status": "needs_human_review"},
                "decision": {"status": "pending", "label_source": "machine", "eligible_for_statistics": False},
            }
        )

    for current, following in zip(events, events[1:]):
        next_hitter = following["hitter"]
        if next_hitter.get("track_id") and next_hitter["track_id"] != current["hitter"].get("track_id"):
            current["receiver"] = {
                "track_id": next_hitter["track_id"],
                "confidence": round(min(0.45, next_hitter.get("confidence", 0.0)), 4),
                "source": "next_hit_candidate",
            }
        current["proposal"] = _propose_from_space(current, following)
    if events:
        events[-1]["proposal"] = _propose_from_space(events[-1], None)
    return events


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


def _deduplicate_seeds(seeds, min_gap_sec=0.35):
    priority = {"spatial_hit_candidate": 2, "trajectory_turn": 1}
    grouped = []
    for seed in sorted(seeds, key=lambda item: item["time_sec"]):
        if grouped and seed["time_sec"] - grouped[-1]["time_sec"] <= min_gap_sec:
            previous = grouped[-1]
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


def _event_trajectory_evidence(track_rows, hit_time_sec):
    after = [item for item in track_rows if hit_time_sec <= item["time_sec"] <= hit_time_sec + 1.8 and item["status"] in {"detected", "reconstructed"}]
    if len(after) < 2:
        return {
            "observation_count": len(after),
            "detected_count": sum(item["status"] == "detected" for item in after),
            "reconstructed_count": sum(item["status"] == "reconstructed" for item in after),
            "outbound_speed_px_s": None,
            "flight_duration_sec": None,
            "quality": "insufficient",
            "limitations": ["Insufficient post-hit shuttle evidence; leave the shot label unknown."],
        }
    speeds = []
    for before, current in zip(after, after[1:]):
        elapsed = current["time_sec"] - before["time_sec"]
        if elapsed > 0:
            speeds.append(_distance(before["image_xy"], current["image_xy"]) / elapsed)
    detected_count = sum(item["status"] == "detected" for item in after)
    reconstructed_count = sum(item["status"] == "reconstructed" for item in after)
    return {
        "observation_count": len(after),
        "detected_count": detected_count,
        "reconstructed_count": reconstructed_count,
        "outbound_speed_px_s": round(max(speeds) if speeds else 0.0, 2),
        "flight_duration_sec": round(after[-1]["time_sec"] - after[0]["time_sec"], 4),
        "image_displacement_px": round(_distance(after[0]["image_xy"], after[-1]["image_xy"]), 2),
        "quality": "moderate" if detected_count >= 3 else "weak",
        "limitations": [
            "2D image-plane motion only; no precise monocular 3D claim.",
            "Reconstructed points are context, not detector measurements.",
        ],
    }


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
