"""Metrics for evaluating persistent four-player tracking from JSON evidence."""

from __future__ import annotations

from math import hypot


def evaluate_doubles_tracking(annotation_rows, prediction_rows, match_distance_m=1.25):
    """Measure detection recall, ID switches, gaps, occlusion recovery and hits.

    Ground truth uses stable human ``person_id``. Predictions use system
    ``track_id``.  The evaluator only matches records whose temporal state is
    ``detected``; predicted/missing locations are intentionally excluded from
    recall and hit-attribution credit.
    """
    prediction_by_frame = {int(row.get("frame", -1)): row for row in prediction_rows}
    previous_track = {}
    missing_run = {}
    longest_missing = {}
    detected_count = 0
    ground_truth_count = 0
    id_switches = 0
    assignments_by_frame = {}
    occlusion_starts = {}
    occlusion_recoveries = 0
    occlusion_opportunities = 0
    hit_total = 0
    hit_correct = 0

    for annotation in sorted(annotation_rows, key=lambda item: int(item.get("frame", 0))):
        frame = int(annotation.get("frame", 0))
        prediction = prediction_by_frame.get(frame, {})
        tracks = ((prediction.get("spatial") or {}).get("tracks") or [])
        detected_tracks = [
            item for item in tracks
            if item.get("status") == "detected" and _point(item.get("court_xy_m"))
        ]
        players = annotation.get("players") or []
        assignments = _assign_players(players, detected_tracks, match_distance_m)
        assignments_by_frame[frame] = assignments
        occluded = set(annotation.get("occluded_person_ids") or [])
        for player in players:
            person_id = player.get("person_id")
            if not person_id:
                continue
            if person_id in occluded:
                occlusion_starts.setdefault(person_id, frame)
                continue
            ground_truth_count += 1
            track_id = assignments.get(person_id)
            if track_id:
                detected_count += 1
                if previous_track.get(person_id) and previous_track[person_id] != track_id:
                    id_switches += 1
                if person_id in occlusion_starts:
                    occlusion_opportunities += 1
                    if previous_track.get(person_id) == track_id:
                        occlusion_recoveries += 1
                    occlusion_starts.pop(person_id, None)
                previous_track[person_id] = track_id
                missing_run[person_id] = 0
            else:
                missing_run[person_id] = missing_run.get(person_id, 0) + 1
                longest_missing[person_id] = max(longest_missing.get(person_id, 0), missing_run[person_id])

        for hit in annotation.get("hits") or []:
            person_id = hit.get("person_id")
            if not person_id or person_id in occluded:
                continue
            hit_total += 1
            predicted_hitter = _candidate_hitter(prediction)
            inverse = {track_id: player_id for player_id, track_id in assignments.items()}
            if inverse.get(predicted_hitter) == person_id:
                hit_correct += 1

    return {
        "ground_truth_player_observations": ground_truth_count,
        "detected_player_observations": detected_count,
        "player_recall": _ratio(detected_count, ground_truth_count),
        "player_miss_rate": 1.0 - _ratio(detected_count, ground_truth_count),
        "id_switches": id_switches,
        "longest_track_interruption_frames": max(longest_missing.values(), default=0),
        "longest_track_interruption_by_person": dict(sorted(longest_missing.items())),
        "occlusion_recovery_opportunities": occlusion_opportunities,
        "occlusion_recoveries_same_track": occlusion_recoveries,
        "occlusion_recovery_rate": _ratio(occlusion_recoveries, occlusion_opportunities),
        "hit_attribution_ground_truth_count": hit_total,
        "hit_attribution_correct_count": hit_correct,
        "hit_attribution_accuracy": _ratio(hit_correct, hit_total),
        "measurement_policy": "only status=detected receives recall or attribution credit",
    }


def _assign_players(players, tracks, gate):
    candidates = []
    for player in players:
        if not player.get("person_id") or not _point(player.get("court_xy_m")):
            continue
        for track in tracks:
            distance = _distance(player["court_xy_m"], track["court_xy_m"])
            if distance <= gate:
                candidates.append((distance, player["person_id"], track["track_id"]))
    remaining_people = {item.get("person_id") for item in players if item.get("person_id")}
    remaining_tracks = {item.get("track_id") for item in tracks if item.get("track_id")}
    assignments = {}
    for _distance_value, person_id, track_id in sorted(candidates):
        if person_id not in remaining_people or track_id not in remaining_tracks:
            continue
        assignments[person_id] = track_id
        remaining_people.remove(person_id)
        remaining_tracks.remove(track_id)
    return assignments


def _candidate_hitter(prediction):
    events = ((prediction.get("spatial") or {}).get("hit_events") or [])
    candidates = [item for item in events if item.get("status") == "candidate" and item.get("hitter_track_id")]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("confidence") or 0.0)).get("hitter_track_id")


def _point(value):
    return value is not None and len(value) >= 2


def _distance(left, right):
    return hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None
