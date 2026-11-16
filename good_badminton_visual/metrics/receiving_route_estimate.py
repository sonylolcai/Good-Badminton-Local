"""Candidate receiving-region statistics from new per-player pose evidence.

This is an uncalibrated kinematic heuristic, never a verified contact detector.
It does not consume legacy shuttle candidates, running scores or LLM grades.
"""
from collections import Counter
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
from .heatmap_grid import receiving_heatmap

VERSION = 'pose-wrist-pair.v4'
SOURCE = 'pose_wrist_pair_estimate.v1'
HALF_SLOT_SOURCES = {'half_court_slot_opening', 'half_court_slot_continuity',
                     'half_court_slot_replacement'}
POLICY = {'min_body_confidence': .5, 'min_location_confidence': .5,
          'min_identity_confidence': .7, 'min_keypoint_confidence': .5,
          'min_torso_pixels': 10, 'max_pose_join_gap_seconds': .15,
          'min_wrist_speed_torso_lengths_per_second': 4,
          'same_player_peak_suppression_seconds': .65,
          'min_pair_gap_seconds': .18, 'max_pair_gap_seconds': 2,
          'pairing': 'adjacent_global_pose_peaks_on_opposite_court_ends_only',
          'occupancy': 'half_court_slot_motion_never_joined_across_occupant_segments',
          'pair_occupancy': 'both_slots_continuously_observed_with_same_occupant_between_peaks',
          'missing_slots': 'absent_slots_are_unknown_at_every_observed_source_timestamp',
          'validation': 'uncalibrated_candidates_not_confirmed_contacts'}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _point(value):
    return (isinstance(value, (list, tuple)) and len(value) >= 2
            and all(_finite(v) for v in value[:2]))


def _slot_occupant(track):
    """Pose absence is not replacement; missing/untrusted tracking is unknown."""
    association = track.get('association') or {}
    segment = track.get('occupancy_segment')
    end = track.get('court_end')
    confidence = (track.get('confidence'),
                  (track.get('location_evidence') or {}).get('confidence'),
                  association.get('identity_confidence'))
    if (track.get('status') != 'detected' or track.get('identity_scope') != 'half_court_slots.v1'
            or association.get('source') not in HALF_SLOT_SOURCES
            or type(segment) is not int or segment < 1 or end not in {'end_a', 'end_b'}
            or track.get('slot_court_end') != end
            or not all(_finite(v) for v in confidence)
            or confidence[0] < .5 or confidence[1] < .5 or confidence[2] < .7):
        return None
    return segment, end


def _slot_continuous(sample, start, end, timelines):
    if sample['identity_scope'] != 'half_court_slots.v1':
        return True
    times, occupants = timelines[sample['track_id']]
    first, stop = bisect_right(times, start) - 1, bisect_right(times, end)
    gap = POLICY['max_pose_join_gap_seconds']
    if first < 0 or start - times[first] > gap or end - times[stop - 1] > gap:
        return False
    expected = sample['occupancy_segment'], sample['end']
    return all(occupants[i] == expected and (i == first or times[i] - times[i - 1] <= gap)
               for i in range(first, stop))


def _pose_sample(row, track):
    if not isinstance(track, dict) or not track.get('track_id') or track.get('status') != 'detected':
        return None
    association = track.get('association') or {}
    source = association.get('source')
    scope = track.get('identity_scope')
    segment = 0
    if source in HALF_SLOT_SOURCES:
        # These identifiers represent occupied slots, not verified people.
        # A returning/replacement occupant inherits the slot but not its motion.
        segment = track.get('occupancy_segment')
        if (scope != 'half_court_slots.v1' or type(segment) is not int or segment < 1
                or track.get('slot_court_end') != track.get('court_end')):
            return None
    elif (scope == 'half_court_slots.v1'
          or source not in {'bytetrack', 'court_association', 'roster_bootstrap'}):
        # Untrusted reconnects remain excluded in legacy person-identity mode.
        return None
    identity = association.get('identity_confidence', {'bytetrack': .95, 'court_association': .85, 'roster_bootstrap': .95}.get(association.get('source'), 0))
    confidence = track.get('confidence')
    location = (track.get('location_evidence') or {}).get('confidence')
    if (not all(_finite(v) for v in (identity, confidence, location))
            or identity < .7 or confidence < .5 or location < .5):
        return None
    time = row.get('time_sec')
    frame = row.get('frame')
    xy = track.get('court_xy_m')
    end = track.get('court_end')
    if (not _finite(time) or time < 0 or type(frame) is not int or frame < 0
            or not _point(xy) or not 0 <= xy[0] <= 6.1 or not 0 <= xy[1] <= 13.4
            or end not in {'end_a', 'end_b'}):
        return None
    local_y = 6.7 - xy[1] if end == 'end_a' else xy[1] - 6.7
    if not 0 <= local_y <= 6.7:
        return None
    pose = track.get('pose') or {}
    points, scores = pose.get('keypoints_image'), pose.get('keypoint_scores')
    if (pose.get('is_current_measurement') is not True or pose.get('format') != 'coco17_image_v1'
            or not isinstance(points, list) or len(points) < 17
            or not isinstance(scores, list) or len(scores) < 17):
        return None
    required = (5, 6, 11, 12)
    if any(not _point(points[i]) or not _finite(scores[i]) or scores[i] < .5 for i in required):
        return None
    shoulder = [(points[5][i] + points[6][i]) / 2 for i in (0, 1)]
    hip = [(points[11][i] + points[12][i]) / 2 for i in (0, 1)]
    scale = math.dist(shoulder, hip)
    if scale < POLICY['min_torso_pixels']:
        return None
    wrists = {i: [(points[i][j] - shoulder[j]) / scale for j in (0, 1)]
              for i in (9, 10) if _point(points[i]) and _finite(scores[i]) and scores[i] >= .5}
    if not wrists:
        return None
    return {'track_id': str(track['track_id']), 'time': time, 'frame': frame,
            'wrists': wrists, 'xy': list(xy[:2]), 'end': end,
            'identity_scope': scope, 'occupancy_segment': segment}


def estimate_receiving_routes(detections_path):
    path = Path(detections_path)
    digest, samples, seen = hashlib.sha256(), {}, set()
    occupancy = {}
    observed_times = set()
    if path.is_file():
        with path.open('rb') as source:
            for raw in source:
                digest.update(raw)
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                time = row.get('time_sec')
                if _finite(time) and time >= 0:
                    observed_times.add(time)
                for track in (row.get('spatial') or {}).get('tracks') or []:
                    if isinstance(track, dict) and track.get('track_id') and _finite(time) and time >= 0:
                        history = occupancy.setdefault(str(track['track_id']), {})
                        token = _slot_occupant(track)
                        # Conflicting duplicate observations do not establish continuity.
                        history[time] = token if time not in history or history[time] == token else None
                    sample = _pose_sample(row, track)
                    if sample is None:
                        continue
                    group = (sample['track_id'], sample['identity_scope'] or '', sample['occupancy_segment'])
                    key = (*group, sample['time'])
                    if key in seen:
                        continue
                    seen.add(key)
                    samples.setdefault(group, []).append(sample)
    # A source row with no entry for a known slot is explicit unknown evidence,
    # even if its neighbors are closer than the permitted measurement gap.
    times = sorted(observed_times)
    slot_ids = {group[0] for group in samples if group[1] == 'half_court_slots.v1'}
    timelines = {tid: (times, [occupancy[tid].get(t) for t in times]) for tid in slot_ids}
    peaks = []
    for _, rows in sorted(samples.items()):
        rows.sort(key=lambda item: item['time'])
        signals = []
        for left, middle, right in zip(rows, rows[1:], rows[2:]):
            if (left['end'] != middle['end'] or middle['end'] != right['end']
                    or not 0 < middle['time'] - left['time'] <= POLICY['max_pose_join_gap_seconds']
                    or not 0 < right['time'] - middle['time'] <= POLICY['max_pose_join_gap_seconds']
                    or not _slot_continuous(middle, left['time'], right['time'], timelines)):
                continue
            hands = set(left['wrists']) & set(middle['wrists']) & set(right['wrists'])
            if not hands:
                continue
            seconds = right['time'] - left['time']
            speed = max(math.dist(left['wrists'][hand], right['wrists'][hand]) / seconds for hand in hands)
            signals.append({**middle, 'signal': speed})
        selected = []
        for left, middle, right in zip(signals, signals[1:], signals[2:]):
            if (middle['signal'] < POLICY['min_wrist_speed_torso_lengths_per_second']
                    or middle['signal'] < left['signal'] or middle['signal'] <= right['signal']
                    or middle['time'] - left['time'] > POLICY['max_pose_join_gap_seconds']
                    or right['time'] - middle['time'] > POLICY['max_pose_join_gap_seconds']
                    or not _slot_continuous(middle, left['time'], right['time'], timelines)):
                continue
            if selected and middle['time'] - selected[-1]['time'] < POLICY['same_player_peak_suppression_seconds']:
                if middle['signal'] > selected[-1]['signal']:
                    selected[-1] = middle
            else:
                selected.append(middle)
        peaks.extend(selected)
    peaks.sort(key=lambda item: (item['time'], item['track_id']))
    events = []
    for actor, receiver in zip(peaks, peaks[1:]):
        seconds = receiver['time'] - actor['time']
        if (actor['end'] == receiver['end'] or
                not POLICY['min_pair_gap_seconds'] <= seconds <= POLICY['max_pair_gap_seconds']):
            continue
        if not all(_slot_continuous(sample, actor['time'], receiver['time'], timelines)
                   for sample in (actor, receiver)):
            continue
        x, y = receiver['xy']
        local_y = 6.7 - y if receiver['end'] == 'end_a' else y - 6.7
        zone = min(2, int(local_y / 6.7 * 3)) * 3 + min(2, int(x / 6.1 * 3))
        events.append({'event_id': f"pose-{actor['track_id']}-{actor['frame']}-{receiver['track_id']}-{receiver['frame']}",
            'actor_track_id': actor['track_id'], 'receiver_track_id': receiver['track_id'],
            'source_time_sec': actor['time'], 'receiver_time_sec': receiver['time'],
            'source_frame': actor['frame'], 'receiver_frame': receiver['frame'],
            'actor_identity_scope': actor['identity_scope'],
            'receiver_identity_scope': receiver['identity_scope'],
            'actor_occupancy_segment': actor['occupancy_segment'],
            'receiver_occupancy_segment': receiver['occupancy_segment'],
            'receiver_zone': zone, 'receiver_xy_m': receiver['xy'],
            'status': 'candidate', 'target_kind': 'estimated_receiver_action_position',
            'source': SOURCE})
    return {'schema_version': 'pose-receiving-route-estimates.v1', 'algorithm_version': VERSION,
            'source': SOURCE, 'source_sha256': digest.hexdigest() if path.is_file() else None,
            'policy': dict(POLICY), 'pose_peak_count': len(peaks), 'events': events}


def receiving_route_estimate_profile(track_id, estimates):
    events = [row for row in estimates['events'] if row['actor_track_id'] == track_id]
    counts = Counter(row['receiver_zone'] for row in events)
    samples = len(events)
    value = -sum((count / samples) * math.log(count / samples) for count in counts.values()) / math.log(9) if samples else None
    return {'schema_version': 'shot-route-profile.v1', 'track_id': track_id,
            'source': SOURCE, 'estimator_version': VERSION,
            'metric': 'opponent_receiving_region_diversity',
            'status': 'estimated' if samples else 'not_collected',
            'sample_count': samples, 'minimum_samples': 1, 'sample_quality': 'small_candidate_sample' if samples < 12 else 'unvalidated_candidate_sample',
            'value': round(max(0, min(1, value)), 4) if value is not None else None,
            'normalized_value': round(max(0, min(100, value * 100)), 1) if value is not None else None,
            'unit': 'normalized_entropy', 'competitive_rating': False,
            'distribution': {str(zone): count for zone, count in sorted(counts.items())},
            'receiving_heatmap': receiving_heatmap(events),
            'event_ids': [row['event_id'] for row in events], 'events': events,
            'source_sha256': estimates.get('source_sha256'), 'policy': estimates['policy'],
            'limitation': 'Unvalidated wrist-action pairing. Contacts and receiver identity are estimates, not confirmed shot chains.'}
