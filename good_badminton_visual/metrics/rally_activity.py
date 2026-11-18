"""Evidence-gated active-play intervals; a candidate after an end is NOT a start.

No score inference. Human terminal facts and machine candidates stay distinct.
"""
import math
from bisect import bisect_left, bisect_right
from .court_contact_gate import inside_quad

VERSION = 'rally-activity.v1'
POLICY = {'restart_pair_min_sec': .18, 'restart_pair_max_sec': 3.0,
          'flight_min_points': 3, 'flight_min_span_px': 40,
          'flight_max_point_gap_sec': .35, 'settle_min_span_sec': .18,
          'settle_radius_px': 24, 'settle_hand_exclusion_px': 60,
          'settle_lookahead_sec': .55, 'descent_lookback_sec': .6}


def _finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def _point(v):
    return isinstance(v, (list, tuple)) and len(v) >= 2 and all(_finite(x) for x in v[:2])


def _observed(rows):
    return sorted((r for r in rows if r.get('status') == 'detected' and _finite(r.get('time_sec'))
                   and _point(r.get('image_xy')) and (r.get('confidence') or 0) >= .35), key=lambda r: r['time_sec'])


def find_settle_terminals(track_rows, person_rows, polygon, pixel_scale=1):
    """Fast descent followed by a compact measured floor-plane cluster.

    This is an explainable terminal candidate, not a monocular height proof.
    No interpolated or outside-court reacquisition point supplies the cluster.
    """
    points = [r for r in _observed(track_rows) if inside_quad(r['image_xy'], polygon)]
    times = [p['time_sec'] for p in points]
    people = sorted((r for r in person_rows if _finite(r.get('time_sec'))), key=lambda r: r['time_sec'])
    person_times = [r['time_sec'] for r in people]
    terminals = []
    for index, current in enumerate(points):
        t, xy = current['time_sec'], current['image_xy']
        before = points[bisect_left(times, t - POLICY['descent_lookback_sec']):index]
        if len(before) < 2:
            continue
        left, right = before[-2:]
        dt = right['time_sec'] - left['time_sec']
        if not 0 < dt <= .16 or t - right['time_sec'] > .35:
            continue
        vy = (right['image_xy'][1] - left['image_xy'][1]) / dt
        speed = math.dist(left['image_xy'], right['image_xy']) / dt
        if vy < 100 * pixel_scale or speed < 200 * pixel_scale:
            continue
        if xy[1] < right['image_xy'][1] - 12 * pixel_scale:
            continue
        after = points[index:bisect_right(times, t + POLICY['settle_lookahead_sec'])]
        cluster = []
        for p in after:
            if math.dist(xy, p['image_xy']) > POLICY['settle_radius_px'] * pixel_scale:
                break
            if cluster and p['time_sec'] - cluster[-1]['time_sec'] > .16:
                break
            cluster.append(p)
        if len(cluster) < 3 or cluster[-1]['time_sec'] - t < POLICY['settle_min_span_sec']:
            continue
        nearby = people[bisect_left(person_times, t - .150001):bisect_right(person_times, t + .150001)]
        hands = [h for r in nearby for p in (r.get('spatial') or {}).get('tracks') or []
                 if p.get('status') == 'detected'
                 for h in ((p.get('location_evidence') or {}).get('hands_image') or {}).values() if _point(h)]
        hand_distance = min((math.dist(xy, h[:2]) for h in hands), default=math.inf)
        if hand_distance <= POLICY['settle_hand_exclusion_px'] * pixel_scale:
            continue
        if terminals and t - terminals[-1]['time_sec'] < 1:
            continue
        terminals.append({'time_sec': t, 'source': 'descent_then_compact_settle', 'status': 'candidate',
                          'confidence': .72, 'eligible_for_scoring': False,
                          'evidence': {'cluster_points': len(cluster), 'cluster_end_sec': cluster[-1]['time_sec'],
                                       'descent_speed_px_s': round(speed, 3), 'descent_vy_px_s': round(vy, 3),
                                       'nearest_hand_distance_px': round(hand_distance, 3) if math.isfinite(hand_distance) else None}})
    return terminals


def _actor(event):
    return event.get('actor_track_id') or (event.get('hitter') or {}).get('track_id')


def _position(event):
    return event.get('actor_xy_m') or (event.get('hitter') or {}).get('court_xy_m')


def _restart_pair(a, b, observed, pixel_scale, observed_times):
    dt = b['hit_time_sec'] - a['hit_time_sec']
    if not POLICY['restart_pair_min_sec'] <= dt <= POLICY['restart_pair_max_sec']:
        return False
    pa, pb = _position(a), _position(b)
    if not _actor(a) or not _actor(b) or _actor(a) == _actor(b) or not _point(pa) or not _point(pb):
        return False
    if not all(0 <= p[0] <= 6.1 and 0 <= p[1] <= 13.4 for p in (pa, pb)) or (pa[1] < 6.7) == (pb[1] < 6.7):
        return False
    run = []
    for p in observed[bisect_left(observed_times, a['hit_time_sec']):bisect_right(observed_times, b['hit_time_sec'])]:
        if run and p['time_sec'] - run[-1]['time_sec'] > POLICY['flight_max_point_gap_sec']:
            run = []
        run.append(p)
        if len(run) >= 3 and max(math.dist(p['image_xy'], q['image_xy']) for q in run[:-1]) >= POLICY['flight_min_span_px'] * pixel_scale:
            return True
    return False


def derive_rally_activity(events, track_rows, terminals, *, pixel_scale=1, video_end_sec=None):
    ordered = sorted((dict(e) for e in events if _finite(e.get('hit_time_sec'))), key=lambda e: e['hit_time_sec'])
    observed = _observed(track_rows)
    observed_times = [p['time_sec'] for p in observed]
    end = video_end_sec if _finite(video_end_sec) else max([r['time_sec'] for r in track_rows if _finite(r.get('time_sec'))] + [0])
    human = [t for t in terminals if t.get('source') in {'human_review', 'user_reported_interval'} and _finite(t.get('time_sec'))]
    strong_sources = {'descent_then_compact_settle', 'ground_facing_boundary_exit', 'implausible_reacquisition_after_descent', 'shuttle_stationary_or_slow'}
    automatic = [t for t in terminals if t.get('source') in strong_sources and (t.get('confidence') or 0) >= .70
                 and _finite(t.get('time_sec')) and not any(abs(t['time_sec'] - h['time_sec']) <= .6 for h in human)]
    boundaries = []
    for t in sorted(human + automatic, key=lambda t: t['time_sec']):
        if 0 <= t['time_sec'] <= end and (not boundaries or t['time_sec'] - boundaries[-1]['time_sec'] > .05):
            boundaries.append(t)
    accepted = [e for e in ordered if not e.get('excluded')]
    assignments = {e['event_id']: {'rally_phase': 'awaiting_restart', 'rally_id': None} for e in ordered}
    rallies = []
    lower = -1e-6
    for interval, terminal in enumerate([*boundaries, None]):
        upper = terminal['time_sec'] if terminal else end
        candidates = [e for e in accepted if lower < e['hit_time_sec'] <= upper]
        pair = next(((a, b) for a, b in zip(candidates, candidates[1:]) if _restart_pair(a, b, observed, pixel_scale, observed_times)), None)
        if pair:
            # At video entry we may already be inside a rally; do not invent a serve.
            start = candidates[0]['hit_time_sec'] if interval == 0 else pair[0]['hit_time_sec']
            group = [e for e in candidates if start <= e['hit_time_sec'] <= upper]
            rid = f'activity_rally_{len(rallies) + 1:04d}'
            for index, e in enumerate(group, 1):
                assignments[e['event_id']] = {'rally_phase': 'in_rally', 'rally_id': rid, 'shot_index_in_rally': index}
            rallies.append({'schema_version': VERSION, 'rally_id': rid, 'status': 'candidate_active_interval',
                'start_time_sec': start, 'end_time_sec': upper, 'shot_count': len(group),
                'start_status': 'video_entry_partial_rally' if interval == 0 else 'opposite_contact_and_flight_candidate_may_miss_serve',
                'start_evidence_shot_ids': [pair[0]['event_id'], pair[1]['event_id']],
                'shot_event_ids': [e['event_id'] for e in group], 'shot_times_sec': [e['hit_time_sec'] for e in group],
                'end_reason': terminal['source'] if terminal else 'video_end_without_confirmed_terminal_event',
                'terminal': dict(terminal) if terminal else {'source': 'unknown'},
                'score': {'status': 'unknown', 'included_in_player_statistics': False}})
        lower = upper
    for e in ordered:
        if e.get('excluded'):
            assignments[e['event_id']] = {'rally_phase': 'human_excluded', 'rally_id': None}
    return {'schema_version': VERSION, 'policy': dict(POLICY), 'rallies': rallies,
            'terminals': boundaries, 'assignments': assignments,
            'active_candidate_count': sum(v['rally_phase'] == 'in_rally' for v in assignments.values()),
            'waiting_restart_count': sum(v['rally_phase'] == 'awaiting_restart' for v in assignments.values()),
            'score_inference': False}
