"""Strict four-corner PERSON scope for contacts, not a shuttle flight crop.

Kept identical to the evaluation-suite copy for independently deployed runtimes.
"""
import math

VERSION = 'calibrated-person-contact-scope.v1'


def _point(v):
    return (isinstance(v, (list, tuple)) and len(v) >= 2
            and all(type(n) in (int, float) and math.isfinite(n) for n in v[:2]))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def valid_quad(polygon):
    if not isinstance(polygon, (list, tuple)) or len(polygon) != 4 or not all(_point(p) for p in polygon):
        return False
    turns = [_cross(polygon[i], polygon[(i + 1) % 4], polygon[(i + 2) % 4]) for i in range(4)]
    return all(t > 1e-7 for t in turns) or all(t < -1e-7 for t in turns)


def inside_quad(point, polygon):
    if not _point(point) or not valid_quad(polygon):
        return False
    sides = [_cross(polygon[i], polygon[(i + 1) % 4], point) for i in range(4)]
    return all(s >= -1e-7 for s in sides) or all(s <= 1e-7 for s in sides)


def person_court_gate(track, polygon):
    base = {'version': VERSION, 'basis': 'current_measured_person_foot_image_xy', 'margin_px': 0}
    if not valid_quad(polygon):
        return {**base, 'status': 'unknown', 'reason': 'missing_or_invalid_four_corners'}
    if not isinstance(track, dict) or track.get('status') != 'detected':
        return {**base, 'status': 'unknown', 'reason': 'no_detected_person'}
    evidence = track.get('location_evidence') or {}
    if evidence.get('is_current_measurement') is False:
        return {**base, 'status': 'unknown', 'reason': 'stale_person_position'}
    point = track.get('image_xy')
    if not _point(point):
        return {**base, 'status': 'unknown', 'reason': 'missing_person_foot_position'}
    confidence = evidence.get('confidence')
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or confidence < .5:
        return {**base, 'status': 'unknown', 'reason': 'low_or_missing_foot_confidence'}
    return {**base, 'status': 'inside' if inside_quad(point, polygon) else 'outside',
            'reason': 'measured_foot_in_quad' if inside_quad(point, polygon) else 'measured_foot_outside_quad',
            'foot_image_xy': list(point[:2])}


def scoped_rows(rows, polygon):
    """Read-only projection; never clip/clamp a person or change detector data."""
    output = []
    for row in rows:
        spatial = row.get('spatial') or {}
        tracks = [t for t in spatial.get('tracks') or [] if person_court_gate(t, polygon)['status'] == 'inside']
        ids = {t.get('track_id') for t in tracks}
        hits = [h for h in spatial.get('hit_events') or [] if h.get('hitter_track_id') in ids]
        output.append({**row, '_court_hit_gate_active': True,
                       '_court_external_tracks': [t for t in spatial.get('tracks') or [] if t not in tracks],
                       'spatial': {**spatial, 'tracks': tracks, 'hit_events': hits}})
    return output


def external_hand_is_nearest(row, shuttle):
    if not _point(shuttle):
        return False
    def nearest(tracks):
        return min((math.dist(shuttle[:2], hand[:2]) for t in tracks if t.get('status') == 'detected'
                    for hand in ((t.get('location_evidence') or {}).get('hands_image') or {}).values()
                    if _point(hand)), default=math.inf)
    external = nearest((row or {}).get('_court_external_tracks') or [])
    internal = nearest(((row or {}).get('spatial') or {}).get('tracks') or [])
    return math.isfinite(external) and external <= internal
