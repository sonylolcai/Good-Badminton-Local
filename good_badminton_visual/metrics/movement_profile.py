"""Descriptive movement axes, separate from competitive skill and legacy scores."""
from __future__ import annotations
import math

VERSION = 'movement-profile.v1'
MIN_OBSERVED_SECONDS = 15.0
MIN_COVERAGE = .6
AXES = ('activity', 'frequency', 'intensity', 'range', 'continuity')
LABELS = ('活跃度', '移动频率', '移动强度', '移动范围', '移动持续性')


def movement_profile(segments, points, *, coverage_ratio, moving_speed=.25, high_speed=2.0):
    observed = sum(s['seconds'] for s in segments)
    active = sum(s['seconds'] for s in segments if s['speed_mps'] >= moving_speed)
    high = sum(s['seconds'] for s in segments if s['speed_mps'] >= high_speed)
    # A bout ends at a detection gap. Hysteresis rejects small posture jitter.
    bouts, current, quiet, previous_end = [], 0.0, 0.0, None
    for segment in segments:
        start = segment['start_time_sec']
        if previous_end is not None and start - previous_end > .001:
            if current >= .5: bouts.append(current)
            current, quiet = 0.0, 0.0
        speed, seconds = segment['speed_mps'], segment['seconds']
        if speed >= moving_speed or (current and speed >= .15):
            current += seconds; quiet = 0.0
        elif current:
            quiet += seconds
            if quiet >= .5:
                if current >= .5: bouts.append(current)
                current, quiet = 0.0, 0.0
        previous_end = start + seconds
    if current >= .5: bouts.append(current)
    zones = {}
    ys = sorted(float(p['court_xy_m'][1]) for p in points if p.get('court_xy_m'))
    lower = bool(ys and ys[len(ys)//2] >= 6.7)
    for point in points:
        xy = point.get('court_xy_m')
        if not xy: continue
        x, y = map(float, xy)
        local_y = y - 6.7 if lower else 6.7 - y
        if not (0 <= x <= 6.1 and 0 <= local_y <= 6.7): continue
        zone = (min(2,int(x/6.1*3)),min(2,int(local_y/6.7*3)))
        zones[zone] = zones.get(zone,0) + 1
    covered = sum(count >= 3 for count in zones.values())
    ready = math.isfinite(coverage_ratio) and coverage_ratio >= MIN_COVERAGE and observed >= MIN_OBSERVED_SECONDS and len(points) >= 30
    raw = (active/observed if observed else None, len(bouts)/(observed/60) if observed else None,
           high/active if active else 0.0, covered, sum(bouts)/len(bouts) if bouts else 0.0)
    scales = (1.0,12.0,1.0,9.0,6.0)
    units = ('ratio','per_minute','ratio','zones','seconds')
    axes = [{ 'id':key,'label':label,'status':'observed' if ready else 'insufficient_measurements',
              'value':round(value,4) if ready and value is not None else None,'unit':unit,
              'normalized_value':round(min(100,max(0,value/scale*100)),1) if ready and value is not None else None }
            for key,label,value,scale,unit in zip(AXES,LABELS,raw,scales,units)]
    return {'schema_version':VERSION,'kind':'movement_description','competitive_rating':False,
            'normalization_version':'provisional-display.v1',
            'normalization_reference':dict(zip(AXES,scales)),
            'observed_seconds':round(observed,3),'coverage_ratio':round(coverage_ratio,4),'axes':axes,
            'policy':'Missing gaps excluded. Axes describe movement, not skill. Range is player position, not reachable shots.'}
