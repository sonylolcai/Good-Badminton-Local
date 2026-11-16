"""Descriptive, time-weighted charts from accepted anonymous track segments.

No interpolation, competitive ranking or inferred rally/landing information.
Every interval carries its original video timestamps for review.
"""
from __future__ import annotations

import math
from .heatmap_grid import cell_index, grid

VERSION = 'personal-performance-charts.v1'
SPEED_BANDS = ((0, .25), (.25, 1), (1, 2), (2, None))


def performance_charts(track_id, segments, speed_segments, *, duration_sec, coverage_ratio, match_mode):
    duration = max(0.0, float(duration_sec or 0))
    # Bound page cost to <=90 bins while preserving the source time extent.
    window = max(15, math.ceil(duration / 90 / 15) * 15)
    timeline = [{'start_time_sec': start, 'end_time_sec': min(start + window, duration),
                 'observed_seconds': 0.0, 'distance_m': 0.0,
                 'speed_observed_seconds': 0.0, 'moving_seconds': 0.0}
                for start in range(0, math.ceil(duration), window)]
    cells = [0.0] * 18  # 3 columns x 6 rows, y=0 is far baseline, y=13.4 near baseline.
    fine_cells = [0.0] * 84
    speed_seconds = [0.0] * len(SPEED_BANDS)

    def pieces(segment):
        start = max(0.0, segment['start_time_sec'])
        end = min(duration, segment['start_time_sec'] + segment['seconds'])
        while start < end - 1e-9:
            index = int(start // window)
            stop = min(end, (index + 1) * window)
            yield index, stop - start
            start = stop

    for segment in segments:
        seconds = sum(dt for _, dt in pieces(segment))
        xy = segment.get('start_court_xy_m')
        fine_index = cell_index(xy)
        if fine_index is not None:
            fine_cells[fine_index] += seconds
        if xy and seconds:
            col = min(2, max(0, int(xy[0] / 6.1 * 3)))
            row = min(5, max(0, int(xy[1] / 13.4 * 6)))
            cells[row * 3 + col] += seconds
        for index, dt in pieces(segment):
            timeline[index]['observed_seconds'] += dt
            timeline[index]['distance_m'] += segment['distance_m'] * dt / segment['seconds']

    for segment in speed_segments:
        seconds = sum(dt for _, dt in pieces(segment))
        speed = segment['speed_mps']
        band = next(i for i, (lo, hi) in enumerate(SPEED_BANDS) if speed >= lo and (hi is None or speed < hi))
        speed_seconds[band] += seconds
        for index, dt in pieces(segment):
            timeline[index]['speed_observed_seconds'] += dt
            if speed >= .25:
                timeline[index]['moving_seconds'] += dt

    observed = sum(cells)
    speed_observed = sum(speed_seconds)
    def measurement(seconds):
        return {'status': 'observed' if seconds > 0 else 'insufficient_measurements',
                'missing_reason': None if seconds > 0 else 'no_contiguous_measurements',
                'observed_seconds': round(seconds, 6)}

    for row in timeline:
        raw_seconds, speed_time = row['observed_seconds'], row['speed_observed_seconds']
        row.update(status='observed' if raw_seconds else 'insufficient_measurements',
                   moving_ratio=row['moving_seconds'] / speed_time if speed_time else None,
                   coverage_ratio=raw_seconds / (row['end_time_sec'] - row['start_time_sec']))
        if not raw_seconds:
            row['distance_m'] = None
        for key, value in row.items():
            if isinstance(value, float):
                row[key] = round(value, 6)

    return {
        'schema_version': VERSION, 'track_id': track_id, 'match_mode': match_mode,
        'scope': 'analyzed_video_including_between_rally_movement',
        'duration_sec': round(duration, 6), 'measurement_coverage_ratio': round(coverage_ratio, 4),
        'source': 'detections.jsonl:spatial.tracks',
        'position_heatmap': {**grid(fine_cells), 'weighting': 'accepted_segment_seconds_at_start_position'},
        'position_distribution': {**measurement(observed), 'coordinate_system': 'standard_badminton_court_m',
            'width_m': 6.1, 'length_m': 13.4, 'columns': 3, 'rows': 6,
            'weighting': 'accepted_segment_seconds_at_start_position',
            'cells': [{'row': i // 3, 'column': i % 3, 'seconds': round(dt, 6),
                       'ratio': round(dt / observed, 6) if observed else None} for i, dt in enumerate(cells)]},
        'speed_distribution': {**measurement(speed_observed), 'unit': 'm/s',
            'statistics_window_sec': .5, 'weighting': 'accepted_speed_window_seconds',
            'bands': [{'min_mps': lo, 'max_mps': hi, 'seconds': round(dt, 6),
                       'ratio': round(dt / speed_observed, 6) if speed_observed else None}
                      for (lo, hi), dt in zip(SPEED_BANDS, speed_seconds)]},
        'activity_timeline': {**measurement(sum(row['observed_seconds'] for row in timeline)),
            'window_seconds': window, 'windows': timeline},
        'comparison': {'status': 'not_evaluated', 'reason': 'comparable_history_not_established'},
        'scoring': {'status': 'not_evaluated', 'reason': 'validated_reference_not_established'},
        'policy': {'max_join_gap_seconds': .5, 'moving_speed_mps': .25,
                   'high_intensity_speed_mps': 2.0,
                   'missing_predicted_reconnected_intervals': 'excluded',
                   'timestamps': 'source_time_sec_preferred_frame_fps_fallback'},
    }
