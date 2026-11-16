"""Fine court bins from measured coordinates; never subdivide legacy counts."""
import math

VERSION = 'court-heatmap-grid.v1'
COLUMNS, HALF_ROWS = 6, 7
WIDTH, HALF_LENGTH = 6.1, 6.7


def cell_index(xy, *, half=False):
    if (not isinstance(xy, (list, tuple)) or len(xy) < 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy[:2])
            or not 0 <= xy[0] <= WIDTH or not 0 <= xy[1] <= HALF_LENGTH * 2):
        return None
    # Receiving depth is measured away from the net for either court end.
    y = abs(xy[1] - HALF_LENGTH) if half else xy[1]
    rows = HALF_ROWS if half else HALF_ROWS * 2
    length = HALF_LENGTH if half else HALF_LENGTH * 2
    return min(rows - 1, int(y / length * rows)) * COLUMNS + min(COLUMNS - 1, int(xy[0] / WIDTH * COLUMNS))


def grid(weights, *, half=False):
    total = sum(weights)
    unit = 'count' if half else 'seconds'
    return {'schema_version': VERSION, 'coordinate_space': 'opponent_half' if half else 'full_court',
            'columns': COLUMNS, 'rows': HALF_ROWS if half else HALF_ROWS * 2, 'half_rows': HALF_ROWS,
            'width_m': WIDTH, 'length_m': HALF_LENGTH if half else HALF_LENGTH * 2,
            'row_origin': 'net' if half else 'far_baseline',
            'status': 'observed' if total else 'insufficient_measurements',
            'sample_count' if half else 'observed_seconds': round(total, 6),
            'cells': [{'row': i // COLUMNS, 'column': i % COLUMNS, unit: round(weight, 6),
                       'ratio': round(weight / total, 6) if total else None}
                      for i, weight in enumerate(weights)]}


def receiving_heatmap(events):
    weights = [0] * (COLUMNS * HALF_ROWS)
    for event in events:
        index = cell_index(event.get('receiver_xy_m'), half=True)
        if index is not None:
            weights[index] += 1
    return {**grid(weights, half=True), 'total_route_samples': len(events),
            'weighting': 'receiving_events_with_measured_court_coordinates'}
