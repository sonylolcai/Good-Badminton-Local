#!/usr/bin/env python3
"""Adapt the existing optimized TrackNet engine to the full-video API contract."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export_raw(detections, frame_count, path):
    """Refuse incomplete sequences or any interpolated point as raw evidence."""
    if [d['frame_index'] for d in detections] != list(range(frame_count)):
        raise ValueError('TrackNet output must cover every source frame exactly once')
    if any(d['status'] not in {'detected', 'missing'} for d in detections):
        raise ValueError('Interpolated or predicted points cannot enter raw TrackNet CSV')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'HeatmapMax','TrajectoryId','SelectionReason','SourceTime'])
        for d in detections:
            writer.writerow([d['frame_index'], int(d['visible']),
                             d['x'] if d['visible'] else 0, d['y'] if d['visible'] else 0,
                             d['confidence'],(d.get('selection') or {}).get('trajectory_id',''),
                             (d.get('selection') or {}).get('reason',''),d.get('source_time_sec','')])
    temporary.replace(path)


def main():
    import cv2
    from badminton_analysis.detection.lightweight_tracknet import LightweightTrackNetDetector
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    video = Path(args.video).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.output_dir).resolve()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError('Cannot open TrackNet input video')
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width,height=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    print('GOOD_BADMINTON_TRACKNET_EVENT=' + json.dumps({'stage': 'inference', 'runtime': 'lightweight', 'total_frames': count}), flush=True)
    detector = LightweightTrackNetDetector(checkpoint, device=args.device)
    detections = detector.predict_video(video, interpolate_isolated_missing=False)
    declared_count = count
    count = detector.last_decoded_frame_count
    if count <= 0:
        raise RuntimeError('TrackNet input contains no decodable video frames')
    raw = out / 'tracknet_raw' / (video.stem + '_ball.csv')
    export_raw(detections, count, raw)
    observed = sum(d['visible'] for d in detections)
    provenance = {
        'runtime': 'LightweightTrackNetDetector', 'device': str(detector.device),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
        'raw_csv_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'source_frame_count': count, 'observed_count': observed,
        'container_declared_frame_count': declared_count,
        'observed_fraction': observed / count if count else None,
        'seq_len': detector.seq_len, 'sliding_step': detector.seq_len,
        'heatmap_threshold': detector.heatmap_threshold,
        'subpixel_centroid': detector.subpixel_centroid,
        'interpolation_in_raw_csv': False, 'accuracy': None,
    }
    (out / 'runtime_provenance.json').write_text(json.dumps(provenance, indent=2))
    print('GOOD_BADMINTON_TRACKNET_EVENT=' + json.dumps({'stage': 'completed', **provenance}), flush=True)


if __name__ == '__main__':
    main()
