"""Shared ball selection and auditable GPU execution profiles (no model imports)."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_ball_detector(sport_id='badminton'):
    return 'yolo' if sport_id == 'tennis' else 'tracknet_v3'


def pose_checkpoint(sport_id='badminton'):
    prefix = 'GOOD_TENNIS_STREAM' if sport_id == 'tennis' else 'GOOD_BADMINTON_STREAM'
    fallback = ROOT / 'weights' / ('yolo11s-pose.pt' if sport_id == 'tennis' else 'yolo11n-pose.pt')
    return Path(os.environ.get(prefix + '_POSE_MODEL') or fallback).expanduser().resolve()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=16)
def _digest(path, size, modified):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_digest(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return None
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def tracknet_checkpoint():
    """Both entry points resolve the same weights; conflicting aliases fail."""
    values = [os.environ.get(key) for key in (
        'GOOD_BADMINTON_TRACKNET_CHECKPOINT', 'GOOD_BADMINTON_STREAM_TRACKNET_MODEL')]
    paths = [Path(v).expanduser().resolve() for v in values if v]
    if len(paths) == 2 and paths[0] != paths[1]:
        hashes = [file_digest(p) for p in paths]
        if None in hashes or hashes[0] != hashes[1]:
            raise ValueError('Batch and stream TrackNet checkpoints disagree; refusing different weights')
    return paths[0] if paths else ROOT / 'weights/tracknetv3/ckpts/TrackNet_best.pt'


def tracknet_device():
    batch = os.environ.get('GOOD_BADMINTON_TRACKNET_DEVICE')
    stream = os.environ.get('GOOD_BADMINTON_STREAM_TRACKNET_DEVICE')
    if batch and stream and batch != stream:
        raise ValueError('Batch and stream TrackNet device settings disagree')
    return batch or stream or 'auto'


def tracknet_runtime():
    runtime = os.environ.get('GOOD_BADMINTON_TRACKNET_RUNTIME', 'lightweight')
    if runtime not in {'lightweight', 'official'}:
        raise ValueError('TrackNet runtime must be lightweight or official')
    return runtime


def execution_profile(sport_id, detector, route='batch'):
    from .tracking.association_policy import association_policy
    profile = {'schema_version': 'gpu-inference-profile.v1', 'sport_id': sport_id,
               'route': route, 'detector': detector,
               'analysis_implementation_sha256': fingerprint({
                   p: file_digest(ROOT / p) for p in (
                       'badminton_analysis/system.py',
                       'badminton_analysis/detection/yolo_pose.py',
                       'badminton_analysis/visualization/player_pose.py',
                       'api/stream_runtime.py',
                       'badminton_analysis/analysis/offline_shot_reconstruction.py')}),
               'person_association_policy': association_policy(sport_id),
               'person_association_implementation_sha256': fingerprint({
                   p: file_digest(ROOT / p) for p in (
                       'badminton_analysis/tracking/association_policy.py',
                       'badminton_analysis/tracking/bytetrack_adapter.py',
                       'badminton_analysis/tracking/person_only.py',
                       'badminton_analysis/analysis/fixed_camera_match.py')}),
               'pose_checkpoint_sha256': file_digest(pose_checkpoint(sport_id))}
    if detector == 'tracknet_v3':
        checkpoint = tracknet_checkpoint()
        runtime = tracknet_runtime() if route == 'batch' else 'lightweight'
        profile.update(runtime=runtime, checkpoint_sha256=file_digest(checkpoint),
                       device=tracknet_device(), window_frames=8,
                       heatmap_threshold=.5, subpixel_centroid=True,
                       raw_interpolation=False,
                       sampling='every_source_frame',
                       window_step=8 if route == 'batch' and runtime == 'lightweight' else 1,
                       background='uniform_video_median_40' if route == 'batch' and runtime == 'lightweight' else
                                  'configured_official' if route == 'batch' else 'first_window_median',
                       implementation_sha256=fingerprint({
                           str(p): file_digest(ROOT / p) for p in (
                               'badminton_analysis/detection/lightweight_tracknet.py',
                               'api/tracknet_stream.py' if route == 'stream' else 'tools/run_lightweight_tracknet_raw.py',
                           )}))
    else:
        profile.update(runtime='disabled' if detector == 'none' else 'yolo')
    return {**profile, 'sha256': fingerprint(profile)}


def inference_contract(sport_id):
    detectors = ['none', 'yolo'] + (['tracknet_v3'] if sport_id == 'badminton' else [])
    return {'schema_version': 'gpu-inference-contract.v1',
            'default_ball_detector': default_ball_detector(sport_id),
            'batch': {d: execution_profile(sport_id, d) for d in detectors},
            'stream': {d: execution_profile(sport_id, d, 'stream') for d in detectors}}
