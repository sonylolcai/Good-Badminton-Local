"""Anonymous CPU aggregation on the inference host; never downloads or runs models."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from functools import lru_cache
import os
from pathlib import Path
import re
import threading
import time

PROTOCOL = 'stream-visual-result.v1'
# The previous implementation uses the same derivation for sessions without the
# anonymous report opt-in. Keep their pinned obligations and cache keys intact.
PREVIOUS_IMPLEMENTATION = '995d6e67cb6113b9875f74a13873adabcb1bedc3419f8ec94dfae574575efd89'
SUPPORTED_IDENTITY_POLICIES = ('individual_tracks.v1', 'half_court_slots.v1', 'anchored_roster.v1')
MAX_RESULT_BYTES = 4 * 1024 * 1024
SESSION_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
CONFIGURATION_KEYS = frozenset(('analysis_sample_hz', 'pose_imgsz', 'shuttle_detector',
    'tracker_backend', 'lock_match_roster', 'roster_stable_frames', 'max_roster_count',
    'expected_player_count', 'roster_discovery_seconds', 'roster_identity_policy',
    'match_mode', 'far_player_enhancement', 'sport_id', 'session_mode', 'calibration_scope',
    'allow_anonymous_report'))

def enabled():
    raw = os.environ.get('GOOD_BADMINTON_VISUAL_RESULT_ENABLED')
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in ('0', 'false', 'no', 'off')

def validate_projection_request(request):
    projection = request.get('visual_result')
    if projection is None:
        return
    if not enabled():
        raise ValueError('visual_result_capability_disabled')
    if projection['implementation_sha256'] != implementation():
        raise ValueError('visual_result_implementation_mismatch')
    configuration = request.get('configuration') or {}
    if (configuration.get('roster_identity_policy', 'individual_tracks.v1') not in SUPPORTED_IDENTITY_POLICIES
            or configuration.get('sport_id', 'badminton') != 'badminton'):
        raise ValueError('visual_result_identity_policy_not_supported')

def now():
    return datetime.now(timezone.utc).isoformat()

def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

@lru_cache(maxsize=1)
def implementation():
    from good_badminton_visual import __file__ as source
    root = Path(source).parent
    manifest = json.loads((root/'implementation.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('visual_result_implementation_mismatch')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest['runtime_sha256']:
        raise ValueError('visual_result_runtime_mismatch')
    producers = ('api/stream_sessions.py', 'badminton_analysis/streaming/engine.py')
    geometry_evidence = {
        name: hashlib.sha256((root.parent/name).read_bytes()).hexdigest() for name in producers
    }
    if geometry_evidence != manifest.get('geometry_evidence_sha256'):
        raise ValueError('visual_result_geometry_evidence_mismatch')
    version = {'files': manifest['files'], 'runtime_sha256': manifest['runtime_sha256'],
               'geometry_evidence_sha256': geometry_evidence}
    if hashlib.sha256(encode(version)).hexdigest() != manifest['sha256']:
        raise ValueError('visual_result_implementation_manifest_invalid')
    return manifest['sha256']

def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items()
                if not k.endswith('_path') and k not in ('output_dir', 'source_paths')}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    return value

def atomic_json(path, value):
    raw = encode(value)
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError('visual_result_too_large')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+'.'+os.urandom(8).hex())
    try:
        with temp.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def _session_video_metadata(root, session):
    video = dict(session.get('video') or {}) if isinstance(session.get('video'), dict) else {}
    bound = session.get('calibration_frame_size')
    if isinstance(bound, list) and len(bound) == 2 and all(type(value) is int and value > 0 for value in bound):
        video['height'], video['width'] = bound
    ckpt_path = Path(root) / 'checkpoint.json'
    if (not video.get('width') or not video.get('height') or not video.get('fps')) and ckpt_path.is_file():
        try:
            ckpt = json.loads(ckpt_path.read_text())
        except (OSError, ValueError):
            ckpt = {}
        ckpt_video = ckpt.get('video') if isinstance(ckpt.get('video'), dict) else {}
        for key in ('width', 'height', 'fps'):
            if not video.get(key) and ckpt_video.get(key):
                video[key] = ckpt_video[key]
        shape = (ckpt.get('temporal_processor_state') or {}).get('frame_shape')
        if (not video.get('width') or not video.get('height')) and isinstance(shape, (list, tuple)) and len(shape) >= 2:
            if shape[0] and shape[1]:
                video['height'], video['width'] = int(shape[0]), int(shape[1])
    return video or None

def input_identity(root, session):
    path = Path(root) / 'events.jsonl'
    stat = path.stat()
    configuration = {k: v for k, v in session.get('configuration', {}).items() if k in CONFIGURATION_KEYS}
    identity = {
        'analysis_session_id': session['analysis_session_id'],
        'events_revision': int(session.get('events_revision', 1)),
        'continuity_epoch': int(session.get('continuity_epoch', 0)),
        'status': session['status'], 'sealed_at': session.get('sealed_at'),
        'expected_last_segment_index': session.get('expected_last_segment_index'),
        'event_file': [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
        'configuration_sha256': hashlib.sha256(encode(configuration)).hexdigest(),
        'roster_identity_policy': configuration.get('roster_identity_policy', 'individual_tracks.v1'),
        'implementation_sha256': (session.get('visual_result') or {}).get('implementation_sha256') or implementation(),
        'visual_result': session.get('visual_result'),
    }
    if session.get('court_corners') is not None:
        identity['court_corners_sha256'] = hashlib.sha256(encode(session['court_corners'])).hexdigest()
    video = _session_video_metadata(root, session)
    if video is not None:
        identity['video'] = video
    return identity

def identity_key(identity):
    return hashlib.sha256(encode(identity)).hexdigest()

def result_path(root, identity):
    return Path(root) / 'visual_result' / (identity_key(identity)+'.json')

def _source_unchanged(root, expected):
    try:
        latest = json.loads((Path(root)/'manifest.json').read_text())
        return latest.get('sealed') is True and input_identity(root, latest) == expected
    except (OSError, ValueError):
        return False

class LocalEvidenceClient:
    """Adapter for the frozen metric implementation; every page is local memory."""
    def __init__(self, events, session_id, revision):
        self.events = sorted(events, key=lambda e: (float(e.get('source_time_sec', 0)), str(e['event_id'])))
        self.positions = {e['event_id']: i for i, e in enumerate(self.events)}
        if len(self.positions) != len(self.events):
            raise ValueError('duplicate_visual_event_identity')
        self.session_id, self.revision = session_id, revision
        self.ledger = type('EvidenceIdentity', (), {'analysis_session_id': session_id})()

    def read_events(self, *, cursor=None, limit=500):
        start = self.positions[cursor]+1 if cursor else 0
        events = self.events[start:start+limit]
        return {'schema_version': 'stream-session.v1', 'analysis_session_id': self.session_id,
                'events': events, 'events_revision': self.revision,
                'next_cursor': events[-1]['event_id'] if events else None}

def compute_visual_result(root, identity, terminal):
    """Spawn-process entry point. Reads local immutable evidence once; no HTTP/DB/model IO."""
    root = Path(root)
    started = time.monotonic()
    if not _source_unchanged(root, identity):
        return {'stale': True}
    session = json.loads((root/'manifest.json').read_text())
    projection = session.get('visual_result')
    pinned_implementation = (projection or {}).get('implementation_sha256')
    active_implementation = implementation()
    compatible_previous = (pinned_implementation == PREVIOUS_IMPLEMENTATION
                           and (session.get('configuration') or {}).get('allow_anonymous_report') is not True)
    if pinned_implementation != active_implementation and not compatible_previous:
        raise ValueError('visual_result_not_requested_or_version_changed')
    if session['status'] not in ('finalized', 'partial'):
        return {'stale': True}
    configuration = {k: v for k, v in session.get('configuration', {}).items() if k in CONFIGURATION_KEYS}
    if configuration.get('sport_id', 'badminton') != 'badminton':
        raise ValueError('visual_result_sport_not_supported')
    events = []
    digest = hashlib.sha256()
    size = 0
    with (root/'events.jsonl').open('rb') as stream:
        for line in stream:
            size += len(line)
            if size > 1024 * 1024 * 1024:
                raise ValueError('visual_result_evidence_too_large')
            digest.update(line)
            if line.strip(): events.append(json.loads(line))
    from good_badminton_visual.derivation import derive_stream_movement_metrics
    from good_badminton_visual.result_contract import validate_metrics
    from good_badminton_visual.candidates import project_candidates
    client = LocalEvidenceClient(events, identity['analysis_session_id'], identity['events_revision'])
    create_request = {'configuration': configuration}
    if session.get('court_corners') is not None:
        create_request['court_corners'] = session['court_corners']
    video_meta = _session_video_metadata(root, session)
    if video_meta is not None:
        create_request['video'] = video_meta
    derived = derive_stream_movement_metrics(root/'visual_result'/identity_key(identity),
        client=client, terminal_status=terminal, create_request=create_request)
    validate_metrics(derived['metrics'])
    candidates, _ = project_candidates(client.events, identity['analysis_session_id'])
    metric_tracks = {player['track_id'] for player in derived['metrics']['players']}
    candidates = [candidate for candidate in candidates if candidate.get('track_id') in metric_tracks]
    result = public({
        'schema_version': PROTOCOL, 'state': 'ready',
        'analysis_session_id': identity['analysis_session_id'],
        'gpu_status': session['status'], 'progress': terminal['progress'],
        'source': {'identity': identity, 'events_sha256': digest.hexdigest(),
                   'event_count': len(events), 'person_observation_count': derived['person_observation_count']},
        'metrics': derived['metrics'], 'track_candidates': candidates,
        'completeness': {'status': 'partial', 'scope': 'movement_and_anonymous_stream_evidence',
                         'missing_stages': ['rally_segmentation', 'hit_moments', 'hitter_attribution']},
        'generated_at': now(), 'aggregation_seconds': round(time.monotonic()-started, 6),
    })
    if not _source_unchanged(root, identity):
        return {'stale': True}
    result['result_sha256'] = hashlib.sha256(encode(result)).hexdigest()
    atomic_json(result_path(root, identity), result)
    return {'ready': True}

def _prepare_cpu_process():
    if hasattr(os, 'nice'): os.nice(5)

class VisualResultService:
    def __init__(self, manager, *, active=None, executor=None):
        self.manager = manager
        self.enabled = enabled() if active is None else active
        self.executor = executor
        self.futures = {}
        self.stopped = threading.Event()
        self.thread = None
        self.pending = {}
        self.lock = threading.Lock()

    def notify(self, session_id):
        if isinstance(session_id, str) and SESSION_ID.fullmatch(session_id):
            with self.lock:
                if session_id not in self.futures: self.pending[session_id] = time.monotonic()

    @property
    def pending_count(self):
        with self.lock: return len(self.pending) + len(self.futures)

    def outstanding_count(self, sessions):
        # Persisted obligations still prevent shutdown if the capability was
        # disabled/restarted while requested summaries were unfinished.
        pending = 0
        for session in sessions:
            if not session.get('visual_result') or session.get('status') not in ('finalized', 'partial'):
                continue
            try:
                if self.read(session)[0] == 202: pending += 1
            except (OSError, ValueError):
                pending += 1
        return max(self.pending_count, pending)

    def _identity(self, session):
        if session.get('status') not in ('finalized', 'partial') or session.get('sealed') is not True:
            return None
        try: return input_identity(self.manager._session_dir(session['analysis_session_id']), session)
        except OSError: return None

    def read(self, session):
        progress = self.manager._compute_progress(session)
        base = {'schema_version': PROTOCOL, 'analysis_session_id': session['analysis_session_id'],
                'state': 'pending', 'gpu_status': session['status'], 'progress': progress}
        if not session.get('visual_result'):
            return 200, {**base, 'state': 'failed', 'error_code': 'visual_result_not_requested'}
        identity = self._identity(session)
        if identity is None:
            if session.get('status') in ('finalized', 'partial') and session.get('sealed') is True:
                return 200, {**base, 'state': 'failed', 'error_code': 'visual_result_evidence_unavailable'}
            return 202, base
        path = result_path(self.manager._session_dir(session['analysis_session_id']), identity)
        try:
            with path.open('rb') as stream: raw = stream.read(MAX_RESULT_BYTES+1)
            if len(raw) > MAX_RESULT_BYTES: raise ValueError('visual_result_too_large')
            result = json.loads(raw)
            if result.get('source', {}).get('identity') != identity: return 202, base
            if result.get('state') == 'ready':
                payload = {k: v for k, v in result.items() if k != 'result_sha256'}
                if hashlib.sha256(encode(payload)).hexdigest() != result.get('result_sha256'):
                    raise ValueError('visual_result_checksum_invalid')
            return 200, result
        except FileNotFoundError: return 202, base

    def tick(self):
        for sid, (future, identity) in list(self.futures.items()):
            if not future.done(): continue
            del self.futures[sid]
            try:
                outcome = future.result()
                if outcome.get('stale'): self.notify(sid)
            except Exception as error:
                try:
                    from concurrent.futures.process import BrokenProcessPool
                    if isinstance(error, BrokenProcessPool):
                        self.executor.shutdown(wait=False, cancel_futures=True)
                        self.executor = None
                    root = self.manager._session_dir(sid)
                    if _source_unchanged(root, identity):
                        detail_code = getattr(error, 'code', None)
                        failure_payload = {
                            'schema_version': PROTOCOL, 'analysis_session_id': sid, 'state': 'failed',
                            'gpu_status': identity['status'], 'source': {'identity': identity},
                            'progress': self.manager._compute_progress(self.manager._load(sid)),
                            'error_code': 'visual_result_aggregation_failed', 'error_type': type(error).__name__,
                        }
                        if isinstance(detail_code, str) and detail_code:
                            failure_payload['detail_code'] = detail_code
                        atomic_json(result_path(root, identity), failure_payload)
                    else:
                        self.notify(sid)
                except Exception:
                    # A failed persistence attempt is not a durable terminal
                    # outcome. Keep the obligation scheduled in this process.
                    self.notify(sid)
                    raise
        # At most one aggregation in flight; do not queue long-lived copies of evidence.
        if self.futures: return
        with self.lock:
            sid = max(self.pending, key=self.pending.get) if self.pending else None
            if sid: del self.pending[sid]
        if sid is None: return
        try:
            session = self.manager._load(sid)
        except Exception:
            self.notify(sid)
            raise
        if session is None or not session.get('visual_result'): return
        identity = self._identity(session)
        if identity is None: return
        try:
            code, _ = self.read(session)
        except (OSError, ValueError):
            # An interrupted/corrupted derived artifact can be regenerated from
            # retained evidence; no raw input or previous revision is replaced.
            code = 202
        if code == 200: return
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                                initializer=_prepare_cpu_process)
        terminal = {'analysis_session_id': sid, 'status': session['status'],
                    'progress': self.manager._compute_progress(session)}
        try:
            self.futures[sid] = (self.executor.submit(compute_visual_result,
                str(self.manager._session_dir(sid)), identity, terminal), identity)
        except Exception:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
            self.notify(sid)
            raise

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.stopped.clear()
        for session in sorted(self.manager._list_sessions(), key=lambda s: s.get('created_at') or ''):
            if session.get('visual_result') and session.get('status') in ('finalized', 'partial'):
                self.notify(session['analysis_session_id'])
        def run():
            while not self.stopped.is_set():
                try: self.tick()
                except Exception:
                    import logging
                    logging.getLogger(__name__).warning('Visual result scheduling will retry')
                self.stopped.wait(1)
        self.thread = threading.Thread(target=run, name='visual-result-scheduler', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread: self.thread.join(timeout=3)
        if self.executor: self.executor.shutdown(wait=False, cancel_futures=True)

def register_visual_result_routes(app, manager, require_api_key):
    from fastapi import Depends, HTTPException
    from fastapi.responses import JSONResponse
    service = VisualResultService(manager)
    manager.visual_result_service = service
    app.state.visual_result_service = service
    if manager.worker_running: service.start()
    app.router.add_event_handler('shutdown', service.stop)

    @app.get('/api/v1/stream-sessions/{session_id}/visual-result', dependencies=[Depends(require_api_key)])
    def visual_result(session_id: str):
        if not SESSION_ID.fullmatch(session_id): raise HTTPException(404, 'session not found')
        session = manager._load(session_id)
        if session is None: raise HTTPException(404, 'session not found')
        if not service.enabled and not session.get('visual_result'):
            raise HTTPException(503, 'GPU visual result admissions are disabled')
        code, result = service.read(session)
        return JSONResponse(result, status_code=code, headers={'Cache-Control': 'private, no-store'})
