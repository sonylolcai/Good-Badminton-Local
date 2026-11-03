"""Durable GPU -> business notifications after inference and media archival."""
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import Request, build_opener, HTTPRedirectHandler

from good_badminton_contracts.gpu_notifications import SCHEMA, encode, event_id, signature
from .stream_models import TERMINAL_SESSION_STATES

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None

class ResultNotifications:
    def __init__(self, manager, archive, *, url, secret, send=None, clock=time.time, archive_results=None, not_before=None, environment_urls=None):
        self.environment_urls = dict(environment_urls or {})
        if not set(self.environment_urls) <= {'test', 'production'}:
            raise ValueError('invalid callback environment')
        for target in [url, *self.environment_urls.values()]:
            parsed = urlsplit(target)
            if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                    or parsed.query or parsed.fragment or len(secret.encode()) < 32):
                raise ValueError('GPU callback requires a credential-free HTTPS URL and a 32-byte secret')
        self.manager, self.archive = manager, archive
        self.url, self.secret, self.clock = url, secret, clock
        self.send = send or self._send
        self.archive_results = (os.environ.get('GPU_RESULT_ARCHIVE_ENABLED') == '1') if archive_results is None else archive_results
        cutoff = os.environ.get('GPU_RESULT_CALLBACK_NOT_BEFORE', '') if not_before is None else not_before
        self.not_before = datetime.fromisoformat(cutoff.replace('Z', '+00:00')) if cutoff else None
        if self.not_before is not None and self.not_before.tzinfo is None:
            raise ValueError('GPU callback activation time requires a timezone')
        self.stopped = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def payload(self, session):
        identity = session.get('media_identity')
        if not identity or session.get('status') not in TERMINAL_SESSION_STATES:
            return None
        if self.not_before is not None:
            created = datetime.fromisoformat(session['created_at'].replace('Z', '+00:00'))
            if created.tzinfo is None:
                raise ValueError('session creation time requires a timezone')
            if created < self.not_before:
                return None  # Existing pre-contract history stays intact; never notify an unrelated business inbox.
        media = self.archive.status(session['analysis_session_id'])
        if not session.get('sealed') and session.get('status') in {'failed', 'cancelled', 'interrupted_needs_rebuild'} and media.get('status') == 'awaiting_seal':
            media = {'status': 'failed', 'error': 'analysis_failed_before_seal'}
        if media.get('status') not in {'ready', 'invalid', 'failed'}:
            return None
        value = {'schema_version': SCHEMA, 'analysis_session_id': session['analysis_session_id'],
                 'edge_session_id': identity['edge_session_id'], 'media_identity': identity,
                 'sport_id': session.get('configuration', {}).get('sport_id', 'badminton'), 'analysis_status': session['status'],
                 'media_status': media['status'], 'revision': media.get('revision'),
                 'manifest_object_key': media.get('manifest_object_key'),
                 'manifest_sha256': media.get('manifest_sha256'), 'error_type': media.get('error'),
                 'total_source_frames': media.get('total_source_frames')}
        value['event_id'] = event_id(value)
        return value

    def _path(self, session_id):
        return self.manager._session_dir(session_id) / 'result-notification.json'

    def _read(self, session_id):
        path = self._path(session_id)
        try:
            return json.loads(path.read_text()) if path.is_file() else {}
        except (OSError, ValueError):
            return {}  # Rebuild and resend the same immutable event; inbox deduplicates.

    def pending_count(self):
        count = 0
        for session in self.manager._list_sessions():
            try:
                value = self.payload(session)
                if value and self._read(session['analysis_session_id']).get('acknowledged_base_event_id') != value['event_id']:
                    count += 1
            except Exception:
                count += 1  # Unknown state cannot permit a destructive shutdown.
        return count

    def _save(self, session_id, value):
        path = self._path(session_id)
        tmp = path.with_suffix('.pending')
        with tmp.open('w') as out:
            json.dump(value, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)

    def _send(self, payload):
        raw = encode(payload)
        timestamp = str(int(self.clock()))
        headers = {'Content-Type': 'application/json', 'X-Gpu-Timestamp': timestamp,
                   'X-Gpu-Signature': signature(self.secret, timestamp, raw)}
        target = self.url
        if self.environment_urls:
            environment = (payload.get('media_identity') or {}).get('environment')
            if environment not in self.environment_urls:
                raise ValueError('callback environment is not enabled')
            target = self.environment_urls[environment]
        request = Request(target, data=raw, headers=headers, method='POST')
        with build_opener(NoRedirect).open(request, timeout=10) as response:
            result = json.loads(response.read(32768))
            if response.status not in {200, 202} or result.get('event_id') != payload['event_id'] or result.get('status') != 'accepted':
                raise IOError('business server did not acknowledge this notification')

    def run_once(self):
        if not self.lock.acquire(blocking=False):
            return
        try:
            for session in self.manager._list_sessions():
                sid = session['analysis_session_id']
                try:
                    value = self.payload(session)
                    if not value:
                        continue
                    state = self._read(sid)
                    base_id = value['event_id']
                    if state.get('blocked') and state.get('base_event_id') == base_id and self.clock() < state.get('blocked_until', 0):
                        continue
                    if state.get('acknowledged_base_event_id') == base_id:
                        continue
                    if state.get('base_event_id') == base_id and self.clock() < state.get('next_attempt_at', 0):
                        continue
                    attempts = state.get('attempts', 0) + 1 if state.get('base_event_id') == base_id else 1
                    previous_payload = state.get('payload') if state.get('base_event_id') == base_id else None
                    state = {'event_id': base_id, 'base_event_id': base_id, 'attempts': attempts, 'payload': value,
                             'next_attempt_at': self.clock() + min(300, 2 ** min(attempts, 8))}
                    self._save(sid, state)  # Persist before I/O; replay is safe after process death.
                    if self.archive_results and value['media_status'] == 'ready':
                        if previous_payload and previous_payload.get('analysis_archive'):
                            value = previous_payload
                        else:
                            from .analysis_archive import upload_analysis_archive
                            value['analysis_archive'] = upload_analysis_archive(self.manager, self.archive.storage, session)
                            value['event_id'] = event_id(value)
                        state.update(event_id=value['event_id'], payload=value)
                        self._save(sid, state)
                    self.send(value)
                    state.update(acknowledged_event_id=value['event_id'], acknowledged_base_event_id=base_id, acknowledged_at=self.clock())
                    self._save(sid, state)
                except Exception as exc:
                    state = self._read(sid)
                    if state.get('attempts'):
                        state['last_error_type'] = type(exc).__name__
                        permanent = isinstance(exc, HTTPError) and 400 <= exc.code < 500 and exc.code not in {408, 409, 425, 429}
                        state['blocked'] = permanent or state['attempts'] >= 8
                        if state['blocked']:
                            state['blocked_until'] = self.clock() + 900  # Low-rate recovery probe after operator/network repair.
                        self._save(sid, state)
                    continue
        finally:
            self.lock.release()

    def blocked_count(self):
        return sum(bool(self._read(session['analysis_session_id']).get('blocked')) for session in self.manager._list_sessions())

    def retry_blocked(self):
        with self.lock:
            self._retry_blocked()

    def _retry_blocked(self):
        for session in self.manager._list_sessions():
            sid = session['analysis_session_id']
            state = self._read(sid)
            if state.get('blocked'):
                state.update(blocked=False, attempts=0, next_attempt_at=0)
                self._save(sid, state)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def loop():
            while not self.stopped.is_set():
                try:
                    self.run_once()
                except Exception:
                    pass
                self.stopped.wait(2)
        self.thread = threading.Thread(target=loop, daemon=True, name='gpu-result-notifications')
        self.thread.start()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=12)
