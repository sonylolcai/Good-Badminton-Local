"""Fixed business-issued capture authorization, signal liveness and durable ends.

This input lease never controls instance power or the service process. Terminal
state is persisted before delivery; result/media callbacks remain independent.
"""
from datetime import datetime, timezone
import hashlib
import json
import re
import time

from .stream_errors import StreamSessionError, invalid_state, session_not_found, validation_error
from .stream_models import TERMINAL_SESSION_STATES
from good_badminton_contracts.gpu_notifications import event_id

POLICY = 'capture-lease.v1'
FIELDS = {'schema_version', 'lease_id', 'match_id', 'edge_session_id', 'device_id', 'camera_id',
          'sport_id', 'started_at', 'expires_at', 'token_sha256'}
REASONS = {'token_expired', 'terminal_restarted', 'input_signal_timeout'}


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('capture timestamp must be a string')
    instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if instant.tzinfo is None:
        raise ValueError('capture timestamp needs timezone')
    return instant.timestamp()


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace('+00:00', 'Z')


def now(manager):
    return manager.capture_clock()


def status(session):
    lease = session['capture_lease']
    return {k: lease[k] for k in ('lease_id', 'started_at', 'expires_at', 'state', 'last_signal_at', 'sequence')} | {
        'analysis_session_id': session['analysis_session_id'], 'capture_lease_version': POLICY,
        'inactivity_timeout_seconds': 120}


def register(manager, sid, body):
    if not isinstance(body, dict) or set(body) != FIELDS or body.get('schema_version') != POLICY:
        raise validation_error('invalid fixed capture lease fields', analysis_session_id=sid)
    try:
        for key in ('lease_id', 'match_id', 'edge_session_id', 'device_id', 'camera_id'):
            if not isinstance(body[key], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', body[key]):
                raise ValueError('invalid capture identity')
        if body['sport_id'] not in {'badminton', 'tennis'} or not re.fullmatch(r'[a-f0-9]{64}', body['token_sha256']):
            raise ValueError('invalid capture sport or token digest')
        started, expires = timestamp(body['started_at']), timestamp(body['expires_at'])
        if expires - started != 1800 or started > now(manager) + 5 or expires <= now(manager):
            raise ValueError('capture lease must have a live fixed 1800 second deadline')
    except (ValueError, TypeError) as exc:
        raise validation_error(str(exc), analysis_session_id=sid) from None
    with manager._lock:
        session = manager._load(sid)
        if session is None:
            raise session_not_found(sid)
        existing = session.get('capture_lease')
        if existing:
            if any(existing.get(k) != body[k] for k in FIELDS):
                raise invalid_state('fixed capture lease cannot be replaced or extended', sid)
            if existing['state'] != 'active' or session.get('sealed') or session['status'] in TERMINAL_SESSION_STATES:
                raise invalid_state('capture lease has ended', sid)
            return status(session)  # A retry never resets liveness.
        identity = session.get('media_identity') or {}
        if identity.get('edge_session_id') and identity['edge_session_id'] != body['edge_session_id']:
            raise invalid_state('capture lease business session mismatch', sid)
        if session.get('configuration', {}).get('sport_id', 'badminton') != body['sport_id']:
            raise invalid_state('capture lease sport mismatch', sid)
        if session.get('sealed') or session['status'] in TERMINAL_SESSION_STATES or session.get('segments'):
            raise invalid_state('capture lease must precede all input', sid)
        # The absolute upload deadline is business-issued, but liveness begins
        # only once this GPU has activated the lease. GPU readiness/creation
        # latency must not consume the terminal's first 120-second signal window.
        lease = dict(body, state='active', last_signal_at=iso(now(manager)), sequence=0)
        session['capture_lease'] = lease
        session['upload_authorization'] = {'token_sha256': body['token_sha256'], 'expires_at': body['expires_at'], 'revoked': False}
        session['retention'] = {'policy': 'persistent_media'}
        manager._save(session)
        return status(session)


def close(manager, session, reason):
    lease = session.get('capture_lease')
    if not lease:
        raise invalid_state('no fixed capture lease exists', session['analysis_session_id'])
    if lease['state'] != 'active':
        return status(session)
    if session.get('sealed') or session['status'] in TERMINAL_SESSION_STATES:
        raise invalid_state('input already ended', session['analysis_session_id'])
    at = iso(now(manager))
    lease.update(state='ended', end_reason=reason, ended_at=at)
    session['upload_authorization']['revoked'] = True
    session.update(capture_incomplete=True, capture_end_reason=reason, capture_ended_at=at, sealed=True, sealed_at=at)
    received = sorted(int(i) for i in session.get('segments', {}))
    # Only a known contiguous prefix can drain safely without inventing a gap
    # or a final unseen fragment. Even then the outcome is explicitly partial.
    can_drain = reason == 'token_expired' and received and received == list(range(received[-1] + 1))
    if can_drain:
        session.update(expected_last_segment_index=received[-1], allow_partial=True, status='draining')
    else:
        session['status'] = 'cancelled'
        manager._cancelled_sessions.add(session['analysis_session_id'])
        session['error'] = {'code': reason, 'message': 'capture input ended', 'retryable': False, 'details': {}}
    payload = {'schema_version': 'capture-lifecycle.v1', 'event_type': 'capture_ended',
               'analysis_session_id': session['analysis_session_id'], 'reason': reason, 'occurred_at': at}
    payload.update({k: lease[k] for k in ('lease_id', 'match_id', 'edge_session_id', 'device_id', 'camera_id', 'sport_id', 'expires_at')})
    payload['event_id'] = event_id(payload)
    # The immutable event itself is the durable outbox. Sender state is a
    # separate file so a retry or callback acknowledgement cannot lose it.
    session['capture_lifecycle_event'] = payload
    manager._append_state_history(session, session['status'], reason)
    manager._emit_status_event(session, session['status'])
    manager._save(session)
    if can_drain:
        manager._notify()
    else:
        manager._finish_trace(session, 'cancelled')
    return status(session)


def enforce_open(manager, session):
    lease = session.get('capture_lease')
    if not lease:
        return
    if lease['state'] == 'active' and not session.get('sealed') and session['status'] not in TERMINAL_SESSION_STATES:
        reason = ('token_expired' if now(manager) >= timestamp(lease['expires_at']) else
                  'input_signal_timeout' if now(manager) - timestamp(lease['last_signal_at']) >= 120 else None)
        if reason:
            close(manager, session, reason)
    if (lease['state'] != 'active' or session.get('sealed') or session['status'] in TERMINAL_SESSION_STATES
            or session.get('upload_authorization', {}).get('revoked')):
        raise StreamSessionError('capture_ended', 'capture authorization is no longer active', 401,
                                 analysis_session_id=session['analysis_session_id'])


def signal(manager, sid, body, token):
    if (not isinstance(body, dict) or set(body) != {'lease_id', 'sequence'}
            or type(body.get('sequence')) is not int or not 1 <= body['sequence'] <= 2**53 - 1):
        raise validation_error('invalid capture signal', analysis_session_id=sid)
    with manager._lock:
        session = manager._load(sid)
        if session is None:
            raise session_not_found(sid)
        lease = session.get('capture_lease')
        if not lease or body['lease_id'] != lease['lease_id']:
            raise StreamSessionError('capture_unauthorized', 'capture lease mismatch', 401, analysis_session_id=sid)
        enforce_open(manager, session)
        if not isinstance(token, str) or not __import__('hmac').compare_digest(hashlib.sha256(token.encode()).hexdigest(), lease['token_sha256']):
            raise StreamSessionError('capture_unauthorized', 'invalid capture token', 401, analysis_session_id=sid)
        if body['sequence'] <= lease['sequence']:
            raise invalid_state('capture signal sequence is stale', sid)
        lease.update(sequence=body['sequence'], last_signal_at=iso(now(manager)))
        manager._save(session)
        return status(session)


def interrupt(manager, sid, body):
    if not isinstance(body, dict) or set(body) != {'lease_id', 'reason'} or body.get('reason') not in REASONS:
        raise validation_error('invalid capture interruption', analysis_session_id=sid)
    with manager._lock:
        session = manager._load(sid)
        if session is None:
            raise session_not_found(sid)
        if session.get('capture_lease', {}).get('lease_id') != body['lease_id']:
            raise invalid_state('capture interruption lease mismatch', sid)
        if body['reason'] == 'token_expired' and now(manager) < timestamp(session['capture_lease']['expires_at']):
            raise invalid_state('capture deadline has not arrived', sid)
        return close(manager, session, body['reason'])


def tick(manager):
    for snapshot in manager._list_sessions():
        if not snapshot.get('capture_lease'):
            continue
        with manager._lock:
            session = manager._load(snapshot['analysis_session_id'])
            if session is None:
                continue
            try:
                enforce_open(manager, session)
            except StreamSessionError as exc:
                if exc.code != 'capture_ended':
                    raise
