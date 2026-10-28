"""Signed, idempotent GPU result notices; never includes video or credentials."""
import hashlib
import hmac
import json
import time

SCHEMA = 'gpu-result-notification.v1'
MAX_BYTES = 32768

def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def event_id(payload):
    return hashlib.sha256(encode({k: v for k, v in payload.items() if k != 'event_id'})).hexdigest()

def signature(secret, timestamp, raw):
    return hmac.new(secret.encode(), (str(timestamp) + '\n' + hashlib.sha256(raw).hexdigest()).encode(), hashlib.sha256).hexdigest()

def verify(secret, timestamp, supplied_signature, raw, *, now=None):
    if len(secret.encode()) < 32 or len(raw) > MAX_BYTES:
        raise ValueError('invalid callback configuration or payload size')
    try:
        instant = int(timestamp)
    except (TypeError, ValueError):
        raise ValueError('invalid callback timestamp') from None
    if abs((time.time() if now is None else now) - instant) > 300:
        raise ValueError('expired callback signature')
    if not hmac.compare_digest(signature(secret, timestamp, raw), supplied_signature or ''):
        raise ValueError('invalid callback signature')
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema_version') != SCHEMA:
        raise ValueError('unsupported GPU notification')
    if value.get('event_id') != event_id(value):
        raise ValueError('notification identity mismatch')
    if value.get('sport_id') not in {'badminton', 'tennis'}:
        raise ValueError('unsupported notification sport')
    return value
