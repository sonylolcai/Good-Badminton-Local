"""Bounded media spool admission, including space promised to future assembly."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil

from .stream_errors import StreamSessionError


@dataclass(frozen=True)
class MediaLimits:
    min_free_bytes: int = 10 * 1024**3
    max_cache_bytes: int = 24 * 1024**3
    max_session_bytes: int = 8 * 1024**3
    max_source_seconds: int = 45 * 60
    max_segments: int = 4096
    max_request_bytes: int = 34 * 1024**2
    max_upload_requests: int = 4

    @classmethod
    def from_environment(cls):
        fields = {
            'min_free_bytes': 'GOOD_BADMINTON_STREAM_MIN_FREE_BYTES',
            'max_cache_bytes': 'GOOD_BADMINTON_STREAM_MAX_CACHE_BYTES',
            'max_session_bytes': 'GOOD_BADMINTON_STREAM_MAX_SESSION_BYTES',
            'max_source_seconds': 'GOOD_BADMINTON_STREAM_MAX_SOURCE_SECONDS',
            'max_segments': 'GOOD_BADMINTON_STREAM_MAX_SEGMENTS',
            'max_request_bytes': 'GOOD_BADMINTON_STREAM_MAX_REQUEST_BYTES',
            'max_upload_requests': 'GOOD_BADMINTON_STREAM_MAX_UPLOAD_REQUESTS',
        }
        defaults = cls()
        values = {}
        for field, variable in fields.items():
            try:
                value = int(os.environ.get(variable, getattr(defaults, field)))
            except (ValueError, TypeError):
                raise ValueError(f'{variable} must be a positive integer') from None
            if value <= 0:
                raise ValueError(f'{variable} must be a positive integer')
            values[field] = value
        return cls(**values)


def tree_bytes(root):
    """Count the API data tree without following symlinks into other volumes."""
    total = 0
    def fail(error):
        raise error
    for parent, _, files in os.walk(root, followlinks=False, onerror=fail):
        for name in files:
            path = Path(parent) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass  # An atomic replace or a completed cache release won the race.
    return total


def pending_assembly_bytes(manager):
    if not (getattr(manager, 'media_archive_enabled', False) or os.environ.get('GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED', '0').lower() in {'1','true'}):
        return 0
    promised = 0
    for session in manager._list_sessions():
        if session.get('local_media_release'):
            continue
        directory = manager._session_dir(session['analysis_session_id'])
        raw = sum(path.stat().st_size for path in (directory / 'segments').glob('*.bin') if path.is_file())
        if not raw:
            continue
        # A partially written master already consumes capacity counted by tree_bytes.
        import json
        import re
        master_size = 0
        try:
            state = json.loads((directory / 'media/status.json').read_text())
            if state.get('status') == 'ready':
                continue  # The committed master needs no additional assembly space.
            revision = state.get('revision', '')
            if re.fullmatch(r'[0-9a-f]{64}', revision):
                master = directory / 'media' / revision / 'master.mp4'
                master_size = master.stat().st_size if master.is_file() else 0
        except (OSError, ValueError):
            pass
        promised += max(0, math.ceil(raw * 1.1) + 1024**2 - master_size)
    return promised


def ensure_capacity(manager, additional_bytes, *, session_id=None, extra_master_bytes=0):
    limits = MediaLimits.from_environment()
    additional_bytes = max(0, int(additional_bytes))
    promised = pending_assembly_bytes(manager)
    required = additional_bytes + max(0, int(extra_master_bytes)) + promised
    free = shutil.disk_usage(manager.data_dir).free
    if free - required < limits.min_free_bytes:
        raise StreamSessionError('gpu_media_storage_full',
            'Media cache has insufficient space including the reserved assembled videos; retry after archives are released',
            507, retryable=True, analysis_session_id=session_id,
            details={'min_free_bytes':limits.min_free_bytes, 'free_bytes':free,
                     'pending_assembly_bytes':promised})
    if tree_bytes(manager.data_dir) + required > limits.max_cache_bytes:
        raise StreamSessionError('gpu_media_cache_limit',
            'Media cache budget reached; retry after archives are released',
            507, retryable=True, analysis_session_id=session_id,
            details={'max_cache_bytes':limits.max_cache_bytes})


def ensure_session_limit(session, metadata):
    limits = MediaLimits.from_environment()
    entries = dict(session.get('segments') or {})
    entries[str(metadata['segment_index'])] = metadata
    if len(entries) > limits.max_segments:
        code, message = 'gpu_media_segment_limit', 'Session segment count limit reached; finish this recording'
    elif sum(e['content_length_bytes'] for e in entries.values()) > limits.max_session_bytes:
        code, message = 'gpu_media_session_size_limit', 'Session byte limit reached; finish this recording'
    else:
        # Source time may start after a long preview. Limit the captured span, not its absolute offset.
        start = min(e['source_start_time_sec'] for e in entries.values())
        end = max(e['source_start_time_sec'] + e['duration_sec'] for e in entries.values())
        if math.isfinite(end - start) and end - start <= limits.max_source_seconds:
            return
        code, message = 'gpu_media_duration_limit', 'Session duration limit reached; finish this recording'
    raise StreamSessionError(code, message, 413, retryable=False,
        analysis_session_id=session['analysis_session_id'],
        details={'max_session_bytes':limits.max_session_bytes, 'max_source_seconds':limits.max_source_seconds,
                 'max_segments':limits.max_segments})


def ensure_segment_write(manager, session, metadata):
    ensure_session_limit(session, metadata)
    size = metadata['content_length_bytes']
    extra_master = 0
    if getattr(manager, 'media_archive_enabled', False) or os.environ.get('GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED', '0').lower() in {'1','true'}:
        path = manager._segment_path(session['analysis_session_id'], metadata['segment_index'])
        old_size = path.stat().st_size if path.is_file() else 0
        extra_master = math.ceil(max(0, size - old_size) * 1.1)
    # Atomic replacement temporarily occupies both the old file and the new bytes.
    ensure_capacity(manager, size, session_id=session['analysis_session_id'], extra_master_bytes=extra_master)


def archived_segment_location(manager, session, segment_index):
    """A released local file is durable only under its unexpired cloud commit receipt."""
    from datetime import datetime, timezone
    import hashlib
    import json
    import re
    release = session.get('local_media_release') or {}
    if release.get('phase') not in {'releasing','released'}:
        return None
    revision = release.get('revision', '')
    if not re.fullmatch(r'[0-9a-f]{64}', revision):
        return None
    try:
        if datetime.now(timezone.utc) >= datetime.fromisoformat(release['expires_at']):
            return None
        root = manager._session_dir(session['analysis_session_id']) / 'media'
        raw = (root / revision / 'manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != release['manifest_sha256']:
            return None
        state = json.loads((root / 'status.json').read_text())
        if state.get('status') != 'ready' or state.get('manifest_sha256') != release['manifest_sha256']:
            return None
        manifest = json.loads(raw)
        entry = manifest['segments'][int(segment_index)]
        original = session['segments'][str(segment_index)]
        if entry['segment_index'] != segment_index or any(entry[k] != original[k] for k in ('sha256','content_length_bytes')):
            return None
        return {'storage':'object_storage','object_key':entry['object_key'],'expires_at':release['expires_at']}
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def health_limits(manager):
    from dataclasses import asdict
    value = asdict(MediaLimits.from_environment())
    value['disk_free_bytes'] = shutil.disk_usage(manager.data_dir).free
    value['disk_above_reserve'] = value['disk_free_bytes'] >= value['min_free_bytes']
    return value
