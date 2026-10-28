"""Commit terminal visual evidence under the business-allocated OSS directory."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from good_badminton_contracts.gpu_notifications import encode
from .media_identity import media_prefix
from .media_storage import sha256_file


def upload_analysis_archive(manager, storage, session):
    sid = session['analysis_session_id']
    root = manager._session_dir(sid)
    # Freeze source bytes before hashing/uploading: terminal status can become
    # visible just before the final trace write. Never commit a mixed revision.
    with tempfile.TemporaryDirectory(prefix='.analysis-snapshot-', dir=root) as temporary:
        files = []
        for role, source in [('events', manager._events_path(sid)), ('trace', manager.trace_path(sid))]:
            if source is None or not Path(source).is_file() or not Path(source).stat().st_size:
                continue
            name = 'events.jsonl' if role == 'events' else 'trace.json'
            target = Path(temporary) / name
            shutil.copyfile(source, target)
            if role == 'trace':
                trace = json.loads(target.read_text())
                if trace.get('terminal_status') != session['status'] or not trace.get('finished_at'):
                    raise IOError('waiting for final trace commit')
            files.append({'role': role, 'name': name, 'sha256': sha256_file(target),
                          'content_length_bytes': target.stat().st_size, 'source': target})
        if not files:
            raise ValueError('terminal visual evidence is unavailable')
        return _commit(manager, storage, session, files)


def _commit(manager, storage, session, files):
    sid = session['analysis_session_id']
    identity = session['media_identity']
    sport = session.get('configuration', {}).get('sport_id', 'badminton')
    summary = {'analysis_session_id': sid, 'media_identity': identity, 'sport_id': sport,
               'analysis_status': session['status'],
               'objects': [{k: v for k, v in item.items() if k != 'source'} for item in files]}
    revision = hashlib.sha256(encode(summary)).hexdigest()
    base = media_prefix(identity) + '/analysis/' + revision
    for item in files:
        item['object_key'] = base + '/' + item['name']
        storage.upload(item['object_key'], item['source'])
        head = storage.head(item['object_key'])
        if head['size_bytes'] != item['content_length_bytes'] or head['sha256'] != item['sha256']:
            raise IOError('analysis object identity mismatch')
    manifest = {**summary, 'schema_version': 'gpu-analysis-archive.v1', 'revision': revision,
                'objects': [{k: v for k, v in item.items() if k != 'source'} for item in files]}
    raw = encode(manifest)
    directory = manager._session_dir(sid) / 'analysis-archive' / revision
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'manifest.json'
    path.write_bytes(raw)
    key = base + '/manifest.json'
    storage.upload(key, path)  # Commit last, after every immutable evidence object.
    head = storage.head(key)
    digest = hashlib.sha256(raw).hexdigest()
    if head['size_bytes'] != len(raw) or head['sha256'] != digest:
        raise IOError('analysis manifest identity mismatch')
    return {'manifest': manifest, 'manifest_object_key': key, 'manifest_sha256': digest, 'content_length_bytes': len(raw)}
