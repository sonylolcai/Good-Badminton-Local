"""Bounded authenticated read of a durable received MP4; never runs inference."""
import hashlib
import re
from fastapi import Depends,HTTPException
from fastapi.responses import Response

MAX_PREVIEW_BYTES=16*1024*1024
VERSION='direct-preview.v1'

def register_preview_routes(app, manager, require_api_key):
    @app.get('/api/v1/stream-sessions/{session_id}/preview/latest.mp4',dependencies=[Depends(require_api_key)])
    def latest_preview(session_id:str):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',session_id):
            raise HTTPException(status_code=404,detail='session not found')
        session=manager.get_session(session_id)
        if not session:raise HTTPException(status_code=404,detail='session not found')
        segments=list((session.get('segments') or {}).values())
        candidates=[s for s in segments if type(s.get('segment_index')) is int and s['segment_index']>=0
                    and s.get('content_type')=='video/mp4' and s.get('sha256') and not s.get('declared_missing')]
        if not candidates:raise HTTPException(status_code=404,detail='no durable MP4 preview yet')
        entry=max(candidates,key=lambda s:s['segment_index'])
        size=entry.get('content_length_bytes')
        if type(size) is not int or not 0<size<=MAX_PREVIEW_BYTES:
            raise HTTPException(status_code=413,detail='preview exceeds read limit')
        path=manager._segment_path(session_id,entry['segment_index'])
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(manager.sessions_dir.resolve()):
                raise ValueError('invalid segment path')
            with path.open('rb') as stream:data=stream.read(MAX_PREVIEW_BYTES+1)
        except (OSError,ValueError):
            raise HTTPException(status_code=409,detail='preview is unavailable or archived') from None
        if len(data)!=size or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise HTTPException(status_code=409,detail='preview receipt does not match stored bytes')
        return Response(content=data,media_type='video/mp4',headers={'Cache-Control':'private, no-store',
            'X-Stream-Segment-Index':str(entry['segment_index']),'X-Content-SHA256':entry['sha256']})
