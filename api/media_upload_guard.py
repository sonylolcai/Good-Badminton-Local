"""Bound multipart spooling before FastAPI parses files into system temp storage."""
from __future__ import annotations

import re
import shutil
import tempfile
import threading
from pathlib import Path

from starlette.responses import JSONResponse
from starlette.formparsers import MultiPartException

from .media_limits import MediaLimits, ensure_capacity
from .stream_errors import StreamSessionError
from .stream_models import SESSION_ID_PATTERN, error_response


class _BodyLimitExceeded(MultiPartException):
    pass


class MediaUploadGuard:
    def __init__(self, app, *, manager):
        self.app, self.manager = app, manager
        self._lock = threading.Lock()
        self._inflight = 0
        self._reserved = 0

    async def __call__(self, scope, receive, send):
        path = scope.get('path','')
        if (scope['type']!='http' or scope.get('method')!='POST'
                or not re.fullmatch(r'/api/v1/stream-sessions/[^/]+/segments/[0-9]+(?:/repair)?/?',path)):
            return await self.app(scope,receive,send)
        limits=MediaLimits.from_environment()
        async def reject(status,code,message,retryable=False):
            session_id=path.split('/')[4]
            if not SESSION_ID_PATTERN.fullmatch(session_id):
                session_id=None
            payload=error_response(code,message,retryable,analysis_session_id=session_id)
            await JSONResponse(payload,status_code=status)(scope,receive,send)
        lengths=[value for name,value in scope.get('headers',[]) if name.lower()==b'content-length']
        if lengths:
            try:
                if len(lengths)!=1 or not lengths[0].isdigit():raise ValueError
                length=int(lengths[0])
            except ValueError:
                return await reject(400,'invalid_content_length','Invalid Content-Length')
            if length>limits.max_request_bytes:
                return await reject(413,'gpu_media_request_limit','Upload request exceeds the bounded segment envelope')
        try:
            with self.manager._lock:
                ensure_capacity(self.manager,0)
        except StreamSessionError as exc:
            return await reject(exc.status_code,exc.code,exc.message,exc.retryable)
        temporary=Path(tempfile.gettempdir())
        same_volume=temporary.stat().st_dev==self.manager.data_dir.stat().st_dev
        reserve=limits.min_free_bytes if same_volume else min(limits.min_free_bytes,64*1024**2)
        with self._lock:
            if self._inflight>=limits.max_upload_requests:
                rejected=(429,'gpu_media_upload_busy','Concurrent upload limit reached',True)
            elif shutil.disk_usage(temporary).free-self._reserved-limits.max_request_bytes<reserve:
                rejected=(507,'gpu_media_temp_storage_full','Temporary upload storage has insufficient free space',True)
            else:
                rejected=None
                self._inflight+=1
                self._reserved+=limits.max_request_bytes
        if rejected:
            return await reject(*rejected)
        received=0
        overflow=False
        async def bounded_receive():
            nonlocal received,overflow
            message=await receive()
            if message['type']=='http.request':
                received+=len(message.get('body',b''))
                if received>limits.max_request_bytes:
                    overflow=True
                    raise _BodyLimitExceeded('Upload request exceeds its media envelope')
            return message
        async def bounded_send(message):
            # Some framework versions map a body reader error to HTTP 400.
            # Suppress that response and return the stable media limit code below.
            if not overflow:
                await send(message)
        try:
            try:
                await self.app(scope,bounded_receive,bounded_send)
            except _BodyLimitExceeded:
                pass
            if overflow:
                await reject(413,'gpu_media_request_limit','Streamed upload exceeds the bounded segment envelope')
        finally:
            with self._lock:
                self._inflight-=1
                self._reserved-=limits.max_request_bytes
