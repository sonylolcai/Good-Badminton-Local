"""Bound multipart spooling before FastAPI parses files into system temp storage."""
from __future__ import annotations

import asyncio
import os
import secrets
from dataclasses import replace
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


class _BodyTimeoutExceeded(MultiPartException):
    pass


_SEGMENT_UPLOAD_PATH_RE = re.compile(r'/api/v1/stream-sessions/([^/]+)/segments/[^/]+(?:/repair)?/?')


class MediaUploadGuard:
    def __init__(self, app, *, manager):
        self.app, self.manager = app, manager
        self._lock = threading.Lock()
        self._inflight = 0
        self._reserved = 0

    def _has_valid_api_key(self, headers: dict[bytes, bytes]) -> bool:
        expected = os.environ.get('GOOD_BADMINTON_API_KEY', '')
        provided = headers.get(b'x-api-key', b'')
        return bool(expected and provided and secrets.compare_digest(provided, expected.encode('utf-8')))

    def _has_valid_stream_upload_access(self, session_id: str, headers: dict[bytes, bytes]) -> bool:
        if self._has_valid_api_key(headers):
            return True
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            return False
        raw_token = headers.get(b'x-upload-token', b'')
        if not raw_token:
            return False
        try:
            upload_token = raw_token.decode('utf-8')
        except UnicodeDecodeError:
            return False
        verifier = getattr(self.manager, 'verify_upload_token', None)
        if not callable(verifier):
            return False
        try:
            return bool(verifier(session_id, upload_token))
        except StreamSessionError:
            return False

    async def __call__(self, scope, receive, send):
        path = scope.get('path','')
        segment_match = _SEGMENT_UPLOAD_PATH_RE.fullmatch(path) if scope.get('type') == 'http' and scope.get('method') == 'POST' else None
        is_job_upload = scope.get('type') == 'http' and scope.get('method') == 'POST' and path in {'/api/v1/jobs', '/api/v1/court/detect'}
        if not is_job_upload and segment_match is None:
            return await self.app(scope,receive,send)
        headers: dict[bytes, bytes] = {}
        for name, value in scope.get('headers', []):
            headers.setdefault(name.lower(), value)
        if is_job_upload:
            if not self._has_valid_api_key(headers):
                return await JSONResponse({'detail':'Invalid API key'}, status_code=401)(scope,receive,send)
        elif not self._has_valid_stream_upload_access(segment_match.group(1), headers):
            return await JSONResponse({'detail':'Invalid session upload credential'}, status_code=401)(scope,receive,send)
        jobs = getattr(self.manager, 'job_manager', None)
        if jobs is not None: jobs.prune_expired_jobs()
        limits=MediaLimits.from_environment()
        if path in {'/api/v1/jobs', '/api/v1/court/detect'}:
            limits = replace(limits, max_request_bytes=int(os.environ.get('GOOD_BADMINTON_JOB_MAX_REQUEST_BYTES', str(1024**3))))
        async def reject(status,code,message,retryable=False):
            parts=path.split('/')
            session_id=parts[4] if len(parts)>4 else ''
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
        temporary=Path(tempfile.gettempdir())
        same_volume=temporary.stat().st_dev==self.manager.data_dir.stat().st_dev
        reserve=limits.min_free_bytes if same_volume else min(limits.min_free_bytes,64*1024**2)
        request_reserve = length if lengths else limits.max_request_bytes
        with self._lock:
            try:
                with self.manager._lock:
                    ensure_capacity(self.manager, self._reserved + request_reserve * 2)
            except StreamSessionError as exc:
                capacity_error = exc
            else:
                capacity_error = None
            if capacity_error is not None:
                rejected = (capacity_error.status_code, capacity_error.code, capacity_error.message, capacity_error.retryable)
            elif self._inflight>=limits.max_upload_requests:
                rejected=(429,'gpu_media_upload_busy','Concurrent upload limit reached',True)
            elif shutil.disk_usage(temporary).free-self._reserved-request_reserve<reserve:
                rejected=(507,'gpu_media_temp_storage_full','Temporary upload storage has insufficient free space',True)
            else:
                rejected=None
                self._inflight+=1
                self._reserved+=request_reserve
        if rejected:
            return await reject(*rejected)
        received=0
        overflow=False
        timed_out=False
        body_complete=False
        response_started=False
        deadline=asyncio.get_running_loop().time()+limits.upload_body_timeout_seconds
        async def bounded_receive():
            nonlocal received,overflow,timed_out,body_complete
            if not body_complete:
                remaining=deadline-asyncio.get_running_loop().time()
                if remaining<=0:
                    timed_out=True
                    raise _BodyTimeoutExceeded('Upload request body timed out')
                try:
                    message=await asyncio.wait_for(receive(),timeout=remaining)
                except asyncio.TimeoutError as exc:
                    timed_out=True
                    raise _BodyTimeoutExceeded('Upload request body timed out') from exc
            else:
                message=await receive()
            if message['type']=='http.request':
                received+=len(message.get('body',b''))
                if received>limits.max_request_bytes:
                    overflow=True
                    raise _BodyLimitExceeded('Upload request exceeds its media envelope')
                if not message.get('more_body',False):
                    body_complete=True
            return message
        async def bounded_send(message):
            nonlocal response_started
            # Some framework versions map a body reader error to HTTP 400.
            # Suppress that response and return the stable media limit/timeout code below.
            if not overflow and not timed_out:
                if message.get('type')=='http.response.start':
                    response_started=True
                await send(message)
        try:
            try:
                await self.app(scope,bounded_receive,bounded_send)
            except (_BodyLimitExceeded,_BodyTimeoutExceeded):
                pass
            if overflow and not response_started:
                await reject(413,'gpu_media_request_limit','Streamed upload exceeds the bounded segment envelope')
            elif timed_out and not response_started:
                await reject(408,'gpu_media_upload_timeout','Upload request timed out while reading body',True)
        finally:
            with self._lock:
                self._inflight-=1
                self._reserved-=request_reserve
