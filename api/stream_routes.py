"""HTTP adapter for the pure, anonymous GPU stream-session contract.

The adapter deliberately knows only how to authenticate, persist and retrieve
stream-session data.  It does not import whole-video jobs, WebUI pipelines,
business identifiers, scoring, rally construction, reports or renderers.
Both the legacy badminton facade and the new sport-fixed GPU apps reuse this
single adapter so their stream contract cannot drift.
"""

from __future__ import annotations

import json
from typing import Callable, Mapping, Optional

from fastapi import Depends, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from .stream_errors import StreamSessionError, validation_error
from badminton_analysis.inference_policy import execution_profile
from .stream_models import MAX_SEGMENT_BYTES, validate_create_request
from .stream_sessions import (
    cancel_session_handler,
    complete_session_handler,
    create_session_handler,
    get_session_status_handler,
    read_events_handler,
    repair_segment_handler,
    submit_segment_handler,
)


def register_stream_routes(
    app,
    *,
    stream_manager,
    require_api_key: Callable,
    require_upload_access: Callable,
    health_payload: Callable[[], Mapping],
    media_archive=None,
) -> None:
    """Attach the versioned stream API to an application composition root."""

    from .direct_preview import register_preview_routes
    register_preview_routes(app,stream_manager,require_api_key)
    from .media_archive import archive_from_environment
    from .media_upload_guard import MediaUploadGuard
    app.add_middleware(MediaUploadGuard, manager=stream_manager)
    media_archive = media_archive or archive_from_environment(stream_manager)
    app.state.media_archive = media_archive
    from .runtime_control import register_runtime_control
    register_runtime_control(app, stream_manager, media_archive, require_api_key)
    if media_archive is not None:
        if stream_manager.worker_running:
            media_archive.start()
        app.router.add_event_handler("shutdown", media_archive.stop)

    @app.exception_handler(StreamSessionError)
    async def handle_stream_session_error(_request, exc):
        return JSONResponse(status_code=exc.status_code, content=exc.to_error_response())

    @app.get("/api/v1/health")
    def health():
        from .media_limits import health_limits
        payload = dict(health_payload())
        return {
            **payload,
            "stream_processing_mode": getattr(stream_manager.processor_factory, "processing_mode", "analysis"),
            "media_limits": health_limits(stream_manager),
            "direct_preview_version": "direct-preview.v1",
            "roster_identity_policies": (["individual_tracks.v1", "half_court_slots.v1"]
                                         if payload.get('sport_id') == 'badminton' else ["individual_tracks.v1"]),
            "media_archive_enabled": media_archive is not None,
            "media_archive_running": media_archive is not None and media_archive.running,
        }

    def require_media_archive(session_id):
        if stream_manager.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        if media_archive is None:
            raise HTTPException(status_code=503, detail="media archive is not configured")
        return media_archive

    @app.get("/api/v1/stream-sessions/{session_id}/media", dependencies=[Depends(require_api_key)])
    def get_media_archive(session_id: str):
        return require_media_archive(session_id).status(session_id)

    @app.post("/api/v1/stream-sessions/{session_id}/media/prepare", dependencies=[Depends(require_api_key)])
    def prepare_media_archive(session_id: str):
        archive = require_media_archive(session_id)
        if not stream_manager.get_session(session_id).get("sealed"):
            raise HTTPException(status_code=409, detail="seal input before preparing media")
        archive.request_retry(session_id)
        return JSONResponse(status_code=202, content=archive.status(session_id))

    @app.get("/api/v1/stream-sessions/{session_id}/media/manifest", dependencies=[Depends(require_api_key)])
    def get_media_manifest(session_id: str):
        from .media_archive import MediaValidationError
        try:
            return require_media_archive(session_id).manifest(session_id)
        except MediaValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/v1/stream-sessions",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def create_stream_session(
        body: dict,
        x_idempotency_key: Optional[str] = Header(default=None),
        x_expected_inference_profile: Optional[str] = Header(default=None),
    ):
        # Reject unsupported payload fields before a manager creates state;
        # profile/mode synchronization happens immediately afterwards, before
        # model runtime construction or persistence.
        try:
            normalized = validate_create_request(body)
        except ValueError as exc:
            raise validation_error(str(exc))
        if x_expected_inference_profile:
            config = normalized['configuration']
            profile = execution_profile(config.get('sport_id', 'badminton'), config['shuttle_detector'], 'stream')
            if profile['sha256'] != x_expected_inference_profile:
                raise HTTPException(status_code=409, detail='GPU stream inference profile does not match the caller')
        validator = getattr(stream_manager.processor_factory, "validate_session_request", None)
        if callable(validator):
            try:
                validator(normalized)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                raise validation_error(str(exc))
        response_status, payload = create_session_handler(
            stream_manager,
            body,
            x_idempotency_key,
        )
        return JSONResponse(status_code=response_status, content=payload)

    @app.post(
        "/api/v1/stream-sessions/{session_id}/segments/{segment_index}",
        dependencies=[Depends(require_upload_access)],
    )
    async def submit_stream_segment(
        session_id: str,
        segment_index: int,
        segment: UploadFile = File(...),
        metadata: str = Form(...),
    ):
        try:
            metadata_body = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise validation_error(
                "metadata must be a JSON object",
                analysis_session_id=session_id,
            ) from exc
        # UploadFile is spooled by Starlette; cap the one segment read so the
        # service never accepts an unbounded request into the engine queue.
        data_bytes = await segment.read(MAX_SEGMENT_BYTES + 1)
        response_status, payload = submit_segment_handler(
            stream_manager,
            session_id,
            segment_index,
            metadata_body,
            data_bytes,
        )
        return JSONResponse(status_code=response_status, content=payload)

    @app.post(
        "/api/v1/stream-sessions/{session_id}/segments/{segment_index}/repair",
        dependencies=[Depends(require_upload_access)],
    )
    async def repair_stream_segment(
        session_id: str,
        segment_index: int,
        segment: UploadFile = File(...),
        metadata: str = Form(...),
        x_grant_token: Optional[str] = Header(default=None),
    ):
        try:
            metadata_body = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise validation_error(
                "metadata must be a JSON object",
                analysis_session_id=session_id,
            ) from exc
        data_bytes = await segment.read(MAX_SEGMENT_BYTES + 1)
        response_status, payload = repair_segment_handler(
            stream_manager,
            session_id,
            segment_index,
            metadata_body,
            data_bytes,
            x_grant_token,
        )
        return JSONResponse(status_code=response_status, content=payload)

    @app.get(
        "/api/v1/stream-sessions/{session_id}",
        dependencies=[Depends(require_api_key)],
    )
    async def get_stream_session(session_id: str):
        response_status, payload = get_session_status_handler(stream_manager, session_id)
        return JSONResponse(status_code=response_status, content=payload)

    @app.post(
        "/api/v1/stream-sessions/{session_id}/gaps/{segment_index}",
        dependencies=[Depends(require_upload_access)],
    )
    async def declare_stream_gap(session_id: str, segment_index: int, body: dict):
        response_status, payload = stream_manager.declare_gap(session_id, segment_index, body)
        return JSONResponse(status_code=response_status, content=payload)

    @app.post(
        "/api/v1/stream-sessions/{session_id}/repairs",
        dependencies=[Depends(require_api_key)],
    )
    async def request_stream_repairs(
        session_id: str,
        body: dict,
        x_idempotency_key: Optional[str] = Header(default=None),
    ):
        response_status, payload = stream_manager.request_repairs(
            session_id, body, x_idempotency_key
        )
        return JSONResponse(status_code=response_status, content=payload)

    @app.get(
        "/api/v1/stream-sessions/{session_id}/repairs",
        dependencies=[Depends(require_api_key)],
    )
    async def list_stream_repairs(session_id: str):
        response_status, payload = stream_manager.list_repairs(session_id)
        return JSONResponse(status_code=response_status, content=payload)

    @app.post(
        "/api/v1/stream-sessions/{session_id}/repairs/{request_id}/abandon",
        dependencies=[Depends(require_api_key)],
    )
    async def abandon_stream_repair(session_id: str, request_id: str):
        response_status, payload = stream_manager.abandon_repair(session_id, request_id)
        return JSONResponse(status_code=response_status, content=payload)

    @app.post(
        "/api/v1/stream-sessions/{session_id}/upload-authorization",
        dependencies=[Depends(require_api_key)],
    )
    async def issue_upload_authorization(session_id: str):
        return stream_manager.issue_upload_token(session_id)

    @app.delete(
        "/api/v1/stream-sessions/{session_id}/upload-authorization",
        dependencies=[Depends(require_api_key)],
    )
    async def revoke_upload_authorization(session_id: str):
        return stream_manager.revoke_upload_token(session_id)

    @app.get(
        "/api/v1/stream-sessions/{session_id}/segments/{segment_index}/receipt",
        dependencies=[Depends(require_api_key)],
    )
    async def get_segment_receipt(session_id: str, segment_index: int):
        return stream_manager.get_segment_receipt(session_id, segment_index)

    @app.get(
        "/api/v1/stream-sessions/{session_id}/events",
        dependencies=[Depends(require_api_key)],
    )
    async def get_stream_events(
        session_id: str,
        cursor: Optional[str] = None,
        limit: int = 100,
    ):
        response_status, payload = read_events_handler(
            stream_manager,
            session_id,
            cursor=cursor,
            limit=limit,
        )
        return JSONResponse(status_code=response_status, content=payload)

    @app.get(
        "/api/v1/stream-sessions/{session_id}/trace",
        dependencies=[Depends(require_api_key)],
    )
    async def get_stream_trace(session_id: str):
        # Reuse the status lookup for the same structured not-found behavior.
        get_session_status_handler(stream_manager, session_id)
        path = stream_manager.trace_path(session_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Stream trace is not available")
        return FileResponse(path, media_type="application/json", filename=path.name)

    @app.get(
        "/api/v1/stream-sessions/{session_id}/candidate-photos/{track_id}",
        dependencies=[Depends(require_api_key)],
    )
    async def get_stream_candidate_photo(session_id: str, track_id: str):
        path = stream_manager.candidate_photo_path(session_id, track_id)
        if path is None:
            raise HTTPException(status_code=404, detail="candidate photo is not available")
        return FileResponse(path, media_type="image/jpeg", filename=f"{track_id}.jpg")

    @app.post(
        "/api/v1/stream-sessions/{session_id}/complete",
        dependencies=[Depends(require_api_key)],
    )
    async def complete_stream_session(session_id: str, body: dict):
        response_status, payload = complete_session_handler(stream_manager, session_id, body)
        return JSONResponse(status_code=response_status, content=payload)

    @app.delete(
        "/api/v1/stream-sessions/{session_id}",
        dependencies=[Depends(require_api_key)],
    )
    async def cancel_stream_session(session_id: str):
        response_status, payload = cancel_session_handler(stream_manager, session_id)
        return JSONResponse(status_code=response_status, content=payload)
