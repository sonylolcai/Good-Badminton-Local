"""HTTP boundary for the GPU-only Good-Badminton analysis service."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from .jobs import AnalysisJobManager
from .stream_errors import StreamSessionError, validation_error
from .stream_models import MAX_SEGMENT_BYTES, validate_create_request
from .stream_runtime import StreamProcessorFactory
from .stream_sessions import (
    StreamSessionManager,
    cancel_session_handler,
    complete_session_handler,
    create_session_handler,
    get_session_status_handler,
    read_events_handler,
    submit_segment_handler,
)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024


def create_app(
    data_dir=None,
    start_worker=True,
    *,
    stream_processor_factory=None,
    stream_manager=None,
):
    data_path = Path(data_dir or os.environ.get("GOOD_BADMINTON_API_DATA_DIR", "api_data")).resolve()
    manager = AnalysisJobManager(data_path, start_worker=start_worker)
    if stream_manager is None:
        stream_processor_factory = stream_processor_factory or StreamProcessorFactory(data_path)
        stream_manager = StreamSessionManager(
            data_path,
            processor_factory=stream_processor_factory,
            start_worker=start_worker,
        )
    app = FastAPI(title="Good-Badminton GPU API", version="1.0.0")
    app.state.job_manager = manager
    app.state.stream_manager = stream_manager

    @app.exception_handler(StreamSessionError)
    async def handle_stream_session_error(_request, exc):
        return JSONResponse(status_code=exc.status_code, content=exc.to_error_response())

    def require_api_key(x_api_key: Optional[str] = Header(default=None)):
        expected = os.environ.get("GOOD_BADMINTON_API_KEY")
        if not expected:
            raise HTTPException(status_code=503, detail="API authentication is not configured")
        if x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok",
            "service": "good-badminton-gpu-api",
            "worker_running": manager.worker_running,
            "stream_worker_running": stream_manager.worker_running,
            "api_auth_configured": bool(os.environ.get("GOOD_BADMINTON_API_KEY")),
        }

    @app.post(
        "/api/v1/stream-sessions",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def create_stream_session(
        body: dict,
        x_idempotency_key: Optional[str] = Header(default=None),
    ):
        # Validate once before touching the model runtime so unknown business
        # fields can never be silently dropped at the GPU boundary.
        try:
            normalized = validate_create_request(body)
        except ValueError as exc:
            raise validation_error(str(exc))
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
        dependencies=[Depends(require_api_key)],
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

    @app.get(
        "/api/v1/stream-sessions/{session_id}",
        dependencies=[Depends(require_api_key)],
    )
    async def get_stream_session(session_id: str):
        response_status, payload = get_session_status_handler(stream_manager, session_id)
        return JSONResponse(status_code=response_status, content=payload)

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

    @app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
    async def create_job(
        video: UploadFile = File(...),
        template: UploadFile = File(...),
        court_corners: str = Form(...),
        options_json: str = Form("{}"),
        x_idempotency_key: Optional[str] = Header(default=None),
    ):
        corners = _parse_corners(court_corners)
        options = _parse_options(options_json)
        if x_idempotency_key is not None and not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", x_idempotency_key):
            raise HTTPException(status_code=422, detail="X-Idempotency-Key must be 16-128 safe characters")
        existing = manager.get_by_idempotency_key(x_idempotency_key)
        if existing is not None:
            return _job_response(existing, receipt_reused=True)
        job_id = os.urandom(16).hex()
        staging_dir = data_path / "staging" / job_id
        staging_dir.mkdir(parents=True, exist_ok=False)
        try:
            video_path = await _save_upload(video, staging_dir, VIDEO_EXTENSIONS, "video")
            template_path = await _save_upload(template, staging_dir, IMAGE_EXTENSIONS, "template")
            job = manager.create_job(video_path, template_path, corners, options, x_idempotency_key)
            destination = data_path / "jobs" / job["job_id"] / "input"
            destination.mkdir(parents=True, exist_ok=True)
            video_target = destination / video_path.name
            template_target = destination / template_path.name
            shutil.move(str(video_path), str(video_target))
            shutil.move(str(template_path), str(template_target))
            shutil.rmtree(staging_dir, ignore_errors=True)
            # The worker is queued after creation.  Store stable final paths in
            # its manifest so a temporary staging directory is never referenced.
            stored = manager.get_job(job["job_id"])
            stored["input"]["video_path"] = str(video_target)
            stored["input"]["template_path"] = str(template_target)
            manager._write_job(stored)
            manager.enqueue(
                job["job_id"],
                video_target,
                template_target,
                corners,
                options,
                destination.parent / "output",
            )
            return _job_response(manager.get_job(job["job_id"]), receipt_reused=False)
        except HTTPException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    @app.get("/api/v1/jobs/by-idempotency/{idempotency_key}", dependencies=[Depends(require_api_key)])
    def get_job_by_idempotency(idempotency_key: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", idempotency_key):
            raise HTTPException(status_code=404, detail="Job not found")
        job = manager.get_by_idempotency_key(idempotency_key)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_response(job, receipt_reused=True)

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
    def get_job(job_id: str):
        job = _require_job(manager, job_id)
        return _job_response(job)

    @app.delete("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
    def cancel_job(job_id: str):
        _require_job(manager, job_id)
        job = manager.cancel_job(job_id)
        return _job_response(job)

    @app.get("/api/v1/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
    def get_result(job_id: str):
        job = _require_job(manager, job_id)
        if job["status"] != "succeeded":
            raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
        response = _job_response(job)
        for name in response["result"]["artifacts"]:
            response["result"]["artifacts"][name]["url"] = f"/api/v1/jobs/{job_id}/artifacts/{name}"
        return response

    @app.get("/api/v1/jobs/{job_id}/performance-trace", dependencies=[Depends(require_api_key)])
    def get_performance_trace(job_id: str):
        """Download the durable timing trace for any terminal task state."""
        _require_job(manager, job_id)
        path = manager.performance_trace_path(job_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Performance trace is not available yet")
        return FileResponse(path, media_type="application/json", filename=path.name)

    @app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_name}", dependencies=[Depends(require_api_key)])
    def get_artifact(job_id: str, artifact_name: str):
        _require_job(manager, job_id)
        path = manager.artifact_path(job_id, artifact_name)
        if path is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(path, media_type=manager._media_type(path), filename=path.name)

    return app


def _require_job(manager, job_id):
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _job_response(job, receipt_reused=None):
    result = job.get("result")
    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "progress": job.get("progress"),
        "error": job.get("error"),
        "execution": job.get("execution"),
        "timing": job.get("timing"),
        "performance_trace": job.get("performance_trace"),
        "tracking": job.get("tracking"),
        "result": result,
        "state_history": job.get("state_history", []),
    }
    if receipt_reused is not None:
        response["receipt"] = {
            "accepted": True,
            "accepted_at": (job.get("request") or {}).get("accepted_at") or job.get("created_at"),
            "reused": receipt_reused,
            "status_url": f"/api/v1/jobs/{job['job_id']}",
            "result_url": f"/api/v1/jobs/{job['job_id']}/result",
            "poll_after_seconds": 2,
        }
    if response["performance_trace"] is not None:
        response["performance_trace"] = {
            **response["performance_trace"],
            "url": f"/api/v1/jobs/{job['job_id']}/performance-trace",
        }
    return response


def _parse_corners(value):
    try:
        corners = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="court_corners must be a JSON array") from exc
    if not isinstance(corners, list) or len(corners) != 4:
        raise HTTPException(status_code=422, detail="court_corners must contain exactly four points")
    normalized = []
    for point in corners:
        if not isinstance(point, list) or len(point) != 2:
            raise HTTPException(status_code=422, detail="Each court corner must be [x, y]")
        try:
            normalized.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Court corner coordinates must be numeric") from exc
    return normalized


def _parse_options(value):
    try:
        received = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="options_json must be a JSON object") from exc
    if not isinstance(received, dict):
        raise HTTPException(status_code=422, detail="options_json must be a JSON object")
    defaults = {
        "language": "zh",
        "pose_family": "yolo-pose",
        "pose_mode": "balanced",
        "audio": False,
        "show_skeletons": True,
        "show_player_trajectories": True,
        "show_court_trajectory": True,
        "show_shuttlecock_trajectory": True,
        "show_player_stats": True,
        "show_pose_roi": False,
        "visualize_positions": True,
        "output_video_style": "skeleton",
        # Data is the production contract. Rendering and media export are
        # explicit diagnostics/review options rather than default compute.
        "generate_annotated_video": False,
        "browser_video_reencode": False,
        # Fixed-camera production keeps a timestamped 10 Hz pose budget.  A
        # caller can still explicitly request 0 for an offline full-frame
        # evidence run, but it is not suitable as the streaming default.
        "pose_imgsz": 960,
        # One shared cadence for all measurement-producing components.  The
        # legacy pose_sample_hz key remains accepted for older business
        # clients, but is normalized to this value below.
        "analysis_sample_hz": 10.0,
        "pose_sample_hz": 10.0,
        "pose_conf": 0.15,
        "far_player_enhancement": False,
        "far_pose_roi": [0.12, 0.30, 0.86, 0.82],
        "match_mode": "singles",
        "lock_match_roster": True,
        "roster_stable_frames": 2,
        # ByteTrack is the production association source. It is invoked only
        # on the same timestamp buckets as pose/shuttle/JSONL measurement;
        # it never turns a 10/15/30 Hz task back into full-frame tracking.
        "tracker_backend": "bytetrack",
        "enable_bytetrack": True,
        # YOLO remains the low-latency default.  TrackNetV3 is an explicit,
        # slower accuracy experiment and must never start from an omitted API
        # option.
        "shuttle_detector": "yolo",
        # When the caller opts out of ball detection, a cautious all-player
        # stability window can still provide review-only rally boundaries.
        "movement_rally_settle_seconds": 0.7,
        # A scene/action model is post-processing only.  It can run only with
        # a ball detector and a separately configured, reviewed checkpoint.
        "enable_huji_play_state": True,
        # Opaque business-session reference only. Participant check IDs remain
        # on the business service and never become visual identity evidence.
        "match_session_ref": None,
    }
    unsupported = set(received).difference(defaults)
    if unsupported:
        raise HTTPException(status_code=422, detail=f"Unsupported options: {sorted(unsupported)}")
    options = {**defaults, **received}
    if "analysis_sample_hz" in received:
        options["pose_sample_hz"] = options["analysis_sample_hz"]
    else:
        options["analysis_sample_hz"] = options["pose_sample_hz"]
    if options["pose_imgsz"] not in {640, 960, 1280}:
        raise HTTPException(status_code=422, detail="pose_imgsz must be 640, 960, or 1280")
    sample_hz = float(options["analysis_sample_hz"])
    if sample_hz < 0.0 or (0.0 < sample_hz < 1.0):
        raise HTTPException(
            status_code=422,
            detail="analysis_sample_hz must be 0 (every source frame) or at least 1",
        )
    if not 0 < float(options["pose_conf"]) <= 1:
        raise HTTPException(status_code=422, detail="pose_conf must be in (0, 1]")
    if options["output_video_style"] not in {"annotated", "skeleton"}:
        raise HTTPException(status_code=422, detail="output_video_style must be annotated or skeleton")
    for key in ("generate_annotated_video", "browser_video_reencode", "enable_huji_play_state"):
        if not isinstance(options[key], bool):
            raise HTTPException(status_code=422, detail=f"{key} must be a JSON boolean")
    if not options["generate_annotated_video"]:
        # There is no media source to transcode. Keep the persisted option
        # truthful so the terminal performance trace explains the omission.
        options["browser_video_reencode"] = False
    if options["match_mode"] not in {"singles", "doubles"}:
        raise HTTPException(status_code=422, detail="match_mode must be singles or doubles")
    if options["tracker_backend"] not in {"court_association", "bytetrack"}:
        raise HTTPException(status_code=422, detail="tracker_backend must be court_association or bytetrack")
    if not isinstance(options["enable_bytetrack"], bool):
        raise HTTPException(status_code=422, detail="enable_bytetrack must be a JSON boolean")
    if options["tracker_backend"] == "bytetrack" and not options["enable_bytetrack"]:
        raise HTTPException(
            status_code=422,
            detail="enable_bytetrack must be true when tracker_backend is bytetrack",
        )
    if options["shuttle_detector"] not in {"none", "yolo", "tracknet_v3"}:
        raise HTTPException(status_code=422, detail="shuttle_detector must be none, yolo, or tracknet_v3")
    if float(options["movement_rally_settle_seconds"]) not in {0.5, 0.7, 1.0}:
        raise HTTPException(
            status_code=422,
            detail="movement_rally_settle_seconds must be 0.5, 0.7, or 1.0",
        )
    if int(options["roster_stable_frames"]) < 1 or int(options["roster_stable_frames"]) > 10:
        raise HTTPException(status_code=422, detail="roster_stable_frames must be between 1 and 10")
    if options["match_session_ref"] is not None and not re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}", str(options["match_session_ref"])
    ):
        raise HTTPException(status_code=422, detail="match_session_ref must be a safe opaque reference")
    options["lock_match_roster"] = bool(options["lock_match_roster"])
    options["enable_bytetrack"] = bool(options["enable_bytetrack"])
    options["roster_stable_frames"] = int(options["roster_stable_frames"])
    options["analysis_sample_hz"] = float(options["analysis_sample_hz"])
    options["pose_sample_hz"] = options["analysis_sample_hz"]
    options["movement_rally_settle_seconds"] = float(options["movement_rally_settle_seconds"])
    options["enable_huji_play_state"] = bool(options["enable_huji_play_state"])
    options["match_session_ref"] = (
        str(options["match_session_ref"]) if options["match_session_ref"] is not None else None
    )
    return options


async def _save_upload(upload, staging_dir, allowed_extensions, field_name):
    filename = Path(upload.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=422, detail=f"{field_name} must have one of: {allowed}")
    path = staging_dir / f"{field_name}{suffix}"
    total = 0
    with path.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                output.close()
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"{field_name} exceeds 10 GiB limit")
            output.write(chunk)
    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"{field_name} is empty")
    return path


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
