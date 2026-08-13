"""HTTP boundary for the GPU-only Good-Badminton analysis service."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .jobs import AnalysisJobManager


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024


def create_app(data_dir=None, start_worker=True):
    data_path = Path(data_dir or os.environ.get("GOOD_BADMINTON_API_DATA_DIR", "api_data")).resolve()
    manager = AnalysisJobManager(data_path, start_worker=start_worker)
    app = FastAPI(title="Good-Badminton GPU API", version="1.0.0")
    app.state.job_manager = manager

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
            "api_auth_configured": bool(os.environ.get("GOOD_BADMINTON_API_KEY")),
        }

    @app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_api_key)])
    async def create_job(
        video: UploadFile = File(...),
        template: UploadFile = File(...),
        court_corners: str = Form(...),
        options_json: str = Form("{}"),
    ):
        corners = _parse_corners(court_corners)
        options = _parse_options(options_json)
        job_id = os.urandom(16).hex()
        staging_dir = data_path / "staging" / job_id
        staging_dir.mkdir(parents=True, exist_ok=False)
        try:
            video_path = await _save_upload(video, staging_dir, VIDEO_EXTENSIONS, "video")
            template_path = await _save_upload(template, staging_dir, IMAGE_EXTENSIONS, "template")
            job = manager.create_job(video_path, template_path, corners, options)
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
            return _job_response(manager.get_job(job["job_id"]))
        except HTTPException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
    def get_job(job_id: str):
        job = _require_job(manager, job_id)
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


def _job_response(job):
    result = job.get("result")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "progress": job.get("progress"),
        "error": job.get("error"),
        "result": result,
    }


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
        "pose_imgsz": 1280,
        "pose_conf": 0.15,
        "far_player_enhancement": False,
        "far_pose_roi": [0.12, 0.30, 0.86, 0.82],
    }
    unsupported = set(received).difference(defaults)
    if unsupported:
        raise HTTPException(status_code=422, detail=f"Unsupported options: {sorted(unsupported)}")
    options = {**defaults, **received}
    if options["pose_imgsz"] not in {640, 960, 1280}:
        raise HTTPException(status_code=422, detail="pose_imgsz must be 640, 960, or 1280")
    if not 0 < float(options["pose_conf"]) <= 1:
        raise HTTPException(status_code=422, detail="pose_conf must be in (0, 1]")
    if options["output_video_style"] not in {"annotated", "skeleton"}:
        raise HTTPException(status_code=422, detail="output_video_style must be annotated or skeleton")
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

    uvicorn.run("api.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))
