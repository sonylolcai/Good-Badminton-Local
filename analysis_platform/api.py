"""HTTP boundary for the standalone Next.js evaluation platform."""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .app_service import (
    clone_run_task,
    compare_run_reports,
    create_run,
    execute_uploaded_run,
    execute_uploaded_stream_run,
    get_run_detail,
    list_manifests,
    list_run_summaries,
    publish_manifest,
    recover_stream_session,
    save_manifest_draft,
    verify_gpu_service,
    verify_published_manifest,
)
from .review import (
    REVIEW_DECISIONS,
    RALLY_TERMINAL_OUTCOMES,
    SHOT_TYPES,
    add_manual_candidate,
    add_manual_rally_terminal,
    analysis_run_label,
    candidate_choices,
    candidate_table,
    candidate_view,
    create_or_load_review_session,
    find_analysis_runs,
    merge_review_candidates,
    review_summary,
    reviewed_rally_table,
    save_human_review,
    split_review_candidate,
)
from .runner import prepare_court_from_video
from .task_control import AnalysisTaskController


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TASKS = AnalysisTaskController()
ManifestKind = Literal["dataset", "dataset_version", "case", "annotation_version", "experiment"]


class ManifestRequest(BaseModel):
    kind: ManifestKind
    manifest: dict[str, Any]


class VerifyManifestRequest(BaseModel):
    kind: ManifestKind
    identity: str = Field(min_length=2, max_length=160)


class CreateRunRequest(BaseModel):
    task: dict[str, Any]


class CloneRunRequest(BaseModel):
    source_run_id: str = Field(min_length=2, max_length=160)
    new_run_id: str = Field(min_length=2, max_length=160)


class CompareRequest(BaseModel):
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    gate_profile: dict[str, Any] | None = None


class GpuVerifyRequest(BaseModel):
    sport_id: Literal["badminton", "tennis"] = "badminton"
    gpu_base_url: str | None = Field(default=None, max_length=500)


class RecoverStreamRequest(GpuVerifyRequest):
    analysis_session_id: str = Field(min_length=2, max_length=160)
    output_dir: str = Field(min_length=1, max_length=1000)


class ReviewOpenRequest(BaseModel):
    analysis_dir: str = Field(min_length=1, max_length=1000)
    reference_video: str | None = Field(default=None, max_length=1000)
    regenerate: bool = False
    selected_shot_id: str | None = Field(default=None, max_length=160)


class ReviewSaveRequest(ReviewOpenRequest):
    shot_id: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=80)
    decision: str = Field(min_length=1, max_length=80)
    reviewer: str = Field(default="", max_length=160)
    note: str = Field(default="", max_length=2000)


class ReviewManualRequest(ReviewOpenRequest):
    time_sec: float = Field(ge=0)
    reviewer: str = Field(default="", max_length=160)


class ReviewMergeRequest(ReviewOpenRequest):
    shot_ids: list[str] = Field(min_length=2)
    reviewer: str = Field(default="", max_length=160)


class ReviewSplitRequest(ReviewOpenRequest):
    shot_id: str = Field(min_length=1, max_length=160)
    split_offset_sec: float = Field(default=0.18, gt=0, le=2)
    reviewer: str = Field(default="", max_length=160)


class RallyTerminalRequest(ReviewOpenRequest):
    time_sec: float = Field(ge=0)
    outcome: str = Field(min_length=1, max_length=80)
    reviewer: str = Field(default="", max_length=160)
    note: str = Field(default="", max_length=2000)


def _origins() -> list[str]:
    configured = os.environ.get("GOOD_BADMINTON_ANALYSIS_API_ALLOWED_ORIGINS", "")
    return [item.strip() for item in configured.split(",") if item.strip()] or [
        "http://127.0.0.1:3100",
        "http://localhost:3100",
    ]


def _store_root() -> Path:
    return Path(os.environ.get(
        "GOOD_BADMINTON_ANALYSIS_STORE",
        _PROJECT_ROOT / "outputs" / "analysis_platform",
    )).resolve()


app = FastAPI(
    title="Good Badminton Evaluation API",
    version="1.0.0",
    description="独立评测平台的数据版本、研究 Run、对比门禁与人工复核 API。",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.exception_handler(ValueError)
@app.exception_handler(FileNotFoundError)
async def validation_error(_: Request, exc: Exception) -> JSONResponse:
    code = "not_found" if isinstance(exc, FileNotFoundError) else "validation_failed"
    http_status = status.HTTP_404_NOT_FOUND if isinstance(exc, FileNotFoundError) else 422
    return JSONResponse(status_code=http_status, content={"error": {"code": code, "message": str(exc)}})


@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    root = _store_root()
    root.mkdir(parents=True, exist_ok=True)
    return {
        "status": "ready",
        "service": "evaluation-platform-api",
        "store_root": str(root),
        "review": {
            "shot_types": SHOT_TYPES,
            "decisions": REVIEW_DECISIONS,
            "rally_terminal_outcomes": RALLY_TERMINAL_OUTCOMES,
        },
    }


@app.get("/api/v1/manifests/{kind}")
def manifests(kind: ManifestKind, draft: bool = Query(False)) -> list[dict[str, Any]]:
    return list_manifests(_store_root(), kind, draft=draft)


@app.post("/api/v1/manifests/drafts")
def save_draft(request: ManifestRequest) -> dict[str, Any]:
    return save_manifest_draft(_store_root(), request.kind, request.manifest)


@app.post("/api/v1/manifests/publish")
def publish(request: ManifestRequest) -> dict[str, Any]:
    return publish_manifest(_store_root(), request.kind, request.manifest)


@app.post("/api/v1/manifests/verify")
def verify_manifest(request: VerifyManifestRequest) -> dict[str, Any]:
    return verify_published_manifest(_store_root(), request.kind, request.identity)


@app.get("/api/v1/runs")
def runs() -> list[dict[str, Any]]:
    return list_run_summaries(_store_root())


@app.get("/api/v1/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    return get_run_detail(_store_root(), run_id)


@app.post("/api/v1/runs")
def freeze_run(request: CreateRunRequest) -> dict[str, Any]:
    return create_run(_store_root(), request.task)


@app.post("/api/v1/runs/clone")
def clone_run(request: CloneRunRequest) -> dict[str, Any]:
    return clone_run_task(_store_root(), request.source_run_id, request.new_run_id)


@app.post("/api/v1/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    return compare_run_reports(request.baseline, request.candidate, gate_profile=request.gate_profile)


@app.post("/api/v1/gpu/verify")
def verify_gpu(request: GpuVerifyRequest) -> dict[str, Any]:
    return verify_gpu_service(request.sport_id, request.gpu_base_url or None)


@app.post("/api/v1/court/detect")
def detect_court(video: UploadFile = File(...)) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="analysis-court-") as temporary:
        source = Path(temporary) / (Path(video.filename or "video.mp4").name)
        with source.open("wb") as output:
            shutil.copyfileobj(video.file, output)
        result = prepare_court_from_video(str(source))
        preview = result.get("preview_bgr")
        preview_data_url = None
        if preview is not None:
            encoded, data = cv2.imencode(".jpg", preview)
            if encoded:
                preview_data_url = "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode("ascii")
        return {"corners": result.get("corners") or [], "preview_data_url": preview_data_url}


def _json_object(value: str, field: str, *, optional: bool = False) -> dict[str, Any] | None:
    if optional and not value.strip():
        return None
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return parsed


def _json_array(value: str, field: str) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must contain valid JSON") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{field} must contain a JSON string array")
    return parsed


def _uploaded_file(root: Path, upload: UploadFile, fallback: str) -> Path:
    path = root / Path(upload.filename or fallback).name
    with path.open("wb") as output:
        shutil.copyfileobj(upload.file, output)
    return path


@app.post("/api/v1/runs/execute")
def execute_run(
    video: UploadFile = File(...),
    template: UploadFile = File(...),
    dataset_version_json: str = Form(...),
    task_json: str = Form(...),
    corners_json: str = Form(...),
    evaluator: str = Form(...),
    evaluator_arguments_json: str = Form("[]"),
    processing_target: Literal["local", "remote_gpu"] = Form("local"),
    gpu_base_url: str = Form(""),
    baseline_json: str = Form(""),
    gate_profile_json: str = Form(""),
) -> dict[str, Any]:
    try:
        corners = json.loads(corners_json)
    except json.JSONDecodeError as exc:
        raise ValueError("corners_json must contain valid JSON") from exc
    handle = _TASKS.start()
    with tempfile.TemporaryDirectory(prefix="analysis-run-") as temporary:
        root = Path(temporary)
        source = _uploaded_file(root, video, "video.mp4")
        court_template = _uploaded_file(root, template, "template.jpg")
        try:
            return execute_uploaded_run(
                _store_root(),
                _json_object(dataset_version_json, "dataset_version_json") or {},
                _json_object(task_json, "task_json") or {},
                source,
                court_template,
                corners,
                evaluator,
                _json_array(evaluator_arguments_json, "evaluator_arguments_json"),
                processing_target=processing_target,
                gpu_base_url=gpu_base_url or None,
                cancel_cb=handle.is_cancelled,
                baseline_report=_json_object(baseline_json, "baseline_json", optional=True),
                gate_profile=_json_object(gate_profile_json, "gate_profile_json", optional=True),
            )
        finally:
            _TASKS.finish(handle)


@app.post("/api/v1/runs/execute-stream")
def execute_stream_run(
    video: UploadFile = File(...),
    dataset_version_json: str = Form(...),
    task_json: str = Form(...),
    corners_json: str = Form(...),
    gpu_base_url: str = Form(""),
) -> dict[str, Any]:
    try:
        corners = json.loads(corners_json)
    except json.JSONDecodeError as exc:
        raise ValueError("corners_json must contain valid JSON") from exc
    handle = _TASKS.start()
    with tempfile.TemporaryDirectory(prefix="analysis-stream-") as temporary:
        source = _uploaded_file(Path(temporary), video, "video.mp4")
        try:
            return execute_uploaded_stream_run(
                _store_root(),
                _json_object(dataset_version_json, "dataset_version_json") or {},
                _json_object(task_json, "task_json") or {},
                source,
                corners,
                gpu_base_url=gpu_base_url or None,
                cancel_cb=handle.is_cancelled,
            )
        finally:
            _TASKS.finish(handle)


@app.post("/api/v1/runs/cancel")
def cancel_run() -> dict[str, Any]:
    return _TASKS.request_cancel() or {"cancel_requested": False, "message": "当前没有运行中的评测任务。"}


@app.post("/api/v1/runs/recover-stream")
def recover_stream(request: RecoverStreamRequest) -> dict[str, Any]:
    return recover_stream_session(
        request.analysis_session_id,
        request.output_dir,
        gpu_base_url=request.gpu_base_url or None,
        sport_id=request.sport_id,
    )


def _allowed_roots() -> tuple[Path, ...]:
    return ((_PROJECT_ROOT / "outputs").resolve(), (_store_root() / "runs").resolve())


def _safe_analysis_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not any(_is_below(path, root) for root in _allowed_roots()):
        raise ValueError("analysis_dir is outside the evaluation data roots")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _safe_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not any(_is_below(path, root) for root in _allowed_roots()):
        raise ValueError("file is outside the evaluation data roots")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@app.get("/api/v1/files")
def file_download(path: str = Query(..., min_length=1, max_length=1200)) -> FileResponse:
    return FileResponse(_safe_file(path))


@app.get("/api/v1/review/runs")
def review_runs() -> list[dict[str, str]]:
    found: list[str] = []
    for root in _allowed_roots():
        found.extend(find_analysis_runs(root))
    return [
        {"label": analysis_run_label(path), "analysis_dir": path}
        for path in dict.fromkeys(found)
    ]


def _review_payload(analysis_dir: Path, reference_video: str | None = None, selected: str | None = None, *, regenerate: bool = False) -> dict[str, Any]:
    session = create_or_load_review_session(str(analysis_dir), reference_video, regenerate=regenerate)
    choices = candidate_choices(session)
    selected = selected if selected in {value for _, value in choices} else (choices[0][1] if choices else None)
    view = candidate_view(session, selected) if selected else None
    return {
        "analysis_dir": str(analysis_dir),
        "summary": review_summary(session),
        "candidates": candidate_table(session),
        "candidate_choices": [{"label": label, "value": value} for label, value in choices],
        "selected": selected,
        "view": view,
        "reviewed_rallies": reviewed_rally_table(session),
    }


@app.post("/api/v1/review/open")
def open_review(request: ReviewOpenRequest) -> dict[str, Any]:
    return _review_payload(
        _safe_analysis_dir(request.analysis_dir),
        request.reference_video,
        request.selected_shot_id,
        regenerate=request.regenerate,
    )


@app.post("/api/v1/review/save")
def save_review(request: ReviewSaveRequest) -> dict[str, Any]:
    analysis_dir = _safe_analysis_dir(request.analysis_dir)
    session = create_or_load_review_session(str(analysis_dir), request.reference_video)
    save_human_review(session, request.shot_id, request.label, request.decision, request.reviewer, request.note)
    return _review_payload(analysis_dir, request.reference_video, request.shot_id)


@app.post("/api/v1/review/manual")
def add_review_candidate(request: ReviewManualRequest) -> dict[str, Any]:
    analysis_dir = _safe_analysis_dir(request.analysis_dir)
    session = create_or_load_review_session(str(analysis_dir), request.reference_video)
    _, candidate = add_manual_candidate(session, request.time_sec, request.reviewer)
    return _review_payload(analysis_dir, request.reference_video, candidate["shot_id"])


@app.post("/api/v1/review/merge")
def merge_review(request: ReviewMergeRequest) -> dict[str, Any]:
    analysis_dir = _safe_analysis_dir(request.analysis_dir)
    session = create_or_load_review_session(str(analysis_dir), request.reference_video)
    _, candidate = merge_review_candidates(session, request.shot_ids, request.reviewer)
    return _review_payload(analysis_dir, request.reference_video, candidate["shot_id"])


@app.post("/api/v1/review/split")
def split_review(request: ReviewSplitRequest) -> dict[str, Any]:
    analysis_dir = _safe_analysis_dir(request.analysis_dir)
    session = create_or_load_review_session(str(analysis_dir), request.reference_video)
    _, first, _ = split_review_candidate(session, request.shot_id, request.split_offset_sec, request.reviewer)
    return _review_payload(analysis_dir, request.reference_video, first["shot_id"])


@app.post("/api/v1/review/rally-terminal")
def add_rally_terminal(request: RallyTerminalRequest) -> dict[str, Any]:
    analysis_dir = _safe_analysis_dir(request.analysis_dir)
    add_manual_rally_terminal(
        str(analysis_dir), request.time_sec, request.outcome, request.reviewer, request.note
    )
    return _review_payload(analysis_dir, request.reference_video)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "analysis_platform.api:app",
        host=os.environ.get("GOOD_BADMINTON_ANALYSIS_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("GOOD_BADMINTON_ANALYSIS_API_PORT", "8010")),
        reload=False,
    )


if __name__ == "__main__":
    main()
