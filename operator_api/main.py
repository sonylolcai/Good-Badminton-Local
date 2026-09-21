"""JSON API for the Next.js operator console.

The API owns business registration and operator actions. Camera terminals do
not use these routes: they use the signed ``/api/v1/edge/*`` protocol.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, HTTPException, Path as ApiPath, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from operator_api.services.operator_backoffice import (  # noqa: E402
    BackofficeError,
    BusinessDatabase,
    OperatorBackoffice,
)


CourtStatus = Literal["active", "maintenance", "inactive"]
CaptureMode = Literal["idle", "preview", "record"]


class CourtRegistration(BaseModel):
    code: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=120)
    sort_order: int = Field(default=0, ge=0, le=10000)
    status: CourtStatus = "active"


class VenueRegistrationRequest(BaseModel):
    """Register a usable venue, creating its tenant only when explicitly asked."""

    tenant_id: str | None = Field(default=None, description="Existing active tenant UUID")
    tenant_name: str | None = Field(default=None, min_length=1, max_length=120, description="New tenant name")
    venue_code: str = Field(min_length=1, max_length=80)
    venue_name: str = Field(min_length=1, max_length=120)
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=64)
    address: str | None = Field(default=None, max_length=500)
    courts: list[CourtRegistration] = Field(min_length=1, max_length=100)


class CreateCourtRequest(CourtRegistration):
    pass


class CourtStatusRequest(BaseModel):
    status: CourtStatus


class EdgeBindingRequest(BaseModel):
    device_code: str = Field(min_length=1, max_length=120)
    camera_code: str = Field(min_length=1, max_length=120)
    credential_version: str = Field(default="v1", min_length=1, max_length=32)


class GpuConfigRequest(BaseModel):
    base_url: str = Field(min_length=1, max_length=500)
    api_key: str = Field(default="", max_length=2000)


class GpuOperationRequest(BaseModel):
    operation: Literal["start", "stop"]


class CaseGpuForwardingRequest(BaseModel):
    enabled: bool


class CourtCaptureModeRequest(BaseModel):
    mode: CaptureMode


class ImagePoint(BaseModel):
    x: float = Field(ge=-100000, le=100000)
    y: float = Field(ge=-100000, le=100000)


class CrossCourtLine(BaseModel):
    """One visible horizontal court line and its standard distance in metres."""

    court_y_m: float = Field(ge=0, le=13.4)
    points: list[ImagePoint] = Field(min_length=2, max_length=2)


class CourtCalibrationRequest(BaseModel):
    """Manual visible corners or enough line evidence to extrapolate them."""

    mode: Literal["manual_corners", "line_evidence"]
    corners: list[ImagePoint] = Field(default_factory=list, max_length=4)
    left_sideline: list[ImagePoint] = Field(default_factory=list, max_length=2)
    right_sideline: list[ImagePoint] = Field(default_factory=list, max_length=2)
    cross_lines: list[CrossCourtLine] = Field(default_factory=list, max_length=2)


def _origins() -> list[str]:
    configured = os.environ.get("GOOD_BADMINTON_OPERATOR_API_ALLOWED_ORIGINS", "")
    return [item.strip() for item in configured.split(",") if item.strip()] or [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]


app = FastAPI(
    title="Good Badminton Operator API",
    version="1.1.0",
    description="场馆、场地、设备接入与 GPU 管理 API。终端视频流请使用独立的 edge-ingest.v1 协议。",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.exception_handler(BackofficeError)
async def business_error_handler(_: Request, exc: BackofficeError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"error": {"code": "business_validation_failed", "message": str(exc)}})


def get_service() -> OperatorBackoffice:
    class PassiveController:
        def snapshot(self) -> dict[str, Any]:
            return {}

        def request_cancel(self) -> None:
            return None

    return OperatorBackoffice(PassiveController())


def get_db() -> BusinessDatabase:
    return BusinessDatabase()


def _tenant(row: list[str]) -> dict[str, str]:
    return {"id": row[0], "name": row[1], "status": row[2]}


def _venue(row: list[str]) -> dict[str, Any]:
    return {
        "id": row[0], "tenant_id": row[1], "code": row[2], "name": row[3],
        "timezone": row[4], "address": row[5] or None, "status": row[6],
        "court_count": int(row[7]),
    }


def _court(row: list[str]) -> dict[str, Any]:
    return {
        "id": row[0], "venue_id": row[1], "code": row[2], "name": row[3],
        "sort_order": int(row[4]), "status": row[5],
    }


def _require_venue(db: BusinessDatabase, venue_id: str) -> dict[str, Any]:
    for row in db.list_venues():
        if row[0] == venue_id:
            return _venue(row)
    raise HTTPException(status_code=404, detail={"code": "venue_not_found", "message": "场馆不存在。"})


def _edge_gateway_base_url() -> str:
    """Return the browser-reachable HTTPS edge gateway origin.

    The operator process uses a separate loopback origin for its server-to-
    server event refresh.  Never emit that internal address into the page.
    """
    return os.environ.get(
        "GOOD_BADMINTON_EDGE_GATEWAY_PUBLIC_URL",
        os.environ.get("GOOD_BADMINTON_EDGE_GATEWAY_URL", "http://127.0.0.1:18080"),
    ).rstrip("/")


def _edge_gateway_internal_url() -> str:
    return os.environ.get(
        "GOOD_BADMINTON_EDGE_GATEWAY_INTERNAL_URL",
        os.environ.get("GOOD_BADMINTON_EDGE_GATEWAY_URL", "http://127.0.0.1:18080"),
    ).rstrip("/")


def _edge_gateway_json(path: str) -> dict[str, Any]:
    try:
        request = UrlRequest(f"{_edge_gateway_internal_url()}{path}", headers={"Accept": "application/json"})
        with urlopen(request, timeout=5) as response:
            import json
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("edge gateway response was not an object")
        return body
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail={"code": "edge_gateway_unavailable", "message": f"无法读取业务网关状态：{exc}"}) from exc


@app.get("/api/v1/system/readiness")
def get_readiness() -> dict[str, Any]:
    return get_db().readiness()


@app.get("/api/v1/dashboard")
def get_dashboard() -> dict[str, Any]:
    db = get_db()
    service = get_service()
    if db.readiness().get("status") != "ready":
        return {"metrics": {}, "overview": service.overview()}
    return {"metrics": db.dashboard(), "overview": service.overview()}


@app.get("/api/v1/tenants")
def list_tenants() -> dict[str, list[dict[str, str]]]:
    return {"tenants": [_tenant(row) for row in get_db().list_tenants()]}


@app.get("/api/v1/venues")
def list_venues() -> dict[str, Any]:
    db = get_db()
    readiness = db.readiness()
    if readiness.get("status") != "ready":
        return {"venues": [], "status": readiness}
    return {"venues": [_venue(row) for row in db.list_venues()], "status": readiness}


@app.post("/api/v1/venue-registrations", status_code=status.HTTP_201_CREATED)
def register_venue(request: VenueRegistrationRequest) -> dict[str, Any]:
    result = get_db().register_venue(
        tenant_id=request.tenant_id,
        tenant_name=request.tenant_name,
        venue_code=request.venue_code,
        venue_name=request.venue_name,
        timezone_name=request.timezone,
        address=request.address,
        courts=[court.model_dump() for court in request.courts],
    )
    return {"registration": result, "message": "场馆与场地已注册。"}


@app.get("/api/v1/venues/{venue_id}")
def get_venue(venue_id: Annotated[str, ApiPath(min_length=1)]) -> dict[str, Any]:
    return {"venue": _require_venue(get_db(), venue_id)}


@app.get("/api/v1/venues/{venue_id}/courts")
def list_venue_courts(venue_id: Annotated[str, ApiPath(min_length=1)]) -> dict[str, Any]:
    db = get_db()
    _require_venue(db, venue_id)
    return {"courts": [_court(row) for row in db.list_courts() if row[1] == venue_id]}


@app.post("/api/v1/venues/{venue_id}/courts", status_code=status.HTTP_201_CREATED)
def create_court(venue_id: Annotated[str, ApiPath(min_length=1)], request: CreateCourtRequest) -> dict[str, Any]:
    db = get_db()
    _require_venue(db, venue_id)
    court_id = db.save_court("", venue_id, request.code, request.name, request.sort_order, request.status)
    return {"court": next(_court(row) for row in db.list_courts() if row[0] == court_id)}


@app.patch("/api/v1/venues/{venue_id}/courts/{court_id}/status")
def update_court_status(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
    request: CourtStatusRequest,
) -> dict[str, Any]:
    db = get_db()
    db.set_court_status(venue_id, court_id, request.status)
    return {"court": next(_court(row) for row in db.list_courts() if row[0] == court_id)}


@app.get("/api/v1/venues/{venue_id}/operations")
def venue_operations(venue_id: Annotated[str, ApiPath(min_length=1)]) -> dict[str, Any]:
    db = get_db()
    _require_venue(db, venue_id)
    snapshot = db.venue_live_operations(venue_id)
    for court in snapshot["courts"]:
        case = court["case"]
        if case:
            if case["preview_available"]:
                case["preview_url"] = f"{_edge_gateway_base_url()}/api/v1/edge/sessions/{case['id']}/preview/latest.mp4"
            else:
                case["preview_url"] = None
    return {
        "summary": {"camera_connected": snapshot["camera_connected"], "active_cases": snapshot["active_cases"], "total_courts": len(snapshot["courts"])},
        "courts": snapshot["courts"],
    }


@app.post("/api/v1/venues/{venue_id}/courts/{court_id}/calibration-candidate")
def create_calibration_candidate(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
    request: CourtCalibrationRequest,
) -> dict[str, Any]:
    """Calculate four candidate corners from the currently visible preview."""
    db = get_db()
    _require_venue(db, venue_id)
    candidate = db.calibration_candidate(venue_id, court_id, request.model_dump())
    return {"candidate": candidate,
            "message": "已生成候选四角。请确认叠加线与画面中的可见场地线重合后再保存。"}


@app.post("/api/v1/venues/{venue_id}/courts/{court_id}/calibration")
def save_calibration(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
    request: CourtCalibrationRequest,
) -> dict[str, Any]:
    """Confirm a preview-derived candidate and unlock only future record sessions."""
    db = get_db()
    _require_venue(db, venue_id)
    calibration = db.save_camera_calibration(venue_id, court_id, request.model_dump())
    return {"calibration": calibration,
            "message": "场地标定已验证。请停止当前预览，再开始采集；后续会话才可向 GPU 推送。"}


@app.post("/api/v1/venues/{venue_id}/courts/{court_id}/capture")
def set_court_capture_mode(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
    request: CourtCaptureModeRequest,
) -> dict[str, Any]:
    """Change capture remotely; the venue Mac applies it on its next heartbeat."""

    db = get_db()
    _require_venue(db, venue_id)
    control = db.set_court_capture_mode(venue_id, court_id, request.mode)
    messages = {
        "idle": "已停止视频采集；终端将继续发送心跳，不再上传视频。",
        "preview": "已请求实时预览；终端将在下一次心跳后开始上传短时预览。",
        "record": "已请求采集；终端将在下一次心跳后开始上传视频。GPU 仍需单独开启。",
    }
    return {"capture": control, "message": messages[request.mode]}


@app.post("/api/v1/venues/{venue_id}/courts/{court_id}/case/gpu-forwarding")
def set_case_gpu_forwarding(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
    request: CaseGpuForwardingRequest,
) -> dict[str, Any]:
    """Switch GPU delivery for subsequent segments of the court's live case."""
    db = get_db()
    _require_venue(db, venue_id)
    case = db.set_case_gpu_forwarding(venue_id, court_id, request.enabled)
    return {
        "case": case,
        "message": "已开启 GPU 推送；后续视频片段会送往 GPU。" if request.enabled
        else "已暂停 GPU 推送；视频预览仍会继续，已暂停期间的片段不会补推。",
    }


@app.get("/api/v1/cases/{case_id}/gpu-events")
def case_gpu_events(case_id: Annotated[str, ApiPath(min_length=1)], limit: int = 100) -> dict[str, Any]:
    """Refresh and return safe, case-scoped GPU execution events.

    The edge gateway owns the GPU transport and persists newly fetched events.
    If it is unavailable, existing persisted events are still returned so an
    operator can distinguish a gateway outage from a blank execution history.
    """
    db = get_db()
    saved = db.case_event_log(case_id, limit)
    try:
        upstream = _edge_gateway_json(f"/api/v1/edge/sessions/{case_id}/gpu-events?limit={max(1, min(limit, 500))}")
        return {"case_id": case_id, "gpu_analysis_session_id": upstream.get("gpu_analysis_session_id"),
                "events": upstream.get("events") or [], "persisted_events": saved,
                "next_cursor": upstream.get("next_cursor"), "source_status": "live"}
    except HTTPException as exc:
        return {"case_id": case_id, "gpu_analysis_session_id": None, "events": [], "persisted_events": saved,
                "next_cursor": None, "source_status": "gateway_unavailable", "warning": exc.detail["message"]}


@app.post("/api/v1/venues/{venue_id}/courts/{court_id}/edge-bindings", status_code=status.HTTP_201_CREATED)
def provision_edge_binding(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
    request: EdgeBindingRequest,
) -> dict[str, str]:
    """Return the one-time terminal secret; never log or persist it in this API."""
    return get_db().provision_edge_camera("", "", venue_id, court_id, request.device_code, request.camera_code, request.credential_version)


@app.get("/api/v1/gpu/status")
def get_gpu_status() -> dict[str, Any]:
    return get_service().gpu_status(check_health=True)


@app.post("/api/v1/gpu/config")
def save_gpu_config(request: GpuConfigRequest) -> dict[str, Any]:
    return get_service().save_gpu_config(request.base_url, request.api_key)


@app.post("/api/v1/gpu/operate")
def operate_gpu(request: GpuOperationRequest) -> dict[str, Any]:
    return get_service().request_gpu_operation(request.operation)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("operator_api.main:app", host="127.0.0.1", port=8000, reload=True)
