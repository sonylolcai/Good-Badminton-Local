"""JSON API for the Next.js operator console.

The API owns business registration and operator actions. Camera terminals do
not use these routes: they use the signed ``/api/v1/edge/*`` protocol.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, File, Form, HTTPException, Path as ApiPath, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from operator_api.services.operator_backoffice import (  # noqa: E402
    BackofficeError,
    BusinessDatabase,
    OperatorBackoffice,
)
from operator_api.services.auth import (  # noqa: E402
    AuthService,
    SESSION_COOKIE,
    principal_has_permission,
)
from operator_api.services.resource_lifecycle import BusinessResourceService, VIDEO_SUFFIXES  # noqa: E402


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


class ReplaySaveRequest(BaseModel):
    seconds: int = Field(default=20, ge=2, le=120)


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


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=1000)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=1000)
    new_password: str = Field(min_length=12, max_length=1000)


class AdminCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=12, max_length=1000)
    role: Literal["platform_admin", "venue_admin"]
    venue_id: str | None = None


class AdminStatusRequest(BaseModel):
    status: Literal["active", "disabled"]


class PlayerCreateRequest(BaseModel):
    nickname: str = Field(min_length=1, max_length=120)


class PlayerUpdateRequest(BaseModel):
    nickname: str = Field(min_length=1, max_length=120)
    status: Literal["active", "disabled"]


class VideoRetentionPolicyRequest(BaseModel):
    enabled: bool
    retention_days: int = Field(default=7, ge=1, le=3650)
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=80)
    daily_run_time: str = Field(default="03:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


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
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-Request-ID"],
)
app.state.auth_override = None
app.state.auth_service_override = None
app.state.resource_service_override = None


def get_auth_service() -> AuthService:
    return app.state.auth_service_override or AuthService(get_db())


def get_resource_service() -> BusinessResourceService:
    return app.state.resource_service_override or BusinessResourceService(get_db())


@app.on_event("startup")
def bootstrap_admin() -> None:
    get_auth_service().bootstrap_if_needed()


@app.middleware("http")
async def require_admin_session(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or path in {"/api/v1/auth/login", "/api/v1/system/readiness"}:
        return await call_next(request)
    if not path.startswith("/api/v1/"):
        return await call_next(request)
    principal = app.state.auth_override
    if principal is None:
        token = request.cookies.get(SESSION_COOKIE, "")
        principal = get_auth_service().authenticate(token) if token else None
    if principal is None:
        return JSONResponse(status_code=401, content={"error": {"code": "authentication_required", "message": "请先登录。"}})
    if principal.get("must_change_password") and path not in {
        "/api/v1/auth/me", "/api/v1/auth/change-password", "/api/v1/auth/logout"
    }:
        return JSONResponse(status_code=403, content={"error": {"code": "password_change_required", "message": "首次登录必须先修改密码。"}})
    platform_only = (
        "/api/v1/admins", "/api/v1/tenants", "/api/v1/venue-registrations",
        "/api/v1/dashboard", "/api/v1/gpu", "/api/v1/cases", "/api/v1/settings",
    )
    if path.startswith(platform_only) and not principal_has_permission(principal, "platform.manage"):
        return JSONResponse(status_code=403, content={"error": {"code": "permission_denied", "message": "仅平台管理员可执行此操作。"}})
    match = re.match(r"^/api/v1/venues/([^/]+)", path)
    if match and not principal_has_permission(principal, "venues.read", match.group(1)):
        return JSONResponse(status_code=403, content={"error": {"code": "venue_scope_denied", "message": "无权访问该场馆。"}})
    request.state.admin = principal
    return await call_next(request)


@app.exception_handler(BackofficeError)
async def business_error_handler(_: Request, exc: BackofficeError) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"error": {"code": "business_validation_failed", "message": str(exc)}})


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": {"code": "validation_failed", "message": str(exc)}})


def get_service() -> OperatorBackoffice:
    class PassiveController:
        def snapshot(self) -> dict[str, Any]:
            return {}

        def request_cancel(self) -> None:
            return None

    return OperatorBackoffice(PassiveController())


def get_db() -> BusinessDatabase:
    return BusinessDatabase()


def _admin(request: Request) -> dict[str, Any]:
    return request.state.admin


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


def _edge_gateway_operator_json(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    key = os.environ.get("GOOD_BADMINTON_EDGE_MASTER_KEY", "")
    if not key:
        raise HTTPException(status_code=503, detail={"code": "edge_gateway_not_configured", "message": "业务网关授权未配置。"})
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = UrlRequest(
        f"{_edge_gateway_internal_url()}{path}", data=data, method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json", "X-Operator-Edge-Key": key},
    )
    try:
        with urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("edge gateway response was not an object")
        return body
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = {"code": "edge_gateway_error", "message": str(exc)}
        raise HTTPException(status_code=exc.code, detail=detail.get("detail", detail)) from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise HTTPException(status_code=502, detail={"code": "edge_gateway_unavailable", "message": f"无法访问业务网关：{exc}"}) from exc


def _edge_gateway_operator_video(path: str) -> tuple[bytes, str]:
    key = os.environ.get("GOOD_BADMINTON_EDGE_MASTER_KEY", "")
    if not key:
        raise HTTPException(status_code=503, detail={"code": "edge_gateway_not_configured", "message": "业务网关授权未配置。"})
    request = UrlRequest(f"{_edge_gateway_internal_url()}{path}", headers={"X-Operator-Edge-Key": key})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read(), response.headers.get_content_type()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail={"code": "edge_gateway_unavailable", "message": f"无法读取保存的录像：{exc}"}) from exc


@app.post("/api/v1/auth/login")
def login(request: LoginRequest) -> JSONResponse:
    try:
        token, principal = get_auth_service().login(request.username, request.password)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail="用户名或密码错误。") from exc
    response = JSONResponse({"admin": principal})
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(os.environ.get("GOOD_BADMINTON_ADMIN_SESSION_HOURS", "12")) * 3600,
        httponly=True,
        secure=os.environ.get("GOOD_BADMINTON_APP_ENV") == "production",
        samesite="strict",
        path="/",
    )
    return response


@app.get("/api/v1/auth/me")
def current_admin(request: Request) -> dict[str, Any]:
    return {"admin": _admin(request)}


@app.post("/api/v1/auth/logout")
def logout(request: Request) -> JSONResponse:
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        get_auth_service().logout(token)
    response = JSONResponse({"logged_out": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.post("/api/v1/auth/change-password")
def change_password(request: Request, payload: PasswordChangeRequest) -> dict[str, bool]:
    try:
        get_auth_service().change_password(_admin(request), payload.current_password, payload.new_password)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail="当前密码错误。") from exc
    return {"changed": True}


@app.get("/api/v1/admins")
def list_admins() -> dict[str, list[dict]]:
    return {"admins": get_auth_service().list_admins()}


@app.post("/api/v1/admins", status_code=status.HTTP_201_CREATED)
def create_admin(request: Request, payload: AdminCreateRequest) -> dict[str, dict]:
    try:
        admin = get_auth_service().create_admin(
            _admin(request), payload.username, payload.password, payload.role, payload.venue_id
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"admin": admin}


@app.patch("/api/v1/admins/{admin_id}")
def update_admin_status(
    request: Request,
    admin_id: Annotated[str, ApiPath(min_length=1)],
    payload: AdminStatusRequest,
) -> dict[str, dict]:
    try:
        admin = get_auth_service().set_admin_status(_admin(request), admin_id, payload.status)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"admin": admin}


@app.get("/api/v1/settings/video-retention")
def get_video_retention_policy() -> dict[str, dict]:
    return {"policy": get_db().get_video_retention_policy()}


@app.patch("/api/v1/settings/video-retention")
def update_video_retention_policy(request: Request, payload: VideoRetentionPolicyRequest) -> dict[str, dict]:
    policy = get_db().update_video_retention_policy(
        enabled=payload.enabled,
        retention_days=payload.retention_days,
        timezone_name=payload.timezone,
        daily_run_time=payload.daily_run_time,
        actor_admin_id=_admin(request)["id"],
    )
    return {"policy": policy}


def _authorized_asset(request: Request, asset_id: str, permission: str) -> tuple[BusinessResourceService, dict]:
    service = get_resource_service()
    try:
        asset = service.database.get_media_asset(asset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not principal_has_permission(_admin(request), permission, asset["venue_id"]):
        raise HTTPException(status_code=403, detail="无权管理该场馆的视频。")
    return service, asset


@app.get("/api/v1/resources")
def list_resources(request: Request, venue_id: str | None = None) -> dict[str, list[dict]]:
    principal = _admin(request)
    service = get_resource_service()
    if principal_has_permission(principal, "platform.manage"):
        return {"resources": service.list_assets(venue_id)}
    allowed = [item["venue_id"] for item in principal.get("roles") or [] if item.get("venue_id")]
    if venue_id and venue_id not in allowed:
        raise HTTPException(status_code=403, detail="无权访问该场馆的视频。")
    venues = [venue_id] if venue_id else allowed
    return {"resources": [item for current in venues for item in service.list_assets(current)]}


@app.post("/api/v1/resources", status_code=status.HTTP_201_CREATED)
async def upload_resource(
    request: Request,
    video: UploadFile = File(...),
    venue_id: str = Form(...),
    player_id: str | None = Form(default=None),
    match_id: str | None = Form(default=None),
) -> dict[str, dict]:
    principal = _admin(request)
    if not principal_has_permission(principal, "resources.upload", venue_id):
        raise HTTPException(status_code=403, detail="无权向该场馆上传视频。")
    filename = Path(video.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES or not (video.content_type or "").startswith("video/"):
        raise HTTPException(status_code=422, detail="仅支持 mp4、mov、mkv、avi 或 webm 视频。")
    db = get_db()
    venue = _require_venue(db, venue_id)
    service = get_resource_service()
    asset_id = str(uuid.uuid4())
    target = service.target_path(asset_id, suffix)
    limit = int(os.environ.get("GOOD_BADMINTON_MAX_VIDEO_UPLOAD_BYTES", str(20 * 1024**3)))
    size = 0
    try:
        with target.open("xb") as output:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail="视频超过允许的上传大小。")
                output.write(chunk)
        asset = service.register_uploaded_video(
            asset_id=asset_id,
            path=target,
            tenant_id=venue["tenant_id"],
            venue_id=venue_id,
            player_id=(player_id or "").strip() or None,
            match_id=(match_id or "").strip() or None,
            media_type=video.content_type or "video/mp4",
            original_filename=filename,
            actor_admin_id=principal["id"],
        )
    except Exception:
        target.unlink(missing_ok=True)
        try:
            target.parent.rmdir()
        except OSError:
            pass
        raise
    return {"resource": asset}


@app.post("/api/v1/resources/{asset_id}/analysis", status_code=status.HTTP_202_ACCEPTED)
async def trigger_resource_analysis(
    request: Request,
    asset_id: Annotated[str, ApiPath(min_length=1)],
    template: UploadFile = File(...),
    corners_json: str = Form(...),
    options_json: str = Form("{}"),
) -> dict[str, dict]:
    service, _asset = _authorized_asset(request, asset_id, "analysis.create")
    suffix = Path(template.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp"}:
        raise HTTPException(status_code=422, detail="场地图仅支持 png、jpg、jpeg 或 bmp。")
    try:
        corners = json.loads(corners_json)
        options = json.loads(options_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="解析参数必须是有效 JSON。") from exc
    if not isinstance(options, dict):
        raise HTTPException(status_code=422, detail="options_json 必须是 JSON 对象。")
    with tempfile.TemporaryDirectory(prefix="business-analysis-") as temporary:
        template_path = Path(temporary) / f"court{suffix}"
        data = await template.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="场地图超过 20 MB。")
        template_path.write_bytes(data)
        result = service.trigger_analysis(asset_id, _admin(request)["id"], template_path, corners, options)
    return {"analysis": result}


@app.delete("/api/v1/resources/{asset_id}/resources")
def delete_resource_files(request: Request, asset_id: Annotated[str, ApiPath(min_length=1)]):
    service, _asset = _authorized_asset(request, asset_id, "resources.delete")
    result = service.delete_asset(asset_id, _admin(request)["id"])
    return JSONResponse(status_code=409 if result["status"] == "partial" else 200, content=result)


@app.delete("/api/v1/resources/{asset_id}")
def delete_resource_record(request: Request, asset_id: Annotated[str, ApiPath(min_length=1)]):
    if not principal_has_permission(_admin(request), "platform.manage"):
        raise HTTPException(status_code=403, detail="仅平台管理员可删除全部数据。")
    service, _asset = _authorized_asset(request, asset_id, "resources.delete")
    result = service.delete_asset(asset_id, _admin(request)["id"], full=True)
    return JSONResponse(status_code=409 if result["status"] == "partial" else 200, content=result)


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
def list_venues(request: Request) -> dict[str, Any]:
    db = get_db()
    readiness = db.readiness()
    if readiness.get("status") != "ready":
        return {"venues": [], "status": readiness}
    venues = [_venue(row) for row in db.list_venues()]
    principal = _admin(request)
    if not principal_has_permission(principal, "platform.manage"):
        allowed = {item["venue_id"] for item in principal.get("roles") or [] if item.get("venue_id")}
        venues = [venue for venue in venues if venue["id"] in allowed]
    return {"venues": venues, "status": readiness}


@app.get("/api/v1/players")
def list_players(request: Request) -> dict[str, list[dict]]:
    return {"players": get_db().list_players()}


@app.post("/api/v1/players", status_code=status.HTTP_201_CREATED)
def create_player(
    request: Request,
    payload: PlayerCreateRequest,
) -> dict[str, dict]:
    principal = _admin(request)
    if not principal_has_permission(principal, "players.manage"):
        raise HTTPException(status_code=403, detail="仅平台管理员可新增球员。")
    return {"player": get_db().create_player(payload.nickname, principal["id"])}


@app.patch("/api/v1/players/{player_id}")
def update_player(
    request: Request,
    player_id: Annotated[str, ApiPath(min_length=1)],
    payload: PlayerUpdateRequest,
) -> dict[str, dict]:
    principal = _admin(request)
    if not principal_has_permission(principal, "players.manage"):
        raise HTTPException(status_code=403, detail="仅平台管理员可修改球员。")
    return {
        "player": get_db().update_player(
            player_id, payload.nickname, payload.status, principal["id"]
        )
    }


@app.get("/api/v1/players/{player_id}/play-records")
def list_player_play_records(
    request: Request,
    player_id: Annotated[str, ApiPath(min_length=1)],
) -> dict[str, list[dict]]:
    if not principal_has_permission(_admin(request), "platform.manage"):
        raise HTTPException(status_code=403, detail="仅平台管理员可查看完整打球记录。")
    return {"records": get_db().list_player_play_records(player_id)}


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


def _case_for_replay(venue_id: str, court_id: str) -> dict[str, Any]:
    db = get_db()
    _require_venue(db, venue_id)
    case = db.case_for_court(venue_id, court_id)
    if not case:
        raise HTTPException(status_code=409, detail={"code": "no_live_case", "message": "摄像头尚未上传可保存的视频片段。"})
    return case


def _replay_payload(case_id: str, replay: dict[str, Any], venue_id: str, court_id: str) -> dict[str, Any]:
    return {
        **replay,
        "url": f"/api/v1/venues/{venue_id}/courts/{court_id}/case/replays/{replay['id']}",
        "case_id": case_id,
    }


@app.post("/api/v1/venues/{venue_id}/courts/{court_id}/case/replays", status_code=status.HTTP_201_CREATED)
def save_court_replay(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
    request: ReplaySaveRequest,
) -> dict[str, Any]:
    """Save the latest contiguous recording window from this court's current case."""
    case = _case_for_replay(venue_id, court_id)
    payload = _edge_gateway_operator_json(
        f"/api/v1/edge/sessions/{case['case_id']}/replays", method="POST", payload=request.model_dump(),
    )
    replay = payload.get("replay")
    if not isinstance(replay, dict):
        raise HTTPException(status_code=502, detail={"code": "edge_gateway_invalid_response", "message": "业务网关没有返回保存的录像。"})
    return {"replay": _replay_payload(case["case_id"], replay, venue_id, court_id), "message": "已保存当前比赛片段。"}


@app.get("/api/v1/venues/{venue_id}/courts/{court_id}/case/replays")
def list_court_replays(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
) -> dict[str, Any]:
    case = _case_for_replay(venue_id, court_id)
    payload = _edge_gateway_operator_json(f"/api/v1/edge/sessions/{case['case_id']}/replays")
    replays = payload.get("replays")
    if not isinstance(replays, list):
        raise HTTPException(status_code=502, detail={"code": "edge_gateway_invalid_response", "message": "业务网关没有返回录像列表。"})
    return {"case_id": case["case_id"], "replays": [_replay_payload(case["case_id"], replay, venue_id, court_id)
                                                        for replay in replays if isinstance(replay, dict)]}


@app.get("/api/v1/venues/{venue_id}/courts/{court_id}/case/replays/{replay_id}")
def play_court_replay(
    venue_id: Annotated[str, ApiPath(min_length=1)],
    court_id: Annotated[str, ApiPath(min_length=1)],
    replay_id: Annotated[str, ApiPath(min_length=21, max_length=21)],
) -> Response:
    case = _case_for_replay(venue_id, court_id)
    video, media_type = _edge_gateway_operator_video(f"/api/v1/edge/sessions/{case['case_id']}/replays/{replay_id}")
    return Response(content=video, media_type=media_type or "video/mp4", headers={"Cache-Control": "private, no-store"})


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
