"""Internal operator console services and Gradio tabs.

It keeps all operator controls on the business side. GPU requests use only
anonymous task references and never forward user, score, team or rating data.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
from hashlib import sha256
from html import escape
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import gradio as gr

from runtime_config import RuntimeConfigurationError, business_api_base_url, load_runtime_environment

from webui.remote_gpu import RemoteAnalysisError, remote_gpu_config
from webui.task_ledger import BusinessTaskLedger


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / ".webui-remote-gpu.env"
_DEFAULT_OPERATOR_CONFIG_PATH = _PROJECT_ROOT / ".webui-operator.env"
_DEFAULT_OPERATIONS_PATH = _PROJECT_ROOT / "outputs" / "operator_backoffice" / "operations.jsonl"
_VENUE_STATUSES = {"active", "inactive"}
_COURT_STATUSES = {"active", "maintenance", "inactive"}
_USER_STATUSES = {"active", "disabled"}
_MEMBERSHIP_ROLES = {"owner", "operator", "viewer"}
_QR_SCOPES = {"venue", "court"}
_ACTIVITY_STATUSES = {"draft", "published", "cancelled", "completed"}
_POST_TYPES = {"official", "match_card", "member_share"}
_POST_STATUSES = {"published", "hidden"}
_CLAIM_STATUSES = {"pending", "confirmed", "rejected", "corrected"}
_DELIVERY_STATUSES = {"pending", "processing", "ready", "failed", "withheld"}


OPERATOR_BACKOFFICE_CSS = """
#component-0 .gradio-container { max-width: 1480px !important; background: #f6f7fb; }
.venue-hero { padding: 28px 30px; border-radius: 24px; color: #fff; background: linear-gradient(125deg, #102b45 0%, #176b75 54%, #d7a23d 145%); box-shadow: 0 14px 36px rgba(18, 53, 74, .18); }
.venue-hero h2 { margin: 0; font-size: 28px; letter-spacing: -.5px; }
.venue-hero p { margin: 8px 0 0; color: rgba(255,255,255,.8); }
.venue-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 18px 0; }
.venue-metric { min-height: 112px; padding: 18px; border: 1px solid #e9edf4; border-radius: 18px; background: #fff; box-shadow: 0 4px 14px rgba(28, 39, 64, .04); }
.venue-metric small { display: block; color: #7d8797; font-size: 13px; }
.venue-metric strong { display: block; margin-top: 10px; color: #15233a; font-size: 30px; line-height: 1; }
.venue-metric em { color: #35a47b; font-style: normal; font-size: 12px; }
.venue-status { display: flex; flex-wrap: wrap; gap: 10px; padding: 14px 16px; border: 1px solid #e8edf3; border-radius: 16px; background: #fff; }
.venue-pill { padding: 7px 10px; border-radius: 999px; background: #eff8f4; color: #277a5b; font-size: 13px; font-weight: 600; }
.venue-pill.warn { background: #fff4e5; color: #b36a14; }
.venue-plain-status { padding: 10px 14px; border-radius: 12px; background: #f0f4fb; color: #516176; font-size: 13px; }
.venue-ops .tab-nav { gap: 4px; padding: 8px 0; border-bottom: 1px solid #e9edf2; }
.venue-ops .tab-nav button { border-radius: 10px 10px 0 0; font-weight: 600; }
.venue-ops .tab-nav button.selected { color: #0f766e; border-color: #0f766e; }
.venue-ops .block-title { margin: 18px 0 8px; }
@media (max-width: 760px) { .venue-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } .venue-hero { padding: 22px; } }
"""


def _status_text(readiness: dict[str, Any]) -> str:
    status = str(readiness.get("status") or "unknown")
    if status == "ready":
        return f"<div class='venue-plain-status'>业务数据库已连接 · {escape(str(readiness.get('database') or '已就绪'))}</div>"
    return f"<div class='venue-plain-status'>业务数据库未就绪 · {escape(str(readiness.get('message') or status))}</div>"


def _dashboard_markup(metrics: dict[str, Any], overview: dict[str, Any]) -> str:
    cards = [
        ("近 7 天开赛", metrics.get("matches_7d", "—"), "真实业务库数据"),
        ("赛果确认率", f"{metrics.get('confirmation_rate_7d', '—')}%", "双方确认后进入榜单"),
        ("已启用场地", metrics.get("active_courts", "—"), "场地码可绑定入口"),
        ("待开展活动", metrics.get("upcoming_activities", "—"), "场馆活动与挑战"),
    ]
    card_html = "".join(f"<div class='venue-metric'><small>{escape(label)}</small><strong>{escape(str(value))}</strong><em>{escape(hint)}</em></div>" for label, value, hint in cards)
    database = overview.get("business_database") or {}
    business_api = overview.get("business_api") or {}
    gpu = overview.get("gpu") or {}
    pills = [
        ("业务库已连接" if database.get("status") == "ready" else "业务库未连接", database.get("status") != "ready"),
        ("业务网关在线" if business_api.get("status") == "ready" else "业务网关不可用", business_api.get("status") != "ready"),
        ("GPU 已配置" if gpu.get("configured") else "GPU 未配置", not gpu.get("configured")),
        ("本机运营模式", True),
    ]
    pill_html = "".join(f"<span class='venue-pill{' warn' if warning else ''}'>{escape(label)}</span>" for label, warning in pills)
    return f"<section class='venue-hero'><h2>羽球空间 · 运营看板</h2><p>一眼掌握场馆入口、对局、活动与 AI 服务状态。</p></section><section class='venue-metrics'>{card_html}</section><section class='venue-status'>{pill_html}</section>"


def _gpu_markup(status: dict[str, Any]) -> str:
    state = str(status.get("status") or "not_checked")
    label = "GPU 服务在线" if state == "ready" else "GPU 服务待检查" if state == "not_checked" else "GPU 服务不可用"
    hint = str(status.get("control_hint") or status.get("message") or "健康检查、任务取消与受控启停均由服务端执行。")
    return f"<section class='venue-status'><span class='venue-pill{' warn' if state not in {'ready', 'not_checked'} else ''}'>{escape(label)}</span><span class='venue-plain-status'>{escape(hint)}</span></section>"


def _match_detail_markup(detail: dict[str, Any]) -> str:
    if detail.get("hint") or detail.get("error"):
        return f"<div class='venue-plain-status'>{escape(str(detail.get('hint') or detail.get('error')))}</div>"
    match = detail.get("match") or []
    return f"<section class='venue-status'><span class='venue-pill'>{escape(str(match[1] if len(match) > 1 else '未知状态'))}</span><span class='venue-plain-status'>赛制：{escape(str(match[2] if len(match) > 2 else '—'))} · 停录回执：{escape(str(match[6] if len(match) > 6 else '待确认'))} · 认领 {len(detail.get('track_claims') or [])} 条 · 待投递 {len(detail.get('deliveries') or [])} 项</span></section>"


class BackofficeError(RuntimeError):
    """An error safe to display to an internal operator."""


def _validate_base_url(value: str) -> str:
    candidate = str(value or "").strip().rstrip("/")
    parsed = urlparse(candidate)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise BackofficeError("GPU 服务地址必须是无账号、无查询参数的 http(s) 基础地址。")
    return candidate


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key.replace("_", "").isalnum():
            values[key] = value.strip()
    return values


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 由运营后台维护；不要提交此文件。",
        "# API Key 只供服务端使用，后台不会读回或显示该值。",
        *[f"{key}={values[key]}" for key in sorted(values)],
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_request(base_url: str, api_key: str, path: str, *, method: str = "GET", timeout: float = 10) -> dict[str, Any]:
    request = Request(
        _validate_base_url(base_url) + path,
        headers={"X-API-Key": api_key} if api_key else {},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", "replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise BackofficeError(f"GPU 服务返回 HTTP {exc.code}：{body}") from exc
    except (URLError, OSError) as exc:
        raise BackofficeError(f"无法连接 GPU 服务：{exc}") from exc
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BackofficeError("GPU 服务响应不是 JSON。") from exc
    if not isinstance(result, dict):
        raise BackofficeError("GPU 服务响应格式无效。")
    return result


class BusinessDatabase:
    """Explicit adapter over the existing PostgreSQL ``business`` schema."""

    def __init__(self, database_url: str | None = None) -> None:
        local_values = _read_env_file(Path(os.environ.get("GOOD_BADMINTON_WEBUI_OPERATOR_CONFIG", _DEFAULT_OPERATOR_CONFIG_PATH)))
        self.database_url = str(
            database_url
            or os.environ.get("GOOD_BADMINTON_BUSINESS_DATABASE_URL", "")
            or local_values.get("GOOD_BADMINTON_BUSINESS_DATABASE_URL", "")
        ).strip()
        try:
            self.connect_timeout_seconds = min(
                10,
                max(1, int(os.environ.get("GOOD_BADMINTON_BUSINESS_DATABASE_CONNECT_TIMEOUT", "3"))),
            )
        except ValueError:
            self.connect_timeout_seconds = 3
        self._readiness_cache: dict[str, Any] | None = None
        self._readiness_checked_monotonic = 0.0

    def readiness(self) -> dict[str, Any]:
        if not self.database_url:
            return {"status": "not_configured", "message": "未配置 GOOD_BADMINTON_BUSINESS_DATABASE_URL；管理页不会写入影子数据。"}
        # ``render_backoffice_tabs()`` reads this state for several tab
        # defaults.  Reusing a very short snapshot prevents one unavailable
        # database from serially blocking the whole WebUI during first paint.
        if self._readiness_cache and time.monotonic() - self._readiness_checked_monotonic < 3:
            return dict(self._readiness_cache)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("select current_database() as database, now() as checked_at")
                row = cursor.fetchone()
            result = {"status": "ready", "database": row["database"], "checked_at": str(row["checked_at"])}
        except BackofficeError as exc:
            result = {"status": "unavailable", "message": str(exc)}
        self._readiness_cache = result
        self._readiness_checked_monotonic = time.monotonic()
        return dict(result)

    def list_venues(self) -> list[list[str]]:
        return self._rows(
            """select v.id::text, v.tenant_id::text, v.code, v.name, v.timezone, coalesce(v.address, ''), v.status, count(c.id)::int
               from business.venues v left join business.courts c on c.venue_id = v.id
               group by v.id order by v.created_at desc"""
        )

    def save_venue(self, venue_id: str, tenant_id: str, code: str, name: str, timezone_name: str, address: str, status: str) -> str:
        if not tenant_id.strip() or not code.strip() or not name.strip():
            raise BackofficeError("租户 ID、场馆编码和场馆名称均为必填项。")
        if status not in _VENUE_STATUSES:
            raise BackofficeError("场馆状态无效。")
        parameters = (code.strip(), name.strip(), timezone_name.strip() or "Asia/Shanghai", address.strip() or None, status)
        if venue_id.strip():
            if not self._execute("update business.venues set code=%s, name=%s, timezone=%s, address=%s, status=%s where id=%s and tenant_id=%s", parameters + (venue_id.strip(), tenant_id.strip())):
                raise BackofficeError("未找到指定场馆，或该场馆不属于该租户。")
            self._audit("venue.updated", "venue", venue_id.strip(), {"code": code.strip(), "status": status})
            return venue_id.strip()
        created_id = str(uuid.uuid4())
        self._execute("insert into business.venues (id, tenant_id, code, name, timezone, address, status) values (%s, %s, %s, %s, %s, %s, %s)", (created_id, tenant_id.strip()) + parameters)
        self._audit("venue.created", "venue", created_id, {"code": code.strip(), "status": status})
        return created_id

    def list_courts(self) -> list[list[str]]:
        return self._rows("select id::text, venue_id::text, code, name, sort_order, status from business.courts order by venue_id, sort_order, name")

    def save_court(self, court_id: str, venue_id: str, code: str, name: str, sort_order: int, status: str) -> str:
        if not venue_id.strip() or not code.strip() or not name.strip():
            raise BackofficeError("场馆 ID、场地编码和场地名称均为必填项。")
        if status not in _COURT_STATUSES:
            raise BackofficeError("场地状态无效。")
        parameters = (code.strip(), name.strip(), max(0, int(sort_order)), status)
        if court_id.strip():
            if not self._execute("update business.courts set code=%s, name=%s, sort_order=%s, status=%s where id=%s and venue_id=%s", parameters + (court_id.strip(), venue_id.strip())):
                raise BackofficeError("未找到指定场地，或该场地不属于该场馆。")
            self._audit("court.updated", "court", court_id.strip(), {"code": code.strip(), "status": status})
            return court_id.strip()
        created_id = str(uuid.uuid4())
        self._execute("insert into business.courts (id, venue_id, code, name, sort_order, status) values (%s, %s, %s, %s, %s, %s)", (created_id, venue_id.strip()) + parameters)
        self._audit("court.created", "court", created_id, {"code": code.strip(), "status": status})
        return created_id

    def list_people(self) -> list[list[str]]:
        return self._rows(
            """select u.id::text, coalesce(u.nickname, ''), u.status, u.profile_visibility,
                      coalesce(string_agg(v.name || ':' || vm.role, ', ' order by v.name), '')
               from business.users u
               left join business.venue_memberships vm on vm.user_id = u.id and vm.status = 'active'
               left join business.venues v on v.id = vm.venue_id
               group by u.id order by u.created_at desc"""
        )

    def save_person(self, user_id: str, nickname: str, status: str, visibility: str) -> str:
        if status not in _USER_STATUSES or visibility not in {"private", "venue"}:
            raise BackofficeError("用户状态或资料可见范围无效。")
        if user_id.strip():
            if not self._execute("update business.users set nickname=%s, status=%s, profile_visibility=%s where id=%s", (nickname.strip() or None, status, visibility, user_id.strip())):
                raise BackofficeError("未找到指定用户。")
            self._audit("user.updated", "user", user_id.strip(), {"status": status, "profile_visibility": visibility})
            return user_id.strip()
        created_id = str(uuid.uuid4())
        self._execute("insert into business.users (id, nickname, status, profile_visibility) values (%s, %s, %s, %s)", (created_id, nickname.strip() or None, status, visibility))
        self._audit("user.created", "user", created_id, {"status": status, "profile_visibility": visibility})
        return created_id

    def assign_membership(self, venue_id: str, user_id: str, role: str, status: str) -> None:
        if role not in _MEMBERSHIP_ROLES or status not in {"active", "revoked"}:
            raise BackofficeError("成员角色或状态无效。")
        if not venue_id.strip() or not user_id.strip():
            raise BackofficeError("场馆 ID 和用户 ID 均为必填项。")
        self._execute(
            """insert into business.venue_memberships (venue_id, user_id, role, status, revoked_at)
               values (%s, %s, %s, %s, case when %s = 'revoked' then now() else null end)
               on conflict (venue_id, user_id) do update set role=excluded.role, status=excluded.status,
               revoked_at=case when excluded.status='revoked' then now() else null end""",
            (venue_id.strip(), user_id.strip(), role, status, status),
        )
        self._audit("venue_membership.updated", "venue_membership", None, {"venue_ref": venue_id.strip(), "member_ref": user_id.strip(), "role": role, "status": status})

    def list_analysis_jobs(self) -> list[list[str]]:
        return self._rows(
            """select j.id::text, j.match_id::text, j.status, j.job_type, coalesce(j.external_analysis_session_id, ''),
                      coalesce(ci.display_name, ''), j.created_at::text, coalesce(j.finished_at::text, '')
               from business.analysis_jobs j left join business.compute_instances ci on ci.id=j.compute_instance_id
               order by j.created_at desc limit 100"""
        )

    def list_compute_instances(self) -> list[list[str]]:
        return self._rows(
            """select id::text, provider, external_instance_id, display_name, status,
                      coalesce(hourly_cost::text, ''), currency, coalesce(last_heartbeat_at::text, '')
               from business.compute_instances order by updated_at desc"""
        )

    def dashboard(self) -> dict[str, Any]:
        """Return only persisted business metrics; no disconnected demo figures."""
        rows = self._rows(
            """select
                 (select count(*) from business.venues where status='active') as active_venues,
                 (select count(*) from business.courts where status='active') as active_courts,
                 (select count(*) from business.matches where started_at >= now() - interval '7 days') as matches_7d,
                 (select count(*) from business.matches where lifecycle_status='result_confirmed' and started_at >= now() - interval '7 days') as confirmed_matches_7d,
                 (select count(*) from business.venue_activities where status='published' and starts_at >= now()) as upcoming_activities,
                 (select count(*) from business.activity_registrations where status='registered') as active_registrations,
                 (select count(*) from business.match_deliveries where status='ready') as ready_deliveries"""
        )
        keys = ["active_venues", "active_courts", "matches_7d", "confirmed_matches_7d", "upcoming_activities", "active_registrations", "ready_deliveries"]
        values = rows[0] if rows else ["0"] * len(keys)
        result = dict(zip(keys, values))
        matches = int(result["matches_7d"] or 0)
        result["confirmation_rate_7d"] = round(int(result["confirmed_matches_7d"] or 0) * 100 / matches, 1) if matches else 0
        return result

    def list_qr_tokens(self) -> list[list[str]]:
        return self._rows(
            """select q.id::text, v.name, coalesce(c.name, ''), q.scope, q.label, q.status,
                      q.created_at::text, coalesce(q.expires_at::text, ''), coalesce(q.revoked_at::text, '')
               from business.qr_tokens q join business.venues v on v.id=q.venue_id
               left join business.courts c on c.id=q.court_id order by q.created_at desc limit 100"""
        )

    def create_qr_token(self, venue_id: str, court_id: str, scope: str, label: str, expires_at: str) -> dict[str, str]:
        if scope not in _QR_SCOPES or not venue_id.strip():
            raise BackofficeError("二维码类型和场馆 ID 均为必填项。")
        if scope == "court" and not court_id.strip():
            raise BackofficeError("场地码必须指定场地 ID。")
        if scope == "venue" and court_id.strip():
            raise BackofficeError("场馆码不能绑定场地 ID。")
        raw_token = secrets.token_urlsafe(24)
        token_digest = sha256(raw_token.encode("utf-8")).hexdigest()
        record_id = str(uuid.uuid4())
        payload = f"badminton://v1/scan/{raw_token}"
        expiry = str(expires_at or "").strip() or None
        self._execute(
            """insert into business.qr_tokens (id, venue_id, court_id, scope, label, token_digest, expires_at)
               values (%s, %s, %s, %s, %s, %s, nullif(%s, '')::timestamptz)""",
            (record_id, venue_id.strip(), court_id.strip() or None, scope, label.strip() or ("场地码" if scope == "court" else "场馆码"), token_digest, expiry or ""),
        )
        self._audit("qr_token.created", "qr_token", record_id, {"venue_ref": venue_id.strip(), "court_ref": court_id.strip() or None, "scope": scope})
        return {"id": record_id, "payload": payload, "message": "二维码已生成。请立即下载或打印；后台只保存摘要，不能再次读取原始令牌。"}

    def revoke_qr_token(self, token_id: str) -> None:
        if not token_id.strip():
            raise BackofficeError("请输入二维码 ID。")
        if not self._execute("update business.qr_tokens set status='revoked', revoked_at=now() where id=%s and status='active'", (token_id.strip(),)):
            raise BackofficeError("未找到可撤销的有效二维码。")
        self._audit("qr_token.revoked", "qr_token", token_id.strip(), {})

    def list_activities(self) -> list[list[str]]:
        return self._rows(
            """select a.id::text, v.name, a.title, a.status, a.starts_at::text,
                      coalesce(a.capacity::text, ''), count(r.user_id) filter (where r.status='registered')::text
               from business.venue_activities a join business.venues v on v.id=a.venue_id
               left join business.activity_registrations r on r.activity_id=a.id
               group by a.id, v.name order by a.starts_at desc limit 100"""
        )

    def save_activity(self, activity_id: str, venue_id: str, title: str, description: str, starts_at: str, deadline: str, capacity: int | None, status: str) -> str:
        if not venue_id.strip() or not title.strip() or not starts_at.strip() or status not in _ACTIVITY_STATUSES:
            raise BackofficeError("场馆、标题、开始时间和有效状态均为必填项。")
        parameters = (title.strip(), description.strip() or None, starts_at.strip(), deadline.strip() or None, int(capacity) if capacity else None, status)
        if activity_id.strip():
            if not self._execute(
                """update business.venue_activities set title=%s, description=%s, starts_at=%s::timestamptz,
                   registration_deadline_at=nullif(%s, '')::timestamptz, capacity=%s, status=%s where id=%s and venue_id=%s""",
                parameters + (activity_id.strip(), venue_id.strip()),
            ):
                raise BackofficeError("未找到活动，或活动不属于该场馆。")
            saved_id = activity_id.strip()
            action = "activity.updated"
        else:
            saved_id = str(uuid.uuid4())
            self._execute(
                """insert into business.venue_activities (id, venue_id, title, description, starts_at, registration_deadline_at, capacity, status)
                   values (%s, %s, %s, %s, %s::timestamptz, nullif(%s, '')::timestamptz, %s, %s)""",
                (saved_id, venue_id.strip()) + parameters,
            )
            action = "activity.created"
        self._audit(action, "activity", saved_id, {"venue_ref": venue_id.strip(), "status": status})
        return saved_id

    def list_community_posts(self) -> list[list[str]]:
        return self._rows(
            """select p.id::text, v.name, p.post_type, p.status, left(coalesce(p.body, ''), 120),
                      coalesce(u.nickname, '运营'), p.created_at::text
               from business.community_posts p join business.venues v on v.id=p.venue_id
               left join business.users u on u.id=p.author_user_id order by p.created_at desc limit 100"""
        )

    def create_community_post(self, venue_id: str, author_user_id: str, post_type: str, body: str) -> str:
        if not venue_id.strip() or not author_user_id.strip() or post_type not in _POST_TYPES or not body.strip():
            raise BackofficeError("场馆、发布运营人员、内容类型和正文均为必填项。")
        saved_id = str(uuid.uuid4())
        self._execute(
            """insert into business.community_posts (id, venue_id, author_user_id, post_type, body, status, visibility)
               values (%s, %s, %s, %s, %s, 'published', 'venue')""",
            (saved_id, venue_id.strip(), author_user_id.strip(), post_type, body.strip()),
        )
        self._audit("community_post.created", "community_post", saved_id, {"venue_ref": venue_id.strip(), "post_type": post_type})
        return saved_id

    def set_community_post_status(self, post_id: str, status: str) -> None:
        if not post_id.strip() or status not in _POST_STATUSES:
            raise BackofficeError("帖子 ID 或状态无效。")
        if not self._execute("update business.community_posts set status=%s, hidden_at=case when %s='hidden' then now() else null end where id=%s", (status, status, post_id.strip())):
            raise BackofficeError("未找到帖子。")
        self._audit("community_post.moderated", "community_post", post_id.strip(), {"status": status})

    def list_matches(self) -> list[list[str]]:
        return self._rows(
            """select m.id::text, v.name, c.name, m.match_format, m.lifecycle_status,
                      coalesce(m.score_summary::text, ''), coalesce(m.external_analysis_session_id, ''),
                      m.started_at::text, coalesce(m.recording_stopped_at::text, '')
               from business.matches m join business.venues v on v.id=m.venue_id join business.courts c on c.id=m.court_id
               order by m.created_at desc limit 100"""
        )

    def match_detail(self, match_id: str | None) -> dict[str, Any]:
        if not match_id:
            return {"hint": "选择一场比赛，查看赛果确认、匿名轨迹认领和个人投递状态。"}
        matches = self._rows(
            """select id::text, lifecycle_status, match_format, coalesce(score_summary::text, ''),
                      coalesce(recording_gateway_receipt, ''), coalesce(external_analysis_session_id, ''),
                      coalesce(recording_stopped_at::text, '') from business.matches where id=%s""", (match_id,)
        )
        if not matches:
            return {"error": "未找到比赛。", "match_id": match_id}
        participants = self._rows("select user_id::text, coalesce(team_id, ''), coalesce(slot_id, ''), result_confirmation, coalesce(confirmed_at::text, '') from business.match_participants where match_id=%s", (match_id,))
        claims = self._rows("select id::text, track_id, user_id::text, status, coalesce(confidence::text, ''), coalesce(evidence_ref, '') from business.track_claims where match_id=%s order by claimed_at", (match_id,))
        deliveries = self._rows("select id::text, user_id::text, delivery_type, status, coalesce(reason, ''), coalesce(available_at::text, '') from business.match_deliveries where match_id=%s order by created_at", (match_id,))
        return {"match": matches[0], "participants": participants, "track_claims": claims, "deliveries": deliveries}

    def review_track_claim(self, claim_id: str, status: str) -> None:
        if not claim_id.strip() or status not in _CLAIM_STATUSES - {"pending"}:
            raise BackofficeError("认领 ID 或审核状态无效。")
        if not self._execute("update business.track_claims set status=%s, reviewed_at=now() where id=%s", (status, claim_id.strip())):
            raise BackofficeError("未找到轨迹认领。")
        self._audit("track_claim.reviewed", "track_claim", claim_id.strip(), {"status": status})

    def set_delivery_status(self, delivery_id: str, status: str, reason: str) -> None:
        if not delivery_id.strip() or status not in _DELIVERY_STATUSES:
            raise BackofficeError("投递 ID 或状态无效。")
        if not self._execute("update business.match_deliveries set status=%s, reason=%s, available_at=case when %s='ready' then now() else available_at end where id=%s", (status, reason.strip() or None, status, delivery_id.strip())):
            raise BackofficeError("未找到投递记录。")
        self._audit("match_delivery.updated", "match_delivery", delivery_id.strip(), {"status": status})

    def list_audit_events(self) -> list[list[str]]:
        return self._rows(
            """select created_at::text, actor_type, action, resource_type, coalesce(resource_id, ''), after_summary::text
               from business.audit_events order by created_at desc limit 100"""
        )

    def _connect(self):
        if not self.database_url:
            raise BackofficeError("未配置 GOOD_BADMINTON_BUSINESS_DATABASE_URL。")
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise BackofficeError("未安装 psycopg；请执行 pip install -r requirements.txt。") from exc
        try:
            return psycopg.connect(
                self.database_url,
                row_factory=dict_row,
                autocommit=True,
                connect_timeout=self.connect_timeout_seconds,
            )
        except Exception as exc:
            raise BackofficeError(f"业务数据库不可用：{exc}") from exc

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[list[str]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            return [["" if value is None else str(value) for value in row.values()] for row in cursor.fetchall()]

    def _execute(self, query: str, params: tuple[Any, ...]) -> int:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def _audit(self, action: str, resource_type: str, resource_id: str | None, after_summary: dict[str, Any]) -> None:
        self._execute("insert into business.audit_events (id, actor_type, action, resource_type, resource_id, after_summary) values (%s, 'system', %s, %s, %s, %s::jsonb)", (str(uuid.uuid4()), action, resource_type, resource_id, json.dumps(after_summary, ensure_ascii=False)))


class OperatorBackoffice:
    """Business-side controls; browser input can never supply a shell command."""

    def __init__(self, active_task_controller: Any, *, config_path: Path = _DEFAULT_CONFIG_PATH, operations_path: Path = _DEFAULT_OPERATIONS_PATH, database: BusinessDatabase | None = None) -> None:
        self.active_task_controller = active_task_controller
        self.config_path = Path(config_path)
        self.operations_path = Path(operations_path)
        self.database = database or BusinessDatabase()
        self._operation_lock = threading.Lock()

    def overview(self) -> dict[str, Any]:
        tasks = BusinessTaskLedger().list_tasks(limit=20)
        counts: dict[str, int] = {}
        for task in tasks:
            status = str(task.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {
            "mode": "local_development_operator",
            "business_database": self.database.readiness(),
            "business_api": self._business_health(),
            "gpu": self.gpu_status(check_health=False),
            "active_webui_task": self.active_task_controller.snapshot(),
            "recent_task_status_counts": counts,
            "notice": "本机运营模式未接入登录/RLS；禁止将此 WebUI 暴露到公网。",
        }

    def _business_health(self) -> dict[str, Any]:
        try:
            load_runtime_environment()
            base_url = business_api_base_url()
        except RuntimeConfigurationError as exc:
            return {"status": "misconfigured", "message": str(exc)}
        try:
            with urlopen(Request(base_url + "/api/v1/health", method="GET"), timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
            return {"status": "ready", "base_url": base_url, "payload": payload}
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            return {"status": "unavailable", "base_url": base_url, "message": str(exc)}

    def gpu_status(self, *, check_health: bool = True) -> dict[str, Any]:
        try:
            config = remote_gpu_config()
        except RemoteAnalysisError as exc:
            return {"status": "misconfigured", "message": str(exc)}
        control_ready = bool(os.environ.get("GOOD_BADMINTON_GPU_CONTROL_COMMAND")) and os.environ.get("GOOD_BADMINTON_ENABLE_GPU_AUTOMATION") == "1"
        output: dict[str, Any] = {
            "configured": True,
            "base_url": config["base_url"],
            "api_key_configured": bool(config["api_key"]),
            "automation": "enabled" if control_ready else "external_managed_health_only",
            "control_hint": None if control_ready else "未配置受控 GPU 自动化适配器；可健康检查和取消任务，不能从页面启动/关闭实例。",
        }
        if not check_health:
            output["status"] = "not_checked"
            return output
        try:
            output["health"] = _json_request(config["base_url"], config["api_key"], "/api/v1/health", timeout=min(float(config["timeout_seconds"]), 10))
            output["status"] = "ready"
        except BackofficeError as exc:
            output.update({"status": "unavailable", "message": str(exc)})
        return output

    def save_gpu_config(self, base_url: str, new_api_key: str) -> dict[str, Any]:
        safe_url = _validate_base_url(base_url)
        values = _read_env_file(self.config_path)
        values["GOOD_BADMINTON_GPU_API_URL"] = safe_url
        replacement = str(new_api_key or "").strip()
        if replacement:
            values["GOOD_BADMINTON_GPU_API_KEY"] = replacement
            os.environ["GOOD_BADMINTON_GPU_API_KEY"] = replacement
        _write_env_file(self.config_path, values)
        # An explicit operator save should take effect in this process too;
        # the key itself never appears in response payloads or audit data.
        os.environ["GOOD_BADMINTON_GPU_API_URL"] = safe_url
        self._record_operation("gpu_config_saved", {"base_url": safe_url, "api_key_replaced": bool(replacement)})
        output = self.gpu_status(check_health=False)
        output["message"] = "配置已保存到服务端本机文件；API Key 不会回显。"
        return output

    def request_gpu_operation(self, operation: str) -> dict[str, Any]:
        if operation not in {"start", "stop"}:
            raise BackofficeError("只支持 start 或 stop。")
        command = os.environ.get("GOOD_BADMINTON_GPU_CONTROL_COMMAND", "").strip()
        if not command or os.environ.get("GOOD_BADMINTON_ENABLE_GPU_AUTOMATION") != "1":
            result = {
                "operation": operation,
                "status": "not_available",
                "message": "当前 GPU 为外部托管。请在服务端预配置 GOOD_BADMINTON_GPU_CONTROL_COMMAND 与 GOOD_BADMINTON_ENABLE_GPU_AUTOMATION=1 后再启用自动化。",
            }
            self._record_operation("gpu_operation_not_available", result)
            return result
        operation_id = uuid.uuid4().hex
        self._record_operation("gpu_operation_requested", {"operation_id": operation_id, "operation": operation})
        try:
            completed = subprocess.run(shlex.split(command, posix=False) + [operation], cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=120, shell=False, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = {"operation_id": operation_id, "operation": operation, "status": "failed", "message": str(exc)}
            self._record_operation("gpu_operation_failed", result)
            return result
        result = {
            "operation_id": operation_id,
            "operation": operation,
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "message": (completed.stdout or completed.stderr or "操作脚本已完成").strip()[:1000],
        }
        self._record_operation("gpu_operation_completed", result)
        return result

    def list_ledger_tasks(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for task in BusinessTaskLedger().list_tasks(limit=100):
            remote = task.get("remote") or {}
            error = task.get("error") or {}
            rows.append([
                str(task.get("task_id") or ""), str(task.get("status") or ""), str(task.get("created_at") or ""), str(task.get("updated_at") or ""),
                str(remote.get("job_id") or ""), str(remote.get("base_url") or ""), str(task.get("output_dir") or ""), str(error.get("message") or "")[:180],
            ])
        return rows

    def task_choices(self) -> list[tuple[str, str]]:
        return [(f"{row[2][:19]} · {row[1]} · {row[0][:12]}", row[0]) for row in self.list_ledger_tasks()]

    def task_detail(self, task_id: str | None) -> dict[str, Any]:
        if not task_id:
            return {"hint": "选择一个业务任务查看远端 Job 与审计历史。"}
        return BusinessTaskLedger().get(str(task_id)) or {"error": "未找到任务账本记录。", "task_id": task_id}

    def cancel_remote_task(self, task_id: str) -> dict[str, Any]:
        task = BusinessTaskLedger().get(task_id)
        if task is None:
            raise BackofficeError("任务账本不存在。")
        remote = task.get("remote") or {}
        job_id = str(remote.get("job_id") or "")
        if not job_id or not remote.get("accepted"):
            raise BackofficeError("该任务尚未获得 GPU 接收回执，不能安全地下发远端取消。")
        try:
            config = remote_gpu_config(str(remote.get("base_url") or "") or None)
            response = _json_request(config["base_url"], config["api_key"], f"/api/v1/jobs/{job_id}", method="DELETE")
        except (RemoteAnalysisError, BackofficeError) as exc:
            message = f"本地已请求中断，但无法确认远端 GPU 是否停止：{exc}"
            BusinessTaskLedger().record_terminal(task_id, status="interrupted_unconfirmed", error={"type": "RemoteCancellationUnconfirmed", "message": message})
            self._record_operation("remote_task_cancel_unconfirmed", {"task_ref": task_id, "job_ref": job_id})
            return {"task_id": task_id, "job_id": job_id, "status": "interrupted_unconfirmed", "message": message}
        phase = str(response.get("status") or "cancelling")
        BusinessTaskLedger().record_remote_event(task_id, {"mode": "remote_gpu", "phase": phase, "job_id": job_id, "cancellation_requested": True})
        if phase == "cancelled":
            BusinessTaskLedger().record_terminal(task_id, status="cancelled")
        self._record_operation("remote_task_cancel_requested", {"task_ref": task_id, "job_ref": job_id, "remote_status": phase})
        return {"task_id": task_id, "job_id": job_id, "status": phase, "message": "已向 GPU 发送取消请求；请刷新任务状态确认最终结果。"}

    def request_local_interrupt(self) -> dict[str, Any]:
        return self.active_task_controller.request_cancel() or {"status": "idle", "message": "当前没有本机 WebUI 分析任务。"}

    def _record_operation(self, event: str, payload: dict[str, Any]) -> None:
        self.operations_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"at": datetime.now(timezone.utc).isoformat(), "event": event, "payload": payload}
        with self._operation_lock, self.operations_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_backoffice_tabs(active_task_controller: Any) -> None:
    """Render the venue-first B-scheme console in the current Gradio tabs."""
    service = OperatorBackoffice(active_task_controller)

    def ready() -> bool:
        return service.database.readiness().get("status") == "ready"

    def refresh_venues():
        status = service.database.readiness()
        return (service.database.list_venues(), service.database.list_courts(), status) if status.get("status") == "ready" else ([], [], status)

    def refresh_dashboard():
        status = service.database.readiness()
        dashboard = service.database.dashboard() if status.get("status") == "ready" else {}
        return _dashboard_markup(dashboard, service.overview()), _status_text(status)

    def save_venue(*values):
        saved_id = service.database.save_venue(*values)
        venues, courts, _ = refresh_venues()
        return venues, courts, f"场馆已保存：`{saved_id}`"

    def save_court(*values):
        saved_id = service.database.save_court(*values)
        venues, courts, _ = refresh_venues()
        return venues, courts, f"场地已保存：`{saved_id}`"

    def refresh_qrs():
        return service.database.list_qr_tokens() if ready() else []

    def create_qr(venue_id, court_id, scope, label, expires_at):
        result = service.database.create_qr_token(venue_id, court_id, scope, label, expires_at)
        try:
            import qrcode
            image = qrcode.make(result["payload"])
        except ImportError:
            image = None
        return refresh_qrs(), image, result["payload"], result["message"]

    def revoke_qr(token_id):
        service.database.revoke_qr_token(token_id)
        return refresh_qrs(), "二维码已撤销；旧码扫描将被业务网关拒绝。"

    def refresh_community():
        if not ready():
            return [], [], service.database.readiness()
        return service.database.list_activities(), service.database.list_community_posts(), service.database.readiness()

    def save_activity(*values):
        saved_id = service.database.save_activity(*values)
        activities, posts, _ = refresh_community()
        return activities, posts, f"活动已保存：`{saved_id}`"

    def create_post(*values):
        saved_id = service.database.create_community_post(*values)
        activities, posts, _ = refresh_community()
        return activities, posts, f"场馆动态已发布：`{saved_id}`"

    def moderate_post(post_id, status):
        service.database.set_community_post_status(post_id, status)
        activities, posts, _ = refresh_community()
        return activities, posts, "动态状态已更新。"

    def refresh_matches(selected=None):
        rows = service.database.list_matches() if ready() else []
        choices = [(f"{row[1]} · {row[2]} · {row[4]} · {row[0][:8]}", row[0]) for row in rows]
        values = {value for _, value in choices}
        selected = selected if selected in values else (choices[0][1] if choices else None)
        return rows, gr.update(choices=choices, value=selected), service.database.match_detail(selected) if ready() else service.database.readiness()

    def review_claim(claim_id, status, selected):
        service.database.review_track_claim(claim_id, status)
        return refresh_matches(selected)

    def update_delivery(delivery_id, status, reason, selected):
        service.database.set_delivery_status(delivery_id, status, reason)
        return refresh_matches(selected)

    def refresh_people():
        status = service.database.readiness()
        return (service.database.list_people() if status.get("status") == "ready" else []), status

    def save_person(*values):
        saved_id = service.database.save_person(*values)
        people, _ = refresh_people()
        return people, f"用户已保存：`{saved_id}`"

    def save_membership(venue_id, user_id, role, status):
        service.database.assign_membership(venue_id, user_id, role, status)
        people, _ = refresh_people()
        return people, "成员角色已保存。"

    def refresh_audit():
        return service.database.list_audit_events() if ready() else []

    def configured_gpu_url() -> str:
        try:
            return str(remote_gpu_config()["base_url"])
        except RemoteAnalysisError:
            return ""

    def save_gpu(base_url, new_api_key):
        return service.save_gpu_config(base_url, new_api_key), gr.update(value="")

    def refresh_gpu_service():
        instances = service.database.list_compute_instances() if ready() else []
        return service.gpu_status(check_health=True), instances

    def refresh_tasks(selected_task_id=None):
        choices = service.task_choices()
        values = {value for _, value in choices}
        selected = selected_task_id if selected_task_id in values else (choices[0][1] if choices else None)
        jobs = service.database.list_analysis_jobs() if ready() else []
        return service.list_ledger_tasks(), gr.update(choices=choices, value=selected), service.task_detail(selected), jobs

    def cancel_control_task(task_id):
        result = service.cancel_remote_task(task_id)
        tasks, choices, detail, jobs = refresh_tasks(task_id)
        return result, tasks, choices, detail, jobs

    with gr.Tab("经营看板"):
        dashboard_output = gr.HTML(value=_dashboard_markup(service.database.dashboard() if ready() else {}, service.overview()), elem_classes=["venue-ops"])
        with gr.Row():
            dashboard_refresh = gr.Button("刷新看板", variant="primary", size="sm")
            dashboard_db_status = gr.HTML(value=_status_text(service.database.readiness()))
        dashboard_refresh.click(refresh_dashboard, outputs=[dashboard_output, dashboard_db_status], show_progress="hidden")

    with gr.Tab("场馆与场地"):
        gr.Markdown("## 场馆、场地与扫码入口\n每一片场地必须绑定到一个场馆；二维码只含一次性不可逆的业务令牌。")
        venue_db_status = gr.JSON(label="数据库连接", value=service.database.readiness())
        with gr.Row():
            venues_table = gr.Dataframe(headers=["场馆 ID", "租户 ID", "编码", "名称", "时区", "地址", "状态", "场地数"], value=[], interactive=False, label="场馆")
            courts_table = gr.Dataframe(headers=["场地 ID", "场馆 ID", "编码", "名称", "排序", "状态"], value=[], interactive=False, label="场地")
        refresh_venue_button = gr.Button("刷新场馆数据", size="sm")
        with gr.Accordion("新增或编辑场馆", open=False):
            venue_id = gr.Textbox(label="场馆 ID（留空则新增）")
            venue_tenant_id = gr.Textbox(label="租户 ID")
            with gr.Row():
                venue_code = gr.Textbox(label="场馆编码")
                venue_name = gr.Textbox(label="场馆名称")
            with gr.Row():
                venue_timezone = gr.Textbox(label="时区", value="Asia/Shanghai")
                venue_status = gr.Dropdown(choices=sorted(_VENUE_STATUSES), value="active", label="状态")
            venue_address = gr.Textbox(label="地址")
            venue_save = gr.Button("保存场馆", variant="primary", size="sm")
            venue_notice = gr.Markdown()
        with gr.Accordion("新增或编辑场地", open=False):
            court_id = gr.Textbox(label="场地 ID（留空则新增）")
            court_venue_id = gr.Textbox(label="所属场馆 ID")
            with gr.Row():
                court_code = gr.Textbox(label="场地编码")
                court_name = gr.Textbox(label="场地名称")
                court_sort_order = gr.Number(label="排序", value=0, precision=0)
            court_status = gr.Dropdown(choices=sorted(_COURT_STATUSES), value="active", label="状态")
            court_save = gr.Button("保存场地", variant="primary", size="sm")
            court_notice = gr.Markdown()
        gr.Markdown("### 场馆码与场地码\n场馆码进入场馆社区；场地码进入对应场地的加入对局流程。生成后请立即下载，系统不会保存明文令牌。")
        qr_table = gr.Dataframe(headers=["二维码 ID", "场馆", "场地", "类型", "标签", "状态", "生成时间", "过期时间", "撤销时间"], value=[], interactive=False, label="二维码生命周期")
        refresh_qr_button = gr.Button("刷新二维码", size="sm")
        with gr.Accordion("生成新二维码", open=False):
            with gr.Row():
                qr_venue_id = gr.Textbox(label="场馆 ID")
                qr_court_id = gr.Textbox(label="场地 ID（仅场地码）")
            with gr.Row():
                qr_scope = gr.Dropdown(choices=["venue", "court"], value="court", label="二维码类型")
                qr_label = gr.Textbox(label="打印标签", value="1 号场")
                qr_expires = gr.Textbox(label="失效时间（可空，ISO 8601）")
            qr_create = gr.Button("生成二维码", variant="primary", size="sm")
            qr_image = gr.Image(label="可下载二维码", type="pil")
            qr_payload = gr.Textbox(label="扫码载荷（仅本次显示）", interactive=False)
            qr_notice = gr.Markdown()
        with gr.Accordion("撤销二维码", open=False):
            qr_revoke_id = gr.Textbox(label="二维码 ID")
            qr_revoke = gr.Button("撤销二维码", variant="stop", size="sm")
            qr_revoke_notice = gr.Markdown()
        refresh_venue_button.click(refresh_venues, outputs=[venues_table, courts_table, venue_db_status], show_progress="hidden")
        venue_save.click(save_venue, inputs=[venue_id, venue_tenant_id, venue_code, venue_name, venue_timezone, venue_address, venue_status], outputs=[venues_table, courts_table, venue_notice])
        court_save.click(save_court, inputs=[court_id, court_venue_id, court_code, court_name, court_sort_order, court_status], outputs=[venues_table, courts_table, court_notice])
        refresh_qr_button.click(refresh_qrs, outputs=[qr_table], show_progress="hidden")
        qr_create.click(create_qr, inputs=[qr_venue_id, qr_court_id, qr_scope, qr_label, qr_expires], outputs=[qr_table, qr_image, qr_payload, qr_notice])
        qr_revoke.click(revoke_qr, inputs=[qr_revoke_id], outputs=[qr_table, qr_revoke_notice])

    with gr.Tab("活动与社区"):
        gr.Markdown("## 场馆活动与社区动态\n活动、挑战和排行榜的运营入口归属于场馆；公开内容可审核，原始视频不进入动态。")
        community_db_status = gr.JSON(label="数据库连接", value=service.database.readiness())
        activities_table = gr.Dataframe(headers=["活动 ID", "场馆", "标题", "状态", "开始时间", "容量", "已报名"], value=[], interactive=False, label="活动")
        posts_table = gr.Dataframe(headers=["动态 ID", "场馆", "类型", "状态", "内容摘要", "作者", "发布时间"], value=[], interactive=False, label="社区动态")
        community_refresh = gr.Button("刷新活动与动态", variant="primary", size="sm")
        with gr.Accordion("新建或编辑活动", open=False):
            activity_id = gr.Textbox(label="活动 ID（留空则新增）")
            activity_venue = gr.Textbox(label="场馆 ID")
            activity_title = gr.Textbox(label="标题")
            activity_description = gr.Textbox(label="活动说明", lines=3)
            with gr.Row():
                activity_start = gr.Textbox(label="开始时间（ISO 8601）")
                activity_deadline = gr.Textbox(label="报名截止（可空）")
                activity_capacity = gr.Number(label="容量（可空）", precision=0)
                activity_status = gr.Dropdown(choices=sorted(_ACTIVITY_STATUSES), value="draft", label="状态")
            activity_save = gr.Button("保存活动", variant="primary", size="sm")
            activity_notice = gr.Markdown()
        with gr.Accordion("发布场馆动态", open=False):
            with gr.Row():
                post_venue = gr.Textbox(label="场馆 ID")
                post_author_user_id = gr.Textbox(label="发布运营人员 ID")
            post_type = gr.Dropdown(choices=sorted(_POST_TYPES), value="official", label="类型")
            post_content = gr.Textbox(label="正文", lines=4)
            post_create = gr.Button("发布动态", variant="primary", size="sm")
            post_notice = gr.Markdown()
        with gr.Accordion("审核动态", open=False):
            moderate_post_id = gr.Textbox(label="动态 ID")
            moderate_post_status = gr.Dropdown(choices=sorted(_POST_STATUSES), value="hidden", label="状态")
            moderate_post_button = gr.Button("更新动态状态", size="sm")
            moderate_notice = gr.Markdown()
        community_refresh.click(refresh_community, outputs=[activities_table, posts_table, community_db_status], show_progress="hidden")
        activity_save.click(save_activity, inputs=[activity_id, activity_venue, activity_title, activity_description, activity_start, activity_deadline, activity_capacity, activity_status], outputs=[activities_table, posts_table, activity_notice])
        post_create.click(create_post, inputs=[post_venue, post_author_user_id, post_type, post_content], outputs=[activities_table, posts_table, post_notice])
        moderate_post_button.click(moderate_post, inputs=[moderate_post_id, moderate_post_status], outputs=[activities_table, posts_table, moderate_notice])

    with gr.Tab("比赛与认领"):
        gr.Markdown("## 赛果、停录回执与角色认领\n一名参赛者提交比分后才能请求停录；仅收到录像网关回执才显示已停录。赛果的另一方确认异步进行，正式榜单/挑战只读取 `result_confirmed`。")
        matches_table = gr.Dataframe(headers=["比赛 ID", "场馆", "场地", "赛制", "状态", "比分", "匿名分析会话", "开始", "停录回执时间"], value=[], interactive=False, label="最近比赛")
        match_select = gr.Dropdown(label="选择比赛")
        match_detail_output = gr.JSON(label="参赛者、认领与投递", value=service.database.match_detail(None))
        match_refresh = gr.Button("刷新比赛状态", variant="primary", size="sm")
        with gr.Accordion("审核 Track ID 认领", open=False):
            claim_id = gr.Textbox(label="认领 ID")
            claim_status = gr.Dropdown(choices=sorted(_CLAIM_STATUSES - {"pending"}), value="confirmed", label="审核结果")
            claim_review = gr.Button("保存认领审核", size="sm")
        with gr.Accordion("个人照片、录像与报告投递", open=False):
            delivery_id = gr.Textbox(label="投递 ID")
            delivery_status = gr.Dropdown(choices=sorted(_DELIVERY_STATUSES), value="processing", label="投递状态")
            delivery_reason = gr.Textbox(label="备注/失败原因（可空）")
            delivery_save = gr.Button("更新投递状态", size="sm")
        match_refresh.click(refresh_matches, inputs=[match_select], outputs=[matches_table, match_select, match_detail_output], show_progress="hidden")
        match_select.change(service.database.match_detail, inputs=[match_select], outputs=[match_detail_output], show_progress="hidden")
        claim_review.click(review_claim, inputs=[claim_id, claim_status, match_select], outputs=[matches_table, match_select, match_detail_output])
        delivery_save.click(update_delivery, inputs=[delivery_id, delivery_status, delivery_reason, match_select], outputs=[matches_table, match_select, match_detail_output])

    with gr.Tab("视频与 GPU"):
        gr.Markdown("## 视频任务与 GPU 服务\nGPU 只接收匿名会话和任务引用；地址与 API Key 只在服务端保存，浏览器不能输入控制命令。")
        gpu_output = gr.JSON(label="GPU 状态", value=service.gpu_status(check_health=False))
        with gr.Row():
            gpu_base_url = gr.Textbox(label="GPU API 基础地址", value=configured_gpu_url())
            gpu_api_key = gr.Textbox(label="新的 API Key（留空则保持原值）", type="password")
        with gr.Row():
            gpu_save = gr.Button("保存服务端配置", variant="primary", size="sm")
            gpu_health = gr.Button("健康检查", size="sm")
            gpu_start = gr.Button("启动 GPU", size="sm")
            gpu_stop = gr.Button("关闭 GPU", variant="stop", size="sm")
        compute_instances_table = gr.Dataframe(headers=["实例 ID", "提供商", "外部实例 ID", "显示名称", "状态", "时价", "币种", "最近心跳"], value=[], interactive=False, label="计算实例")
        ledger_table = gr.Dataframe(headers=["业务任务", "状态", "创建时间", "更新时间", "GPU Job", "GPU 地址", "结果目录", "错误"], value=service.list_ledger_tasks(), interactive=False, label="业务任务账本")
        database_jobs_table = gr.Dataframe(headers=["分析任务", "对局", "状态", "类型", "GPU 会话", "计算实例", "创建时间", "完成时间"], value=[], interactive=False, label="业务库分析任务")
        control_task = gr.Dropdown(choices=service.task_choices(), label="已受理远端任务")
        task_detail_output = gr.JSON(label="任务详情", value=service.task_detail(None))
        active_task_output = gr.JSON(label="当前本机任务", value=active_task_controller.snapshot() or {"status": "idle"})
        interrupt_result = gr.JSON(label="取消结果", value={"status": "idle"})
        with gr.Row():
            tasks_refresh = gr.Button("刷新任务", size="sm")
            local_interrupt = gr.Button("中断本机分析", variant="stop", size="sm")
            remote_interrupt = gr.Button("取消远端 GPU Job", variant="stop", size="sm")
        gpu_save.click(save_gpu, inputs=[gpu_base_url, gpu_api_key], outputs=[gpu_output, gpu_api_key])
        gpu_health.click(refresh_gpu_service, outputs=[gpu_output, compute_instances_table], show_progress="hidden")
        gpu_start.click(lambda: service.request_gpu_operation("start"), outputs=[gpu_output])
        gpu_stop.click(lambda: service.request_gpu_operation("stop"), outputs=[gpu_output])
        tasks_refresh.click(refresh_tasks, inputs=[control_task], outputs=[ledger_table, control_task, task_detail_output, database_jobs_table], show_progress="hidden")
        local_interrupt.click(service.request_local_interrupt, outputs=[active_task_output], show_progress="hidden")
        remote_interrupt.click(cancel_control_task, inputs=[control_task], outputs=[interrupt_result, ledger_table, control_task, task_detail_output, database_jobs_table], show_progress="hidden")
        control_task.change(service.task_detail, inputs=[control_task], outputs=[task_detail_output], show_progress="hidden")

    with gr.Tab("权限与审计"):
        gr.Markdown("## 人员、场馆角色与审计\n当前为本机开发运营模式；上线前必须由业务 API 实施登录、角色校验和行级权限，不能直接公开此 Gradio 页面。")
        people_db_status = gr.JSON(label="数据库连接", value=service.database.readiness())
        people_table = gr.Dataframe(headers=["用户 ID", "昵称", "状态", "资料可见范围", "有效场馆角色"], value=[], interactive=False, label="球友与运营人员")
        audit_table = gr.Dataframe(headers=["时间", "操作者类型", "操作", "资源类型", "资源 ID", "摘要"], value=[], interactive=False, label="运营审计")
        with gr.Row():
            refresh_people_button = gr.Button("刷新人员", size="sm")
            audit_refresh = gr.Button("刷新审计", size="sm")
        with gr.Accordion("新增或编辑用户", open=False):
            person_id = gr.Textbox(label="用户 ID（留空则新增）")
            person_nickname = gr.Textbox(label="昵称")
            with gr.Row():
                person_status = gr.Dropdown(choices=sorted(_USER_STATUSES), value="active", label="状态")
                person_visibility = gr.Dropdown(choices=["private", "venue"], value="private", label="资料可见范围")
            person_save = gr.Button("保存用户", variant="primary", size="sm")
            person_notice = gr.Markdown()
        with gr.Accordion("授予场馆角色", open=False):
            membership_venue_id = gr.Textbox(label="场馆 ID")
            membership_user_id = gr.Textbox(label="用户 ID")
            with gr.Row():
                membership_role = gr.Dropdown(choices=sorted(_MEMBERSHIP_ROLES), value="viewer", label="角色")
                membership_status = gr.Dropdown(choices=["active", "revoked"], value="active", label="状态")
            membership_save = gr.Button("保存场馆角色", variant="primary", size="sm")
            membership_notice = gr.Markdown()
        refresh_people_button.click(refresh_people, outputs=[people_table, people_db_status], show_progress="hidden")
        audit_refresh.click(refresh_audit, outputs=[audit_table], show_progress="hidden")
        person_save.click(save_person, inputs=[person_id, person_nickname, person_status, person_visibility], outputs=[people_table, person_notice])
        membership_save.click(save_membership, inputs=[membership_venue_id, membership_user_id, membership_role, membership_status], outputs=[people_table, membership_notice])
