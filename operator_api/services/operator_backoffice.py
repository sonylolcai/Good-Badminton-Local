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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


from runtime_config import RuntimeConfigurationError, business_api_base_url, load_runtime_environment

from operator_api.services.remote_gpu import RemoteAnalysisError, remote_gpu_config
from operator_api.services.task_ledger import BusinessTaskLedger


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
_CHINA_TIMEZONE = ZoneInfo("Asia/Shanghai")
_COURT_WIDTH_M = 6.10
_COURT_LENGTH_M = 13.40


def _format_china_time(value: Any) -> str:
    """Format a stored timestamp for a human-facing operator table.

    PostgreSQL ``timestamptz`` values and service contracts remain UTC.  This
    function is intentionally used only at the Gradio display boundary, so a
    browser or deployment host in another timezone cannot change what a China
    venue operator reads.
    """
    if value is None or value == "":
        return ""
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(_CHINA_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def _as_text(value: Any) -> str | None:
    """Normalize PostgreSQL timestamp values for the JSON operator contract."""
    return None if value is None else str(value)


def _point(value: Any, field: str) -> tuple[float, float]:
    """Read one browser image point without accepting implicit coordinates."""
    if not isinstance(value, dict):
        raise BackofficeError(f"{field} 必须是图像坐标。")
    try:
        x, y = float(value["x"]), float(value["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackofficeError(f"{field} 必须包含有效的 x、y 坐标。") from exc
    if not all(abs(item) < 100000 for item in (x, y)):
        raise BackofficeError(f"{field} 超出可接受图像范围。")
    return x, y


def _line(points: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(points, list) or len(points) != 2:
        raise BackofficeError(f"{field} 必须标注两个可见点。")
    x1, y1 = _point(points[0], f"{field}[0]")
    x2, y2 = _point(points[1], f"{field}[1]")
    a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
    if a * a + b * b < 1e-8:
        raise BackofficeError(f"{field} 的两个点不能重合。")
    return a, b, c


def _intersection(first: tuple[float, float, float], second: tuple[float, float, float], field: str) -> tuple[float, float]:
    a1, b1, c1 = first
    a2, b2, c2 = second
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-8:
        raise BackofficeError(f"{field} 的两条线近平行，无法计算交点。")
    return ((b1 * c2 - b2 * c1) / determinant, (c1 * a2 - c2 * a1) / determinant)


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve the small 8x8 homography system without adding a CV dependency."""
    size = len(values)
    augmented = [list(row) + [values[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            raise BackofficeError("可见场地线不足以确定透视关系。")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [item / divisor for item in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [item - factor * pivot_item for item, pivot_item in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(size)]


def _inverse_3x3(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-10:
        raise BackofficeError("标注的场地线无法形成有效的透视关系。")
    return [
        [(e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant],
        [(f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant],
        [(d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant],
    ]


def _project(matrix: list[list[float]], x: float, y: float) -> list[float]:
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denominator) < 1e-10:
        raise BackofficeError("计算出的虚拟角位于无效透视位置。")
    return [
        round((matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator, 2),
        round((matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator, 2),
    ]


def _validate_corners(corners: list[list[float]]) -> list[list[float]]:
    if len(corners) != 4:
        raise BackofficeError("场地标定必须产生四个角。")
    points = [(float(item[0]), float(item[1])) for item in corners]
    if len({(round(x, 4), round(y, 4)) for x, y in points}) != 4:
        raise BackofficeError("四个场地角必须互不重合。")
    area = sum(points[index][0] * points[(index + 1) % 4][1] - points[(index + 1) % 4][0] * points[index][1] for index in range(4)) / 2
    if abs(area) < 100:
        raise BackofficeError("场地标定面积过小，无法用于空间分析。")
    return [[round(x, 2), round(y, 2)] for x, y in points]


def build_calibration_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """Create, but never persist, a preview-derived court calibration candidate.

    Line evidence supports an occluded near baseline: the user supplies the two
    sidelines plus any two *visible* horizontal court lines with their known
    standard-court distances. Their intersections determine a homography and
    extrapolate the unseen four corners. Two sidelines plus only one horizontal
    line are intentionally rejected because that geometry is underdetermined.
    """
    mode = str(payload.get("mode") or "")
    if mode == "manual_corners":
        corners = _validate_corners([list(_point(item, f"corners[{index}]")) for index, item in enumerate(payload.get("corners") or [])])
        return {"method": mode, "court_corners": corners, "evidence": {"mode": mode, "corners": payload.get("corners")}}
    if mode != "line_evidence":
        raise BackofficeError("标定方式必须是四角标注或可见场地线标注。")

    left = _line(payload.get("left_sideline"), "左边线")
    right = _line(payload.get("right_sideline"), "右边线")
    cross_lines = payload.get("cross_lines")
    if not isinstance(cross_lines, list) or len(cross_lines) != 2:
        raise BackofficeError("请标注两条可见的横向场地线；近端底线不可见时可选择其他已知横线。")
    parsed_cross_lines: list[tuple[float, tuple[float, float, float], dict[str, Any]]] = []
    for index, item in enumerate(cross_lines):
        if not isinstance(item, dict):
            raise BackofficeError("横向场地线格式无效。")
        try:
            court_y = float(item["court_y_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackofficeError("横向场地线必须声明其距远端底线的标准距离。") from exc
        if not 0.0 <= court_y <= _COURT_LENGTH_M:
            raise BackofficeError("横向场地线的标准距离不在场地范围内。")
        parsed_cross_lines.append((court_y, _line(item.get("points"), f"横线[{index}]"), item))
    parsed_cross_lines.sort(key=lambda item: item[0])
    if abs(parsed_cross_lines[0][0] - parsed_cross_lines[1][0]) < 1e-6:
        raise BackofficeError("两条横向场地线必须对应标准场地上的不同位置。")

    first_y, first_line, _ = parsed_cross_lines[0]
    second_y, second_line, _ = parsed_cross_lines[1]
    image_points = [
        _intersection(left, first_line, "左边线与第一横线"),
        _intersection(right, first_line, "右边线与第一横线"),
        _intersection(right, second_line, "右边线与第二横线"),
        _intersection(left, second_line, "左边线与第二横线"),
    ]
    court_points = [(0.0, first_y), (_COURT_WIDTH_M, first_y), (_COURT_WIDTH_M, second_y), (0.0, second_y)]
    equations: list[list[float]] = []
    values: list[float] = []
    for (u, v), (x, y) in zip(image_points, court_points):
        equations.extend([[u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v], [0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v]])
        values.extend([x, y])
    h = _solve_linear_system(equations, values)
    image_to_court = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]
    court_to_image = _inverse_3x3(image_to_court)
    corners = _validate_corners([_project(court_to_image, 0.0, 0.0), _project(court_to_image, _COURT_WIDTH_M, 0.0), _project(court_to_image, _COURT_WIDTH_M, _COURT_LENGTH_M), _project(court_to_image, 0.0, _COURT_LENGTH_M)])
    return {"method": mode, "court_corners": corners, "evidence": {"mode": mode, "left_sideline": payload.get("left_sideline"), "right_sideline": payload.get("right_sideline"), "cross_lines": cross_lines}}


OPERATOR_BACKOFFICE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#component-0 .gradio-container { max-width: 1480px !important; background: #fcfcfc; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.saas-shell { max-width: 1280px; margin: 0 auto; padding: 24px; }
.saas-eyebrow { margin: 0 0 8px; color: #6b7280; font-size: 13px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; }
.saas-heading { margin: 0; color: #111827; font-size: 32px; font-weight: 700; letter-spacing: -0.02em; }
.saas-subtitle { margin: 8px 0 24px; color: #4b5563; font-size: 15px; line-height: 1.5; }
.saas-section { margin: 32px 0 16px; color: #111827; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; }
.saas-note { padding: 16px 20px; border: 1px solid #e5e7eb; border-radius: 12px; background: #f9fafb; color: #374151; font-size: 14px; line-height: 1.5; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }

.ops-summary { display: grid; grid-template-columns: 1.5fr repeat(4, minmax(0, 1fr)); gap: 16px; margin: 24px 0; }
.ops-summary-card { display: flex; flex-direction: column; justify-content: center; min-height: 130px; padding: 20px; border: 1px solid #e5e7eb; border-radius: 16px; background: #ffffff; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03); transition: transform 0.2s ease, box-shadow 0.2s ease; }
.ops-summary-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04); }
.ops-summary-card.primary { color: #ffffff; border: none; background: linear-gradient(135deg, #0f172a 0%, #334155 100%); }
.ops-summary-card small { display: block; color: #6b7280; font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.ops-summary-card.primary small { color: #94a3b8; }
.ops-summary-card strong { display: block; margin-top: auto; color: #111827; font-size: 32px; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; }
.ops-summary-card.primary strong { color: #f8fafc; }
.ops-summary-card span { display: block; margin-top: 8px; color: #9ca3af; font-size: 13px; font-weight: 400; }
.ops-summary-card.primary span { color: #cbd5e1; }

.ops-service-strip { display: flex; flex-wrap: wrap; gap: 12px; padding: 16px 20px; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); align-items: center; }
.ops-service { display: inline-flex; align-items: center; padding: 6px 12px; border-radius: 9999px; background: #dcfce7; color: #166534; font-size: 13px; font-weight: 500; }
.ops-service::before { content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #16a34a; margin-right: 8px; }
.ops-service.warn { background: #fef9c3; color: #854d0e; }
.ops-service.warn::before { background-color: #eab308; }
.ops-service.neutral { background: #f3f4f6; color: #4b5563; }
.ops-service.neutral::before { background-color: #9ca3af; }

.ops-action-card { display: flex; flex-direction: column; height: 100%; padding: 0; border: 1px solid #e5e7eb; border-radius: 16px; background: #ffffff; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); overflow: hidden; }
.ops-action-card h3 { margin: 0; padding: 20px 20px 8px; color: #111827; font-size: 18px; font-weight: 600; border-bottom: 1px solid transparent; }
.ops-action-card p { margin: 0; padding: 0 20px 16px; color: #6b7280; font-size: 14px; line-height: 1.5; border-bottom: 1px solid #f3f4f6; }
.ops-action-card .wrap { padding: 20px; flex-grow: 1; display: flex; flex-direction: column; gap: 16px; background: #fafafa; }

.venue-hero { padding: 40px; border-radius: 24px; color: #ffffff; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); box-shadow: 0 20px 25px -5px rgba(37, 99, 235, 0.2), 0 10px 10px -5px rgba(37, 99, 235, 0.1); }
.venue-hero h2 { margin: 0; font-size: 36px; font-weight: 700; letter-spacing: -0.02em; }
.venue-hero p { margin: 12px 0 0; color: #bfdbfe; font-size: 16px; }

.venue-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 24px 0; }
.venue-metric { min-height: 120px; padding: 20px; border: 1px solid #e5e7eb; border-radius: 16px; background: #ffffff; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05); }
.venue-metric small { display: block; color: #6b7280; font-size: 14px; font-weight: 500; margin-bottom: 8px; }
.venue-metric strong { display: block; margin-top: auto; color: #111827; font-size: 32px; font-weight: 700; line-height: 1; }
.venue-metric em { display: inline-block; margin-top: 8px; padding: 2px 8px; border-radius: 9999px; background: #dcfce7; color: #166534; font-style: normal; font-size: 12px; font-weight: 500; }

.venue-status { display: flex; flex-wrap: wrap; gap: 12px; padding: 16px; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff; align-items: center; }
.venue-pill { padding: 6px 12px; border-radius: 9999px; background: #dcfce7; color: #166534; font-size: 14px; font-weight: 500; }
.venue-pill.warn { background: #fef9c3; color: #854d0e; }
.venue-plain-status { padding: 8px 16px; border-radius: 8px; background: #f3f4f6; color: #374151; font-size: 14px; font-weight: 500; }

.venue-ops .tab-nav { gap: 8px; padding: 12px 0; border-bottom: 1px solid #e5e7eb; }
.venue-ops .tab-nav button { padding: 10px 16px; border-radius: 8px 8px 0 0; font-weight: 500; color: #6b7280; border: 1px solid transparent; border-bottom: none; transition: all 0.2s ease; }
.venue-ops .tab-nav button:hover { color: #374151; background: #f9fafb; }
.venue-ops .tab-nav button.selected { color: #2563eb; background: #ffffff; border-color: #e5e7eb; border-bottom-color: #ffffff; margin-bottom: -1px; }
.venue-ops .block-title { margin: 24px 0 12px; color: #111827; font-weight: 600; }

/* Gradio specific overrides for a cleaner look */
.gradio-container button.primary { background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; border: none !important; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important; color: white !important; font-weight: 600 !important; border-radius: 8px !important; transition: all 0.2s ease !important; }
.gradio-container button.primary:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3) !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select { border-radius: 8px !important; border-color: #d1d5db !important; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important; transition: border-color 0.2s ease, box-shadow 0.2s ease !important; }
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important; }

@media (max-width: 1024px) { .ops-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); } .ops-summary-card.primary { grid-column: span 2; } }
@media (max-width: 768px) { .venue-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } .venue-hero { padding: 32px 24px; } .ops-summary { grid-template-columns: 1fr; } .ops-summary-card.primary { grid-column: span 1; } }
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
    return f"<section class='saas-shell'><section class='venue-hero'><h2>羽球空间 · 运营看板</h2><p>一眼掌握场馆入口、对局、活动与 AI 服务状态。</p></section><section class='venue-metrics'>{card_html}</section><section class='venue-status'>{pill_html}</section></section>"


def _gpu_markup(status: dict[str, Any]) -> str:
    state = str(status.get("status") or "not_checked")
    label = "GPU 服务在线" if state == "ready" else "GPU 服务待检查" if state == "not_checked" else "GPU 服务不可用"
    hint = str(status.get("control_hint") or status.get("message") or "健康检查、任务取消与受控启停均由服务端执行。")
    return f"<section class='venue-status'><span class='venue-pill{' warn' if state not in {'ready', 'not_checked'} else ''}'>{escape(label)}</span><span class='venue-plain-status'>{escape(hint)}</span></section>"


def _single_venue_markup(snapshot: dict[str, Any], readiness: dict[str, Any], gpu: dict[str, Any]) -> str:
    """Render a human-facing control summary without exposing raw diagnostics."""
    venue_name = str(snapshot.get("venue_name") or "尚未选择场馆")
    venue_code = str(snapshot.get("venue_code") or "—")
    total_courts = int(snapshot.get("total_courts") or 0)
    active_courts = int(snapshot.get("active_courts") or 0)
    terminal_online = int(snapshot.get("terminal_online") or 0)
    camera_online = int(snapshot.get("camera_online") or 0)
    analysis_active = int(snapshot.get("analysis_active") or 0)
    database_ready = readiness.get("status") == "ready"
    gpu_state = str(gpu.get("status") or "misconfigured")
    cards = [
        ("当前场馆", venue_name, f"场馆编码 · {venue_code}", "primary"),
        ("可用场地", f"{active_courts} / {total_courts}", "维护中的场地不会接入新会话", ""),
        ("终端 / 摄像头", f"{terminal_online} / {camera_online}", "在线数量；首次部署前显示为 0", ""),
        ("正在解析", str(analysis_active), "已由终端发起且业务服务已受理", ""),
        ("GPU 服务", "在线" if gpu_state == "ready" else "待检查" if gpu_state == "not_checked" else "未就绪", "由业务服务器受控访问", ""),
    ]
    card_html = "".join(
        f"<div class='ops-summary-card {style}'><small>{escape(label)}</small><strong>{escape(value)}</strong><span>{escape(hint)}</span></div>"
        for label, value, hint, style in cards
    )
    services = [
        ("业务数据库已连接" if database_ready else "业务数据库未连接", not database_ready),
        ("GPU 可用" if gpu_state == "ready" else "GPU 尚未确认", gpu_state not in {"ready", "not_checked"}),
        ("解析会话运行中" if analysis_active else "暂无解析会话", False),
    ]
    service_html = "".join(
        f"<span class='ops-service{' warn' if warning else ' neutral' if label == '暂无解析会话' else ''}'>{escape(label)}</span>"
        for label, warning in services
    )
    return (
        "<section class='saas-shell'><p class='saas-eyebrow'>VENUE OPERATIONS</p>"
        "<h2 class='saas-heading'>单馆运营总控</h2>"
        "<p class='saas-subtitle'>用一处后台完成场地可用性、设备接入、AI 解析与 GPU 服务管理。</p>"
        f"<section class='ops-summary'>{card_html}</section><section class='ops-service-strip'>{service_html}</section></section>"
    )


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

    def readiness(self) -> dict[str, Any]:
        if not self.database_url:
            return {"status": "not_configured", "message": "未配置 GOOD_BADMINTON_BUSINESS_DATABASE_URL；管理页不会写入影子数据。"}
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("select current_database() as database, now() as checked_at")
                row = cursor.fetchone()
            return {"status": "ready", "database": row["database"], "checked_at": str(row["checked_at"])}
        except BackofficeError as exc:
            return {"status": "unavailable", "message": str(exc)}

    def list_venues(self) -> list[list[str]]:
        return self._rows(
            """select v.id::text, v.tenant_id::text, v.code, v.name, v.timezone, coalesce(v.address, ''), v.status, count(c.id)::int
               from business.venues v left join business.courts c on c.venue_id = v.id
               group by v.id order by v.created_at desc"""
        )

    def list_tenants(self) -> list[list[str]]:
        """Return tenant choices for venue registration; no user identity data."""
        return self._rows(
            "select id::text, name, status from business.tenants order by name, id"
        )

    def register_venue(self, *, tenant_id: str | None, tenant_name: str | None,
                       venue_code: str, venue_name: str, timezone_name: str,
                       address: str | None, courts: list[dict[str, Any]]) -> dict[str, Any]:
        """Create a complete tenant/venue/court registration in one transaction.

        A venue cannot be created without an owning tenant and at least one
        court.  This method is intentionally separate from the legacy granular
        ``save_*`` methods so the public operator API has one predictable setup
        workflow and no partially registered venue if a court insert fails.
        """
        safe_tenant_id = str(tenant_id or "").strip()
        safe_tenant_name = str(tenant_name or "").strip()
        safe_venue_code = str(venue_code or "").strip()
        safe_venue_name = str(venue_name or "").strip()
        safe_timezone = str(timezone_name or "").strip() or "Asia/Shanghai"
        if bool(safe_tenant_id) == bool(safe_tenant_name):
            raise BackofficeError("请选择已有租户，或填写一个新的租户名称（二者只能选其一）。")
        if not safe_venue_code or not safe_venue_name:
            raise BackofficeError("场馆编码和场馆名称均为必填项。")
        if not courts:
            raise BackofficeError("注册场馆时至少需要创建一个场地。")

        normalized_courts: list[tuple[str, str, int, str]] = []
        seen_codes: set[str] = set()
        for position, court in enumerate(courts):
            code = str(court.get("code") or "").strip()
            name = str(court.get("name") or "").strip()
            status = str(court.get("status") or "active").strip()
            try:
                sort_order = max(0, int(court.get("sort_order", position)))
            except (TypeError, ValueError) as exc:
                raise BackofficeError("场地排序必须是非负整数。") from exc
            if not code or not name or status not in _COURT_STATUSES:
                raise BackofficeError("每个场地都必须有编码、名称和有效状态。")
            normalized_code = code.casefold()
            if normalized_code in seen_codes:
                raise BackofficeError("同一场馆内的场地编码不能重复。")
            seen_codes.add(normalized_code)
            normalized_courts.append((code, name, sort_order, status))

        with self._connect() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    if safe_tenant_id:
                        cursor.execute(
                            "select id::text, name from business.tenants where id=%s and status='active'",
                            (safe_tenant_id,),
                        )
                        tenant = cursor.fetchone()
                        if not tenant:
                            raise BackofficeError("未找到可用租户。")
                        saved_tenant_id = str(tenant["id"])
                        saved_tenant_name = str(tenant["name"])
                        tenant_created = False
                    else:
                        saved_tenant_id = str(uuid.uuid4())
                        saved_tenant_name = safe_tenant_name
                        cursor.execute(
                            "insert into business.tenants (id, name, status) values (%s, %s, 'active')",
                            (saved_tenant_id, saved_tenant_name),
                        )
                        tenant_created = True

                    saved_venue_id = str(uuid.uuid4())
                    cursor.execute(
                        """insert into business.venues (id, tenant_id, code, name, timezone, address, status)
                           values (%s, %s, %s, %s, %s, %s, 'active')""",
                        (saved_venue_id, saved_tenant_id, safe_venue_code, safe_venue_name, safe_timezone, str(address or "").strip() or None),
                    )
                    saved_courts: list[dict[str, Any]] = []
                    for code, name, sort_order, status in normalized_courts:
                        court_id = str(uuid.uuid4())
                        cursor.execute(
                            """insert into business.courts (id, venue_id, code, name, sort_order, status)
                               values (%s, %s, %s, %s, %s, %s)""",
                            (court_id, saved_venue_id, code, name, sort_order, status),
                        )
                        saved_courts.append({"id": court_id, "code": code, "name": name, "sort_order": sort_order, "status": status})
                    cursor.execute(
                        """insert into business.audit_events
                           (id, tenant_id, venue_id, actor_type, action, resource_type, resource_id, after_summary)
                           values (%s, %s, %s, 'system', 'venue.registration_created', 'venue', %s, %s::jsonb)""",
                        (
                            str(uuid.uuid4()), saved_tenant_id, saved_venue_id, saved_venue_id,
                            json.dumps({"tenant_created": tenant_created, "venue_code": safe_venue_code, "court_count": len(saved_courts)}, ensure_ascii=False),
                        ),
                    )
        return {
            "tenant": {"id": saved_tenant_id, "name": saved_tenant_name, "created": tenant_created},
            "venue": {"id": saved_venue_id, "tenant_id": saved_tenant_id, "code": safe_venue_code, "name": safe_venue_name, "timezone": safe_timezone, "address": str(address or "").strip() or None, "status": "active"},
            "courts": saved_courts,
        }

    def venue_options(self) -> list[tuple[str, str]]:
        """Return human-readable venue choices for the operations console."""
        rows = self._rows("select id::text, name, code from business.venues order by name, code")
        return [(f"{row[1]} · {row[2]}", row[0]) for row in rows]

    def venue_control_snapshot(self, venue_id: str | None) -> dict[str, Any]:
        """Return the one-venue state needed by the operator's first screen.

        The query deliberately selects the most recent terminal, camera,
        calibration and ingest session per court.  The operations page therefore
        reports a usable decision state instead of asking an operator to infer it
        from multiple UUID-heavy tables.
        """
        if not venue_id:
            return {"rows": [], "court_options": [], "total_courts": 0, "active_courts": 0,
                    "terminal_online": 0, "camera_online": 0, "analysis_active": 0}
        rows = self._rows(
            """select v.name as venue_name, v.code as venue_code, c.id::text as court_id,
                      c.name as court_name, c.code as court_code, c.status as court_status,
                      coalesce(d.id::text, '') as device_id, coalesce(d.device_code, '') as device_code,
                      coalesce(d.status, '未绑定') as device_status,
                      coalesce(d.last_heartbeat_at::text, '') as device_heartbeat, coalesce(cam.id::text, '') as camera_id,
                      coalesce(cam.camera_code, '') as camera_code, coalesce(cam.status, '未绑定') as camera_status,
                      coalesce(cam.last_heartbeat_at::text, '') as camera_heartbeat,
                      coalesce(cal.quality_status, '未标定') as calibration_status,
                      coalesce(session.status, '未开始') as session_status
               from business.venues v
               join business.courts c on c.venue_id = v.id
               left join lateral (
                   select * from business.edge_devices
                   where court_id = c.id order by updated_at desc limit 1
               ) d on true
               left join lateral (
                   select * from business.cameras
                   where court_id = c.id and (d.id is null or edge_device_id = d.id)
                   order by updated_at desc limit 1
               ) cam on true
               left join lateral (
                   select quality_status from business.camera_calibrations
                   where camera_id = cam.id order by version desc limit 1
               ) cal on true
               left join lateral (
                   select status from business.edge_ingest_sessions
                   where camera_id = cam.id order by updated_at desc limit 1
               ) session on true
               where v.id = %s
               order by c.sort_order, c.name""",
            (venue_id,),
        )
        control_rows: list[list[str]] = []
        court_options: list[tuple[str, str]] = []
        for row in rows:
            court_id, court_name, court_code, court_status = row[2:6]
            device_status, camera_status, calibration_status, analysis_status = row[8], row[12], row[14], row[15]
            heartbeat = _format_china_time(row[13] or row[9]) or "尚未收到心跳"
            if court_status != "active":
                action = "场地不可用：先恢复为可用状态"
            elif device_status != "online" or camera_status != "active":
                action = "待接入：绑定终端和摄像头后等待心跳"
            elif calibration_status != "validated":
                action = "待标定：完成标定校验后才能开始解析"
            elif analysis_status in {"receiving", "relaying", "processing"}:
                action = "解析进行中：关注 GPU 转发和结果回执"
            else:
                action = "已就绪：由终端发起受签名的解析会话"
            control_rows.append([court_name, court_code, court_status, device_status, camera_status, calibration_status, analysis_status, heartbeat, action])
            court_options.append((f"{court_name} · {court_code} · {court_status}", court_id))
        return {
            "venue_name": rows[0][0] if rows else "未找到场馆",
            "venue_code": rows[0][1] if rows else "—",
            "rows": control_rows,
            "court_options": court_options,
            "total_courts": len(rows),
            "active_courts": sum(row[5] == "active" for row in rows),
            "terminal_online": sum(row[8] == "online" for row in rows),
            "camera_online": sum(row[12] == "active" for row in rows),
            "analysis_active": sum(row[15] in {"receiving", "relaying", "processing"} for row in rows),
        }

    def venue_live_operations(self, venue_id: str) -> dict[str, Any]:
        """Return one named operational record for every court in a venue.

        The UI must not infer "connected" from a historic ``online`` string.
        A camera is connected only when both terminal and camera heartbeats are
        fresh.  The case ID is the business-owned edge ingest session ID.
        """
        rows = self._dict_rows(
            """select c.id::text as court_id, c.code as court_code, c.name as court_name, c.status as court_status,
                      coalesce(d.id::text, '') as device_id, coalesce(d.device_code, '') as device_code,
                      coalesce(d.status, 'unbound') as device_status, d.last_heartbeat_at as device_heartbeat_at,
                      coalesce(cam.id::text, '') as camera_id, coalesce(cam.camera_code, '') as camera_code,
                      coalesce(cam.status, 'unbound') as camera_status, cam.last_heartbeat_at as camera_heartbeat_at,
                      coalesce(cal.quality_status, 'unconfigured') as calibration_status,
                      s.id::text as case_id, s.status as case_status, s.gpu_analysis_session_id,
                      s.gpu_status, s.gpu_forwarding_enabled, s.preview_available, s.last_preview_at,
                      coalesce(s.received_segment_count, 0) as received_segment_count,
                      coalesce(s.forwarded_segment_count, 0) as forwarded_segment_count,
                      s.last_received_at, s.last_forwarded_at, s.error_code, s.error_message,
                      coalesce(control.desired_mode, 'idle') as capture_mode,
                      coalesce(control.revision, 0) as capture_revision,
                      control.updated_at as capture_updated_at,
                      case when d.status='online' and cam.status='active'
                                and d.last_heartbeat_at >= now() - interval '90 seconds'
                                and cam.last_heartbeat_at >= now() - interval '90 seconds'
                           then true else false end as camera_connected
               from business.courts c
               left join lateral (
                   select * from business.edge_devices where court_id=c.id
                   order by updated_at desc limit 1
               ) d on true
               left join lateral (
                   select * from business.cameras where court_id=c.id
                     and (d.id is null or edge_device_id=d.id)
                   order by updated_at desc limit 1
               ) cam on true
               left join lateral (
                   select quality_status from business.camera_calibrations
                   where camera_id=cam.id order by version desc limit 1
               ) cal on true
               left join lateral (
                   select * from business.edge_ingest_sessions where camera_id=cam.id
                   order by updated_at desc limit 1
               ) s on true
               left join business.edge_capture_controls control on control.court_id=c.id
               where c.venue_id=%s order by c.sort_order, c.name""",
            (venue_id,),
        )
        courts: list[dict[str, Any]] = []
        for row in rows:
            case_id = str(row.get("case_id") or "")
            connected = bool(row.get("camera_connected"))
            courts.append({
                "court": {"id": row["court_id"], "code": row["court_code"], "name": row["court_name"], "status": row["court_status"]},
                "camera": {
                    "connected": connected, "device_id": row["device_id"] or None, "device_code": row["device_code"] or None,
                    "device_status": row["device_status"], "device_heartbeat_at": _as_text(row.get("device_heartbeat_at")),
                    "camera_id": row["camera_id"] or None, "camera_code": row["camera_code"] or None,
                    "camera_status": row["camera_status"], "camera_heartbeat_at": _as_text(row.get("camera_heartbeat_at")),
                    "calibration_status": row["calibration_status"],
                },
                "capture": {
                    "mode": row["capture_mode"], "revision": int(row["capture_revision"] or 0),
                    "updated_at": _as_text(row.get("capture_updated_at")),
                },
                "case": None if not case_id else {
                    "id": case_id, "status": row["case_status"], "gpu_analysis_session_id": row.get("gpu_analysis_session_id") or None,
                    "gpu_status": row.get("gpu_status") or None, "gpu_forwarding_enabled": bool(row.get("gpu_forwarding_enabled")),
                    "preview_available": bool(row.get("preview_available")), "last_preview_at": _as_text(row.get("last_preview_at")),
                    "received_segment_count": int(row.get("received_segment_count") or 0),
                    "forwarded_segment_count": int(row.get("forwarded_segment_count") or 0),
                    "last_received_at": _as_text(row.get("last_received_at")), "last_forwarded_at": _as_text(row.get("last_forwarded_at")),
                    "error": None if not row.get("error_code") else {"code": row["error_code"], "message": row.get("error_message") or ""},
                },
            })
        return {"courts": courts, "camera_connected": sum(1 for court in courts if court["camera"]["connected"]),
                "active_cases": sum(1 for court in courts if court["case"] and court["case"]["status"] in {"requested", "receiving", "relaying", "processing"})}

    def calibration_candidate(self, venue_id: str, court_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Calculate a candidate only for a court with a live business preview."""
        rows = self._dict_rows(
            """select cam.id::text as camera_id
               from business.courts c
               join business.cameras cam on cam.court_id=c.id
               join business.edge_ingest_sessions s on s.camera_id=cam.id
               where c.id=%s and c.venue_id=%s and s.status in ('requested', 'receiving')
                 and s.preview_available=true
               order by s.updated_at desc limit 1""",
            (court_id, venue_id),
        )
        if not rows:
            raise BackofficeError("请先开启预览并等待业务服务器收到可播放的视频片段，再标注场地。")
        candidate = build_calibration_candidate(payload)
        return {"camera_id": rows[0]["camera_id"], **candidate}

    def save_camera_calibration(self, venue_id: str, court_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist only an explicitly confirmed, preview-derived calibration.

        The validated record is immutable evidence for the next record session.
        It never mutates the active preview session, which deliberately has no
        calibration ID; the operator must stop preview and start recording.
        """
        candidate = self.calibration_candidate(venue_id, court_id, payload)
        saved_calibration_id = str(uuid.uuid4())
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """select coalesce(max(version), 0) + 1 as next_version
                   from business.camera_calibrations where camera_id=%s""",
                (candidate["camera_id"],),
            )
            version = int(cursor.fetchone()["next_version"])
            cursor.execute(
                """insert into business.camera_calibrations
                   (id, camera_id, version, court_corners, quality_status, evidence)
                   values (%s, %s, %s, %s::jsonb, 'validated', %s::jsonb)""",
                (saved_calibration_id, candidate["camera_id"], version,
                 json.dumps(candidate["court_corners"]), json.dumps(candidate["evidence"])),
            )
        result = {"id": saved_calibration_id, "camera_id": candidate["camera_id"], "version": version,
                  "quality_status": "validated", "court_corners": candidate["court_corners"], "method": candidate["method"]}
        self._audit("camera.calibration_validated", "camera_calibration", saved_calibration_id,
                    {"venue_ref": venue_id, "court_ref": court_id, "version": version, "method": candidate["method"]})
        return result

    def set_court_capture_mode(self, venue_id: str, court_id: str, mode: str) -> dict[str, Any]:
        """Set the desired state which the unattended venue Mac polls.

        This does not contact the Mac directly.  A signed heartbeat picks up
        the new revision within its normal polling interval, so no public
        inbound path to a venue network is ever required.
        """

        if mode not in {"idle", "preview", "record"}:
            raise BackofficeError("采集模式无效。")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """select c.id::text from business.courts c
                   join business.venues v on v.id=c.venue_id
                   where c.id=%s and c.venue_id=%s and c.status='active' and v.status='active'""",
                (court_id, venue_id),
            )
            if not cursor.fetchone():
                raise BackofficeError("场地不存在、未启用，或不属于指定场馆。")
            cursor.execute(
                """insert into business.edge_capture_controls (court_id, desired_mode, revision, updated_at)
                   values (%s, %s, 1, now())
                   on conflict (court_id) do update
                     set desired_mode=excluded.desired_mode,
                         revision=business.edge_capture_controls.revision + 1,
                         updated_at=now()
                   returning desired_mode, revision, updated_at""",
                (court_id, mode),
            )
            row = cursor.fetchone()
        result = {"court_id": court_id, "mode": str(row["desired_mode"]),
                  "revision": int(row["revision"]), "updated_at": _as_text(row["updated_at"])}
        self._audit("court.capture_mode_updated", "court", court_id, result)
        return result

    def case_for_court(self, venue_id: str, court_id: str) -> dict[str, Any] | None:
        rows = self._dict_rows(
            """select s.id::text as case_id, s.court_id::text as court_id, s.status, s.gpu_analysis_session_id,
                      s.gpu_status, s.gpu_forwarding_enabled, s.preview_available, s.last_preview_at,
                      s.received_segment_count, s.forwarded_segment_count, s.last_received_at, s.last_forwarded_at,
                      s.error_code, s.error_message
               from business.edge_ingest_sessions s join business.courts c on c.id=s.court_id
               where s.court_id=%s and c.venue_id=%s order by s.updated_at desc limit 1""",
            (court_id, venue_id),
        )
        return rows[0] if rows else None

    def set_case_gpu_forwarding(self, venue_id: str, court_id: str, enabled: bool) -> dict[str, Any]:
        """Enable/disable delivery of *future* video segments for the live case."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """update business.edge_ingest_sessions s set gpu_forwarding_enabled=%s, updated_at=now()
                   from business.courts c
                   where s.court_id=c.id and c.venue_id=%s and s.court_id=%s
                     and s.status in ('requested', 'receiving', 'relaying', 'processing')
                     and s.calibration_id is not null
                     and s.id=(select id from business.edge_ingest_sessions
                               where court_id=%s order by updated_at desc limit 1)
                   returning s.id::text as case_id, s.status, s.gpu_analysis_session_id, s.gpu_status,
                             s.gpu_forwarding_enabled, s.preview_available, s.received_segment_count,
                             s.forwarded_segment_count, s.last_received_at, s.last_forwarded_at,
                             s.error_code, s.error_message""",
                (enabled, venue_id, court_id, court_id),
            )
            row = cursor.fetchone()
        if not row:
            pending_calibration = self._dict_rows(
                """select 1 from business.edge_ingest_sessions s join business.courts c on c.id=s.court_id
                   where s.court_id=%s and c.venue_id=%s
                     and s.status in ('requested', 'receiving', 'relaying', 'processing')
                     and s.calibration_id is null limit 1""",
                (court_id, venue_id),
            )
            if pending_calibration:
                raise BackofficeError("当前是未标定的预览会话。请先根据预览保存并验证四角，再停止预览并重新开始采集。")
            raise BackofficeError("该场地没有可控制的实时 case；请先等待终端创建视频会话。")
        return dict(row)

    def case_event_log(self, case_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self._dict_rows(
            """select id::text, source, event_id, level, event_type, coalesce(message, '') as message,
                      payload, occurred_at, created_at
               from business.edge_ingest_event_logs where edge_ingest_session_id=%s
               order by occurred_at desc, id desc limit %s""",
            (case_id, max(1, min(int(limit), 500))),
        )

    def set_court_status(self, venue_id: str, court_id: str, status: str) -> None:
        if status not in _COURT_STATUSES or not venue_id or not court_id:
            raise BackofficeError("请选择场地，并指定有效的场地状态。")
        if not self._execute("update business.courts set status=%s where id=%s and venue_id=%s", (status, court_id, venue_id)):
            raise BackofficeError("未找到指定场地，或该场地不属于当前场馆。")
        self._audit("court.status_updated", "court", court_id, {"venue_ref": venue_id, "status": status})

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

    def list_venue_court_overview(self) -> list[list[str]]:
        """Return the business-facing venue/court management table.

        The old pair of raw tables made operators compare UUIDs manually.  One
        row now represents one court in one venue, with the operational status
        needed to decide whether that court can be managed or prepared for a
        camera deployment.  Technical IDs stay available in edit forms/API,
        rather than dominating the first screen.
        """
        return self._rows(
            """select v.name as venue_name, v.code as venue_code, v.status as venue_status,
                      c.name as court_name, c.code as court_code, c.status as court_status,
                      coalesce(d.status, '未绑定') as edge_device_status,
                      coalesce(cam.status, '未绑定') as camera_status,
                      coalesce(s.status, '未开始') as analysis_status
               from business.venues v join business.courts c on c.venue_id=v.id
               left join business.edge_devices d on d.court_id=c.id
               left join business.cameras cam on cam.edge_device_id=d.id and cam.court_id=c.id
               left join lateral (
                   select status from business.edge_ingest_sessions
                   where camera_id=cam.id order by updated_at desc limit 1
               ) s on true
               order by v.name, c.sort_order, c.name, d.device_code, cam.camera_code"""
        )

    def list_venue_court_runtime(self) -> list[list[str]]:
        """One global operations table, grouped by venue instead of separate pages."""
        return self._rows(
            """select v.name as venue_name, v.code as venue_code, c.name as court_name, c.code as court_code,
                      coalesce(d.id::text, ''), coalesce(d.device_code, ''), coalesce(d.status, 'unbound'),
                      coalesce(d.last_heartbeat_at::text, ''), coalesce(cam.id::text, ''), coalesce(cam.camera_code, ''),
                      coalesce(cam.status, 'unbound'), coalesce(cam.last_heartbeat_at::text, ''),
                      coalesce(s.status, 'idle'), coalesce(s.received_segment_count::text, '0'),
                      coalesce(s.forwarded_segment_count::text, '0'), coalesce(s.gpu_analysis_session_id, ''),
                      coalesce(s.gpu_status, ''), coalesce(s.error_message, ''), coalesce(s.updated_at::text, '')
               from business.venues v join business.courts c on c.venue_id=v.id
               left join business.edge_devices d on d.court_id=c.id
               left join business.cameras cam on cam.edge_device_id=d.id and cam.court_id=c.id
               left join lateral (
                   select * from business.edge_ingest_sessions
                   where camera_id=cam.id order by updated_at desc limit 1
               ) s on true
               order by v.name, c.sort_order, c.name, d.device_code, cam.camera_code"""
        )

    def provision_edge_camera(
        self, device_id: str, camera_id: str, venue_id: str, court_id: str,
        device_code: str, camera_code: str, credential_version: str,
    ) -> dict[str, str]:
        """Create one terminal binding and reveal its derived secret once.

        RTSP credentials stay on the terminal.  ``cameras`` keeps an opaque
        ``edge-managed`` marker solely because the existing schema requires a
        stream reference; the marker contains no RTSP endpoint or secret.
        """
        required = {"场馆 ID": venue_id, "场地 ID": court_id, "设备编码": device_code, "摄像头编码": camera_code, "凭据版本": credential_version}
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise BackofficeError("、".join(missing) + " 为必填项。")
        master_key = os.environ.get("GOOD_BADMINTON_EDGE_MASTER_KEY", "").strip()
        try:
            from business_gateway.edge_contract import derive_device_secret
            derive_device_secret(master_key, "validation", credential_version.strip())
        except (ImportError, ValueError) as exc:
            raise BackofficeError("未配置足够强度的 GOOD_BADMINTON_EDGE_MASTER_KEY；不能生成终端配对凭据。") from exc
        saved_device_id = device_id.strip() or str(uuid.uuid4())
        saved_camera_id = camera_id.strip() or str(uuid.uuid4())
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("select id from business.courts where id=%s and venue_id=%s", (court_id.strip(), venue_id.strip()))
            if not cursor.fetchone():
                raise BackofficeError("场地不存在，或不属于指定场馆。")
            cursor.execute("select id from business.edge_devices where id=%s", (saved_device_id,))
            if cursor.fetchone():
                cursor.execute("update business.edge_devices set venue_id=%s, court_id=%s, device_code=%s, credential_version=%s, status='offline' where id=%s", (venue_id.strip(), court_id.strip(), device_code.strip(), credential_version.strip(), saved_device_id))
            else:
                cursor.execute("insert into business.edge_devices (id, venue_id, court_id, device_code, credential_version) values (%s, %s, %s, %s, %s)", (saved_device_id, venue_id.strip(), court_id.strip(), device_code.strip(), credential_version.strip()))
            marker = f"edge-managed:{saved_device_id}:{saved_camera_id}".encode("utf-8")
            cursor.execute("select id from business.cameras where id=%s", (saved_camera_id,))
            if cursor.fetchone():
                cursor.execute("update business.cameras set court_id=%s, edge_device_id=%s, camera_code=%s, stream_ref_ciphertext=%s, status='offline' where id=%s", (court_id.strip(), saved_device_id, camera_code.strip(), marker, saved_camera_id))
            else:
                cursor.execute("insert into business.cameras (id, court_id, edge_device_id, camera_code, stream_ref_ciphertext) values (%s, %s, %s, %s, %s)", (saved_camera_id, court_id.strip(), saved_device_id, camera_code.strip(), marker))
        self._audit("edge_camera.provisioned", "edge_device", saved_device_id, {"venue_ref": venue_id.strip(), "court_ref": court_id.strip(), "camera_ref": saved_camera_id, "credential_version": credential_version.strip()})
        return {"schema_version": "edge-ingest.v1", "device_id": saved_device_id, "camera_id": saved_camera_id, "credential_version": credential_version.strip(), "device_secret": derive_device_secret(master_key, saved_device_id, credential_version.strip()), "message": "请立即将此凭据写入终端；后台不会在审计或列表中保存/显示该密钥。"}

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

    def list_players(self, venue_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE vm.venue_id = %s" if venue_id else ""
        params = (venue_id,) if venue_id else ()
        return self._dict_rows(
            "SELECT u.id::text, coalesce(u.nickname, '') AS nickname, u.status, "
            "u.profile_visibility, vm.venue_id::text, v.name AS venue_name "
            "FROM business.users u JOIN business.venue_memberships vm ON vm.user_id=u.id "
            "AND vm.status='active' JOIN business.venues v ON v.id=vm.venue_id "
            f"{where} ORDER BY u.created_at DESC",
            params,
        )

    def create_player(self, venue_id: str, nickname: str, actor_admin_id: str) -> dict[str, str]:
        if not venue_id.strip() or not nickname.strip():
            raise BackofficeError("场馆和球员名称均为必填项。")
        player_id = str(uuid.uuid4())
        with self._connect() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.execute("SELECT name FROM business.venues WHERE id=%s AND status='active'", (venue_id,))
            venue = cursor.fetchone()
            if venue is None:
                raise BackofficeError("未找到可用场馆。")
            cursor.execute(
                "INSERT INTO business.users (id, nickname, status, profile_visibility) "
                "VALUES (%s, %s, 'active', 'venue')",
                (player_id, nickname.strip()),
            )
            cursor.execute(
                "INSERT INTO business.venue_memberships (venue_id, user_id, role, status) "
                "VALUES (%s, %s, 'viewer', 'active')",
                (venue_id, player_id),
            )
            cursor.execute(
                "INSERT INTO business.audit_events "
                "(id, actor_type, actor_admin_account_id, action, resource_type, resource_id, venue_id, after_summary) "
                "VALUES (%s, 'admin', %s, 'player.created', 'user', %s, %s, %s::jsonb)",
                (str(uuid.uuid4()), actor_admin_id, player_id, venue_id, json.dumps({"nickname": nickname.strip()}, ensure_ascii=False)),
            )
        return {"id": player_id, "nickname": nickname.strip(), "status": "active", "venue_id": venue_id}

    def update_player(self, venue_id: str, player_id: str, nickname: str, status: str, actor_admin_id: str) -> dict[str, str]:
        if status not in _USER_STATUSES or not nickname.strip():
            raise BackofficeError("球员名称或状态无效。")
        with self._connect() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.execute(
                "UPDATE business.users u SET nickname=%s, status=%s, updated_at=now() "
                "FROM business.venue_memberships vm WHERE u.id=%s AND vm.user_id=u.id "
                "AND vm.venue_id=%s AND vm.status='active'",
                (nickname.strip(), status, player_id, venue_id),
            )
            if cursor.rowcount != 1:
                raise BackofficeError("未找到该场馆中的球员。")
            cursor.execute(
                "INSERT INTO business.audit_events "
                "(id, actor_type, actor_admin_account_id, action, resource_type, resource_id, venue_id, after_summary) "
                "VALUES (%s, 'admin', %s, 'player.updated', 'user', %s, %s, %s::jsonb)",
                (str(uuid.uuid4()), actor_admin_id, player_id, venue_id, json.dumps({"nickname": nickname.strip(), "status": status}, ensure_ascii=False)),
            )
        return {"id": player_id, "nickname": nickname.strip(), "status": status, "venue_id": venue_id}

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

    def register_media_asset(
        self,
        *,
        asset_id: str,
        tenant_id: str,
        venue_id: str,
        player_id: str | None,
        match_id: str | None,
        media_type: str,
        original_filename: str,
        uploaded_at: datetime,
        location_id: str,
        location_ref: str,
        sha256_digest: str,
        size_bytes: int,
        actor_admin_id: str,
    ) -> dict[str, Any]:
        with self._connect() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.execute(
                "SELECT tenant_id::text FROM business.venues WHERE id=%s AND status='active'",
                (venue_id,),
            )
            venue = cursor.fetchone()
            if venue is None or venue["tenant_id"] != tenant_id:
                raise BackofficeError("未找到可用场馆。")
            if player_id:
                cursor.execute(
                    "SELECT 1 FROM business.venue_memberships WHERE venue_id=%s AND user_id=%s AND status='active'",
                    (venue_id, player_id),
                )
                if cursor.fetchone() is None:
                    raise BackofficeError("球员不属于该场馆。")
            cursor.execute(
                "INSERT INTO business.managed_media_resources "
                "(id, tenant_id, venue_id, player_id, match_id, asset_type, media_type, original_filename, upload_succeeded_at) "
                "VALUES (%s, %s, %s, %s, %s, 'video', %s, %s, %s)",
                (asset_id, tenant_id, venue_id, player_id, match_id, media_type, original_filename, uploaded_at),
            )
            cursor.execute(
                "INSERT INTO business.managed_media_resource_locations "
                "(id, media_asset_id, storage_backend, location_ref, sha256, size_bytes) "
                "VALUES (%s, %s, 'local_disk', %s, %s, %s)",
                (location_id, asset_id, location_ref, sha256_digest, size_bytes),
            )
            cursor.execute(
                "INSERT INTO business.audit_events "
                "(id, actor_type, actor_admin_account_id, action, resource_type, resource_id, tenant_id, venue_id, after_summary) "
                "VALUES (%s, 'admin', %s, 'media.uploaded', 'media_asset', %s, %s, %s, %s::jsonb)",
                (str(uuid.uuid4()), actor_admin_id, asset_id, tenant_id, venue_id,
                 json.dumps({"original_filename": original_filename, "size_bytes": size_bytes}, ensure_ascii=False)),
            )
        return {
            "id": asset_id,
            "tenant_id": tenant_id,
            "venue_id": venue_id,
            "player_id": player_id,
            "match_id": match_id,
            "media_type": media_type,
            "original_filename": original_filename,
            "upload_succeeded_at": uploaded_at.isoformat(),
            "status": "active",
        }

    def list_media_assets(self, venue_id: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE a.venue_id=%s" if venue_id else ""
        params = (venue_id,) if venue_id else ()
        return self._dict_rows(
            "SELECT a.id::text, a.tenant_id::text, a.venue_id::text, a.player_id::text, a.match_id::text, "
            "a.media_type, coalesce(a.original_filename, '') AS original_filename, "
            "a.upload_succeeded_at::text, a.status, "
            "coalesce(sum(l.size_bytes) FILTER (WHERE l.deletion_status <> 'deleted'), 0)::bigint AS size_bytes "
            "FROM business.managed_media_resources a LEFT JOIN business.managed_media_resource_locations l ON l.media_asset_id=a.id "
            f"{where} GROUP BY a.id ORDER BY a.upload_succeeded_at DESC",
            params,
        )

    def get_media_asset(self, asset_id: str) -> dict[str, Any]:
        rows = self._dict_rows(
            "SELECT a.id::text, a.tenant_id::text, a.venue_id::text, a.player_id::text, a.match_id::text, "
            "a.media_type, coalesce(a.original_filename, '') AS original_filename, "
            "a.upload_succeeded_at::text, a.status, l.id::text AS location_id, l.storage_backend, "
            "l.location_ref, l.deletion_status FROM business.managed_media_resources a "
            "LEFT JOIN business.managed_media_resource_locations l ON l.media_asset_id=a.id WHERE a.id=%s "
            "ORDER BY l.created_at",
            (asset_id,),
        )
        if not rows:
            raise FileNotFoundError(f"media asset not found: {asset_id}")
        asset = {key: rows[0][key] for key in (
            "id", "tenant_id", "venue_id", "player_id", "match_id", "media_type",
            "original_filename", "upload_succeeded_at", "status",
        )}
        asset["locations"] = [
            {key: row[key] for key in ("location_id", "storage_backend", "location_ref", "deletion_status")}
            for row in rows if row["location_id"]
        ]
        return asset

    def mark_media_location(self, location_id: str, status: str, error: str | None = None) -> None:
        self._execute(
            "UPDATE business.managed_media_resource_locations SET deletion_status=%s, deletion_error=%s, "
            "deleted_at=CASE WHEN %s='deleted' THEN now() ELSE NULL END, updated_at=now() WHERE id=%s",
            (status, error, status, location_id),
        )

    def mark_media_asset_status(self, asset_id: str, status: str) -> None:
        self._execute(
            "UPDATE business.managed_media_resources SET status=%s, updated_at=now() WHERE id=%s",
            (status, asset_id),
        )

    def expired_media_asset_ids(self, cutoff: datetime) -> list[str]:
        return [
            row["id"]
            for row in self._dict_rows(
                "SELECT id::text FROM business.managed_media_resources "
                "WHERE asset_type='video' AND upload_succeeded_at <= %s "
                "AND status IN ('active', 'delete_failed') ORDER BY upload_succeeded_at, id",
                (cutoff,),
            )
        ]

    def audit_media_resource_deletion(self, asset_id: str, actor_admin_id: str | None, result: dict) -> None:
        rows = self._dict_rows(
            "SELECT tenant_id, venue_id FROM business.managed_media_resources WHERE id=%s",
            (asset_id,),
        )
        if not rows:
            return
        self._execute(
            "INSERT INTO business.audit_events "
            "(id, actor_type, actor_admin_account_id, action, resource_type, resource_id, tenant_id, venue_id, after_summary) "
            "VALUES (%s, %s, %s, 'media.resources_deleted', 'media_asset', %s, %s, %s, %s::jsonb)",
            (
                str(uuid.uuid4()), "admin" if actor_admin_id else "system", actor_admin_id, asset_id,
                rows[0]["tenant_id"], rows[0]["venue_id"], json.dumps(result, ensure_ascii=False),
            ),
        )

    def get_video_retention_policy(self) -> dict[str, Any]:
        rows = self._dict_rows(
            "SELECT enabled, retention_days, timezone, daily_run_time::text, "
            "last_started_at::text, last_completed_at::text, updated_at::text "
            "FROM business.video_retention_policy WHERE id=1"
        )
        if not rows:
            raise BackofficeError("视频保留策略尚未初始化。")
        return rows[0]

    def update_video_retention_policy(
        self,
        *,
        enabled: bool,
        retention_days: int,
        timezone_name: str,
        daily_run_time: str,
        actor_admin_id: str,
    ) -> dict[str, Any]:
        try:
            ZoneInfo(timezone_name)
            parsed_time = datetime.strptime(daily_run_time, "%H:%M").time()
        except (ValueError, TypeError) as exc:
            raise BackofficeError("时区或每日执行时间无效。") from exc
        if not 1 <= retention_days <= 3650:
            raise BackofficeError("视频保留天数必须在 1 到 3650 之间。")
        with self._connect() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.execute(
                "UPDATE business.video_retention_policy SET enabled=%s, retention_days=%s, timezone=%s, "
                "daily_run_time=%s, updated_by_admin_id=%s, updated_at=now() WHERE id=1",
                (enabled, retention_days, timezone_name, parsed_time, actor_admin_id),
            )
            cursor.execute(
                "INSERT INTO business.audit_events "
                "(id, actor_type, actor_admin_account_id, action, resource_type, resource_id, after_summary) "
                "VALUES (%s, 'admin', %s, 'video_retention.updated', 'video_retention_policy', NULL, %s::jsonb)",
                (str(uuid.uuid4()), actor_admin_id, json.dumps({
                    "enabled": enabled, "retention_days": retention_days,
                    "timezone": timezone_name, "daily_run_time": daily_run_time,
                }, ensure_ascii=False)),
            )
        return self.get_video_retention_policy()

    def claim_video_retention_run(self) -> dict[str, Any] | None:
        rows = self._dict_rows(
            "UPDATE business.video_retention_policy SET last_started_at=now() WHERE id=1 AND enabled=true "
            "AND (now() AT TIME ZONE timezone)::time >= daily_run_time "
            "AND (last_completed_at IS NULL OR (last_completed_at AT TIME ZONE timezone)::date "
            "< (now() AT TIME ZONE timezone)::date) "
            "AND (last_started_at IS NULL OR last_started_at < now() - interval '1 hour') "
            "RETURNING enabled, retention_days, timezone, daily_run_time::text, last_started_at::text",
        )
        return rows[0] if rows else None

    def complete_video_retention_run(self) -> None:
        self._execute(
            "UPDATE business.video_retention_policy SET last_completed_at=now(), updated_at=now() WHERE id=1",
            (),
        )

    def delete_media_asset(self, asset_id: str, actor_admin_id: str) -> None:
        with self._connect() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.execute("SELECT tenant_id, venue_id FROM business.managed_media_resources WHERE id=%s", (asset_id,))
            asset = cursor.fetchone()
            if asset is None:
                raise FileNotFoundError(f"media asset not found: {asset_id}")
            cursor.execute("DELETE FROM business.analysis_jobs WHERE input_media_asset_id=%s", (asset_id,))
            cursor.execute("DELETE FROM business.managed_media_resources WHERE id=%s", (asset_id,))
            cursor.execute(
                "INSERT INTO business.audit_events "
                "(id, actor_type, actor_admin_account_id, action, resource_type, resource_id, tenant_id, venue_id, after_summary) "
                "VALUES (%s, 'admin', %s, 'media.deleted', 'media_asset', %s, %s, %s, '{}'::jsonb)",
                (str(uuid.uuid4()), actor_admin_id, asset_id, asset["tenant_id"], asset["venue_id"]),
            )

    def create_manual_analysis_job(self, asset_id: str, actor_admin_id: str) -> str:
        job_id = str(uuid.uuid4())
        self._execute(
            "INSERT INTO business.analysis_jobs "
            "(id, status, job_type, input_media_asset_id, requested_by_admin_id, trigger_type) "
            "VALUES (%s, 'uploading', 'video_analysis', %s, %s, 'manual')",
            (job_id, asset_id, actor_admin_id),
        )
        return job_id

    def bind_manual_analysis_job(self, job_id: str, asset_id: str, remote_job_id: str) -> None:
        with self._connect() as connection, connection.transaction(), connection.cursor() as cursor:
            cursor.execute(
                "UPDATE business.analysis_jobs SET status='queued', external_analysis_session_id=%s "
                "WHERE id=%s AND input_media_asset_id=%s",
                (remote_job_id, job_id, asset_id),
            )
            cursor.execute(
                "INSERT INTO business.managed_media_resource_locations "
                "(id, media_asset_id, storage_backend, location_ref) VALUES (%s, %s, 'gpu_http', %s)",
                (str(uuid.uuid4()), asset_id, f"job:{remote_job_id}"),
            )

    def fail_manual_analysis_job(self, job_id: str, message: str) -> None:
        self._execute(
            "UPDATE business.analysis_jobs SET status='failed', finished_at=now() WHERE id=%s",
            (job_id,),
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
            return psycopg.connect(self.database_url, row_factory=dict_row, autocommit=True)
        except Exception as exc:
            raise BackofficeError(f"业务数据库不可用：{exc}") from exc

    def _rows(self, query: str, params: tuple[Any, ...] = ()) -> list[list[str]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            # `_rows` feeds Gradio tables and JSON detail panels only.  Keep
            # machine-facing endpoints on `_dict_rows` in UTC, while ensuring
            # every displayed PostgreSQL datetime is China Standard Time.
            return [["" if value is None else _format_china_time(value) for value in row.values()] for row in cursor.fetchall()]

    def _dict_rows(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

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
    # The production operator surface is the Next.js SaaS console.  Keep the
    # older Gradio console available for local diagnostics, but import its
    # optional dependency only when that console is explicitly rendered.
    import gradio as gr
    """Render the venue-first B-scheme console in the current Gradio tabs."""
    service = OperatorBackoffice(active_task_controller)

    def ready() -> bool:
        return service.database.readiness().get("status") == "ready"

    def refresh_venues():
        status = service.database.readiness()
        return (service.database.list_venue_court_overview(), status) if status.get("status") == "ready" else ([], status)

    def refresh_runtime():
        status = service.database.readiness()
        return (service.database.list_venue_court_runtime() if status.get("status") == "ready" else []), status

    def provision_edge_camera(*values):
        result = service.database.provision_edge_camera(*values)
        runtime, _ = refresh_runtime()
        return runtime, result, "终端与摄像头已绑定。请仅在受控终端写入刚生成的密钥。"

    def refresh_dashboard():
        status = service.database.readiness()
        dashboard = service.database.dashboard() if status.get("status") == "ready" else {}
        return _dashboard_markup(dashboard, service.overview()), _status_text(status)

    def save_venue(*values):
        saved_id = service.database.save_venue(*values)
        venues, _ = refresh_venues()
        return venues, f"场馆已保存：`{saved_id}`"

    def save_court(*values):
        saved_id = service.database.save_court(*values)
        venues, _ = refresh_venues()
        return venues, f"场地已保存：`{saved_id}`"

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

    def refresh_single_venue(venue_id: str | None, selected_court_id: str | None = None):
        readiness = service.database.readiness()
        gpu = service.gpu_status(check_health=False)
        if readiness.get("status") != "ready":
            return _single_venue_markup({}, readiness, gpu), [], gr.update(choices=[], value=None), _status_text(readiness)
        snapshot = service.database.venue_control_snapshot(venue_id)
        court_choices = snapshot["court_options"]
        allowed = {value for _, value in court_choices}
        selected = selected_court_id if selected_court_id in allowed else (court_choices[0][1] if court_choices else None)
        return (
            _single_venue_markup(snapshot, readiness, gpu),
            snapshot["rows"],
            gr.update(choices=court_choices, value=selected),
            _status_text(readiness),
        )

    def set_single_venue_court_status(venue_id: str, court_id: str, status: str):
        service.database.set_court_status(venue_id, court_id, status)
        markup, rows, courts, readiness_markup = refresh_single_venue(venue_id, court_id)
        return markup, rows, courts, readiness_markup, "场地状态已保存；状态变更已经写入业务库并记录审计。"

    def provision_single_venue_camera(venue_id: str, court_id: str, device_code: str, camera_code: str, credential_version: str):
        result = service.database.provision_edge_camera("", "", venue_id, court_id, device_code, camera_code, credential_version)
        markup, rows, courts, readiness_markup = refresh_single_venue(venue_id, court_id)
        pairing_text = json.dumps(result, ensure_ascii=False, indent=2)
        return markup, rows, courts, readiness_markup, pairing_text, "终端与摄像头已绑定。请仅将下方一次性凭据写入受控终端。"

    def refresh_control_gpu():
        return _gpu_markup(service.gpu_status(check_health=True))

    def save_control_gpu(base_url: str, api_key: str):
        result = service.save_gpu_config(base_url, api_key)
        return _gpu_markup(result), gr.update(value=""), "GPU 服务地址已保存；密钥不会回显。"

    def control_gpu_operation(operation: str):
        result = service.request_gpu_operation(operation)
        return _gpu_markup(service.gpu_status(check_health=False)), str(result.get("message") or "GPU 操作已提交。")

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

    initial_readiness = service.database.readiness()
    initial_venue_choices = service.database.venue_options() if initial_readiness.get("status") == "ready" else []
    initial_venue_id = initial_venue_choices[0][1] if initial_venue_choices else None
    initial_snapshot = service.database.venue_control_snapshot(initial_venue_id) if initial_venue_id else {}
    initial_court_choices = initial_snapshot.get("court_options", [])
    initial_court_id = initial_court_choices[0][1] if initial_court_choices else None
    initial_gpu = service.gpu_status(check_health=False)

    with gr.Tab("运营总控"):
        control_summary = gr.HTML(value=_single_venue_markup(initial_snapshot, initial_readiness, initial_gpu), elem_classes=["saas-shell"])
        with gr.Row():
            control_venue = gr.Dropdown(choices=initial_venue_choices, value=initial_venue_id, label="正在管理的球馆", scale=5)
            control_refresh = gr.Button("刷新实时状态", variant="primary", size="sm", scale=1)
        control_db_status = gr.HTML(value=_status_text(initial_readiness))
        gr.HTML("<h3 class='saas-section'>场地监控与解析状态</h3><p class='saas-note'>每行就是一个场地。状态来自业务数据库：设备和摄像头由终端心跳更新；解析状态来自业务服务受理的会话。没有真实终端接入时会明确显示“待接入”，不会伪造在线数据。</p>")
        control_table = gr.Dataframe(
            headers=["场地", "场地编码", "可用性", "终端", "摄像头", "标定", "解析", "最近心跳", "处理建议"],
            value=initial_snapshot.get("rows", []), interactive=False, label="当前球馆场地状态",
        )
        gr.HTML("<h3 class='saas-section'>管理操作</h3>")
        with gr.Row():
            with gr.Column(elem_classes=["ops-action-card"]):
                gr.HTML("<h3>场地可用性</h3><p>运营人员可以把场地设置为可用、维护或停用。变更会立即写入业务库。</p>")
                with gr.Group(elem_classes=["wrap"]):
                    control_court_status_target = gr.Dropdown(choices=initial_court_choices, value=initial_court_id, label="目标场地")
                    control_court_status = gr.Dropdown(choices=sorted(_COURT_STATUSES), value="active", label="设置为")
                    control_court_save = gr.Button("保存场地状态", variant="primary", size="sm")
                    control_court_notice = gr.Markdown()
            with gr.Column(elem_classes=["ops-action-card"]):
                gr.HTML("<h3>终端与摄像头接入</h3><p>首次安装时为场地创建受控终端配对凭据。RTSP 地址和密码始终留在现场终端。</p>")
                with gr.Group(elem_classes=["wrap"]):
                    with gr.Row():
                        control_device_code = gr.Textbox(label="终端编码", value="court-01-edge")
                        control_camera_code = gr.Textbox(label="摄像头编码", value="camera-01")
                    control_credential_version = gr.Textbox(label="凭据版本", value="v1")
                    control_provision = gr.Button("生成一次性配对凭据", variant="primary", size="sm")
                    control_pairing = gr.Textbox(label="一次性终端配对数据（复制后妥善保存）", lines=6, interactive=False)
                    control_pairing_notice = gr.Markdown()
            with gr.Column(elem_classes=["ops-action-card"]):
                gr.HTML("<h3>GPU 服务</h3><p>配置只保存在业务服务器。健康检查、受控启停和任务取消均不会将密钥发给浏览器。</p>")
                with gr.Group(elem_classes=["wrap"]):
                    control_gpu_output = gr.HTML(value=_gpu_markup(initial_gpu))
                    control_gpu_base_url = gr.Textbox(label="GPU API 地址", value=configured_gpu_url())
                    control_gpu_api_key = gr.Textbox(label="新的 API Key（留空则不改）", type="password")
                    with gr.Row():
                        control_gpu_save = gr.Button("保存配置", variant="primary", size="sm")
                        control_gpu_health = gr.Button("健康检查", size="sm")
                    with gr.Row():
                        control_gpu_start = gr.Button("启动 GPU", size="sm")
                        control_gpu_stop = gr.Button("关闭 GPU", variant="stop", size="sm")
                    control_gpu_notice = gr.Markdown()
        gr.HTML("<p class='saas-note'>解析由已接入的终端向业务服务器发起签名会话，再由业务服务器中继给 GPU；后台负责展示状态与配置服务。没有终端视频流时，不能从页面凭空启动解析。</p>")

        control_refresh.click(refresh_single_venue, inputs=[control_venue, control_court_status_target], outputs=[control_summary, control_table, control_court_status_target, control_db_status], show_progress="hidden")
        control_venue.change(refresh_single_venue, inputs=[control_venue, control_court_status_target], outputs=[control_summary, control_table, control_court_status_target, control_db_status], show_progress="hidden")
        control_court_save.click(set_single_venue_court_status, inputs=[control_venue, control_court_status_target, control_court_status], outputs=[control_summary, control_table, control_court_status_target, control_db_status, control_court_notice], show_progress="hidden")
        control_provision.click(provision_single_venue_camera, inputs=[control_venue, control_court_status_target, control_device_code, control_camera_code, control_credential_version], outputs=[control_summary, control_table, control_court_status_target, control_db_status, control_pairing, control_pairing_notice], show_progress="hidden")
        control_gpu_save.click(save_control_gpu, inputs=[control_gpu_base_url, control_gpu_api_key], outputs=[control_gpu_output, control_gpu_api_key, control_gpu_notice], show_progress="hidden")
        control_gpu_health.click(refresh_control_gpu, outputs=[control_gpu_output], show_progress="hidden")
        control_gpu_start.click(lambda: control_gpu_operation("start"), outputs=[control_gpu_output, control_gpu_notice], show_progress="hidden")
        control_gpu_stop.click(lambda: control_gpu_operation("stop"), outputs=[control_gpu_output, control_gpu_notice], show_progress="hidden")

    with gr.Tab("业务概览"):
        dashboard_output = gr.HTML(value=_dashboard_markup(service.database.dashboard() if ready() else {}, service.overview()), elem_classes=["venue-ops"])
        with gr.Row():
            dashboard_refresh = gr.Button("刷新看板", variant="primary", size="sm")
            dashboard_db_status = gr.HTML(value=_status_text(service.database.readiness()))
        dashboard_refresh.click(refresh_dashboard, outputs=[dashboard_output, dashboard_db_status], show_progress="hidden")

    with gr.Tab("场馆与场地"):
        gr.Markdown("## 场馆与场地\n一张表管理全部球馆和场地：先看场地是否可用、终端/摄像头是否已接入、是否正在解析；技术 ID 只在新增或编辑时使用。")
        venue_db_status = gr.JSON(label="数据库连接", value=service.database.readiness())
        venues_table = gr.Dataframe(
            headers=["球馆", "球馆编码", "球馆状态", "场地", "场地编码", "场地状态", "终端状态", "摄像头状态", "解析状态"],
            value=[], interactive=False, label="球馆 / 场地状态总表",
        )
        refresh_venue_button = gr.Button("刷新球馆与场地状态", variant="primary", size="sm")
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
        refresh_venue_button.click(refresh_venues, outputs=[venues_table, venue_db_status], show_progress="hidden")
        venue_save.click(save_venue, inputs=[venue_id, venue_tenant_id, venue_code, venue_name, venue_timezone, venue_address, venue_status], outputs=[venues_table, venue_notice])
        court_save.click(save_court, inputs=[court_id, court_venue_id, court_code, court_name, court_sort_order, court_status], outputs=[venues_table, court_notice])
        refresh_qr_button.click(refresh_qrs, outputs=[qr_table], show_progress="hidden")
        qr_create.click(create_qr, inputs=[qr_venue_id, qr_court_id, qr_scope, qr_label, qr_expires], outputs=[qr_table, qr_image, qr_payload, qr_notice])
        qr_revoke.click(revoke_qr, inputs=[qr_revoke_id], outputs=[qr_table, qr_revoke_notice])

    with gr.Tab("场馆运行"):
        gr.Markdown("## 场馆与场地运行总览\n这是一个统一运营后台，不为每个球馆创建独立页面。表格按场馆、场地展示终端在线状态、当前解析会话和 GPU 转发结果。")
        runtime_db_status = gr.JSON(label="数据库连接", value=service.database.readiness())
        runtime_table = gr.Dataframe(
            headers=["场馆", "场馆编码", "场地", "场地编码", "设备 ID", "设备编码", "设备状态", "设备最近心跳", "摄像头 ID", "摄像头编码", "摄像头状态", "摄像头最近心跳", "解析状态", "已收分片", "已转 GPU", "GPU 会话", "GPU 状态", "最近错误", "状态更新时间"],
            value=[], interactive=False, label="按场馆 / 场地的实时运行状态",
        )
        runtime_refresh = gr.Button("刷新运行状态", variant="primary", size="sm")
        with gr.Accordion("绑定终端与摄像头（首次部署）", open=False):
            gr.Markdown("终端只提交设备 ID、摄像头 ID、时间戳、nonce 和签名。RTSP 地址/密码保留在终端，后台不保存。生成后只复制一次密钥到受控终端。")
            with gr.Row():
                edge_device_id = gr.Textbox(label="设备 ID（留空则生成）")
                edge_camera_id = gr.Textbox(label="摄像头 ID（留空则生成）")
            with gr.Row():
                edge_venue_id = gr.Textbox(label="场馆 ID")
                edge_court_id = gr.Textbox(label="场地 ID")
            with gr.Row():
                edge_device_code = gr.Textbox(label="设备编码", value="court-01-edge")
                edge_camera_code = gr.Textbox(label="摄像头编码", value="camera-01")
                edge_credential_version = gr.Textbox(label="凭据版本", value="v1")
            edge_provision = gr.Button("生成终端配对凭据", variant="primary", size="sm")
            edge_pairing = gr.JSON(label="一次性终端配对数据", value={})
            edge_notice = gr.Markdown()
        runtime_refresh.click(refresh_runtime, outputs=[runtime_table, runtime_db_status], show_progress="hidden")
        edge_provision.click(provision_edge_camera, inputs=[edge_device_id, edge_camera_id, edge_venue_id, edge_court_id, edge_device_code, edge_camera_code, edge_credential_version], outputs=[runtime_table, edge_pairing, edge_notice])

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
