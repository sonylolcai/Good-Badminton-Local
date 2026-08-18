"""Bounded player-report handoff for the post-match analysis stage.

This module deliberately separates measured match data from language-model
wording. It first writes a deterministic evidence package and prompt. When an
OpenAI-compatible report service is configured, it makes *one* bounded request
for the whole match so four-player doubles does not multiply end-of-match
latency. A report-service failure never rewrites or invalidates raw tracking
evidence; callers receive an explicit status instead.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib import error, request


REPORT_SCHEMA_VERSION = "1.0"
DEFAULT_TIMEOUT_SECONDS = 75


def generate_performance_report(output_dir, metadata_path, spatial_summary_path):
    """Write report evidence and optionally request one bounded LLM summary.

    ``GOOD_BADMINTON_LLM_BASE_URL``, ``GOOD_BADMINTON_LLM_API_KEY`` and
    ``GOOD_BADMINTON_LLM_MODEL`` must all be configured for an LLM response.
    The absence of those settings is surfaced as ``not_configured`` rather
    than being presented as an athlete report.
    """
    output_dir = Path(output_dir)
    metadata = _read_json(metadata_path)
    summary = _read_json(spatial_summary_path)
    report_dir = output_dir / "derived"
    report_dir.mkdir(parents=True, exist_ok=True)

    evidence = build_report_evidence(metadata, summary)
    evidence_path = report_dir / "player_performance_input_v1.json"
    _write_json(evidence_path, evidence)

    prompt = build_chinese_prompt(evidence)
    prompt_path = report_dir / "player_performance_prompt_zh.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "player_performance_report",
        "status": "not_configured",
        "evidence_path": str(evidence_path),
        "prompt_path": str(prompt_path),
        "generated_at": _utc_timestamp(),
        "policy": (
            "Report language may explain measured evidence only. It must label "
            "insufficient coverage, unknown score, inferred ball events, and "
            "unclaimed track identities rather than guessing them."
        ),
    }
    llm_config = _llm_config_from_environment()
    if llm_config is None:
        result["reason"] = "LLM service is not configured on this worker"
    else:
        started = time.monotonic()
        try:
            content = _request_openai_compatible_report(llm_config, prompt)
            result.update(
                {
                    "status": "succeeded",
                    "model": llm_config["model"],
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "content": content,
                }
            )
        except (OSError, ValueError, error.URLError, TimeoutError) as exc:
            result.update(
                {
                    "status": "failed",
                    "model": llm_config["model"],
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "reason": str(exc),
                }
            )

    result_path = report_dir / "player_performance_report_v1.json"
    _write_json(result_path, result)
    return {**result, "report_path": str(result_path)}


def build_report_evidence(metadata, spatial_summary):
    """Build compact, traceable inputs without manufacturing ability scores."""
    metadata = metadata or {}
    spatial_summary = spatial_summary or {}
    players = []
    for item in spatial_summary.get("player_style_inputs") or []:
        detected = int(item.get("detected_frames") or 0)
        predicted = int(item.get("predicted_frames") or 0)
        missing = int(item.get("missing_frames") or 0)
        total = detected + predicted + missing
        players.append(
            {
                "track_id": item.get("track_id"),
                "person_id": item.get("person_id"),
                "team_id": item.get("team_id"),
                "movement": {
                    "distance_m": item.get("distance_m"),
                    "zone_frames": item.get("zone_frames") or {},
                },
                "tracking_coverage": {
                    "detected_frames": detected,
                    "predicted_frames": predicted,
                    "missing_frames": missing,
                    "measured_ratio": round(detected / total, 4) if total else 0.0,
                },
                "limitations": [
                    "Track identity is a visual track_id until a post-match human claim binds person_id.",
                    "Predicted and missing positions are not detector measurements.",
                    "Distance and court occupancy describe movement/space only; they do not prove technique or match outcome.",
                ],
            }
        )
    pose = ((metadata.get("models") or {}).get("pose") or {})
    shuttle = ((metadata.get("models") or {}).get("shuttlecock_detection") or {})
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "player_performance_evidence",
        "match": {
            "mode": ((spatial_summary.get("match") or {}).get("mode")),
            "video_duration_sec": ((metadata.get("video") or {}).get("duration_sec")),
            "video_fps": ((metadata.get("video") or {}).get("fps")),
            "score_policy": spatial_summary.get("score_policy"),
        },
        "measurement_plan": {
            "pose_imgsz": pose.get("imgsz"),
            "pose_sample_hz": pose.get("sample_hz"),
            "pose_processed_frame_count": pose.get("processed_frame_count"),
            "shuttle_primary_source": shuttle.get("primary_source"),
            "shuttle_measurement_kind": shuttle.get("measurement_kind"),
        },
        "players": players,
        "rally_evidence": {
            "candidate_rally_count": len(spatial_summary.get("rallies") or []),
            "policy": "Rally and score conclusions remain unknown unless their evidence confidence permits them.",
        },
    }


def build_chinese_prompt(evidence):
    """Return one schema-constrained Chinese prompt for all visual tracks."""
    serialized = json.dumps(evidence, ensure_ascii=False, indent=2)
    return (
        "你是羽毛球赛后运动表现分析助手。仅依据下列 JSON 中的测量证据输出报告；"
        "不要把 track_id 当作真实姓名，不要把 predicted/missing 当作真实动作，不要依据不完整球路"
        "推断胜负、失误率或技术等级。\n\n"
        "请用中文输出严格 JSON：{\"players\":[{\"track_id\":string,\"summary\":string,"
        "\"movement_observations\":[string],\"training_suggestions\":[string],"
        "\"confidence_and_limits\":[string]}],\"match_limits\":[string]}。\n\n"
        "当 measured_ratio 较低、球路候选不足或 score_policy 为 unknown 时，明确说明无法得出什么结论；"
        "建议应可执行，但不得诊断伤病。\n\n"
        "测量证据：\n"
        f"{serialized}\n"
    )


def _llm_config_from_environment():
    base_url = os.environ.get("GOOD_BADMINTON_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("GOOD_BADMINTON_LLM_API_KEY", "").strip()
    model = os.environ.get("GOOD_BADMINTON_LLM_MODEL", "").strip()
    if not (base_url and api_key and model):
        return None
    timeout = int(os.environ.get("GOOD_BADMINTON_LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    if timeout <= 0 or timeout > 180:
        raise ValueError("GOOD_BADMINTON_LLM_TIMEOUT_SECONDS must be between 1 and 180")
    return {"base_url": base_url.rstrip("/"), "api_key": api_key, "model": model, "timeout": timeout}


def _request_openai_compatible_report(config, prompt):
    endpoint = config["base_url"]
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    payload = json.dumps(
        {
            "model": config["model"],
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Return only valid JSON. Never invent unavailable sports evidence."},
                {"role": "user", "content": prompt},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(http_request, timeout=config["timeout"]) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response does not contain choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty")
    return content


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
