"""Metric definitions and explicit release-gate evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


METRIC_DEFINITIONS = {
    "person.recall": ("ratio", "up"),
    "person.far_player_recall": ("ratio", "up"),
    "person.longest_detection_gap_sec": ("sec", "down"),
    "person.false_positive_count": ("count", "down"),
    "tracking.player_recall": ("ratio", "up"),
    "tracking.id_switches": ("count", "down"),
    "tracking.longest_interruption_frames": ("frame", "down"),
    "tracking.occlusion_recovery_rate": ("ratio", "up"),
    "tracking.hit_owner_accuracy": ("ratio", "up"),
    "shuttle.visible_recall": ("ratio", "up"),
    "shuttle.not_visible_fp_rate": ("ratio", "down"),
    "shuttle.localization_error_px_p50": ("px", "down"),
    "shuttle.longest_raw_gap_sec": ("sec", "down"),
    "shuttle.inferred_visible_point_count": ("count", "observe"),
    "perf.batch_realtime_factor": ("ratio", "down"),
    "perf.p95_segment_latency_sec": ("sec", "down"),
    "perf.backlog_seconds_at_video_end": ("sec", "down"),
    "perf.finalize_seconds": ("sec", "down"),
    "perf.llm_seconds": ("sec", "down"),
    "perf.final_tail_seconds": ("sec", "down"),
    "perf.llm_request_count": ("count", "down"),
}

_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def metric_definition(metric_key: str) -> dict[str, str]:
    try:
        unit, direction = METRIC_DEFINITIONS[metric_key]
    except KeyError as exc:
        raise ValueError(f"unregistered metric: {metric_key}") from exc
    return {"metric_definition_version": "1.0.0", "unit": unit, "direction": direction}


def validate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen = set()
    for index, row in enumerate(rows):
        key = row.get("metric_key")
        definition = metric_definition(str(key))
        identity = (row.get("scope"), row.get("scope_id"), key)
        if identity in seen:
            raise ValueError(f"duplicate metric row: {identity}")
        seen.add(identity)
        if row.get("metric_definition_version") != definition["metric_definition_version"]:
            raise ValueError(f"metric version mismatch for {key}")
        if row.get("unit") != definition["unit"]:
            raise ValueError(f"metric unit mismatch for {key}")
        if row.get("scope") not in {"run", "slice", "case"}:
            raise ValueError(f"metrics[{index}].scope is invalid")
        if row.get("status") not in {
            "valid",
            "preliminary",
            "insufficient_data",
            "not_applicable",
        }:
            raise ValueError(f"metrics[{index}].status is invalid")
        if row.get("status") == "valid" and (
            isinstance(row.get("value"), bool) or not isinstance(row.get("value"), (int, float))
        ):
            raise ValueError(f"metrics[{index}].value must be numeric when status=valid")


def evaluate_gate(
    comparison: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate explicit thresholds against candidate values and deltas."""
    if not profile:
        return {"profile_id": None, "status": "PASS", "checks": []}
    if (comparison.get("comparability") or {}).get("status") != "comparable":
        return {
            "profile_id": profile.get("profile_id"),
            "status": "FAIL",
            "checks": [
                {
                    "status": "FAIL",
                    "reason": "Runs are not comparable",
                    "issues": (comparison.get("comparability") or {}).get("issues", []),
                }
            ],
        }

    deltas = comparison.get("metric_deltas", []) + comparison.get("slice_deltas", []) + comparison.get("case_deltas", [])
    checks = []
    for rule in profile.get("checks") or []:
        target = next(
            (
                row
                for row in deltas
                if row["metric_key"] == rule.get("metric_key")
                and row["scope"] == rule.get("scope", "run")
                and row["scope_id"] == rule.get("scope_id", "all")
            ),
            None,
        )
        if target is None:
            status = "FAIL" if rule.get("required", True) else "WARN"
            checks.append({"status": status, "rule": dict(rule), "reason": "Required metric is missing"})
            continue

        violations = []
        candidate = target["candidate"]
        if rule.get("minimum") is not None and candidate < float(rule["minimum"]):
            violations.append(f"candidate {candidate} is below minimum {rule['minimum']}")
        if rule.get("maximum") is not None and candidate > float(rule["maximum"]):
            violations.append(f"candidate {candidate} is above maximum {rule['maximum']}")
        if rule.get("max_regression") is not None and target["classification"] == "regressed":
            if abs(float(target["delta"])) > float(rule["max_regression"]):
                violations.append(
                    f"regression {abs(float(target['delta']))} exceeds {rule['max_regression']}"
                )
        status = rule.get("violation_status", "FAIL") if violations else "PASS"
        if status not in _STATUS_ORDER:
            raise ValueError("violation_status must be PASS, WARN, or FAIL")
        checks.append(
            {
                "status": status,
                "rule": dict(rule),
                "candidate": candidate,
                "delta": target["delta"],
                "reason": "; ".join(violations) if violations else "Rule passed",
            }
        )
    overall = max((item["status"] for item in checks), key=_STATUS_ORDER.get, default="PASS")
    return {"profile_id": profile.get("profile_id"), "status": overall, "checks": checks}
