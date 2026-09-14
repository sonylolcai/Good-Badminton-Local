"""Comparable Run deltas without a hidden aggregate score."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .gates import evaluate_gate, metric_definition


def compare_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    gate_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    issues = _comparability_issues(baseline, candidate)
    report: dict[str, Any] = {
        "schema_version": "comparison-report.v1",
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "comparability": {
            "status": "not_comparable" if issues else "comparable",
            "issues": issues,
        },
        "metric_deltas": [],
        "slice_deltas": [],
        "case_deltas": [],
        "improved_cases": [],
        "regressed_cases": [],
    }
    if issues:
        report["gate_result"] = evaluate_gate(report, gate_profile)
        return report

    baseline_rows = _valid_rows(baseline.get("metrics") or [])
    candidate_rows = _valid_rows(candidate.get("metrics") or [])
    for identity in sorted(set(baseline_rows).intersection(candidate_rows)):
        before = baseline_rows[identity]
        after = candidate_rows[identity]
        metric_key = identity[2]
        direction = metric_definition(metric_key)["direction"]
        delta = float(after["value"]) - float(before["value"])
        row = {
            "scope": identity[0],
            "scope_id": identity[1],
            "metric_key": metric_key,
            "metric_definition_version": before["metric_definition_version"],
            "unit": before["unit"],
            "direction": direction,
            "baseline": before["value"],
            "candidate": after["value"],
            "delta": round(delta, 12),
            "classification": _classification(delta, direction),
            "sample_count": after.get("sample_count"),
            "eligible_sample_count": after.get("eligible_sample_count"),
        }
        target = {
            "run": "metric_deltas",
            "slice": "slice_deltas",
            "case": "case_deltas",
        }[identity[0]]
        report[target].append(row)

    case_states: dict[str, set[str]] = {}
    for row in report["case_deltas"]:
        case_states.setdefault(row["scope_id"], set()).add(row["classification"])
    report["regressed_cases"] = sorted(
        case_id for case_id, states in case_states.items() if "regressed" in states
    )
    report["improved_cases"] = sorted(
        case_id
        for case_id, states in case_states.items()
        if "regressed" not in states and "improved" in states
    )
    report["gate_result"] = evaluate_gate(report, gate_profile)
    return report


def _comparability_issues(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> list[dict[str, str]]:
    issues = []
    for field in ("dataset_version", "dataset_manifest_sha256"):
        if not baseline.get(field) or baseline.get(field) != candidate.get(field):
            issues.append({"code": f"{field}_mismatch", "message": f"{field} must match"})

    baseline_rows = _rows_by_identity(baseline.get("metrics") or [])
    candidate_rows = _rows_by_identity(candidate.get("metrics") or [])
    common = set(baseline_rows).intersection(candidate_rows)
    if not common:
        issues.append({"code": "no_common_metrics", "message": "Runs have no common metric rows"})
    for identity in sorted(common):
        before = baseline_rows[identity]
        after = candidate_rows[identity]
        definition = metric_definition(identity[2])
        if before.get("metric_definition_version") != after.get("metric_definition_version"):
            issues.append(
                {
                    "code": "metric_version_mismatch",
                    "message": f"{identity[2]} uses different metric definition versions",
                }
            )
        elif before.get("metric_definition_version") != definition["metric_definition_version"]:
            issues.append(
                {
                    "code": "unsupported_metric_version",
                    "message": f"{identity[2]} does not use the registered metric definition version",
                }
            )
        if before.get("unit") != after.get("unit"):
            issues.append({"code": "metric_unit_mismatch", "message": f"{identity[2]} uses different units"})
        elif before.get("unit") != definition["unit"]:
            issues.append(
                {
                    "code": "unsupported_metric_unit",
                    "message": f"{identity[2]} does not use the registered unit",
                }
            )

    performance_keys = {identity for identity in common if identity[2].startswith("perf.")}
    if performance_keys:
        for field in ("hardware_fingerprint", "execution_mode"):
            if not baseline.get(field) or baseline.get(field) != candidate.get(field):
                issues.append(
                    {
                        "code": f"performance_{field}_mismatch",
                        "message": f"performance comparison requires matching {field}",
                    }
                )
    return issues


def _rows_by_identity(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        identity = (str(row.get("scope")), str(row.get("scope_id")), str(row.get("metric_key")))
        if identity in indexed:
            raise ValueError(f"duplicate metric row: {identity}")
        metric_definition(identity[2])
        indexed[identity] = row
    return indexed


def _valid_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        identity: row
        for identity, row in _rows_by_identity(rows).items()
        if row.get("status") == "valid"
        and not isinstance(row.get("value"), bool)
        and isinstance(row.get("value"), (int, float))
    }


def _classification(delta: float, direction: str) -> str:
    if abs(delta) <= 1e-12 or direction == "observe":
        return "unchanged"
    improved = delta > 0 if direction == "up" else delta < 0
    return "improved" if improved else "regressed"
