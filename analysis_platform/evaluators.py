"""Thin adapters around the repository's existing evaluation scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ADAPTER_VERSION = "1.0.0"
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_SPECS = {
    "far_player": ("evaluation.far_player.benchmark_far_player", "directory"),
    "doubles": ("evaluation.doubles.run_doubles_evaluation", "file"),
    "shuttle_tracknet_ab": ("evaluation.shuttle_tracknet_ab.run_ab_benchmark", "directory"),
    "performance_streaming": ("evaluation.performance.performance_gate", "file"),
}


class EvaluationAdapterError(RuntimeError):
    pass


def execute_evaluator(
    name: str,
    arguments: Sequence[str],
    output: str | Path,
) -> dict[str, Any]:
    """Run one existing CLI unchanged, then normalize its generated report."""
    module, output_kind = _spec(name)
    output_path = Path(output).resolve()
    if any(argument in {"--output", "--output-dir"} for argument in arguments):
        raise ValueError("output flags are owned by the adapter")
    if output_kind == "directory":
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "report.json"
        output_arguments = ["--output-dir", str(output_path)]
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path = output_path
        output_arguments = ["--output", str(output_path)]

    command = [sys.executable, "-m", module, *map(str, arguments), *output_arguments]
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout or "no evaluator output").strip()
        raise EvaluationAdapterError(
            f"{name} evaluator failed with exit code {completed.returncode}: {diagnostic[-4000:]}"
        )
    if not report_path.is_file():
        raise EvaluationAdapterError(f"{name} evaluator did not create {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationAdapterError(f"{name} evaluator report is invalid JSON: {report_path}") from exc
    return normalize_report(name, report, report_path=report_path)


def normalize_report(
    name: str,
    report: Mapping[str, Any],
    *,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert one existing report into versioned Metric Result rows."""
    _spec(name)
    if not isinstance(report, Mapping):
        raise ValueError("evaluator report must be a JSON object")
    normalizers = {
        "far_player": _normalize_far_player,
        "doubles": _normalize_doubles,
        "shuttle_tracknet_ab": _normalize_shuttle,
        "performance_streaming": _normalize_performance,
    }
    metrics = normalizers[name](report)
    source_path = str(Path(report_path).resolve()) if report_path is not None else None
    source_sha256 = (
        _sha256_file(Path(report_path))
        if report_path is not None
        else hashlib.sha256(
            json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    return {
        "schema_version": "evaluation-adapter.v1",
        "adapter": {"name": name, "version": ADAPTER_VERSION},
        "source_report": {"path": source_path, "sha256": source_sha256},
        "metrics": metrics,
        "metric_count": len(metrics),
    }


def _normalize_far_player(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, result in sorted((report.get("methods") or {}).items()):
        raw = result.get("metrics") or {}
        timing = result.get("timing") or {}
        frame_count = _count(timing.get("evaluated_frame_count"))
        ground_truth_count = _count(raw.get("far_player_ground_truth_count"))
        scope_id = f"method:{method}"
        rows.extend(
            _metrics(
                scope_id,
                [
                    ("person.far_player_recall", raw.get("far_player_recall"), "ratio", ground_truth_count),
                    ("person.longest_detection_gap_sec", raw.get("longest_consecutive_miss_seconds"), "sec", ground_truth_count),
                    ("person.false_positive_count", raw.get("false_positive_count"), "count", frame_count),
                ],
            )
        )
    return rows


def _normalize_doubles(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("metrics") or {}
    player_samples = _count(raw.get("ground_truth_player_observations"))
    occlusion_samples = _count(raw.get("occlusion_recovery_opportunities"))
    hit_samples = _count(raw.get("hit_attribution_ground_truth_count"))
    return _metrics(
        "all",
        [
            ("tracking.player_recall", raw.get("player_recall"), "ratio", player_samples),
            ("tracking.id_switches", raw.get("id_switches"), "count", player_samples),
            ("tracking.longest_interruption_frames", raw.get("longest_track_interruption_frames"), "frame", player_samples),
            ("tracking.occlusion_recovery_rate", raw.get("occlusion_recovery_rate"), "ratio", occlusion_samples),
            ("tracking.hit_owner_accuracy", raw.get("hit_attribution_accuracy"), "ratio", hit_samples),
        ],
    )


def _normalize_shuttle(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, raw in sorted((report.get("methods") or {}).items()):
        if not isinstance(raw, Mapping):
            continue
        visible_samples = _count(raw.get("visible_ground_truth_count"))
        invisible_samples = _count(raw.get("not_visible_ground_truth_count"))
        localization_samples = _count(raw.get("true_positive_count"))
        status = "preliminary" if "rectified" in method else "valid"
        rows.extend(
            _metrics(
                f"method:{method}",
                [
                    ("shuttle.visible_recall", raw.get("raw_detection_recall"), "ratio", visible_samples),
                    ("shuttle.not_visible_fp_rate", raw.get("false_positive_rate"), "ratio", invisible_samples),
                    ("shuttle.localization_error_px_p50", raw.get("median_localization_error_px"), "px", localization_samples),
                    ("shuttle.longest_raw_gap_sec", raw.get("longest_consecutive_raw_miss_seconds"), "sec", visible_samples),
                    ("shuttle.inferred_visible_point_count", raw.get("inferred_points_on_visible_ground_truth"), "count", visible_samples),
                ],
                status=status,
            )
        )
    return rows


def _normalize_performance(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    mappings = {
        "batch.observed_realtime_factor": ("perf.batch_realtime_factor", "ratio"),
        "segment.p95_seconds": ("perf.p95_segment_latency_sec", "sec"),
        "queue.backlog_seconds_at_video_end": ("perf.backlog_seconds_at_video_end", "sec"),
        "finalize.seconds": ("perf.finalize_seconds", "sec"),
        "llm.seconds": ("perf.llm_seconds", "sec"),
        "final_tail.seconds": ("perf.final_tail_seconds", "sec"),
        "llm.request_count": ("perf.llm_request_count", "count"),
    }
    rows = []
    for check in report.get("checks") or []:
        mapped = mappings.get(check.get("name"))
        actual = check.get("actual")
        if mapped is None or isinstance(actual, bool) or not isinstance(actual, (int, float)):
            continue
        metric_key, unit = mapped
        rows.extend(_metrics("all", [(metric_key, actual, unit, 1)]))
    return rows


def _metrics(
    scope_id: str,
    values: Sequence[tuple[str, Any, str, int]],
    *,
    status: str = "valid",
) -> list[dict[str, Any]]:
    rows = []
    for metric_key, value, unit, sample_count in values:
        metric_status = status if value is not None else "insufficient_data"
        rows.append(
            {
                "metric_key": metric_key,
                "metric_definition_version": "1.0.0",
                "scope": "slice" if scope_id.startswith("method:") else "run",
                "scope_id": scope_id,
                "value": value,
                "unit": unit,
                "sample_count": sample_count,
                "eligible_sample_count": sample_count if value is not None else 0,
                "status": metric_status,
            }
        )
    return rows


def _spec(name: str) -> tuple[str, str]:
    try:
        return _SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluator: {name}") from exc


def _count(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
