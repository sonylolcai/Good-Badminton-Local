"""Evaluate immutable performance traces against a versioned production budget.

This module deliberately separates two kinds of evidence:

* a completed ``performance_trace.json`` proves the parameters and observed
  batch-stage timings for one job;
* a ``stream_replay`` report proves the streaming SLO only after the future
  segment API has actually been exercised.

Keeping those claims separate prevents a fast short batch job from being
mistaken for proof that a 15-minute live match can meet its final-tail budget.
The implementation uses only the Python standard library so it can run on the
GPU worker immediately after every relevant deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PASS = "pass"
WARN = "warn"
FAIL = "fail"
_STATUS_ORDER = {PASS: 0, WARN: 1, FAIL: 2}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _elapsed_between(started_at: Any, finished_at: Any) -> Optional[float]:
    started = _parse_timestamp(started_at)
    finished = _parse_timestamp(finished_at)
    if started is None or finished is None:
        return None
    return round(max(0.0, (finished - started).total_seconds()), 3)


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _new_check(
    name: str,
    status: str,
    message: str,
    expected: Any = None,
    actual: Any = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "expected": expected,
        "actual": actual,
    }


def _overall_status(checks: Iterable[Dict[str, Any]]) -> str:
    return max((item["status"] for item in checks), key=lambda status: _STATUS_ORDER[status], default=PASS)


def _stage_seconds(trace: Dict[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for stage in ((trace.get("timing") or {}).get("stages") or []):
        if not isinstance(stage, dict):
            continue
        name = stage.get("name")
        elapsed = _number(stage.get("elapsed_seconds"))
        if not name or elapsed is None:
            continue
        values[str(name)] = round(values.get(str(name), 0.0) + max(0.0, elapsed), 3)
    return values


def summarize_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comparable observations from one immutable job trace."""
    task = trace.get("task") or {}
    progress = trace.get("progress") or {}
    stages = _stage_seconds(trace)
    fps = None
    for stage in ((trace.get("timing") or {}).get("stages") or []):
        details = stage.get("details") if isinstance(stage, dict) else None
        if isinstance(details, dict) and _number(details.get("fps")):
            fps = _number(details.get("fps"))
            break
    total_frames = _number(progress.get("total_frames"))
    duration = (total_frames / fps) if total_frames and fps else None
    wall_seconds = _elapsed_between(task.get("started_at"), task.get("finished_at"))
    return {
        "job_status": task.get("status"),
        "wall_seconds": wall_seconds,
        "source_frame_count": int(total_frames) if total_frames is not None else None,
        "source_fps": fps,
        "source_duration_seconds": round(duration, 3) if duration is not None else None,
        "batch_realtime_factor": round(wall_seconds / duration, 4) if wall_seconds is not None and duration else None,
        "stage_seconds": stages,
        "options": trace.get("options") or {},
    }


def evaluate_trace(trace: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check production parameters and collect honest batch-only observations."""
    summary = summarize_trace(trace)
    options = summary["options"]
    expected = profile.get("production_options") or {}
    checks: List[Dict[str, Any]] = []

    if summary["job_status"] != "succeeded":
        checks.append(_new_check(
            "trace.job_status", FAIL, "Only a completed successful job can establish a performance baseline.",
            "succeeded", summary["job_status"],
        ))

    for option_name in ("pose_imgsz", "pose_sample_hz", "shuttle_detector"):
        required = expected.get(option_name)
        actual = options.get(option_name)
        if actual != required:
            checks.append(_new_check(
                f"production_option.{option_name}", FAIL,
                "Observed job does not use the locked production parameter.", required, actual,
            ))
        else:
            checks.append(_new_check(
                f"production_option.{option_name}", PASS,
                "Observed job uses the locked production parameter.", required, actual,
            ))

    if summary["source_duration_seconds"] is None:
        checks.append(_new_check(
            "trace.source_duration", WARN,
            "Trace has no source FPS/frame-count pair, so real-time factor cannot be calculated.",
        ))
    else:
        factor = summary["batch_realtime_factor"]
        checks.append(_new_check(
            "batch.observed_realtime_factor", WARN,
            "Batch timing is observational only; it cannot prove the streaming SLA until a segment replay is recorded.",
            "<= 1.0 preferred before streaming proof", factor,
        ))

    checks.append(_new_check(
        "streaming_slo.evidence", WARN,
        "No stream-replay report supplied. Batch traces must not be advertised as streaming-SLO proof.",
    ))
    return checks


def evaluate_regression(trace: Dict[str, Any], baseline: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fail only like-for-like trace regressions, never different source videos."""
    current = summarize_trace(trace)
    reference = summarize_trace(baseline)
    checks: List[Dict[str, Any]] = []
    current_duration = current["source_duration_seconds"]
    baseline_duration = reference["source_duration_seconds"]
    if current_duration is None or baseline_duration is None or abs(current_duration - baseline_duration) > 0.01:
        return [_new_check(
            "regression.comparability", WARN,
            "Regression comparison skipped because the traces do not identify the same source duration.",
            baseline_duration, current_duration,
        )]

    limit = float((profile.get("regression") or {}).get("max_stage_regression_percent", 15.0))
    for name, current_seconds in current["stage_seconds"].items():
        baseline_seconds = reference["stage_seconds"].get(name)
        if baseline_seconds is None or baseline_seconds <= 0:
            continue
        percent = round((current_seconds - baseline_seconds) / baseline_seconds * 100.0, 2)
        status = FAIL if percent > limit else PASS
        checks.append(_new_check(
            f"regression.stage.{name}", status,
            "Stage regression exceeds the profile limit." if status == FAIL else "Stage is within the regression limit.",
            f"<= {limit}%", f"{percent}%",
        ))
    if not checks:
        checks.append(_new_check(
            "regression.stage", WARN,
            "No common completed stages were available for comparison.",
        ))
    return checks


def evaluate_stream_replay(stream: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate the future 1-2 second segment replay contract when available."""
    slo = profile.get("streaming_slo") or {}
    checks: List[Dict[str, Any]] = []
    if stream.get("kind") != "good_badminton_stream_replay_benchmark":
        return [_new_check(
            "streaming_slo.report_kind", FAIL,
            "Stream replay report has an unexpected kind.",
            "good_badminton_stream_replay_benchmark", stream.get("kind"),
        )]

    values = {
        "segment.p95_seconds": ("max_segment_p95_seconds", ((stream.get("segments") or {}).get("p95_end_to_end_seconds"))),
        "queue.backlog_seconds_at_video_end": ("max_backlog_seconds_at_video_end", ((stream.get("queue") or {}).get("backlog_seconds_at_video_end"))),
        "finalize.seconds": ("max_finalize_seconds", ((stream.get("completion") or {}).get("finalize_seconds"))),
        "llm.seconds": ("max_llm_seconds", ((stream.get("llm") or {}).get("elapsed_seconds"))),
        "final_tail.seconds": ("max_final_tail_seconds", ((stream.get("completion") or {}).get("final_tail_seconds"))),
    }
    for check_name, (limit_name, actual) in values.items():
        limit = _number(slo.get(limit_name))
        actual_number = _number(actual)
        if limit is None or actual_number is None:
            checks.append(_new_check(check_name, FAIL, "Required stream-replay measurement is missing.", limit, actual))
        elif actual_number > limit:
            checks.append(_new_check(check_name, FAIL, "Streaming SLO limit exceeded.", limit, actual_number))
        else:
            checks.append(_new_check(check_name, PASS, "Streaming SLO limit met.", limit, actual_number))

    llm_count = _number(((stream.get("llm") or {}).get("request_count")))
    max_requests = _number((profile.get("production_options") or {}).get("max_llm_requests_per_match"))
    if llm_count is None:
        checks.append(_new_check("llm.request_count", FAIL, "Stream replay must report LLM request count.", max_requests, None))
    elif max_requests is not None and llm_count > max_requests:
        checks.append(_new_check("llm.request_count", FAIL, "More than one match-level LLM request is forbidden.", max_requests, llm_count))
    else:
        checks.append(_new_check("llm.request_count", PASS, "Match-level LLM request budget met.", max_requests, llm_count))
    return checks


def build_report(
    trace: Dict[str, Any],
    profile: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    stream_replay: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checks = evaluate_trace(trace, profile)
    if baseline is not None:
        checks.extend(evaluate_regression(trace, baseline, profile))
    if stream_replay is not None:
        # A completed stream replay supersedes the two deliberately cautious
        # batch-only observations.  Keep those raw observations in
        # ``observed_trace`` for diagnosis, but do not let a warning that says
        # "batch is not streaming proof" downgrade actual streaming proof.
        checks = [
            check for check in checks
            if check["name"] not in {"streaming_slo.evidence", "batch.observed_realtime_factor"}
        ]
        checks.extend(evaluate_stream_replay(stream_replay, profile))
    return {
        "schema_version": "1.0",
        "kind": "good_badminton_performance_gate",
        "profile_id": profile.get("profile_id"),
        "status": _overall_status(checks),
        "observed_trace": summarize_trace(trace),
        "checks": checks,
        "streaming_slo_proven": stream_replay is not None and _overall_status(checks) == PASS,
        "policy": (
            "A batch trace records real work but cannot by itself prove streaming capability. "
            "Only a 1-2 second segment replay may set streaming_slo_proven=true."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Good-Badminton performance budgets.")
    parser.add_argument("--trace", required=True, type=Path, help="Completed performance_trace.json")
    parser.add_argument("--profile", required=True, type=Path, help="Versioned JSON performance profile")
    parser.add_argument("--baseline", type=Path, help="Optional prior trace for like-for-like stage regression")
    parser.add_argument("--stream-replay", type=Path, help="Optional 1-2 second segment replay benchmark JSON")
    parser.add_argument("--output", type=Path, help="Write the resulting gate report to this path")
    args = parser.parse_args(argv)

    report = build_report(
        _load_json(args.trace),
        _load_json(args.profile),
        baseline=_load_json(args.baseline) if args.baseline else None,
        stream_replay=_load_json(args.stream_replay) if args.stream_replay else None,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 2 if report["status"] == FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
