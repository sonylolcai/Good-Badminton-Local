"""Versioned streaming SLO budget comparison and gate command.

Aligns with the batch performance gate: a stream replay report must meet the
segment P95, backlog, finalize, LLM and final-tail budgets before any
streaming_slo_proven claim.  A replay alone never sets streaming_slo_proven=true.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .trace import BENCHMARK_KIND, load_trace, validate_trace


PASS = "pass"
WARN = "warn"
FAIL = "fail"
_ORDER = {PASS: 0, WARN: 1, FAIL: 2}


DEFAULT_STREAMING_SLO = {
    "segment_duration_seconds": 2.0,
    "max_segment_p95_seconds": 1.5,
    "max_backlog_seconds_at_video_end": 360.0,
    "max_finalize_seconds": 120.0,
    "max_llm_seconds": 75.0,
    "max_final_tail_seconds": 600.0,
}


def _number(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check(name, status, message, expected=None, actual=None):
    return {"name": name, "status": status, "message": message, "expected": expected, "actual": actual}


def compare_stream_benchmark(benchmark: Dict[str, Any], slo: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Compare a stream replay benchmark against the streaming SLO budget."""
    slo = slo or DEFAULT_STREAMING_SLO
    checks: List[Dict[str, Any]] = []
    if benchmark.get("kind") != BENCHMARK_KIND:
        return [_check("benchmark.kind", FAIL, "unexpected benchmark kind", BENCHMARK_KIND, benchmark.get("kind"))]

    values = {
        "segment.p95_seconds": ("max_segment_p95_seconds", ((benchmark.get("segments") or {}).get("p95_end_to_end_seconds"))),
        "queue.backlog_seconds_at_video_end": ("max_backlog_seconds_at_video_end", ((benchmark.get("queue") or {}).get("backlog_seconds_at_video_end"))),
        "finalize.seconds": ("max_finalize_seconds", ((benchmark.get("completion") or {}).get("finalize_seconds"))),
        "llm.seconds": ("max_llm_seconds", ((benchmark.get("llm") or {}).get("elapsed_seconds"))),
        "final_tail.seconds": ("max_final_tail_seconds", ((benchmark.get("completion") or {}).get("final_tail_seconds"))),
    }
    for check_name, (limit_name, actual) in values.items():
        limit = _number(slo.get(limit_name))
        actual_number = _number(actual)
        if actual_number is None:
            checks.append(_check(check_name, WARN, "measurement is absent in this replay", limit, actual))
        elif limit is None:
            checks.append(_check(check_name, WARN, "no budget configured", None, actual_number))
        elif actual_number > limit:
            checks.append(_check(check_name, FAIL, "streaming SLO budget exceeded", limit, actual_number))
        else:
            checks.append(_check(check_name, PASS, "streaming SLO budget met", limit, actual_number))
    return checks


def _overall(checks):
    return max((item["status"] for item in checks), key=lambda status: _ORDER[status], default=PASS)


def build_gate_report(benchmark: Dict[str, Any], slo: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    validation_errors = validate_trace(benchmark)
    checks = []
    if validation_errors:
        checks.append(
            _check(
                "trace.valid",
                FAIL,
                "; ".join(validation_errors),
                "valid streaming trace",
                "invalid",
            )
        )
    else:
        checks.append(
            _check("trace.valid", PASS, "streaming trace schema is complete", "valid", "valid")
        )
    checks.extend(compare_stream_benchmark(benchmark, slo))
    status = _overall(checks)
    return {
        "schema_version": "1.0",
        "kind": "good_badminton_streaming_gate",
        "status": status,
        "streaming_slo_proven": bool(benchmark.get("streaming_slo_proven")) and status == PASS,
        "checks": checks,
        "policy": (
            "A replay records real storage/engine work but cannot prove live-input SLO; "
            "streaming_slo_proven only becomes true after a real input end-to-end test (task F)."
        ),
    }


def gate(benchmark_path: Path, slo_path: Optional[Path] = None, output: Optional[Path] = None) -> int:
    benchmark = load_trace(benchmark_path)
    slo = None
    if slo_path:
        slo = json.loads(slo_path.read_text(encoding="utf-8"))
    report = build_gate_report(benchmark, slo)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 2 if report["status"] == FAIL else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gate a streaming replay benchmark against the SLO budget.")
    parser.add_argument("--benchmark", required=True, type=Path, help="stream replay benchmark JSON")
    parser.add_argument("--slo", type=Path, help="optional JSON budget (defaults to the frozen v1 budget)")
    parser.add_argument("--output", type=Path, help="write the gate report to this path")
    args = parser.parse_args(argv)
    return gate(args.benchmark, args.slo, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
