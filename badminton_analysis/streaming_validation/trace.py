"""Machine-readable streaming-replay trace schema, writer and validator.

This is the streaming counterpart of the batch performance_trace.json.  A replay
report records real per-stage wall-clock work plus config and resource peaks, but
it is always marked is_replay=true and streaming_slo_proven=false until a real
input end-to-end test passes (task F).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


BENCHMARK_KIND = "good_badminton_stream_replay_benchmark"
TRACE_SCHEMA_VERSION = "1.0"

# Stages that every replay trace must carry (even with zero elapsed time).
REQUIRED_STAGES = [
    "segment", "upload", "gpu_receive", "queue_wait", "decode", "sample",
    "pose_inference", "tracking", "event_write", "aggregate", "seal", "download",
]
# Stages that only apply when the corresponding switch is enabled.
OPTIONAL_STAGES = ["ball_inference", "annotate_encode"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def p50(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 6)
    return round((ordered[middle - 1] + ordered[middle]) / 2.0, 6)


def p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(round(0.95 * len(ordered))) - 1)
    return round(ordered[index], 6)


def new_stage(name, status="ok", **fields):
    """Build one stage record with explicit defaults for the required fields."""
    stage = {
        "name": name,
        "status": status,
        "started_at": fields.get("started_at"),
        "finished_at": fields.get("finished_at"),
        "elapsed_seconds": fields.get("elapsed_seconds"),
        "wait_seconds": fields.get("wait_seconds", 0.0),
        "is_parallel": bool(fields.get("is_parallel", False)),
        "parallel_group": fields.get("parallel_group"),
        "segment_count": int(fields.get("segment_count", 0)),
        "frame_count": int(fields.get("frame_count", 0)),
        "error": fields.get("error"),
    }
    return stage


def new_benchmark(config):
    """Return an empty benchmark skeleton the harness fills in."""
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "kind": BENCHMARK_KIND,
        "recorded_at": utc_now(),
        "is_replay": True,
        "streaming_slo_proven": False,
        "config": dict(config or {}),
        "segments": {
            "count": 0,
            "p50_end_to_end_seconds": None,
            "p95_end_to_end_seconds": None,
            "throughput_segments_per_sec": None,
        },
        "queue": {"backlog_seconds_at_video_end": 0.0},
        "completion": {"finalize_seconds": None, "final_tail_seconds": None},
        "llm": {"elapsed_seconds": 0.0, "request_count": 0, "run": False},
        "stages": [],
        "resource_peaks": {},
        "reliability": {},
        "errors": [],
    }


def write_trace(path, benchmark):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def load_trace(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_trace(benchmark) -> List[str]:
    """Return a list of human-readable validation problems (empty means valid)."""
    errors: List[str] = []
    if not isinstance(benchmark, dict):
        return ["trace root must be a JSON object"]
    if benchmark.get("kind") != BENCHMARK_KIND:
        errors.append("trace.kind must be good_badminton_stream_replay_benchmark")
    if benchmark.get("schema_version") != TRACE_SCHEMA_VERSION:
        errors.append("trace.schema_version must be 1.0")
    stages = benchmark.get("stages")
    if not isinstance(stages, list):
        errors.append("trace.stages must be a list")
        return errors
    by_name: Dict[str, Any] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            errors.append("each stage must be an object")
            continue
        name = stage.get("name")
        if not name:
            errors.append("stage is missing a name")
            continue
        if name in by_name:
            errors.append(f"duplicate stage name: {name}")
        by_name[name] = stage
        if stage.get("status") not in {"ok", "failed", "skipped"}:
            errors.append(f"stage {name} has an invalid status")
        if stage.get("status") == "ok" and stage.get("elapsed_seconds") is None:
            errors.append(f"stage {name} is ok but has no elapsed_seconds")
        if stage.get("status") == "failed" and not stage.get("error"):
            errors.append(f"stage {name} is failed but has no error")
    for required in REQUIRED_STAGES:
        if required not in by_name:
            errors.append(f"missing required stage: {required}")
    config = benchmark.get("config")
    if not isinstance(config, dict) or not config.get("sample_hz") or not config.get("resolution"):
        errors.append("trace.config must include sample_hz and resolution")
    return errors

