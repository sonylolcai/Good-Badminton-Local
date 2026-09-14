"""Streaming replay performance and reliability gate (task D).

This package owns the replay harness, trace schema/validator and SLO gate. It
never modifies the analysis core, api routes or business client; it only adds
observation wrappers and produces honest, machine-readable benchmark reports.
"""

from .benchmark import TimingAccumulator, TimingDecoder, TimingProcessor, run_stream_replay
from .baseline import DEFAULT_STREAMING_SLO, compare_stream_benchmark, build_gate_report, gate
from .reliability import run_reliability_scenarios
from .trace import (
    BENCHMARK_KIND,
    TRACE_SCHEMA_VERSION,
    new_benchmark,
    new_stage,
    p50,
    p95,
    validate_trace,
    write_trace,
    load_trace,
)

__all__ = [
    "BENCHMARK_KIND",
    "TRACE_SCHEMA_VERSION",
    "DEFAULT_STREAMING_SLO",
    "TimingAccumulator",
    "TimingDecoder",
    "TimingProcessor",
    "run_stream_replay",
    "compare_stream_benchmark",
    "build_gate_report",
    "gate",
    "run_reliability_scenarios",
    "new_benchmark",
    "new_stage",
    "p50",
    "p95",
    "validate_trace",
    "write_trace",
    "load_trace",
]

