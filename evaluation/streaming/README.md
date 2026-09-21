# Streaming replay performance & reliability gate

Task D provides a repeatable harness that replays a local video as 1-2 second
segments through the real StreamSessionManager + AnalysisEngine and writes a
machine-readable benchmark report.

## What it proves / does not prove

- A replay proves throughput, per-stage cost, and cross-segment correctness.
- It never proves a live-camera SLO: `streaming_slo_proven` stays `false`. Only a
  real-input end-to-end test (task F) may set it `true`.

## Usage

```bash
python -m evaluation.streaming.baseline --benchmark path/to/benchmark.json --output gate.json
```

The benchmark report uses `kind = "good_badminton_stream_replay_benchmark"` and
is consumable by the existing batch gate:

```bash
python -m evaluation.performance.performance_gate \
    --trace path/to/performance_trace.json \
    --profile evaluation/performance/rtx_3090_production_v1.json \
    --stream-replay path/to/benchmark.json
```

## Benchmark report

- `segments.p95_end_to_end_seconds` (budget 1.5s),
- `queue.backlog_seconds_at_video_end` (360s),
- `completion.finalize_seconds` (120s),
- `llm.elapsed_seconds` / `llm.request_count` (75s / <=1),
- `completion.final_tail_seconds` (600s).

The harness persists `end_to_end_trace.json`. Task F's production session trace
also records GPU receive, queue wait, segment open, the combined
decode/sample/model/tracking stage, event/checkpoint write and finalize. A stage
that the current model stack cannot separate is attributed honestly to the
combined stage rather than split using invented timings. Invalid/missing trace
evidence is a hard gate failure.

## Reliability scenarios

`run_reliability_scenarios` covers duplicate, out-of-order, missing-segment,
restart-restore, slow-upload and backlog-at-seal behaviour.
