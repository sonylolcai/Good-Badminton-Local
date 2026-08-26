# Continuous analysis core

This package is task A's transport-independent boundary. It does not register
FastAPI routes and does not change the existing complete-file pipeline.

## Integration contract for task B

Task B should persist the HTTP receipt first, then decode the accepted fragment
and call the same `AnalysisEngine` instance for the lifetime of the session:

```python
descriptor = SegmentDescriptor(...)
segment = OpenCVSegmentDecoder().decode(segment_path, descriptor)
result = engine.process_segment(segment)

persist_events(result.events)
persist_checkpoint(result.checkpoint)
```

The storage layer buffers out-of-order fragments. `AnalysisEngine` intentionally
accepts only `next_expected_segment_index`; this prevents the model/tracker from
crossing an evidence gap. A repeated processed index with the same digest returns
`reused=True`; a different digest raises `SegmentConflictError`.

On process restart, task B creates fresh processor instances and restores the
durable checkpoint:

```python
engine = AnalysisEngine.restore(
    checkpoint,
    measurement_processor=fixed_camera_processor,
    temporal_processor=tracknet_processor_or_none,
)
```

If processor state cannot be restored, `engine.status` becomes
`interrupted_needs_rebuild`. The caller must expose that state and must not silently
restart the tracker with new Track IDs.

For an accepted fragment that fails during processing, the API owns the recovery
decision: it records a durable failed-segment receipt, discards the unsafe engine,
and starts a **fresh** engine epoch after the gap. Later evidence remains available,
but its track IDs are namespaced by continuity epoch and the session can only finish
as `partial`. The engine's `begin_after_gap()` method exists solely for that fresh
epoch boundary; it must never be used to pretend that a tracker bridged lost media.

## Sampling boundary

- `measurement_processor` is called only at the configured 10/15/30Hz source-time
  buckets. Pose, YOLO shuttle, ByteTrack and measurement JSONL belong here.
- `temporal_processor` is optional and receives every decoded source frame. It is
  the future integration point for TrackNetV3's continuous 8-frame window.
- Both processors must return `ProcessorEvent` and JSON-serializable state.

## Replay boundary

`FileReplayAdapter` reads a complete local video lazily and exposes virtual 1–2
second segments without holding a complete segment in memory. It is for deterministic
analysis tests only. Task C owns real fMP4/FFmpeg segmentation, byte hashes and the
business-side retry ledger.
