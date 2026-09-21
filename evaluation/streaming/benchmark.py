"""Replay harness that produces a streaming benchmark report.

The harness drives accepted segments through the real StreamSessionManager and
AnalysisEngine with a lightweight processor, timing the measurable stages at the
storage boundary plus a wrapped decoder/processor.  It never claims a real-input
SLO: streaming_slo_proven stays false and is_replay stays true.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from badminton_analysis.streaming import OpenCVSegmentDecoder

from api.stream_sessions import StreamSessionManager
from .trace import new_benchmark, new_stage, p50, p95, utc_now, write_trace


class TimingAccumulator:
    """Shared wall-clock buckets filled by the wrapped decoder and processors."""

    def __init__(self):
        self.decode_seconds = 0.0
        self.decode_frames = 0
        self.process_seconds = 0.0
        self.process_frames = 0
        self.finalize_seconds = 0.0


class TimingDecoder:
    """Wrap a decoder so each lazy frame decode is timed."""

    def __init__(self, decoder, accumulator: TimingAccumulator):
        self._decoder = decoder
        self._accumulator = accumulator

    def decode(self, segment_path, descriptor):
        segment = self._decoder.decode(segment_path, descriptor)
        return _TimedFrameSegment(segment, self._accumulator)


class _TimedFrameSegment:
    def __init__(self, segment, accumulator: TimingAccumulator):
        self.descriptor = segment.descriptor
        self._frames = segment.frames
        self._accumulator = accumulator

    @property
    def frames(self):
        iterator = iter(self._frames)
        while True:
            started = time.perf_counter()
            try:
                packet = next(iterator)
            except StopIteration:
                break
            self._accumulator.decode_seconds += time.perf_counter() - started
            self._accumulator.decode_frames += 1
            yield packet


class TimingProcessor:
    """Wrap a StatefulFrameProcessor so frame/finalize work is timed."""

    def __init__(self, processor, accumulator: TimingAccumulator):
        self._processor = processor
        self._accumulator = accumulator

    def process_frame(self, frame, context):
        started = time.perf_counter()
        result = self._processor.process_frame(frame, context)
        self._accumulator.process_seconds += time.perf_counter() - started
        self._accumulator.process_frames += 1
        return result

    def finalize(self, context):
        started = time.perf_counter()
        result = self._processor.finalize(context)
        self._accumulator.finalize_seconds += time.perf_counter() - started
        return result

    def snapshot_state(self):
        return self._processor.snapshot_state()

    def restore_state(self, state):
        return self._processor.restore_state(state)


def _create_request(config):
    return {
        "schema_version": "stream-session.v1",
        "camera_id": "bench-camera",
        "calibration_id": "bench-calibration",
        # This harness has no business camera registry. Keep its fixed
        # synthetic calibration explicit so it remains compatible with the
        # production stream-session contract.
        "court_corners": config.get(
            "court_corners",
            [[0.0, 0.0], [64.0, 0.0], [64.0, 64.0], [0.0, 64.0]],
        ),
        "analysis_mode": "person_only",
        "configuration": {
            "analysis_sample_hz": int(config["sample_hz"]),
            "pose_imgsz": int(config.get("pose_imgsz", 960)),
            "shuttle_detector": config.get("shuttle_detector", "none"),
            "generate_annotated_video": bool(config.get("generate_annotated_video", False)),
            "preserve_audio": False,
        },
    }


def run_stream_replay(
    data_dir,
    segments,
    processor_factory: Callable[[], Any],
    config,
    *,
    session_key="benchmark-stream-0001",
):
    """Run one replay and return a benchmark dict plus the manager for inspection.

    ``segments`` is an iterable of ``(segment_index, raw_bytes, metadata)`` tuples
    in source order.  The caller owns real segmentation (task C in production);
    this harness only measures the GPU-side storage/engine chain.
    """
    data_dir = Path(data_dir)
    accumulator = TimingAccumulator()
    decoder = TimingDecoder(OpenCVSegmentDecoder(), accumulator)

    def wrapped_factory(session=None):
        produced = (
            processor_factory(session)
            if StreamSessionManager._factory_accepts_session(processor_factory)
            else processor_factory()
        )
        if produced is None:
            return None
        if isinstance(produced, tuple):
            measurement = produced[0]
            temporal = produced[1] if len(produced) > 1 else None
        else:
            measurement = produced
            temporal = None
        return (
            TimingProcessor(measurement, accumulator) if measurement is not None else None,
            TimingProcessor(temporal, accumulator) if temporal is not None else None,
        )

    manager = StreamSessionManager(
        data_dir, processor_factory=wrapped_factory, decoder=decoder, start_worker=False
    )
    benchmark = new_benchmark(config)
    benchmark["session_id"] = None

    status, created = manager.create_session(_create_request(config), session_key)
    session_id = created["analysis_session_id"]
    benchmark["session_id"] = session_id

    per_segment_e2e: List[float] = []
    receive_total = 0.0
    process_total = 0.0
    replay_started = time.perf_counter()
    for index, raw, metadata in segments:
        receive_started = time.perf_counter()
        manager.receive_segment(session_id, index, metadata, raw)
        receive_ended = time.perf_counter()
        manager.drain()
        process_ended = time.perf_counter()
        receive_total += receive_ended - receive_started
        process_total += process_ended - receive_ended
        per_segment_e2e.append(process_ended - receive_started)

    last_segment_received_at = time.perf_counter()
    complete_started = time.perf_counter()
    manager.complete(
        session_id,
        {"schema_version": "stream-session.v1", "expected_last_segment_index": len(per_segment_e2e) - 1, "allow_partial": False},
    )
    finalize_seconds = time.perf_counter() - complete_started
    replay_total_seconds = time.perf_counter() - replay_started

    status_payload = manager.get_status(session_id)[1]
    progress = status_payload["progress"]
    backlog = progress["backlog_duration_sec"]

    benchmark["segments"]["count"] = len(per_segment_e2e)
    benchmark["segments"]["p50_end_to_end_seconds"] = p50(per_segment_e2e)
    benchmark["segments"]["p95_end_to_end_seconds"] = p95(per_segment_e2e)
    benchmark["segments"]["throughput_segments_per_sec"] = (
        round(len(per_segment_e2e) / replay_total_seconds, 6) if replay_total_seconds > 0 else None
    )
    benchmark["queue"]["backlog_seconds_at_video_end"] = round(backlog, 3)
    benchmark["completion"]["finalize_seconds"] = round(finalize_seconds, 6)
    benchmark["completion"]["final_tail_seconds"] = round(time.perf_counter() - last_segment_received_at, 6)
    benchmark["llm"]["elapsed_seconds"] = 0.0
    benchmark["llm"]["request_count"] = 0
    benchmark["llm"]["run"] = False

    benchmark["stages"] = [
        new_stage("segment", "skipped", error="segment production is owned by task C; not measured here"),
        new_stage("upload", "skipped", error="no network hop in an in-process replay"),
        new_stage("gpu_receive", "ok", elapsed_seconds=round(receive_total, 6), segment_count=len(per_segment_e2e)),
        new_stage("queue_wait", "skipped", error="single-session replay has no queue contention"),
        new_stage("decode", "ok", elapsed_seconds=round(accumulator.decode_seconds, 6), frame_count=accumulator.decode_frames),
        new_stage("sample", "skipped", error="sampling cadence is inside the engine; not separately hooked"),
        new_stage("pose_inference", "ok", elapsed_seconds=round(accumulator.process_seconds, 6), frame_count=accumulator.process_frames),
        new_stage("ball_inference", "skipped", error="ball inference disabled or not separately hooked"),
        new_stage("tracking", "skipped", error="tracking is inside the engine; not separately hooked"),
        new_stage("event_write", "skipped", error="event/checkpoint writes are inside the storage layer; not separately hooked"),
        new_stage("annotate_encode", "skipped", error="annotated video export is disabled"),
        new_stage("aggregate", "ok", elapsed_seconds=round(accumulator.finalize_seconds, 6)),
        new_stage("seal", "ok", elapsed_seconds=round(finalize_seconds, 6)),
        new_stage("download", "skipped", error="no artifact download in an in-process replay"),
    ]
    benchmark["resource_peaks"] = {"note": "resource peaks require the real GPU harness (task F)"}
    benchmark["errors"] = []
    write_trace(data_dir / "end_to_end_trace.json", benchmark)
    return benchmark, manager
