"""Stateful, transport-independent continuous analysis engine."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from .models import (
    ANALYSIS_SAMPLE_RATES,
    EngineNotRunnableError,
    FinalizationContext,
    FinalizationResult,
    FrameContext,
    FrameSegment,
    ProcessorEvent,
    ProcessorExecutionError,
    SegmentConflictError,
    SegmentOrderError,
    SegmentProcessingResult,
    STREAM_SCHEMA_VERSION,
)
from .processor import StatefulFrameProcessor


CHECKPOINT_VERSION = "continuous-analysis.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AnalysisEngine:
    """Process ordered segments while retaining processor state in memory.

    The API/storage layer may persist `checkpoint()` after every accepted
    result.  Restoring a checkpoint is conservative: if either processor cannot
    restore its state, the engine enters `interrupted_needs_rebuild` and refuses
    more frames instead of silently issuing new Track IDs.
    """

    def __init__(
        self,
        analysis_session_id: str,
        analysis_sample_hz: int,
        measurement_processor: StatefulFrameProcessor,
        *,
        temporal_processor: Optional[StatefulFrameProcessor] = None,
    ) -> None:
        if not analysis_session_id:
            raise ValueError("analysis_session_id is required")
        if analysis_sample_hz not in ANALYSIS_SAMPLE_RATES:
            raise ValueError(f"analysis_sample_hz must be one of {sorted(ANALYSIS_SAMPLE_RATES)}")
        self.analysis_session_id = str(analysis_session_id)
        self.analysis_sample_hz = int(analysis_sample_hz)
        self.measurement_processor = measurement_processor
        self.temporal_processor = temporal_processor

        self.status = "running"
        self.restore_error: Optional[str] = None
        self.next_expected_segment_index = 0
        self.processed_segments: dict[int, str] = {}
        self.last_source_time_sec = -1.0
        self.last_measurement_bucket = -1
        self.last_event_source_time_sec = -1.0
        self.source_frames = 0
        self.measurement_frames = 0
        self.event_sequence = 0
        # A segment-level failure creates a hard evidence gap.  The API may
        # intentionally start a fresh processor after that gap so later video
        # is still useful, but it must never be represented as one continuous
        # tracker timeline.
        self.continuity_epoch = 0
        self.gap_segment_indexes: list[int] = []

    @classmethod
    def restore(
        cls,
        checkpoint: Mapping[str, Any],
        measurement_processor: StatefulFrameProcessor,
        *,
        temporal_processor: Optional[StatefulFrameProcessor] = None,
    ) -> "AnalysisEngine":
        try:
            engine = cls(
                analysis_session_id=str(checkpoint.get("analysis_session_id") or "invalid"),
                analysis_sample_hz=int(checkpoint.get("analysis_sample_hz") or 10),
                measurement_processor=measurement_processor,
                temporal_processor=temporal_processor,
            )
            engine._restore_checkpoint(checkpoint)
        except Exception as exc:  # recovery must fail closed, regardless of processor implementation
            engine = cls(
                analysis_session_id=str(checkpoint.get("analysis_session_id") or "invalid"),
                analysis_sample_hz=10,
                measurement_processor=measurement_processor,
                temporal_processor=temporal_processor,
            )
            engine.status = "interrupted_needs_rebuild"
            engine.restore_error = f"{type(exc).__name__}: {exc}"
        return engine

    def process_segment(self, segment: FrameSegment) -> SegmentProcessingResult:
        descriptor = segment.descriptor
        if self.status != "running":
            raise EngineNotRunnableError(f"engine is not runnable: {self.status}")

        previous_digest = self.processed_segments.get(descriptor.segment_index)
        if previous_digest is not None:
            if previous_digest != descriptor.sha256:
                raise SegmentConflictError(
                    f"segment {descriptor.segment_index} was already processed with a different digest"
                )
            return SegmentProcessingResult(
                analysis_session_id=self.analysis_session_id,
                segment_index=descriptor.segment_index,
                reused=True,
                source_frames=0,
                measurement_frames=0,
                events=(),
                checkpoint=self.checkpoint(),
            )

        if descriptor.segment_index != self.next_expected_segment_index:
            raise SegmentOrderError(
                f"expected segment {self.next_expected_segment_index}, got {descriptor.segment_index}"
            )
        if descriptor.source_start_time_sec + 1e-6 < self.last_source_time_sec:
            raise SegmentOrderError("segment source time moves backwards")

        segment_source_frames = 0
        segment_measurement_frames = 0
        events: list[dict] = []
        frame_iterator = iter(segment.frames)
        try:
            for packet in frame_iterator:
                if packet.segment_index != descriptor.segment_index:
                    raise SegmentOrderError(
                        f"frame belongs to segment {packet.segment_index}, expected {descriptor.segment_index}"
                    )
                if packet.source_time_sec + 1e-9 < self.last_source_time_sec:
                    raise SegmentOrderError("frame source time moves backwards")

                measurement_bucket = int(math.floor(packet.source_time_sec * self.analysis_sample_hz + 1e-9))
                is_measurement = measurement_bucket > self.last_measurement_bucket
                context = FrameContext(
                    analysis_session_id=self.analysis_session_id,
                    segment_index=descriptor.segment_index,
                    source_frame_index=packet.source_frame_index,
                    source_time_sec=float(packet.source_time_sec),
                    is_measurement_frame=is_measurement,
                    measurement_bucket=measurement_bucket,
                )

                if self.temporal_processor is not None:
                    events.extend(
                        self._run_processor(self.temporal_processor, packet.frame, context)
                    )
                if is_measurement:
                    events.extend(
                        self._run_processor(self.measurement_processor, packet.frame, context)
                    )
                    self.last_measurement_bucket = measurement_bucket
                    self.measurement_frames += 1
                    segment_measurement_frames += 1

                self.last_source_time_sec = float(packet.source_time_sec)
                self.source_frames += 1
                segment_source_frames += 1
        except Exception as exc:
            self.status = "interrupted_needs_rebuild"
            self.restore_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (SegmentOrderError, SegmentConflictError)):
                raise
            raise ProcessorExecutionError(self.restore_error) from exc
        finally:
            close = getattr(frame_iterator, "close", None)
            if callable(close):
                close()

        self.processed_segments[descriptor.segment_index] = descriptor.sha256
        self.next_expected_segment_index += 1
        checkpoint = self.checkpoint()
        return SegmentProcessingResult(
            analysis_session_id=self.analysis_session_id,
            segment_index=descriptor.segment_index,
            reused=False,
            source_frames=segment_source_frames,
            measurement_frames=segment_measurement_frames,
            events=tuple(events),
            checkpoint=checkpoint,
        )

    def finalize(self) -> FinalizationResult:
        if self.status == "finalized":
            return FinalizationResult(
                analysis_session_id=self.analysis_session_id,
                status=self.status,
                events=(),
                checkpoint=self.checkpoint(),
            )
        if self.status != "running":
            raise EngineNotRunnableError(f"engine cannot finalize from {self.status}")

        context = FinalizationContext(
            analysis_session_id=self.analysis_session_id,
            last_segment_index=max(-1, self.next_expected_segment_index - 1),
            last_source_time_sec=max(0.0, self.last_source_time_sec),
            source_frames=self.source_frames,
            measurement_frames=self.measurement_frames,
        )
        events: list[dict] = []
        try:
            if self.temporal_processor is not None:
                events.extend(self._materialize_many(self.temporal_processor.finalize(context), context))
            events.extend(self._materialize_many(self.measurement_processor.finalize(context), context))
            events.append(
                self._materialize_event(
                    ProcessorEvent(
                        event_type="session_finalized",
                        evidence_state="finalized",
                        confidence=1.0,
                        source_time_sec=context.last_source_time_sec,
                        segment_index=max(0, context.last_segment_index),
                        data={
                            "status": "finalized",
                            "processed_segments": self.next_expected_segment_index,
                            "source_frames": self.source_frames,
                            "measurement_frames": self.measurement_frames,
                        },
                    ),
                    context,
                )
            )
            self.status = "finalized"
            checkpoint = self.checkpoint()
        except Exception as exc:
            self.status = "interrupted_needs_rebuild"
            self.restore_error = f"{type(exc).__name__}: {exc}"
            raise ProcessorExecutionError(self.restore_error) from exc

        return FinalizationResult(
            analysis_session_id=self.analysis_session_id,
            status=self.status,
            events=tuple(events),
            checkpoint=checkpoint,
        )

    def begin_after_gap(
        self,
        *,
        next_segment_index: int,
        source_time_sec: float,
        continuity_epoch: int,
        gap_segment_indexes: Iterable[int],
    ) -> None:
        """Start a fresh processor epoch after known-unusable source media.

        This method is deliberately for a *new* engine with fresh processors.
        It advances only the transport ordering/timing cursor; it does not
        invent a bridge between pre-gap and post-gap tracks.
        """
        if self.processed_segments or self.source_frames or self.measurement_frames:
            raise ValueError("begin_after_gap requires a fresh analysis engine")
        if int(next_segment_index) < 0 or float(source_time_sec) < 0:
            raise ValueError("gap recovery cursor must be non-negative")
        if int(continuity_epoch) < 1:
            raise ValueError("continuity_epoch must be at least 1 after a gap")
        self.next_expected_segment_index = int(next_segment_index)
        # The first source frame after the gap may have exactly this timestamp.
        self.last_source_time_sec = float(source_time_sec)
        self.last_measurement_bucket = int(
            math.floor(float(source_time_sec) * self.analysis_sample_hz + 1e-9)
        ) - 1
        self.continuity_epoch = int(continuity_epoch)
        self.gap_segment_indexes = sorted({int(index) for index in gap_segment_indexes})

    def progress(self) -> dict:
        return {
            "schema_version": STREAM_SCHEMA_VERSION,
            "analysis_session_id": self.analysis_session_id,
            "status": self.status,
            "next_expected_segment_index": self.next_expected_segment_index,
            "processed_segments": len(self.processed_segments),
            "last_source_time_sec": max(0.0, self.last_source_time_sec),
            "source_frames": self.source_frames,
            "measurement_frames": self.measurement_frames,
            "continuity_epoch": self.continuity_epoch,
            "gap_segment_indexes": list(self.gap_segment_indexes),
            "restore_error": self.restore_error,
        }

    def checkpoint(self) -> dict:
        try:
            measurement_state = dict(self.measurement_processor.snapshot_state())
            temporal_state = (
                dict(self.temporal_processor.snapshot_state())
                if self.temporal_processor is not None
                else None
            )
            payload = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "schema_version": STREAM_SCHEMA_VERSION,
                "analysis_session_id": self.analysis_session_id,
                "analysis_sample_hz": self.analysis_sample_hz,
                "status": self.status,
                "restore_error": self.restore_error,
                "next_expected_segment_index": self.next_expected_segment_index,
                "processed_segments": {
                    str(index): digest for index, digest in sorted(self.processed_segments.items())
                },
                "last_source_time_sec": self.last_source_time_sec,
                "last_measurement_bucket": self.last_measurement_bucket,
                "last_event_source_time_sec": self.last_event_source_time_sec,
                "source_frames": self.source_frames,
                "measurement_frames": self.measurement_frames,
                "event_sequence": self.event_sequence,
                "continuity_epoch": self.continuity_epoch,
                "gap_segment_indexes": list(self.gap_segment_indexes),
                "measurement_processor_state": measurement_state,
                "temporal_processor_state": temporal_state,
            }
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
            return payload
        except Exception as exc:
            self.status = "interrupted_needs_rebuild"
            self.restore_error = f"{type(exc).__name__}: {exc}"
            raise ProcessorExecutionError(
                f"processor state is not safely checkpointable: {self.restore_error}"
            ) from exc

    def _restore_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")
        if checkpoint.get("schema_version") != STREAM_SCHEMA_VERSION:
            raise ValueError("checkpoint schema does not match stream-session.v1")
        if str(checkpoint.get("analysis_session_id")) != self.analysis_session_id:
            raise ValueError("checkpoint session does not match engine session")
        if int(checkpoint.get("analysis_sample_hz")) != self.analysis_sample_hz:
            raise ValueError("checkpoint sampling rate does not match engine")
        if checkpoint.get("status") not in {"running", "finalized"}:
            raise ValueError(f"checkpoint is not safely restorable: {checkpoint.get('status')}")

        temporal_state = checkpoint.get("temporal_processor_state")
        if temporal_state is not None and self.temporal_processor is None:
            raise ValueError("checkpoint requires a temporal processor")
        self.measurement_processor.restore_state(checkpoint.get("measurement_processor_state") or {})
        if self.temporal_processor is not None:
            self.temporal_processor.restore_state(temporal_state or {})

        self.status = str(checkpoint["status"])
        self.restore_error = checkpoint.get("restore_error")
        self.next_expected_segment_index = int(checkpoint["next_expected_segment_index"])
        self.processed_segments = {
            int(index): str(digest)
            for index, digest in dict(checkpoint.get("processed_segments") or {}).items()
        }
        self.last_source_time_sec = float(checkpoint["last_source_time_sec"])
        self.last_measurement_bucket = int(checkpoint["last_measurement_bucket"])
        self.last_event_source_time_sec = float(checkpoint["last_event_source_time_sec"])
        self.source_frames = int(checkpoint["source_frames"])
        self.measurement_frames = int(checkpoint["measurement_frames"])
        self.event_sequence = int(checkpoint["event_sequence"])
        self.continuity_epoch = int(checkpoint.get("continuity_epoch", 0))
        self.gap_segment_indexes = sorted(
            {int(index) for index in checkpoint.get("gap_segment_indexes", [])}
        )

    def _run_processor(
        self,
        processor: StatefulFrameProcessor,
        frame: Any,
        context: FrameContext,
    ) -> list[dict]:
        return self._materialize_many(processor.process_frame(frame, context), context)

    def _materialize_many(
        self,
        events: Iterable[ProcessorEvent],
        context: Any,
    ) -> list[dict]:
        return [self._materialize_event(event, context) for event in (events or ())]

    def _materialize_event(self, event: ProcessorEvent, context: Any) -> dict:
        if not isinstance(event, ProcessorEvent):
            raise TypeError("processors must emit ProcessorEvent instances")
        source_time_sec = (
            float(event.source_time_sec)
            if event.source_time_sec is not None
            else float(context.source_time_sec if hasattr(context, "source_time_sec") else context.last_source_time_sec)
        )
        segment_index = (
            int(event.segment_index)
            if event.segment_index is not None
            else int(context.segment_index if hasattr(context, "segment_index") else max(0, context.last_segment_index))
        )
        if source_time_sec + 1e-9 < self.last_event_source_time_sec:
            raise ValueError("processor event source time moves backwards")

        event_id_seed = (
            f"{self.analysis_session_id}:{self.event_sequence}:{segment_index}:"
            f"{source_time_sec:.9f}:{event.event_type}"
        )
        event_id = "evt_" + hashlib.sha256(event_id_seed.encode("utf-8")).hexdigest()[:24]
        self.event_sequence += 1
        self.last_event_source_time_sec = source_time_sec
        data = dict(event.data)
        data.setdefault("continuity_epoch", self.continuity_epoch)
        payload = {
            "schema_version": STREAM_SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event.event_type,
            "analysis_session_id": self.analysis_session_id,
            "source_time_sec": source_time_sec,
            "segment_index": segment_index,
            "emitted_at": _utc_now(),
            "confidence": float(event.confidence),
            "evidence_state": event.evidence_state,
            "data": data,
        }
        # Processor adapters must normalize NumPy/tensor values before crossing
        # the event boundary.  Failing here is safer than writing a partially
        # serializable event that task B cannot persist or replay.
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return payload
