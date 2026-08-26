"""Continuous, segment-oriented analysis primitives.

The package is intentionally transport independent.  `api/stream_*` owns durable
HTTP/session storage, while task F will connect those modules to this engine and
the existing fixed-camera model stack.
"""

from .engine import AnalysisEngine, CHECKPOINT_VERSION
from .models import (
    ANALYSIS_SAMPLE_RATES,
    EVIDENCE_STATES,
    EVENT_TYPES,
    STREAM_SCHEMA_VERSION,
    EngineNotRunnableError,
    FinalizationContext,
    FinalizationResult,
    FrameContext,
    FramePacket,
    FrameSegment,
    ProcessorEvent,
    ProcessorExecutionError,
    SegmentConflictError,
    SegmentDescriptor,
    SegmentOrderError,
    SegmentProcessingResult,
    StreamingAnalysisError,
)
from .processor import StatefulFrameProcessor
from .replay import FileReplayAdapter, OpenCVSegmentDecoder, ReplayDecodeError, ReplayUsageError

__all__ = [
    "ANALYSIS_SAMPLE_RATES",
    "AnalysisEngine",
    "CHECKPOINT_VERSION",
    "EVIDENCE_STATES",
    "EVENT_TYPES",
    "EngineNotRunnableError",
    "FileReplayAdapter",
    "FinalizationContext",
    "FinalizationResult",
    "FrameContext",
    "FramePacket",
    "FrameSegment",
    "OpenCVSegmentDecoder",
    "ProcessorEvent",
    "ProcessorExecutionError",
    "ReplayDecodeError",
    "ReplayUsageError",
    "STREAM_SCHEMA_VERSION",
    "SegmentConflictError",
    "SegmentDescriptor",
    "SegmentOrderError",
    "SegmentProcessingResult",
    "StatefulFrameProcessor",
    "StreamingAnalysisError",
]
