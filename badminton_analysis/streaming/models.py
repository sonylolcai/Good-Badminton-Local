"""Data contracts shared by the continuous analysis core and its adapters.

The HTTP/storage layer owns durable receipts.  These models deliberately stop at
the analysis boundary so task B can persist them without importing FastAPI into
the computer-vision package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


STREAM_SCHEMA_VERSION = "stream-session.v1"
ANALYSIS_SAMPLE_RATES = frozenset({10, 15, 30})
EVENT_TYPES = frozenset(
    {
        "person_observation",
        "shuttle_observation",
        "interaction_candidate",
        "session_status",
        "session_finalized",
    }
)
EVIDENCE_STATES = frozenset(
    {"detected", "predicted", "missing", "derived", "candidate", "finalized"}
)


class StreamingAnalysisError(RuntimeError):
    """Base error for deterministic stream-processing failures."""


class SegmentOrderError(StreamingAnalysisError):
    """Raised when the analysis core receives a segment before its predecessor."""


class SegmentConflictError(StreamingAnalysisError):
    """Raised when an already processed index is reused with different bytes."""


class EngineNotRunnableError(StreamingAnalysisError):
    """Raised when processing is attempted after finalization or unsafe recovery."""


class ProcessorExecutionError(StreamingAnalysisError):
    """Raised after a processor failure makes in-memory continuity uncertain."""


@dataclass(frozen=True)
class SegmentDescriptor:
    """Integrity and source-time metadata for one ordered media fragment."""

    segment_index: int
    source_start_time_sec: float
    duration_sec: float
    sha256: str
    idempotency_key: str
    content_type: str = "video/mp4"
    content_length_bytes: int = 1
    schema_version: str = STREAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STREAM_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")
        if self.source_start_time_sec < 0:
            raise ValueError("source_start_time_sec must be non-negative")
        if not 0 < self.duration_sec <= 10:
            raise ValueError("duration_sec must be greater than 0 and no more than 10")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
        if not self.idempotency_key:
            raise ValueError("idempotency_key is required")
        if not self.content_type:
            raise ValueError("content_type is required")
        if self.content_length_bytes <= 0:
            raise ValueError("content_length_bytes must be positive")


@dataclass(frozen=True)
class FramePacket:
    """One decoded source frame with an absolute match timestamp."""

    frame: Any
    source_frame_index: int
    source_time_sec: float
    segment_index: int

    def __post_init__(self) -> None:
        if self.source_frame_index < 0:
            raise ValueError("source_frame_index must be non-negative")
        if self.source_time_sec < 0:
            raise ValueError("source_time_sec must be non-negative")
        if self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")


@dataclass(frozen=True)
class FrameSegment:
    """A one-shot, lazily decoded sequence for one segment descriptor."""

    descriptor: SegmentDescriptor
    frames: Iterable[FramePacket]


@dataclass(frozen=True)
class FrameContext:
    """Stable timing information supplied to stateful frame processors."""

    analysis_session_id: str
    segment_index: int
    source_frame_index: int
    source_time_sec: float
    is_measurement_frame: bool
    measurement_bucket: int


@dataclass(frozen=True)
class FinalizationContext:
    analysis_session_id: str
    last_segment_index: int
    last_source_time_sec: float
    source_frames: int
    measurement_frames: int


@dataclass(frozen=True)
class ProcessorEvent:
    """An unstamped event emitted by a model/tracker processor.

    `source_time_sec` and `segment_index` may be omitted to use the current
    frame/finalization context.  Delayed events may provide explicit values, but
    the engine rejects values that would move the public event timeline backwards.
    """

    event_type: str
    evidence_state: str
    confidence: float
    data: Mapping[str, Any] = field(default_factory=dict)
    source_time_sec: Optional[float] = None
    segment_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {self.event_type}")
        if self.evidence_state not in EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence_state: {self.evidence_state}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.source_time_sec is not None and self.source_time_sec < 0:
            raise ValueError("source_time_sec must be non-negative")
        if self.segment_index is not None and self.segment_index < 0:
            raise ValueError("segment_index must be non-negative")


@dataclass(frozen=True)
class SegmentProcessingResult:
    analysis_session_id: str
    segment_index: int
    reused: bool
    source_frames: int
    measurement_frames: int
    events: tuple[dict, ...]
    checkpoint: dict


@dataclass(frozen=True)
class FinalizationResult:
    analysis_session_id: str
    status: str
    events: tuple[dict, ...]
    checkpoint: dict
