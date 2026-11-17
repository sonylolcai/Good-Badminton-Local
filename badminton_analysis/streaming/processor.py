"""Protocols for model/tracker implementations used by the streaming engine."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol

from .models import FinalizationContext, FrameContext, ProcessorEvent


class StatefulFrameProcessor(Protocol):
    """A processor whose identity/tracker state survives segment boundaries.

    The measurement processor is called only at the configured 10/15/30 Hz
    buckets.  A separately injected temporal processor, such as a future
    TrackNet adapter, is called for every decoded source frame.  Keeping these
    roles separate prevents an optional temporal model from silently changing
    the baseline person-analysis cadence.
    """

    def process_frame(self, frame: Any, context: FrameContext) -> Iterable[ProcessorEvent]:
        ...

    def finalize(self, context: FinalizationContext) -> Iterable[ProcessorEvent]:
        ...

    def snapshot_state(self) -> Mapping[str, Any]:
        ...

    def restore_state(self, state: Mapping[str, Any]) -> None:
        ...
