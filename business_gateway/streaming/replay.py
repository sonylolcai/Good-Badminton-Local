"""Development replay entry point using the production stream client contract."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .client import StreamSessionClient
from .models import SegmentMetadata
from .segmenter import GrowingVideoSegmenter


def replay_video(
    video_path: Path,
    *,
    client: StreamSessionClient,
    segmenter: GrowingVideoSegmenter,
    create_request: dict[str, Any],
    create_idempotency_key: str,
    realtime: bool = False,
    overwrite_segments: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay a complete file as incrementally delivered, independently decodable pieces."""

    accepted_started = time.perf_counter()
    accepted_wall = _utc_now()
    create_response = client.create_session(
        create_request,
        idempotency_key=create_idempotency_key,
    )
    _emit_progress(
        progress_callback,
        {
            "stage": "gpu_session_accepted",
            "analysis_session_id": create_response["analysis_session_id"],
            "create_response": create_response,
        },
    )
    create_elapsed = time.perf_counter() - accepted_started
    last_index = -1
    segmentation_elapsed = 0.0
    upload_elapsed = 0.0
    uploaded_bytes = 0
    uploaded_segments = 0
    iterator = iter(segmenter.iter_input(
        str(Path(video_path)),
        realtime=realtime,
        overwrite=overwrite_segments,
    ))
    while True:
        segment_started = time.perf_counter()
        try:
            artifact = next(iterator)
        except StopIteration:
            segmentation_elapsed += time.perf_counter() - segment_started
            break
        segmentation_elapsed += time.perf_counter() - segment_started
        metadata = SegmentMetadata.from_file(
            artifact.path,
            segment_index=artifact.segment_index,
            source_start_time_sec=artifact.source_start_time_sec,
            duration_sec=artifact.duration_sec,
            idempotency_prefix=create_idempotency_key,
            content_type=artifact.content_type,
            court_corners=create_request["court_corners"],
        )
        upload_started = time.perf_counter()
        receipt = client.submit_segment(artifact.path, metadata)
        upload_elapsed += time.perf_counter() - upload_started
        uploaded_bytes += metadata.content_length_bytes
        uploaded_segments += 1
        last_index = artifact.segment_index
        _emit_progress(
            progress_callback,
            {
                "stage": "segment_accepted",
                "analysis_session_id": create_response["analysis_session_id"],
                "segment_index": artifact.segment_index,
                "uploaded_segments": uploaded_segments,
                "uploaded_bytes": uploaded_bytes,
                "receipt": receipt,
            },
        )
    if last_index < 0:
        raise RuntimeError("video replay produced no segments")

    seal_started = time.perf_counter()
    completion = client.complete(last_index, allow_partial=False)
    seal_elapsed = time.perf_counter() - seal_started
    _emit_progress(
        progress_callback,
        {
            "stage": "session_sealed",
            "analysis_session_id": create_response["analysis_session_id"],
            "expected_last_segment_index": last_index,
            "completion": completion,
        },
    )

    result_started = time.perf_counter()
    status = client.get_status()
    result_return_elapsed = time.perf_counter() - result_started
    _emit_progress(
        progress_callback,
        {
            "stage": "gpu_status_after_seal",
            "analysis_session_id": create_response["analysis_session_id"],
            "gpu_status": status,
        },
    )
    gpu_trace: dict[str, Any] | None = None
    trace_error: str | None = None
    try:
        gpu_trace = client.get_trace()
    except Exception as exc:
        # A trace fetch must never change delivery semantics.  Persist the
        # missing evidence as an explicit error so the performance gate fails
        # closed instead of pretending the trace exists.
        trace_error = f"{type(exc).__name__}: {exc}"

    trace_path = client.ledger.path.parent / "end_to_end_trace.json"
    _write_end_to_end_trace(
        trace_path,
        analysis_session_id=create_response["analysis_session_id"],
        accepted_at=accepted_wall,
        finished_at=_utc_now(),
        realtime=realtime,
        status=status,
        gpu_trace=gpu_trace,
        gpu_trace_error=trace_error,
        stages={
            "create_session": create_elapsed,
            "segment_or_wait_for_fragment": segmentation_elapsed,
            "segment_upload": upload_elapsed,
            "seal_session": seal_elapsed,
            "result_status_return": result_return_elapsed,
        },
        uploaded_segments=uploaded_segments,
        uploaded_bytes=uploaded_bytes,
    )
    return {
        "analysis_session_id": create_response["analysis_session_id"],
        "last_segment_index": last_index,
        "completion": completion,
        "status": status,
        "end_to_end_trace": str(trace_path),
        "ledger": client.ledger.snapshot(),
    }


def _emit_progress(callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    """Persisting observers are best-effort and never change delivery semantics."""

    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # The receipt ledger remains the authoritative transport record.  A
        # dashboard write error must not cause a duplicate segment submission.
        return


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_end_to_end_trace(
    path: Path,
    *,
    analysis_session_id: str,
    accepted_at: str,
    finished_at: str,
    realtime: bool,
    status: dict[str, Any],
    gpu_trace: dict[str, Any] | None,
    gpu_trace_error: str | None,
    stages: dict[str, float],
    uploaded_segments: int,
    uploaded_bytes: int,
) -> None:
    measured = {
        name: {"elapsed_seconds": round(max(0.0, float(elapsed)), 6), "calls": 1}
        for name, elapsed in stages.items()
    }
    payload = {
        "schema_version": "1.0",
        "kind": "good_badminton_business_gpu_end_to_end_trace",
        "analysis_session_id": analysis_session_id,
        "accepted_at": accepted_at,
        "finished_at": finished_at,
        "input_kind": "realtime_file_replay" if realtime else "development_file_replay",
        "streaming_slo_proven": False,
        "terminal_status_at_trace_fetch": status.get("status"),
        "business_stages": measured,
        "delivery": {
            "uploaded_segments": int(uploaded_segments),
            "uploaded_bytes": int(uploaded_bytes),
        },
        "execution_topology": {
            "business_segment_production_and_upload": "serial_in_file_replay_adapter",
            "business_upload_vs_gpu_processing": "may_overlap_across_segments",
            "gpu_execution": "see_gpu_trace_execution_topology",
            "llm": "not_part_of_video_analysis_service",
        },
        "gpu_trace": gpu_trace,
        "gpu_trace_error": gpu_trace_error,
        "notes": [
            "A local file replay cannot prove a live-camera streaming SLO.",
            "streaming_slo_proven remains false until a real camera-to-business-to-GPU acceptance run passes the production gate.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
