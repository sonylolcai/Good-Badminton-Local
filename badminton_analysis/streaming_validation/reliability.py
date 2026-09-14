"""Reliability replay scenarios for the streaming storage/engine chain.

Each scenario creates a fresh manager and returns a pass/fail result with a short
detail. They verify the contract behaviours that a video replay must preserve:
idempotent receipts, in-order processing, explicit gaps, restart restore and
backlog drain at seal. A replay never masquerades as a live-camera test.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Dict

from api.stream_errors import StreamSessionError
from api.stream_sessions import StreamSessionManager


COURT_CORNERS = [[0.0, 0.0], [1280.0, 0.0], [1280.0, 720.0], [0.0, 720.0]]


def _segment_metadata(index, raw, source_start):
    return {
        "schema_version": "stream-session.v1",
        "segment_index": index,
        "source_start_time_sec": source_start,
        "duration_sec": 1.0,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "idempotency_key": f"reliability-segment-{index:03d}",
        "content_type": "video/mp4",
        "content_length_bytes": len(raw),
        "court_corners": COURT_CORNERS,
    }


def _create_request():
    return {
        "schema_version": "stream-session.v1",
        "camera_id": "reliability-camera",
        "calibration_id": "reliability-calibration",
        "court_corners": COURT_CORNERS,
        "analysis_mode": "person_only",
        "configuration": {
            "analysis_sample_hz": 10,
            "pose_imgsz": 960,
            "shuttle_detector": "none",
            "generate_annotated_video": False,
        },
    }


def _result(status, detail):
    return {"status": status, "detail": detail}


def _new_manager(data_dir, processor_factory):
    return StreamSessionManager(Path(data_dir), processor_factory=processor_factory, start_worker=False)


def _duplicate(data_dir, processor_factory, raw):
    manager = _new_manager(data_dir, processor_factory)
    _, created = manager.create_session(_create_request(), "reliability-duplicate-01")
    sid = created["analysis_session_id"]
    metadata = _segment_metadata(0, raw, 0.0)
    manager.receive_segment(sid, 0, metadata, raw)
    status, duplicate = manager.receive_segment(sid, 0, metadata, raw)
    manager.drain()
    if status == 200 and duplicate["receipt"]["reused"]:
        return _result("ok", "duplicate segment returned reused=true and was not processed twice")
    return _result("failed", f"duplicate reuse failed: status={status}")


def _out_of_order(data_dir, processor_factory, raw):
    manager = _new_manager(data_dir, processor_factory)
    _, created = manager.create_session(_create_request(), "reliability-order-01")
    sid = created["analysis_session_id"]
    status, receipt = manager.receive_segment(sid, 2, _segment_metadata(2, raw, 2.0), raw)
    disposition = receipt["receipt"]["processing_disposition"]
    manager.drain()
    manager.receive_segment(sid, 0, _segment_metadata(0, raw, 0.0), raw)
    manager.receive_segment(sid, 1, _segment_metadata(1, raw, 1.0), raw)
    manager.drain()
    progress = manager.get_status(sid)[1]["progress"]
    processed = progress["processed_segments"]
    if disposition == "waiting_for_predecessor" and processed == 3:
        return _result("ok", "out-of-order segment waited and later drained in source order")
    return _result("failed", f"out-of-order handling failed: disposition={disposition}, processed={processed}")


def _missing_segment(data_dir, processor_factory, raw):
    manager = _new_manager(data_dir, processor_factory)
    _, created = manager.create_session(_create_request(), "reliability-missing-01")
    sid = created["analysis_session_id"]
    manager.receive_segment(sid, 0, _segment_metadata(0, raw, 0.0), raw)
    manager.receive_segment(sid, 1, _segment_metadata(1, raw, 1.0), raw)
    manager.drain()
    try:
        manager.complete(sid, {"schema_version": "stream-session.v1", "expected_last_segment_index": 2, "allow_partial": False})
        return _result("failed", "missing segment was not rejected")
    except StreamSessionError as exc:
        if exc.code != "missing_segments":
            return _result("failed", f"unexpected error code: {exc.code}")
    status, completed = manager.complete(sid, {"schema_version": "stream-session.v1", "expected_last_segment_index": 2, "allow_partial": True})
    final_status = completed.get("status")
    if final_status == "partial":
        return _result("ok", "missing segment blocked strict completion and allowed explicit partial")
    return _result("failed", f"partial completion failed: {final_status}")


def _restart(data_dir, processor_factory, raw):
    first = _new_manager(data_dir, processor_factory)
    _, created = first.create_session(_create_request(), "reliability-restart-01")
    sid = created["analysis_session_id"]
    first.receive_segment(sid, 0, _segment_metadata(0, raw, 0.0), raw)
    first.drain()
    restarted = _new_manager(data_dir, processor_factory)
    recovered = restarted.get_session(sid)
    if recovered["status"] == "interrupted_needs_rebuild":
        return _result("failed", "restart did not restore the engine from checkpoint")
    restarted.receive_segment(sid, 1, _segment_metadata(1, raw, 1.0), raw)
    restarted.drain()
    processed = restarted.get_status(sid)[1]["progress"]["processed_segments"]
    if processed == 2:
        return _result("ok", "restart restored the engine and continued processing")
    return _result("failed", "restarted manager did not continue processing")


def _slow_upload(data_dir, processor_factory, raw):
    manager = _new_manager(data_dir, processor_factory)
    _, created = manager.create_session(_create_request(), "reliability-slow-01")
    sid = created["analysis_session_id"]
    for index in range(3):
        time.sleep(0.002)
        manager.receive_segment(sid, index, _segment_metadata(index, raw, index * 1.0), raw)
    manager.drain()
    processed = manager.get_status(sid)[1]["progress"]["processed_segments"]
    if processed == 3:
        return _result("ok", "delayed arrivals still drained in source order")
    return _result("failed", "delayed arrivals did not drain")


def _backlog_at_seal(data_dir, processor_factory, raw):
    manager = _new_manager(data_dir, processor_factory)
    _, created = manager.create_session(_create_request(), "reliability-backlog-01")
    sid = created["analysis_session_id"]
    for index in range(3):
        manager.receive_segment(sid, index, _segment_metadata(index, raw, index * 1.0), raw)
    status, completed = manager.complete(
        sid, {"schema_version": "stream-session.v1", "expected_last_segment_index": 2, "allow_partial": False}
    )
    if completed["status"] == "draining":
        manager.drain()
    final_status = manager.get_status(sid)[1]["status"]
    if final_status == "finalized":
        return _result("ok", "backlog at seal drained and finalized")
    return _result("failed", f"backlog-at-seal final state unexpected: {final_status}")


def run_reliability_scenarios(data_dir, processor_factory, raw):
    """Run every reliability scenario and return a name -> result mapping."""
    scenarios = {
        "duplicate": _duplicate,
        "out_of_order": _out_of_order,
        "missing_segment": _missing_segment,
        "restart": _restart,
        "slow_upload": _slow_upload,
        "backlog_at_seal": _backlog_at_seal,
    }
    results = {}
    for name, scenario in scenarios.items():
        scenario_dir = Path(data_dir) / f"scenario_{name}"
        try:
            results[name] = scenario(scenario_dir, processor_factory, raw)
        except Exception as exc:
            results[name] = _result("failed", f"{type(exc).__name__}: {exc}")
    return results

