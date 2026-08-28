"""Persistent, single-GPU stream sessions wired to task A's AnalysisEngine.

This module persists session/receipt/checkpoint state and connects the accepted
segments to the continuous analysis core: decode with OpenCVSegmentDecoder, run
the per-session AnalysisEngine, persist its events and checkpoint, and restore
the engine from the checkpoint after a restart.  Route registration on api.app
is deliberately left to task F; the handler functions at the bottom are thin,
route-ready wrappers.

Task F supplies the real StatefulFrameProcessor factory; task B owns the engine
lifecycle, decode, checkpoint atomicity, restore flow and event persistence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import queue
import re
import shutil
import threading
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from badminton_analysis.streaming import (
    AnalysisEngine,
    OpenCVSegmentDecoder,
    ReplayDecodeError,
    SegmentDescriptor,
    StreamingAnalysisError,
)

from .stream_errors import (
    StreamSessionError,
    invalid_state,
    session_not_found,
    validation_error,
)
from .stream_models import (
    SCHEMA_VERSION,
    IDEMPOTENCY_KEY_PATTERN,
    MAX_SEGMENT_BYTES,
    TERMINAL_SESSION_STATES,
    complete_session_response,
    create_session_response,
    events_response,
    segment_receipt_response,
    session_status_response,
    utc_now,
    validate_complete_request,
    validate_create_request,
    validate_segment_metadata,
    new_session_id,
)


class StreamSessionManager:
    """Store sessions on disk and process one GPU segment at a time.

    Sessions live under api_data/stream_sessions, separate from the legacy
    complete-file jobs directory.  Each analysis_session_id owns exactly one
    AnalysisEngine instance (cached in memory); engines are never shared across
    sessions.  The engine's checkpoint is written atomically after every accepted
    result and restored on restart so track/event identity survives a process
    boundary; restore failure is reported as interrupted_needs_rebuild, never as
    a silent new track identity.
    """

    def __init__(
        self,
        data_dir,
        *,
        processor_factory=None,
        engine_factory=None,
        decoder=None,
        retention_hours=None,
        segment_timeout_seconds=None,
        hard_timeout_handler=None,
        start_worker=True,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.sessions_dir = self.data_dir / "stream_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.processor_factory = processor_factory
        self.engine_factory = engine_factory
        self.decoder = decoder if decoder is not None else OpenCVSegmentDecoder()
        configured_retention = (
            retention_hours
            if retention_hours is not None
            else os.environ.get("GOOD_BADMINTON_STREAM_RETENTION_HOURS", "72")
        )
        self.retention_hours = float(configured_retention)
        if self.retention_hours <= 0:
            raise ValueError("stream retention hours must be positive")
        configured_timeout = (
            segment_timeout_seconds
            if segment_timeout_seconds is not None
            else os.environ.get("GOOD_BADMINTON_STREAM_SEGMENT_TIMEOUT_SECONDS", "10")
        )
        self.segment_timeout_seconds = float(configured_timeout)
        if self.segment_timeout_seconds <= 0:
            raise ValueError("stream segment timeout must be positive")
        self.hard_timeout_handler = hard_timeout_handler
        self._engines = {}
        self._event_ids = {}
        # Cancellation cannot always interrupt a model call already executing
        # inside a third-party runtime.  It must, however, win the commit race:
        # no late result may resurrect a cancelled session or become evidence.
        self._cancelled_sessions = set()
        self._queue = queue.Queue()
        self._lock = threading.RLock()
        self._worker = None
        self._watchdog = None
        self._active_segment = None
        self._timed_out_segments = set()
        self._recover_sessions()
        if start_worker:
            self.start()

    # -- paths ----------------------------------------------------------
    def _session_dir(self, session_id):
        return self.sessions_dir / session_id

    def _manifest_path(self, session_id):
        return self._session_dir(session_id) / "manifest.json"

    def _segment_path(self, session_id, index):
        return self._session_dir(session_id) / "segments" / f"{int(index):06d}.bin"

    def _events_path(self, session_id):
        return self._session_dir(session_id) / "events.jsonl"

    def _checkpoint_path(self, session_id):
        return self._session_dir(session_id) / "checkpoint.json"

    def trace_path(self, session_id):
        path = self._session_dir(session_id) / "end_to_end_trace.json"
        return path if path.is_file() else None

    def candidate_photo_path(self, session_id, track_id):
        """Return one protected anonymous full-body crop, if it exists."""
        if self._load(session_id) is None:
            return None
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(track_id or "")):
            return None
        path = (self._session_dir(session_id) / "candidate_photos" / f"{track_id}.jpg").resolve()
        parent = (self._session_dir(session_id) / "candidate_photos").resolve()
        if path.parent != parent or not path.is_file():
            return None
        return path

    # -- persistence ----------------------------------------------------
    def _load(self, session_id):
        path = self._manifest_path(session_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, session):
        session["updated_at"] = utc_now()
        path = self._manifest_path(session["analysis_session_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps(session, ensure_ascii=False, indent=2) + "\n"
        with self._lock:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)

    def _load_checkpoint(self, session_id):
        path = self._checkpoint_path(session_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_checkpoint(self, session_id, checkpoint):
        """Atomically persist the engine checkpoint after an accepted result."""
        path = self._checkpoint_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n"
        with self._lock:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)

    def _load_trace(self, session_id):
        path = self.trace_path(session_id)
        if path is None:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_trace(self, session_id, trace):
        path = self._session_dir(session_id) / "end_to_end_trace.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        # The receive thread, serial worker, and timeout watchdog can all
        # update timing evidence.  Keep each replacement atomic and mutually
        # exclusive so a watchdog failure cannot corrupt the trace file.
        with self._lock:
            trace["updated_at"] = utc_now()
            temporary.write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)

    def _initialize_trace(self, session):
        configuration = dict(session.get("configuration") or {})
        trace = {
            "schema_version": "1.0",
            "kind": "good_badminton_stream_end_to_end_trace",
            "analysis_session_id": session["analysis_session_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "terminal_status": None,
            "streaming_slo_proven": False,
            "configuration": configuration,
            "execution_topology": {
                "current_execution_model": "single_gpu_worker_serial_segments",
                "segments": "serialized_on_one_gpu_worker",
                "within_segment": "decode_sample_models_tracking_are_currently_serial",
                "client_upload_vs_gpu_processing": "may_overlap_for_different_segments",
                "llm": "business_service_only_not_in_gpu_trace",
            },
            "segments": {},
            "stages": {
                "gpu_receive": {"elapsed_seconds": 0.0, "calls": 0},
                "queue_wait": {"elapsed_seconds": 0.0, "calls": 0},
                "segment_open": {"elapsed_seconds": 0.0, "calls": 0},
                "decode_sample_models_tracking": {"elapsed_seconds": 0.0, "calls": 0},
                "event_checkpoint_write": {"elapsed_seconds": 0.0, "calls": 0},
                "finalize": {"elapsed_seconds": 0.0, "calls": 0},
            },
            "stage_attribution": {
                "decode": "included_in_decode_sample_models_tracking",
                "sample": "included_in_decode_sample_models_tracking",
                "pose_inference": "included_in_decode_sample_models_tracking",
                "ball_inference": (
                    "disabled" if configuration.get("shuttle_detector") == "none"
                    else "included_in_decode_sample_models_tracking"
                ),
                "tracking": "included_in_decode_sample_models_tracking",
                "event_write": "measured_as_event_checkpoint_write",
                "annotate_encode": "disabled_by_stream_contract",
                "result_return": "business_client_transport; not measurable on GPU filesystem",
            },
            "notes": [
                "This trace does not claim live-camera SLO proof.",
                "Fine-grained model attribution requires the real GPU profiler gate; combined time is measured honestly rather than split by estimation.",
            ],
        }
        self._save_trace(session["analysis_session_id"], trace)

    @staticmethod
    def _seconds_between(started_at, finished_at):
        try:
            return max(
                0.0,
                (datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                 - datetime.fromisoformat(started_at.replace("Z", "+00:00"))).total_seconds(),
            )
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _trace_stage(self, trace, name, elapsed_seconds):
        stage = trace["stages"][name]
        stage["elapsed_seconds"] = round(
            float(stage.get("elapsed_seconds", 0.0)) + max(0.0, float(elapsed_seconds)),
            6,
        )
        stage["calls"] = int(stage.get("calls", 0)) + 1

    def _finish_trace(self, session, terminal_status):
        trace = self._load_trace(session["analysis_session_id"])
        if trace is None:
            return
        trace["terminal_status"] = terminal_status
        trace["finished_at"] = utc_now()
        trace["summary"] = {
            "received_segments": self._received_count(session),
            "processed_segments": self._processed_count(session),
            "source_duration_seconds": self._received_duration(session),
            "wall_seconds_since_accept": round(
                self._seconds_between(session["created_at"], trace["finished_at"]),
                6,
            ),
        }
        self._save_trace(session["analysis_session_id"], trace)

    def _list_sessions(self):
        sessions = []
        for path in self.sessions_dir.glob("*/manifest.json"):
            try:
                sessions.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sessions

    def cleanup_expired_terminal_sessions(self, *, dry_run=True, now=None):
        """List or remove only terminal session directories older than the TTL.

        Cleanup is never triggered implicitly by an API request. Operations can
        run this method from a scheduled maintenance command with ``dry_run``
        first, avoiding deletion of active/backlogged evidence. Returned paths
        are resolved and verified as direct children of ``stream_sessions``.
        """

        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        cutoff = current - timedelta(hours=self.retention_hours)
        candidates = []
        for session in self._list_sessions():
            if session.get("status") not in TERMINAL_SESSION_STATES:
                continue
            try:
                updated = datetime.fromisoformat(
                    str(session.get("updated_at") or "").replace("Z", "+00:00")
                )
            except ValueError:
                continue
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            if updated > cutoff:
                continue
            session_id = str(session.get("analysis_session_id") or "")
            directory = self._session_dir(session_id).resolve()
            if directory.parent != self.sessions_dir.resolve() or not directory.is_dir():
                continue
            candidates.append(
                {
                    "analysis_session_id": session_id,
                    "status": session["status"],
                    "updated_at": session.get("updated_at"),
                    "path": str(directory),
                    "deleted": False,
                }
            )
        if not dry_run:
            for candidate in candidates:
                shutil.rmtree(candidate["path"])
                candidate["deleted"] = True
        return candidates

    def _find_by_idempotency_key(self, key):
        if not key:
            return None
        for path in self.sessions_dir.glob("*/manifest.json"):
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (session.get("request") or {}).get("idempotency_key") == key:
                return session
        return None

    # -- derived state --------------------------------------------------
    @staticmethod
    def _received_indexes(session):
        return {int(k) for k in session.get("segments", {})}

    def _received_count(self, session):
        return len(session.get("segments", {}))

    def _processed_count(self, session):
        return sum(1 for entry in session.get("segments", {}).values() if entry.get("processed"))

    def _failed_indexes(self, session):
        """Return accepted fragments whose evidence could not be processed.

        A receipt proves the GPU owns the bytes, not that inference succeeded.
        These indexes are therefore intentionally separate from transport-level
        ``missing_segment_indexes``.
        """
        return sorted(
            int(index)
            for index, entry in session.get("segments", {}).items()
            if entry.get("processing_failed")
        )

    def _failed_count(self, session):
        return len(self._failed_indexes(session))

    @staticmethod
    def _entry_settled(entry):
        return bool(entry.get("processed") or entry.get("processing_failed"))

    def _received_duration(self, session):
        return sum(float(entry.get("duration_sec", 0.0)) for entry in session.get("segments", {}).values())

    def _processed_duration(self, session):
        return sum(
            float(entry.get("duration_sec", 0.0))
            for entry in session.get("segments", {}).values()
            if entry.get("processed")
        )

    def _failed_duration(self, session):
        return sum(
            float(entry.get("duration_sec", 0.0))
            for entry in session.get("segments", {}).values()
            if entry.get("processing_failed")
        )

    def _missing_indexes(self, session):
        received = self._received_indexes(session)
        expected_last = session.get("expected_last_segment_index")
        if expected_last is not None:
            upper = int(expected_last)
        else:
            upper = max(received) if received else -1
        return sorted(index for index in range(0, upper + 1) if index not in received)

    def _next_expected_index(self, session):
        index = 0
        while True:
            entry = session.get("segments", {}).get(str(index))
            if entry is None or not self._entry_settled(entry):
                return index
            index += 1
        return index

    def _compute_progress(self, session):
        received = self._received_count(session)
        processed = self._processed_count(session)
        failed = self._failed_count(session)
        received_duration = self._received_duration(session)
        processed_duration = self._processed_duration(session)
        failed_duration = self._failed_duration(session)
        return {
            "received_segments": received,
            "processed_segments": processed,
            "failed_segments": failed,
            "received_duration_sec": round(received_duration, 6),
            "processed_source_time_sec": round(processed_duration, 6),
            "failed_source_time_sec": round(failed_duration, 6),
            "backlog_duration_sec": round(
                max(0.0, received_duration - processed_duration - failed_duration), 6
            ),
            "next_expected_segment_index": self._next_expected_index(session),
            "missing_segment_indexes": self._missing_indexes(session),
            "failed_segment_indexes": self._failed_indexes(session),
        }

    def _derive_status(self, session):
        if session.get("status") in TERMINAL_SESSION_STATES:
            return session["status"]
        if session.get("sealed"):
            return "draining"
        if self._processed_count(session) > 0 or self._failed_count(session) > 0:
            return "running"
        if self._received_count(session) > 0:
            return "queued"
        return "accepted"

    @staticmethod
    def _derive_stage(session, progress):
        status = session.get("status")
        if status == "failed" or status == "interrupted_needs_rebuild":
            return "failed"
        if status == "cancelled":
            return "cancelled"
        if status in {"finalized", "partial"}:
            return "complete"
        has_processing_errors = bool(progress["failed_segment_indexes"])
        if session.get("sealed"):
            return "draining_with_errors" if has_processing_errors else "draining_backlog"
        if progress["missing_segment_indexes"]:
            return "waiting_for_predecessor"
        if progress["received_segments"] > 0:
            return "analyzing_with_errors" if has_processing_errors else "analyzing"
        return "awaiting_segments"

    def _status_response(self, session):
        progress = self._compute_progress(session)
        track_candidates, artifacts = self._project_status_evidence(
            session["analysis_session_id"]
        )
        return session_status_response(
            session,
            progress,
            track_candidates,
            artifacts,
            self._derive_stage(session, progress),
            session.get("error"),
            session.get("processing_errors"),
        )

    def _project_status_evidence(self, session_id):
        """Build the frozen status projection from durable public events.

        The manager does not reach into processor private state. This keeps
        status reconstruction valid after restart and makes events.jsonl the
        auditable source for both live candidates and final artifacts.
        """
        candidates = {}
        artifacts = []
        events = sorted(
            self._load_events(session_id),
            key=lambda event: (
                float(event.get("source_time_sec", 0.0)),
                str(event.get("event_id") or ""),
            ),
        )
        for event in events:
            data = event.get("data") or {}
            for candidate in data.get("track_candidates") or []:
                if candidate.get("track_id"):
                    candidates[str(candidate["track_id"])] = dict(candidate)

            if event.get("event_type") == "person_observation":
                track = data.get("track") or {}
                track_id = track.get("track_id")
                if track_id:
                    current = candidates.get(str(track_id), {})
                    quality = track.get("quality") or {}
                    confidence = max(0.0, min(1.0, float(track.get("confidence") or 0.0)))
                    coverage = max(
                        0.0,
                        min(1.0, float(quality.get("coverage_rate") or 0.0)),
                    )
                    source_time = max(0.0, float(event.get("source_time_sec") or 0.0))
                    candidate = {
                        "track_id": str(track_id),
                        "state": str(track.get("lifecycle_state") or "candidate"),
                        "first_source_time_sec": float(
                            current.get("first_source_time_sec", source_time)
                        ),
                        "last_source_time_sec": source_time,
                        "detected_coverage": coverage,
                        "confidence": min(confidence, coverage),
                    }
                    photo = data.get("candidate_photo")
                    if isinstance(photo, dict) and str(photo.get("track_id") or "") == str(track_id):
                        candidate["candidate_photo"] = {
                            "source_time_sec": float(photo.get("source_time_sec") or source_time),
                            "capture_quality": float(photo.get("capture_quality") or 0.0),
                            "frontal_score": float(photo.get("frontal_score") or 0.0),
                            "view_label": str(photo.get("view_label") or "not_assessed"),
                            "selection_policy": str(photo.get("selection_policy") or "quality_only_v1"),
                            "media_type": "image/jpeg",
                            "fetch_path": (
                                f"/api/v1/stream-sessions/{session_id}/candidate-photos/{track_id}"
                            ),
                        }
                    elif current.get("candidate_photo"):
                        candidate["candidate_photo"] = dict(current["candidate_photo"])
                    candidates[str(track_id)] = candidate

            if event.get("event_type") == "session_finalized":
                received_artifacts = data.get("artifacts")
                if isinstance(received_artifacts, list):
                    artifacts = [dict(item) for item in received_artifacts]
        return [candidates[key] for key in sorted(candidates)], artifacts

    # -- engine lifecycle ----------------------------------------------
    @staticmethod
    def _factory_accepts_session(factory):
        """Return whether a factory supports the task-F session-aware seam."""
        try:
            parameters = inspect.signature(factory).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(
            parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            }
            for parameter in parameters
        )

    def _make_processors(self, session):
        """Return (measurement_processor, temporal_processor) from the factory."""
        if self.processor_factory is None:
            return None, None
        produced = (
            self.processor_factory(session)
            if self._factory_accepts_session(self.processor_factory)
            else self.processor_factory()
        )
        if produced is None:
            return None, None
        if isinstance(produced, tuple):
            measurement = produced[0]
            temporal = produced[1] if len(produced) > 1 else None
        else:
            measurement = produced
            temporal = None
        return measurement, temporal

    def _create_engine(self, session):
        measurement, temporal = self._make_processors(session)
        if self.engine_factory is not None:
            return self.engine_factory(session, measurement, temporal)
        return AnalysisEngine(
            analysis_session_id=session["analysis_session_id"],
            analysis_sample_hz=int(session["configuration"]["analysis_sample_hz"]),
            measurement_processor=measurement,
            temporal_processor=temporal,
        )

    def _restore_engine(self, session, checkpoint):
        measurement, temporal = self._make_processors(session)
        if measurement is None:
            raise RuntimeError("processor_factory is not configured; cannot restore engine state")
        if self.engine_factory is not None:
            try:
                parameters = inspect.signature(self.engine_factory).parameters.values()
                positional = [
                    parameter
                    for parameter in parameters
                    if parameter.kind in {
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    }
                ]
                accepts_checkpoint = any(
                    parameter.kind == inspect.Parameter.VAR_POSITIONAL
                    for parameter in parameters
                ) or len(positional) >= 4
            except (TypeError, ValueError):
                accepts_checkpoint = True
            if not accepts_checkpoint:
                raise RuntimeError(
                    "engine_factory must accept (session, measurement, temporal, checkpoint) "
                    "to support restart recovery"
                )
            return self.engine_factory(session, measurement, temporal, checkpoint)
        return AnalysisEngine.restore(checkpoint, measurement, temporal_processor=temporal)

    def _get_engine(self, session):
        """Return the session's exclusive engine, creating it on first use."""
        session_id = session["analysis_session_id"]
        engine = self._engines.get(session_id)
        if engine is None:
            if self.processor_factory is None:
                return None
            engine = self._create_engine(session)
            self._engines[session_id] = engine
        return engine

    @staticmethod
    def _descriptor_from_entry(entry):
        return SegmentDescriptor(
            segment_index=entry["segment_index"],
            source_start_time_sec=entry["source_start_time_sec"],
            duration_sec=entry["duration_sec"],
            sha256=entry["sha256"],
            idempotency_key=entry["idempotency_key"],
            content_type=entry["content_type"],
            content_length_bytes=entry["content_length_bytes"],
        )

    # -- events ---------------------------------------------------------
    def _seen_event_ids(self, session_id):
        seen = self._event_ids.get(session_id)
        if seen is not None:
            return seen
        seen = set()
        for event in self._load_events(session_id):
            if event.get("event_id"):
                seen.add(event["event_id"])
        self._event_ids[session_id] = seen
        return seen

    def _append_event(self, session, event_type, source_time_sec, segment_index, confidence, evidence_state, data):
        """Append a storage-owned session_status/finalized event."""
        session["event_seq"] = int(session.get("event_seq", 0)) + 1
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": f"evt_{session['analysis_session_id']}_{session['event_seq']:06d}",
            "event_type": event_type,
            "analysis_session_id": session["analysis_session_id"],
            "source_time_sec": float(source_time_sec),
            "segment_index": int(segment_index),
            "emitted_at": utc_now(),
            "confidence": float(confidence),
            "evidence_state": evidence_state,
            "data": dict(data or {}),
        }
        self._write_events(session["analysis_session_id"], [event])
        return event

    @staticmethod
    def _epoch_track_id(track_id, continuity_epoch):
        """Make a new tracker epoch impossible to confuse with the old one."""
        if int(continuity_epoch) <= 0:
            return str(track_id)
        raw = str(track_id)
        suffix = raw[6:] if raw.startswith("track_") else raw
        return f"track_epoch{int(continuity_epoch)}_{suffix}"

    def _tag_epoch_events(self, events, continuity_epoch):
        """Keep post-gap evidence usable without inventing cross-gap identity."""
        tagged = []
        for raw_event in events or ():
            event = deepcopy(raw_event)
            data = event.setdefault("data", {})
            data["continuity_epoch"] = int(continuity_epoch)
            if int(continuity_epoch) > 0:
                track = data.get("track")
                if isinstance(track, dict) and track.get("track_id"):
                    local_track_id = str(track["track_id"])
                    track["tracker_local_id"] = local_track_id
                    track["track_id"] = self._epoch_track_id(
                        local_track_id, continuity_epoch
                    )
                for key in ("track_candidates", "track_profiles"):
                    for candidate in data.get(key) or ():
                        if isinstance(candidate, dict) and candidate.get("track_id"):
                            local_track_id = str(candidate["track_id"])
                            candidate["tracker_local_id"] = local_track_id
                            candidate["track_id"] = self._epoch_track_id(
                                local_track_id, continuity_epoch
                            )
            tagged.append(event)
        return tagged

    def _append_engine_events(self, session_id, events, *, continuity_epoch=0):
        """Write deterministic engine events with an explicit continuity epoch."""
        self._write_events(
            session_id,
            self._tag_epoch_events(events, continuity_epoch),
        )

    def _write_events(self, session_id, events):
        if not events:
            return
        seen = self._seen_event_ids(session_id)
        path = self._events_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for event in events:
                event_id = event.get("event_id")
                if not event_id or event_id in seen:
                    continue
                seen.add(event_id)
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _emit_status_event(self, session, status, source_time_sec=0.0, segment_index=0, extra=None):
        data = {"status": status}
        if extra:
            data.update(extra)
        self._append_event(session, "session_status", source_time_sec, segment_index, 1.0, "derived", data)

    def _append_state_history(self, session, status, event):
        session.setdefault("state_history", []).append({
            "at": utc_now(), "status": status, "event": event,
        })

    # -- lifecycle ------------------------------------------------------
    def create_session(self, body, idempotency_key=None):
        if idempotency_key is not None and not IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key):
            raise validation_error("X-Idempotency-Key must be 16-128 safe characters")
        try:
            request = validate_create_request(body)
        except ValueError as exc:
            raise validation_error(str(exc))
        with self._lock:
            existing = self._find_by_idempotency_key(idempotency_key)
            if existing is not None:
                if _requests_equivalent(existing, request):
                    return 200, create_session_response(existing)
                raise invalid_state(
                    "idempotency key was reused with different request content",
                    existing["analysis_session_id"],
                )
            now = utc_now()
            session = {
                "schema_version": SCHEMA_VERSION,
                "analysis_session_id": new_session_id(),
                "status": "accepted",
                "created_at": now,
                "updated_at": now,
                "sealed": False,
                "sealed_at": None,
                "expected_last_segment_index": None,
                "allow_partial": None,
                "camera_id": request["camera_id"],
                "calibration_id": request["calibration_id"],
                # The business service owns the camera profile.  This is a
                # per-session input copy, not a GPU-side calibration registry.
                "court_corners": request["court_corners"],
                "client_reference": request.get("client_reference"),
                "configuration": request["configuration"],
                "request": {"idempotency_key": idempotency_key, "accepted_at": now},
                "retention": {
                    "policy": "terminal_session_ttl",
                    "hours": self.retention_hours,
                    "cleanup_anchor": "updated_at_after_terminal",
                },
                "segments": {},
                "error": None,
                "processing_errors": [],
                "continuity_epoch": 0,
                "event_seq": 0,
                "state_history": [{"at": now, "status": "accepted", "event": "durably_stored"}],
            }
            self._append_event(session, "session_status", 0.0, 0, 1.0, "derived", {"status": "accepted"})
            self._save(session)
            self._initialize_trace(session)
            self._notify()
            return 202, create_session_response(session)

    def get_session(self, session_id):
        return self._load(session_id)

    def receive_segment(self, session_id, path_index, metadata_body, data_bytes):
        session = self._load(session_id)
        if session is None:
            raise session_not_found(session_id)
        try:
            metadata = validate_segment_metadata(metadata_body)
        except ValueError as exc:
            raise validation_error(str(exc), analysis_session_id=session_id)
        path_index = int(path_index)
        if metadata["segment_index"] != path_index:
            raise validation_error(
                f"path segment_index {path_index} does not match metadata segment_index {metadata['segment_index']}",
                analysis_session_id=session_id,
            )
        actual_sha256 = hashlib.sha256(data_bytes).hexdigest()
        actual_length = len(data_bytes)
        if actual_sha256 != metadata["sha256"]:
            raise validation_error("metadata sha256 does not match the segment bytes", analysis_session_id=session_id)
        if actual_length != metadata["content_length_bytes"]:
            raise validation_error("metadata content_length_bytes does not match the segment bytes", analysis_session_id=session_id)
        if actual_length > MAX_SEGMENT_BYTES:
            raise validation_error(f"segment exceeds the {MAX_SEGMENT_BYTES} byte limit", analysis_session_id=session_id)
        if metadata["court_corners"] != session.get("court_corners"):
            raise validation_error(
                "segment court_corners must exactly match the session court_corners",
                analysis_session_id=session_id,
            )

        with self._lock:
            session = self._load(session_id)
            if session is None:
                raise session_not_found(session_id)
            if session.get("sealed"):
                raise StreamSessionError(
                    "segment_after_seal", "session input is already sealed", 409,
                    analysis_session_id=session_id,
                )
            if session.get("status") in TERMINAL_SESSION_STATES:
                raise invalid_state(f"cannot add a segment to a {session['status']} session", session_id)
            key = str(path_index)
            existing = session["segments"].get(key)
            if existing is not None:
                same_content = (
                    existing["sha256"] == metadata["sha256"]
                    and existing["content_length_bytes"] == metadata["content_length_bytes"]
                )
                same_idempotency_key = (
                    existing.get("idempotency_key") == metadata["idempotency_key"]
                )
                if same_content and same_idempotency_key:
                    session["status"] = self._derive_status(session)
                    self._save(session)
                    receipt = {
                        "accepted": True,
                        "reused": True,
                        "received_at": existing["received_at"],
                        "sha256": existing["sha256"],
                        "content_length_bytes": existing["content_length_bytes"],
                        "processing_disposition": "already_accepted",
                    }
                    return 200, segment_receipt_response(session, path_index, receipt)
                if same_content:
                    raise invalid_state(
                        f"segment_index {path_index} was retried with a different idempotency key",
                        session_id,
                        details={
                            "segment_index": path_index,
                            "accepted_idempotency_key": existing.get("idempotency_key"),
                            "submitted_idempotency_key": metadata["idempotency_key"],
                        },
                    )
                raise StreamSessionError(
                    "segment_hash_conflict",
                    f"segment_index {path_index} was already accepted with a different SHA-256 digest",
                    409,
                    details={
                        "segment_index": path_index,
                        "accepted_sha256": existing["sha256"],
                        "submitted_sha256": metadata["sha256"],
                    },
                    analysis_session_id=session_id,
                )
            segment_path = self._segment_path(session_id, path_index)
            segment_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = segment_path.with_suffix(".bin.tmp")
            temporary.write_bytes(data_bytes)
            os.replace(temporary, segment_path)
            received_at = utc_now()
            disposition = (
                "queued"
                if path_index == self._next_expected_index(session)
                else "waiting_for_predecessor"
            )
            session["segments"][key] = {
                "segment_index": path_index,
                "source_start_time_sec": metadata["source_start_time_sec"],
                "duration_sec": metadata["duration_sec"],
                "sha256": metadata["sha256"],
                "idempotency_key": metadata["idempotency_key"],
                "content_type": metadata["content_type"],
                "content_length_bytes": metadata["content_length_bytes"],
                "court_corners": metadata["court_corners"],
                "received_at": received_at,
                "reused": False,
                "processing_disposition": disposition,
                "processed": False,
                "processing_failed": False,
            }
            session["status"] = self._derive_status(session)
            self._append_state_history(session, session["status"], "segment_received")
            self._save(session)
            trace = self._load_trace(session_id)
            if trace is not None:
                trace["segments"][key] = {
                    "segment_index": path_index,
                    "received_at": received_at,
                    "content_length_bytes": actual_length,
                    "source_start_time_sec": metadata["source_start_time_sec"],
                    "duration_sec": metadata["duration_sec"],
                    "processing_started_at": None,
                    "processing_finished_at": None,
                    "queue_wait_seconds": None,
                    "segment_open_seconds": None,
                    "decode_sample_models_tracking_seconds": None,
                    "event_checkpoint_write_seconds": None,
                }
                self._trace_stage(trace, "gpu_receive", 0.0)
                self._save_trace(session_id, trace)
            receipt = {
                "accepted": True,
                "reused": False,
                "received_at": received_at,
                "sha256": metadata["sha256"],
                "content_length_bytes": metadata["content_length_bytes"],
                "processing_disposition": disposition,
            }
            if disposition == "queued":
                self._notify()
            return 202, segment_receipt_response(session, path_index, receipt)

    def get_status(self, session_id):
        session = self._load(session_id)
        if session is None:
            raise session_not_found(session_id)
        return 200, self._status_response(session)

    def read_events(self, session_id, cursor=None, limit=100):
        session = self._load(session_id)
        if session is None:
            raise session_not_found(session_id)
        # The frozen contract orders the event log by evidence time, not by
        # whichever processor/thread happened to append first.  ``event_id`` is
        # the deterministic tie-breaker and therefore also makes cursor paging
        # stable across retries and process restarts.
        events = sorted(
            self._load_events(session_id),
            key=lambda event: (
                float(event.get("source_time_sec", 0.0)),
                str(event.get("event_id") or ""),
            ),
        )
        limit = max(1, min(int(limit), 1000))
        start = 0
        if cursor:
            start = len(events)
            for position, event in enumerate(events):
                if event.get("event_id") == cursor:
                    start = position + 1
                    break
        page = events[start:start + limit]
        next_cursor = page[-1]["event_id"] if page else None
        return 200, events_response(session, page, next_cursor)

    def _load_events(self, session_id):
        path = self._events_path(session_id)
        if not path.is_file():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def complete(self, session_id, body):
        session = self._load(session_id)
        if session is None:
            raise session_not_found(session_id)
        try:
            request = validate_complete_request(body)
        except ValueError as exc:
            raise validation_error(str(exc), analysis_session_id=session_id)
        with self._lock:
            session = self._load(session_id)
            if session is None:
                raise session_not_found(session_id)
            if session.get("status") in TERMINAL_SESSION_STATES:
                if session["status"] in {"finalized", "partial"}:
                    return 202, complete_session_response(
                        session,
                        session["status"],
                        session["sealed_at"],
                        self._missing_indexes(session),
                        self._failed_indexes(session),
                    )
                raise invalid_state(f"cannot complete a {session['status']} session", session_id)
            if session.get("sealed"):
                if (
                    session.get("expected_last_segment_index") != request["expected_last_segment_index"]
                    or session.get("allow_partial") != request["allow_partial"]
                ):
                    raise invalid_state("session is already sealed with a different completion request", session_id)
                return 202, complete_session_response(
                    session,
                    session["status"],
                    session["sealed_at"],
                    self._missing_indexes(session),
                    self._failed_indexes(session),
                )
            expected_last = request["expected_last_segment_index"]
            allow_partial = request["allow_partial"]
            received = self._received_indexes(session)
            missing = sorted(index for index in range(0, expected_last + 1) if index not in received)
            if missing and not allow_partial:
                raise StreamSessionError(
                    "missing_segments",
                    "session cannot be sealed while required segments are missing",
                    409,
                    retryable=True,
                    details={"expected_last_segment_index": expected_last, "missing_segment_indexes": missing},
                    analysis_session_id=session_id,
                )
            session["sealed"] = True
            session["sealed_at"] = utc_now()
            session["expected_last_segment_index"] = expected_last
            session["allow_partial"] = allow_partial
            failures = self._failed_indexes(session)
            if allow_partial and missing:
                self._finalize_partial(session, missing, failures)
                session = self._load(session_id) or session
            elif self._processed_count(session) + self._failed_count(session) >= self._received_count(session):
                if failures:
                    self._finalize_partial(session, missing, failures)
                else:
                    self._finalize_complete(session)
                session = self._load(session_id) or session
            else:
                session["status"] = "draining"
                self._append_state_history(session, "draining", "input_sealed")
                self._emit_status_event(session, "draining")
                self._save(session)
                self._notify()
            return 202, complete_session_response(
                session,
                session["status"],
                session["sealed_at"],
                missing,
                self._failed_indexes(session),
            )

    def cancel(self, session_id):
        session = self._load(session_id)
        if session is None:
            raise session_not_found(session_id)
        with self._lock:
            session = self._load(session_id)
            if session is None:
                raise session_not_found(session_id)
            if session["status"] == "cancelled":
                return 200, self._status_response(session)
            if session["status"] in TERMINAL_SESSION_STATES:
                raise invalid_state(f"cannot cancel a {session['status']} session", session_id)
            session["status"] = "cancelled"
            session["sealed"] = True
            self._cancelled_sessions.add(session_id)
            session["error"] = {
                "code": "cancelled_by_operator",
                "message": "session processing was cancelled by an authorized operator",
                "retryable": False,
                "details": {},
            }
            self._append_state_history(session, "cancelled", "cancellation_requested")
            self._emit_status_event(session, "cancelled")
            self._save(session)
            self._finish_trace(session, "cancelled")
            return 200, self._status_response(session)

    # -- processing -----------------------------------------------------
    def start(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop, name="good-badminton-stream-worker", daemon=True
        )
        self._worker.start()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="good-badminton-stream-watchdog",
            daemon=True,
        )
        self._watchdog.start()

    @property
    def worker_running(self):
        return self._worker is not None and self._worker.is_alive()

    def _notify(self):
        self._queue.put(True)

    def _worker_loop(self):
        while True:
            self._queue.get()
            try:
                self._process_ready_work()
            finally:
                self._queue.task_done()

    def _watchdog_loop(self):
        """Escalate an in-process model hang to a supervised API restart.

        Python cannot safely kill an arbitrary CUDA/OpenCV call in another
        thread. The only truthful recovery is: durably mark the one segment as
        failed, terminate this API process, then let the deployment supervisor
        restore the prior checkpoint and resume from the next source segment.
        """
        # Keep the 10-second segment SLO observable within roughly one tenth
        # of a second instead of adding a half-second polling delay.
        sleep_seconds = 0.1
        while True:
            time.sleep(sleep_seconds)
            with self._lock:
                active = dict(self._active_segment) if self._active_segment else None
            if active is None:
                continue
            elapsed = time.monotonic() - active["started_monotonic"]
            if elapsed < self.segment_timeout_seconds:
                continue
            key = (active["session_id"], active["index"])
            if key in self._timed_out_segments:
                continue
            self._timed_out_segments.add(key)
            self._persist_timeout_then_restart(active, elapsed)

    def _persist_timeout_then_restart(self, active, elapsed_seconds):
        session_id = active["session_id"]
        index = int(active["index"])
        session = self._load(session_id)
        if session is None or session.get("status") in TERMINAL_SESSION_STATES:
            return
        entry = session.get("segments", {}).get(str(index))
        if entry is None or self._entry_settled(entry):
            return
        failure = {
            "segment_index": index,
            "phase": "segment_timeout",
            "type": "SegmentProcessingTimeout",
            "message": (
                f"segment exceeded the {self.segment_timeout_seconds:.1f}s processing limit; "
                "GPU API will restart and continue from the next segment"
            ),
            "at": utc_now(),
            "source_start_time_sec": float(entry.get("source_start_time_sec", 0.0)),
            "duration_sec": float(entry.get("duration_sec", 0.0)),
            "elapsed_seconds": round(float(elapsed_seconds), 6),
        }
        with self._lock:
            latest = self._load(session_id)
            if latest is None or latest.get("status") in TERMINAL_SESSION_STATES:
                return
            latest_entry = latest.get("segments", {}).get(str(index))
            if latest_entry is None or self._entry_settled(latest_entry):
                return
            latest_entry["processing_failed"] = True
            latest_entry["processed"] = False
            latest_entry["failure"] = failure
            latest["continuity_epoch"] = int(latest.get("continuity_epoch", 0)) + 1
            latest["processing_errors"] = list(latest.get("processing_errors") or ()) + [failure]
            latest["status"] = self._derive_status(latest)
            self._append_state_history(latest, latest["status"], "segment_timeout_restart")
            self._emit_status_event(
                latest,
                "segment_timeout_restart",
                source_time_sec=float(entry.get("source_start_time_sec", 0.0)),
                segment_index=index,
                extra={
                    "segment_failure": failure,
                    "continuity_epoch": latest["continuity_epoch"],
                    "recovery_action": "supervised_gpu_api_restart",
                },
            )
            self._save(latest)
        trace = self._load_trace(session_id)
        if trace is not None:
            trace["segments"].setdefault(str(index), {}).update(
                {
                    "processing_finished_at": utc_now(),
                    "outcome": "timeout_restart",
                    "failure": failure,
                }
            )
            self._save_trace(session_id, trace)

        if callable(self.hard_timeout_handler):
            self.hard_timeout_handler(session_id, index, failure)
            return
        # Exit code 75 is handled by deploy/start_gpu_api_container.sh.  Do not
        # try to keep executing CUDA/OpenCV calls after a watchdog timeout.
        os._exit(75)

    def drain(self):
        """Synchronously process all currently-ready work (used by tests)."""
        self._process_ready_work()

    def _process_ready_work(self):
        while True:
            target = self._next_ready_segment()
            if target is None:
                break
            session_id, index = target
            before = self._load(session_id)
            self._process_segment(session_id, index)
            after = self._load(session_id)
            # A decoder/engine failure must either advance or terminate the
            # session.  If neither happened, stop rather than spin forever on
            # the same ready segment and consume a CPU core.
            if before == after:
                break

    def _next_ready_segment(self):
        sessions = sorted(self._list_sessions(), key=lambda item: item.get("created_at", ""))
        for session in sessions:
            if session.get("status") in TERMINAL_SESSION_STATES:
                continue
            # Scheduling is derived from durable receipt state, not the count
            # of successful results.  A failed segment is settled as a known
            # evidence gap, allowing later fragments to run in a new epoch.
            next_index = self._next_expected_index(session)
            entry = session.get("segments", {}).get(str(next_index))
            if entry is not None and not self._entry_settled(entry):
                return session["analysis_session_id"], next_index
        return None

    def _process_segment(self, session_id, index):
        session = self._load(session_id)
        if (
            session is None
            or session.get("status") in TERMINAL_SESSION_STATES
            or self._is_cancelled(session_id)
        ):
            return
        entry = session.get("segments", {}).get(str(index))
        if entry is None or self._entry_settled(entry):
            return
        if self.processor_factory is None:
            self._continue_after_segment_failure(
                session_id,
                index,
                RuntimeError("processor_factory is not configured; cannot process segments"),
                phase="processor_factory",
            )
            return
        # Creating a processor can load model dependencies lazily.  Treat a
        # failure there exactly like decode/inference failure: record it on this
        # session and keep the queue worker alive for subsequent sessions.
        try:
            engine = self._get_engine(session)
        except Exception as exc:
            self._continue_after_segment_failure(session_id, index, exc, phase="engine_create")
            return
        if engine is None:
            self._continue_after_segment_failure(
                session_id,
                index,
                RuntimeError("unable to create the analysis engine"),
                phase="engine_create",
            )
            return
        segment_path = self._segment_path(session_id, index)
        if not segment_path.is_file():
            self._continue_after_segment_failure(
                session_id,
                index,
                RuntimeError(f"segment {index} bytes are missing from durable storage"),
                phase="segment_storage",
            )
            return
        descriptor = self._descriptor_from_entry(entry)
        processing_started_at = utc_now()
        trace = self._load_trace(session_id)
        if trace is not None:
            segment_trace = trace["segments"].setdefault(str(index), {})
            segment_trace["processing_started_at"] = processing_started_at
            queue_wait = self._seconds_between(entry.get("received_at"), processing_started_at)
            segment_trace["queue_wait_seconds"] = round(queue_wait, 6)
            self._trace_stage(trace, "queue_wait", queue_wait)
            self._save_trace(session_id, trace)
        open_started = time.perf_counter()
        try:
            segment = self.decoder.decode(segment_path, descriptor)
        except (ReplayDecodeError, OSError) as exc:
            self._continue_after_segment_failure(
                session_id,
                index,
                RuntimeError(f"segment {index} could not be decoded: {exc}"),
                phase="decode",
            )
            return
        open_elapsed = time.perf_counter() - open_started
        if self._is_cancelled(session_id):
            return
        processing_started = time.perf_counter()
        self._set_active_segment(session_id, index)
        try:
            result = engine.process_segment(segment)
        except StreamingAnalysisError as exc:
            self._clear_active_segment(session_id, index)
            self._continue_after_segment_failure(session_id, index, exc, phase="engine")
            return
        except Exception as exc:
            self._clear_active_segment(session_id, index)
            self._continue_after_segment_failure(session_id, index, exc, phase="engine")
            return
        processing_elapsed = time.perf_counter() - processing_started
        commit_started = time.perf_counter()
        with self._lock:
            latest = self._load(session_id)
            if latest is None or latest.get("status") in TERMINAL_SESSION_STATES:
                return
            self._append_engine_events(
                session_id,
                result.events,
                continuity_epoch=int(latest.get("continuity_epoch", 0)),
            )
            self._save_checkpoint(session_id, result.checkpoint)
            latest_entry = latest.get("segments", {}).get(str(index))
            if latest_entry is None:
                self._fail_session(
                    session_id,
                    RuntimeError(f"segment {index} receipt disappeared before commit"),
                )
                return
            latest_entry["processed"] = True
            latest_entry["processing_failed"] = False
            latest["status"] = self._derive_status(latest)
            self._append_state_history(latest, latest["status"], "segment_processed")
            self._save(latest)
        commit_elapsed = time.perf_counter() - commit_started
        trace = self._load_trace(session_id)
        if trace is not None:
            segment_trace = trace["segments"].setdefault(str(index), {})
            segment_trace.update(
                {
                    "processing_finished_at": utc_now(),
                    "segment_open_seconds": round(open_elapsed, 6),
                    "decode_sample_models_tracking_seconds": round(processing_elapsed, 6),
                    "event_checkpoint_write_seconds": round(commit_elapsed, 6),
                    "source_frames": int(result.source_frames),
                    "measurement_frames": int(result.measurement_frames),
                }
            )
            self._trace_stage(trace, "segment_open", open_elapsed)
            self._trace_stage(trace, "decode_sample_models_tracking", processing_elapsed)
            self._trace_stage(trace, "event_checkpoint_write", commit_elapsed)
            self._save_trace(session_id, trace)
        self._clear_active_segment(session_id, index)
        self._finalize_if_ready(session_id)

    def _set_active_segment(self, session_id, index):
        with self._lock:
            self._active_segment = {
                "session_id": str(session_id),
                "index": int(index),
                "started_monotonic": time.monotonic(),
            }

    def _clear_active_segment(self, session_id, index):
        with self._lock:
            active = self._active_segment
            if (
                active is not None
                and active.get("session_id") == str(session_id)
                and int(active.get("index", -1)) == int(index)
            ):
                self._active_segment = None

    def _continue_after_segment_failure(self, session_id, index, exc, *, phase):
        """Persist one unusable fragment and resume at the following one.

        The old engine is discarded because its third-party model/tracker may
        have mutated before raising.  A fresh engine begins a new continuity
        epoch after the failed source-time interval.  This is intentionally a
        partial result, never a claim that post-gap tracks are the same people.
        """
        session = self._load(session_id)
        if session is None or session.get("status") in TERMINAL_SESSION_STATES:
            return
        entry = session.get("segments", {}).get(str(index))
        if entry is None or self._entry_settled(entry):
            return

        failure = {
            "segment_index": int(index),
            "phase": str(phase),
            "type": type(exc).__name__,
            "message": str(exc),
            "at": utc_now(),
            "source_start_time_sec": float(entry.get("source_start_time_sec", 0.0)),
            "duration_sec": float(entry.get("duration_sec", 0.0)),
        }
        next_epoch = int(session.get("continuity_epoch", 0)) + 1
        try:
            replacement = self._create_engine(session)
            begin_after_gap = getattr(replacement, "begin_after_gap", None)
            if not callable(begin_after_gap):
                raise RuntimeError(
                    "engine does not support safe continuation after a segment failure"
                )
            begin_after_gap(
                next_segment_index=int(index) + 1,
                source_time_sec=(
                    float(entry.get("source_start_time_sec", 0.0))
                    + float(entry.get("duration_sec", 0.0))
                ),
                continuity_epoch=next_epoch,
                gap_segment_indexes=self._failed_indexes(session) + [int(index)],
            )
            recovery_checkpoint = replacement.checkpoint()
        except Exception as recovery_exc:
            self._fail_session(
                session_id,
                RuntimeError(
                    f"segment {index} failed ({type(exc).__name__}: {exc}); "
                    f"safe continuation could not start: {type(recovery_exc).__name__}: {recovery_exc}"
                ),
            )
            return

        with self._lock:
            latest = self._load(session_id)
            if latest is None or latest.get("status") in TERMINAL_SESSION_STATES:
                return
            latest_entry = latest.get("segments", {}).get(str(index))
            if latest_entry is None or self._entry_settled(latest_entry):
                return
            latest_entry["processing_failed"] = True
            latest_entry["processed"] = False
            latest_entry["failure"] = failure
            latest["continuity_epoch"] = next_epoch
            errors = list(latest.get("processing_errors") or ())
            errors.append(failure)
            latest["processing_errors"] = errors
            latest["status"] = self._derive_status(latest)
            self._append_state_history(latest, latest["status"], "segment_failed_continue")
            self._emit_status_event(
                latest,
                "segment_failed_continue",
                source_time_sec=float(entry.get("source_start_time_sec", 0.0)),
                segment_index=int(index),
                extra={
                    "segment_failure": failure,
                    "continuity_epoch": next_epoch,
                    "identity_policy": (
                        "post-gap tracker IDs use a new continuity epoch; "
                        "they are not automatically linked to pre-gap tracks"
                    ),
                },
            )
            self._save_checkpoint(session_id, recovery_checkpoint)
            self._save(latest)
            self._engines[session_id] = replacement

        trace = self._load_trace(session_id)
        if trace is not None:
            segment_trace = trace["segments"].setdefault(str(index), {})
            segment_trace.update(
                {
                    "processing_finished_at": utc_now(),
                    "outcome": "failed_continue",
                    "failure": failure,
                }
            )
            self._save_trace(session_id, trace)
        self._finalize_if_ready(session_id)

    def _is_cancelled(self, session_id):
        if session_id in self._cancelled_sessions:
            return True
        latest = self._load(session_id)
        return latest is not None and latest.get("status") == "cancelled"

    def _sync_engine_terminal_status(self, session, engine):
        """Mirror an engine terminal state onto the session, returning True if terminal."""
        if engine.status == "interrupted_needs_rebuild":
            session["status"] = "interrupted_needs_rebuild"
            session["error"] = {
                "code": "interrupted_needs_rebuild",
                "message": engine.restore_error or "cross-segment continuity cannot be restored",
                "retryable": False,
                "details": {},
            }
            self._append_state_history(session, "interrupted_needs_rebuild", "engine_unrunnable")
            self._finish_trace(session, "interrupted_needs_rebuild")
            return True
        return False

    def _finalize_if_ready(self, session_id):
        session = self._load(session_id)
        if session is None or session.get("status") in TERMINAL_SESSION_STATES:
            return
        if not session.get("sealed"):
            return
        missing = self._missing_indexes(session)
        failures = self._failed_indexes(session)
        if session.get("allow_partial") and missing:
            self._finalize_partial(session, missing, failures)
            return
        if self._processed_count(session) + self._failed_count(session) >= self._received_count(session):
            if failures:
                self._finalize_partial(session, missing, failures)
            else:
                self._finalize_complete(session)

    def _finalize_complete(self, session):
        engine = self._engines.get(session["analysis_session_id"])
        if engine is None:
            session["status"] = "finalized"
            self._append_state_history(session, "finalized", "finalized")
            self._save(session)
            self._finish_trace(session, "finalized")
            return
        finalize_started = time.perf_counter()
        try:
            result = engine.finalize()
        except Exception as exc:
            if self._sync_engine_terminal_status(session, engine):
                self._save(session)
            else:
                self._fail_session(session["analysis_session_id"], exc)
            return
        session_id = session["analysis_session_id"]
        with self._lock:
            latest = self._load(session_id)
            if latest is None or latest.get("status") in TERMINAL_SESSION_STATES:
                return
            self._append_engine_events(
                session_id,
                result.events,
                continuity_epoch=int(latest.get("continuity_epoch", 0)),
            )
            self._save_checkpoint(session_id, result.checkpoint)
            latest["status"] = "finalized"
            self._append_state_history(latest, "finalized", "finalized")
            self._save(latest)
        finalize_elapsed = time.perf_counter() - finalize_started
        trace = self._load_trace(session_id)
        if trace is not None:
            self._trace_stage(trace, "finalize", finalize_elapsed)
            trace["terminal_status"] = "finalized"
            trace["finished_at"] = utc_now()
            trace["summary"] = {
                "received_segments": self._received_count(latest),
                "processed_segments": self._processed_count(latest),
                "source_duration_seconds": self._received_duration(latest),
                "wall_seconds_since_accept": round(
                    self._seconds_between(latest["created_at"], trace["finished_at"]),
                    6,
                ),
            }
            self._save_trace(session_id, trace)

    def _finalize_partial(self, session, missing, failed=()):
        engine = self._engines.get(session["analysis_session_id"])
        if engine is not None:
            try:
                result = engine.finalize()
            except Exception as exc:
                if self._sync_engine_terminal_status(session, engine):
                    self._save(session)
                else:
                    self._fail_session(session["analysis_session_id"], exc)
                return
            # AnalysisEngine owns deterministic evidence event IDs, but its
            # terminal event says "finalized".  For a gapped session the API
            # owns the one truthful terminal event: "partial" plus the exact
            # missing indexes.  Processor flush events are still preserved.
            flush_events = [
                event for event in result.events
                if event.get("event_type") != "session_finalized"
            ]
        session_id = session["analysis_session_id"]
        with self._lock:
            latest = self._load(session_id)
            if latest is None or latest.get("status") in TERMINAL_SESSION_STATES:
                return
            if engine is not None:
                self._append_engine_events(
                    session_id,
                    flush_events,
                    continuity_epoch=int(latest.get("continuity_epoch", 0)),
                )
                self._save_checkpoint(session_id, result.checkpoint)
            latest["status"] = "partial"
            self._append_state_history(latest, "partial", "finalized_partial")
            self._append_event(
                latest, "session_finalized", self._processed_duration(latest),
                self._processed_count(latest), 1.0, "finalized",
                {
                    "status": "partial",
                    "missing_segment_indexes": list(missing),
                    "failed_segment_indexes": list(failed),
                    "processing_errors": list(latest.get("processing_errors") or ()),
                    "artifacts": [],
                },
            )
            self._save(latest)
            self._finish_trace(latest, "partial")

    def _fail_session(self, session_id, exc):
        session = self._load(session_id)
        if session is None or session.get("status") in TERMINAL_SESSION_STATES:
            return
        session["status"] = "failed"
        session["error"] = {
            "code": "engine_failed",
            "message": str(exc),
            "retryable": False,
            "details": {"type": type(exc).__name__},
        }
        self._append_state_history(session, "failed", "engine_failed")
        self._emit_status_event(session, "failed")
        self._save(session)
        self._finish_trace(session, "failed")

    # -- recovery -------------------------------------------------------
    def _recover_engine_after_failed_cursor(self, session, next_expected_index):
        """Build a fresh epoch only when the saved engine is before a failure."""
        cursor = int(next_expected_index)
        failed = []
        while True:
            entry = session.get("segments", {}).get(str(cursor))
            if entry is None or not entry.get("processing_failed"):
                break
            failed.append(cursor)
            cursor += 1
        if not failed:
            return None
        last_entry = session["segments"][str(failed[-1])]
        replacement = self._create_engine(session)
        begin_after_gap = getattr(replacement, "begin_after_gap", None)
        if not callable(begin_after_gap):
            raise RuntimeError("engine does not support safe continuation after a segment failure")
        begin_after_gap(
            next_segment_index=cursor,
            source_time_sec=(
                float(last_entry.get("source_start_time_sec", 0.0))
                + float(last_entry.get("duration_sec", 0.0))
            ),
            continuity_epoch=max(1, int(session.get("continuity_epoch", 1))),
            gap_segment_indexes=self._failed_indexes(session),
        )
        self._save_checkpoint(session["analysis_session_id"], replacement.checkpoint())
        return replacement

    def _recover_sessions(self):
        for path in self.sessions_dir.glob("*/manifest.json"):
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if session.get("status") in TERMINAL_SESSION_STATES:
                continue
            session_id = session["analysis_session_id"]
            checkpoint = self._load_checkpoint(session_id)
            processed = self._processed_count(session)
            if processed > 0 and checkpoint is None:
                self._mark_interrupted(session, "checkpoint missing after restart; continuity cannot be restored")
                continue
            if checkpoint is not None:
                try:
                    replacement = self._recover_engine_after_failed_cursor(
                        session,
                        int(checkpoint.get("next_expected_segment_index", 0)),
                    )
                except Exception as exc:
                    self._mark_interrupted(session, f"gap recovery failed: {exc}")
                    continue
                if replacement is not None:
                    engine = replacement
                else:
                    try:
                        engine = self._restore_engine(session, checkpoint)
                    except Exception as exc:
                        self._mark_interrupted(session, f"engine restore failed: {exc}")
                        continue
                    if engine.status == "interrupted_needs_rebuild":
                        self._mark_interrupted(session, engine.restore_error or "processor state could not be restored")
                        continue
                self._engines[session_id] = engine
                for index in (checkpoint.get("processed_segments") or {}):
                    entry = session.get("segments", {}).get(str(index))
                    if entry is not None:
                        entry["processed"] = True
                session["status"] = self._derive_status(session)
                self._append_state_history(session, session["status"], "recovered_after_restart")
                self._save(session)
            else:
                try:
                    engine = self._recover_engine_after_failed_cursor(session, 0)
                except Exception as exc:
                    self._mark_interrupted(session, f"gap recovery failed: {exc}")
                    continue
                if engine is not None:
                    self._engines[session_id] = engine
                session["status"] = "queued" if self._received_count(session) > 0 else "accepted"
                self._append_state_history(session, session["status"], "recovered_after_restart")
                self._save(session)

    def _mark_interrupted(self, session, message):
        session["status"] = "interrupted_needs_rebuild"
        session["error"] = {
            "code": "interrupted_needs_rebuild",
            "message": message,
            "retryable": False,
            "details": {},
        }
        self._append_state_history(session, "interrupted_needs_rebuild", "service_restarted")
        self._save(session)
        self._finish_trace(session, "interrupted_needs_rebuild")


def _requests_equivalent(existing, request):
    return (
        existing.get("camera_id") == request["camera_id"]
        and existing.get("calibration_id") == request["calibration_id"]
        and existing.get("court_corners") == request["court_corners"]
        and existing.get("client_reference") == request.get("client_reference")
        and existing.get("configuration") == request["configuration"]
    )


# -- route-ready handlers (wired into api.app by task F) --------------------

def create_session_handler(manager, body, idempotency_key=None):
    """POST /api/v1/stream-sessions"""
    return manager.create_session(body, idempotency_key)


def submit_segment_handler(manager, session_id, segment_index, metadata_body, data_bytes):
    """POST /api/v1/stream-sessions/{id}/segments/{index}"""
    return manager.receive_segment(session_id, segment_index, metadata_body, data_bytes)


def get_session_status_handler(manager, session_id):
    """GET /api/v1/stream-sessions/{id}"""
    return manager.get_status(session_id)


def read_events_handler(manager, session_id, cursor=None, limit=100):
    """GET /api/v1/stream-sessions/{id}/events"""
    return manager.read_events(session_id, cursor=cursor, limit=limit)


def complete_session_handler(manager, session_id, body):
    """POST /api/v1/stream-sessions/{id}/complete"""
    return manager.complete(session_id, body)


def cancel_session_handler(manager, session_id):
    """DELETE /api/v1/stream-sessions/{id}"""
    return manager.cancel(session_id)
