"""Durable business-side records for remote GPU analysis tasks.

The GPU API owns inference artifacts. The caller (currently the WebUI backend,
later the business service) owns the submission audit trail. One JSON document
per task ensures a restarted local process never loses a confirmed remote job.
"""

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


# These are business-side terminal facts.  A UI restart must not turn any of
# them back into a candidate for remote polling.  ``submission_unconfirmed``
# means no remote receipt was ever recovered, so a user must explicitly decide
# whether to submit the source video again.
TERMINAL_TASK_STATUSES = frozenset({
    "succeeded", "failed", "cancelled", "interrupted_unconfirmed",
    "local_fallback", "submission_unconfirmed",
})


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class BusinessTaskLedger:
    """Append-only event history plus a convenient current task snapshot."""

    def __init__(self, root=None):
        configured_root = root or os.environ.get(
            "GOOD_BADMINTON_TASK_LEDGER_DIR", "outputs/business_tasks"
        )
        self.root = Path(configured_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def start_task(self, *, output_dir, remote_base_url):
        task_id = uuid.uuid4().hex
        task = {
            "schema_version": "1.0",
            "task_id": task_id,
            "status": "submitting",
            "created_at": utc_now(),
            "updated_at": None,
            "output_dir": str(Path(output_dir)),
            "remote": {
                "base_url": remote_base_url,
                "job_id": None,
                "accepted": False,
                "accepted_at": None,
                "last_poll_at": None,
                "poll_count": 0,
            },
            "history": [],
            "error": None,
        }
        self._append(task, "submission_started", status="submitting")
        self._write(task)
        return task_id

    def record_remote_event(self, task_id, event):
        """Store every lifecycle callback. API keys and headers are excluded."""
        task = self.get(task_id)
        if task is None:
            return
        clean = {
            key: value for key, value in event.items()
            if key not in {"api_key", "authorization", "headers"}
        }
        remote = task["remote"]
        job_id = clean.get("job_id")
        if job_id:
            remote["job_id"] = job_id
        phase = clean.get("phase")
        if phase == "accepted":
            remote["accepted"] = True
            remote["accepted_at"] = clean.get("accepted_at") or utc_now()
            event_name = "remote_receipt_confirmed"
            task["status"] = "accepted"
        elif phase in {"queued", "running", "cancelling", "cancelled", "succeeded", "failed"}:
            remote["last_poll_at"] = utc_now()
            remote["poll_count"] = int(remote.get("poll_count", 0)) + 1
            event_name = "remote_status_polled"
            task["status"] = phase
        elif phase == "downloading":
            event_name = "result_download_started"
            task["status"] = "downloading"
        elif phase == "artifact_downloading":
            event_name = "artifact_download_started"
            task["status"] = "downloading"
        elif phase == "artifact_downloaded":
            event_name = "artifact_downloaded"
            task["status"] = "downloading"
        elif phase == "local_metadata_persisted":
            event_name = "result_metadata_persisted"
            task["status"] = "downloading"
        elif phase == "downloaded":
            event_name = "result_downloaded"
            task["status"] = "downloaded"
        else:
            event_name = "remote_submission_progress"
        self._append(task, event_name, status=task["status"], details=clean)
        self._write(task)

    def record_terminal(self, task_id, *, status, error=None, details=None):
        task = self.get(task_id)
        if task is None:
            return
        task["status"] = status
        task["error"] = error
        self._append(task, "task_terminal", status=status, details=details)
        self._write(task)

    def archive_performance_trace(self, task_id, trace_path):
        """Copy a completed GPU timing trace into durable business storage.

        The downloaded analysis directory is a working/result cache and can
        later be pruned.  The ledger owns this immutable copy so historical
        performance comparisons do not depend on GPU job retention or a
        browser session remaining alive.
        """
        task = self.get(task_id)
        source = Path(trace_path) if trace_path else None
        if task is None or source is None or not source.is_file():
            return None
        payload = source.read_bytes()
        try:
            trace = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("performance trace must be a UTF-8 JSON document") from exc

        archive_dir = self.root / "performance_traces"
        archive_dir.mkdir(parents=True, exist_ok=True)
        destination = archive_dir / f"{task_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        archived_at = utc_now()
        record = {
            "schema_version": str(trace.get("schema_version") or "1.0"),
            "relative_path": str(destination.relative_to(self.root).as_posix()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "archived_at": archived_at,
            "source_job_id": ((trace.get("task") or {}).get("job_id")),
            "source_status": ((trace.get("task") or {}).get("status")),
        }
        task["performance_trace"] = record
        self._append(task, "performance_trace_archived", status=task["status"], details=record)
        self._write(task)
        return record

    def finalize_end_to_end_trace(self, task_id):
        """Persist one immutable, cross-service timing trace for a terminal task.

        ``performance_trace.json`` is deliberately owned by the GPU API and
        measures only what happens after the server has accepted an upload.
        The business task ledger is the only component that can also observe
        client upload, receipt confirmation, status polling and artifact
        retrieval.  This method joins those two sources *after* the business
        task has reached a terminal state.

        The document says what was actually serialized, rather than promising
        future streaming overlap: the present API accepts a whole media
        container, uses one GPU worker queue, then downloads artifacts after a
        terminal result.  Polling is concurrent control-plane observation, not
        concurrent inference.  Keeping that distinction in the artifact makes
        performance investigations reproducible and prevents UI progress from
        being mistaken for useful GPU parallelism.
        """
        task = self.get(task_id)
        if task is None:
            return None
        existing = task.get("end_to_end_trace")
        if existing:
            return existing
        terminal = {"succeeded", "failed", "cancelled", "interrupted_unconfirmed"}
        if task.get("status") not in terminal:
            raise ValueError("end-to-end trace can only be finalized for a terminal task")

        trace = self._build_end_to_end_trace(task)
        payload = (json.dumps(trace, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

        archive_dir = self.root / "end_to_end_traces"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{task_id}.json"
        self._atomic_write_bytes(archive_path, payload)

        result_copy = None
        output_dir = Path(task.get("output_dir") or "")
        if str(output_dir):
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                result_path = output_dir / "end_to_end_trace.json"
                self._atomic_write_bytes(result_path, payload)
                result_copy = str(result_path)
            except OSError:
                # The durable business ledger is still authoritative when a
                # transient result directory cannot be written or was pruned.
                result_copy = None

        record = {
            "schema_version": "1.0",
            "relative_path": str(archive_path.relative_to(self.root).as_posix()),
            "result_copy_path": result_copy,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "recorded_at": trace["recorded_at"],
            "remote_job_id": (task.get("remote") or {}).get("job_id"),
            "task_status": task.get("status"),
        }
        task["end_to_end_trace"] = record
        self._append(task, "end_to_end_trace_archived", status=task["status"], details=record)
        self._write(task)
        return record

    def record_event(self, task_id, event, *, details=None):
        """Record a business-side recovery or local-fallback event."""
        task = self.get(task_id)
        if task is None:
            return
        self._append(task, event, status=task["status"], details=details)
        self._write(task)

    def pending_tasks(self):
        for task in self.list_tasks():
            if task.get("status") not in TERMINAL_TASK_STATUSES:
                yield task

    def list_tasks(self, limit=None):
        """Return every durable task, newest first, for restart-safe history.

        The event stream remains the detailed audit trail.  This method only
        provides a deterministic index for the WebUI and future business
        service; it never contacts the GPU API or changes task state.
        """
        tasks = []
        for path in self.root.glob("*.json"):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(task, dict) or not task.get("task_id"):
                continue
            tasks.append(task)
        tasks.sort(
            key=lambda task: str(task.get("updated_at") or task.get("created_at") or ""),
            reverse=True,
        )
        return tasks[:max(0, int(limit))] if limit is not None else tasks

    def get(self, task_id):
        path = self._path(task_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _append(self, task, event, *, status, details=None):
        now = utc_now()
        task["updated_at"] = now
        entry = {"at": now, "event": event, "status": status}
        if details:
            entry["details"] = details
        task["history"].append(entry)

    def _path(self, task_id):
        return self.root / f"{task_id}.json"

    def _write(self, task):
        path = self._path(task["task_id"])
        payload = json.dumps(task, ensure_ascii=False, indent=2)
        with self._lock:
            self._atomic_write_bytes(path, (payload + "\n").encode("utf-8"))

    @staticmethod
    def _atomic_write_bytes(destination, payload):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)

    def _build_end_to_end_trace(self, task):
        """Build a terminal trace using only persisted, auditable evidence."""
        history = list(task.get("history") or [])
        remote = dict(task.get("remote") or {})
        performance_trace = self._read_archived_performance_trace(task)
        timing = dict((performance_trace or {}).get("timing") or {})
        execution = dict((performance_trace or {}).get("execution") or {})
        stages = list(timing.get("stages") or [])

        submission = self._first_event(history, "submission_started")
        upload_events = [
            item for item in history
            if item.get("event") == "remote_submission_progress"
            and ((item.get("details") or {}).get("phase") == "uploading")
        ]
        upload_start = upload_events[0] if upload_events else submission
        completed_upload = self._last_matching_upload(upload_events)
        receipt = self._first_event(history, "remote_receipt_confirmed")
        remote_terminal = self._last_remote_terminal_poll(history)
        result_download_start = self._first_event(history, "result_download_started")
        result_download_end = self._last_event(history, "result_downloaded")
        metadata_persisted = self._last_event(history, "result_metadata_persisted")
        terminal = self._last_event(history, "task_terminal")
        artifact_events = self._artifact_transfer_summary(history)

        server_finished_at = ((performance_trace or {}).get("task") or {}).get("finished_at")
        server_started_at = ((performance_trace or {}).get("task") or {}).get("started_at")
        server_created_at = ((performance_trace or {}).get("task") or {}).get("created_at")
        receipt_server_at = ((receipt or {}).get("details") or {}).get("accepted_at") or remote.get("accepted_at")

        trace = {
            "schema_version": "1.0",
            "kind": "good_badminton_end_to_end_trace",
            "recorded_at": utc_now(),
            "task": {
                "business_task_id": task.get("task_id"),
                "status": task.get("status"),
                "created_at": task.get("created_at"),
                "output_dir": task.get("output_dir"),
                "remote_job_id": remote.get("job_id"),
                "remote_base_url": remote.get("base_url"),
            },
            "time_basis": {
                "client_events": "business ledger UTC timestamps",
                "remote_events": "GPU API UTC timestamps copied from its terminal performance trace",
                "clock_note": "Cross-host durations are diagnostic only; same-host stage durations are authoritative.",
            },
            "timeline": {
                "submission": self._event_time_summary(submission),
                "upload": {
                    "started_at": self._at(upload_start),
                    "content_uploaded_at": self._at(completed_upload),
                    "remote_accepted_at": receipt_server_at,
                    "receipt_observed_at": self._at(receipt),
                    "media_bytes": self._upload_total_bytes(upload_events),
                    "content_upload_seconds": self._duration(self._at(upload_start), self._at(completed_upload)),
                    "receipt_round_trip_seconds": self._duration(self._at(upload_start), self._at(receipt)),
                },
                "remote_gpu": {
                    "server_created_at": server_created_at,
                    "server_started_at": server_started_at,
                    "server_finished_at": server_finished_at,
                    "server_wall_seconds": self._duration(server_started_at, server_finished_at),
                    "timing": timing,
                    "stages": stages,
                    "component_metrics": execution.get("analysis_metrics"),
                },
                "polling": {
                    "poll_count": int(remote.get("poll_count") or 0),
                    "first_poll_at": self._at(self._first_event(history, "remote_status_polled")),
                    "last_poll_at": remote.get("last_poll_at"),
                    "terminal_observed_at": self._at(remote_terminal),
                    "remote_completion_observation_seconds": self._duration(
                        server_finished_at, self._at(remote_terminal),
                    ),
                    "note": "Polling is asynchronous status observation; it does not share GPU compute work.",
                },
                "result_transfer": {
                    "started_at": self._at(result_download_start),
                    "finished_at": self._at(result_download_end),
                    "elapsed_seconds": self._duration(
                        self._at(result_download_start), self._at(result_download_end),
                    ),
                    "artifacts": artifact_events,
                },
                "local_persistence": {
                    "metadata_persisted_at": self._at(metadata_persisted),
                    "business_terminal_at": self._at(terminal),
                    "post_download_finalize_seconds": self._duration(
                        self._at(result_download_end), self._at(terminal),
                    ),
                },
                "end_to_end": {
                    "started_at": task.get("created_at"),
                    "finished_at": self._at(terminal),
                    "elapsed_seconds": self._duration(task.get("created_at"), self._at(terminal)),
                },
            },
            "execution_topology": self._execution_topology(
                stages=stages,
                has_remote_job=bool(remote.get("job_id")),
                has_download=bool(result_download_start or result_download_end),
                has_polling=bool(remote.get("poll_count")),
            ),
            "sources": {
                "business_task_ledger": str(self._path(task["task_id"])),
                "archived_gpu_performance_trace": ((task.get("performance_trace") or {}).get("relative_path")),
                "raw_history": history,
            },
            "error": task.get("error"),
        }
        return trace

    def _read_archived_performance_trace(self, task):
        record = task.get("performance_trace") or {}
        relative_path = record.get("relative_path")
        if not relative_path:
            return None
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _event_time_summary(event):
        return {"at": BusinessTaskLedger._at(event)} if event else None

    @staticmethod
    def _at(event):
        return event.get("at") if event else None

    @staticmethod
    def _first_event(history, event_name):
        return next((item for item in history if item.get("event") == event_name), None)

    @staticmethod
    def _last_event(history, event_name):
        return next((item for item in reversed(history) if item.get("event") == event_name), None)

    @staticmethod
    def _last_matching_upload(events):
        for item in reversed(events):
            details = item.get("details") or {}
            total = details.get("total_upload_bytes")
            if total and details.get("uploaded_bytes") == total:
                return item
        return events[-1] if events else None

    @staticmethod
    def _upload_total_bytes(events):
        for item in reversed(events):
            total = (item.get("details") or {}).get("total_upload_bytes")
            if total is not None:
                return int(total)
        return None

    @staticmethod
    def _last_remote_terminal_poll(history):
        terminal = {"succeeded", "failed", "cancelled"}
        for item in reversed(history):
            if item.get("event") != "remote_status_polled":
                continue
            if ((item.get("details") or {}).get("phase")) in terminal:
                return item
        return None

    @staticmethod
    def _artifact_transfer_summary(history):
        artifacts = []
        active = {}
        for item in history:
            if item.get("event") == "artifact_download_started":
                details = item.get("details") or {}
                key = str(details.get("artifact") or details.get("relative_path") or len(active))
                active[key] = {"artifact": details.get("artifact"), "relative_path": details.get("relative_path"), "started_at": item.get("at")}
            elif item.get("event") == "artifact_downloaded":
                details = item.get("details") or {}
                key = str(details.get("artifact") or details.get("relative_path") or len(active))
                item_summary = active.pop(key, {"artifact": details.get("artifact"), "relative_path": details.get("relative_path")})
                item_summary.update({
                    "finished_at": item.get("at"),
                    "size_bytes": details.get("size_bytes"),
                })
                item_summary["elapsed_seconds"] = BusinessTaskLedger._duration(
                    item_summary.get("started_at"), item_summary.get("finished_at"),
                )
                artifacts.append(item_summary)
        artifacts.extend(active.values())
        return artifacts

    @staticmethod
    def _duration(started_at, finished_at):
        if not started_at or not finished_at:
            return None
        try:
            start = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(finished_at).replace("Z", "+00:00"))
        except ValueError:
            return None
        return round(max(0.0, (end - start).total_seconds()), 6)

    @staticmethod
    def _execution_topology(*, stages, has_remote_job, has_download, has_polling):
        """Describe *current* scheduling semantics, not a target architecture."""
        nodes = [
            {
                "id": "client_upload",
                "domain": "business_client",
                "state": "executed" if has_remote_job else "not_observed",
                "depends_on": [],
                "parallel_group": "ingress_serial",
                "relationship": "The API queues work after the complete multipart upload is accepted.",
            },
            {
                "id": "gpu_queue_and_analysis",
                "domain": "gpu_api_single_worker",
                "state": "executed" if has_remote_job else "not_observed",
                "depends_on": ["client_upload"],
                "parallel_group": "gpu_serial",
                "relationship": "One GPU API worker dequeues and runs the recorded top-level stages in order.",
                "serialized_stages": [stage.get("name") for stage in stages],
            },
            {
                "id": "status_polling",
                "domain": "business_client_control_plane",
                "state": "executed" if has_polling else "not_observed",
                "depends_on": ["gpu_queue_and_analysis"],
                "parallel_group": "control_plane_parallel",
                "relationship": "Runs concurrently with remote GPU work only to observe status; it does not reduce the compute critical path.",
            },
            {
                "id": "artifact_download_and_local_persist",
                "domain": "business_client",
                "state": "executed" if has_download else "not_observed",
                "depends_on": ["gpu_queue_and_analysis"],
                "parallel_group": "egress_serial",
                "relationship": "Artifacts are requested after a terminal remote result and are downloaded one by one before local metadata is finalized.",
            },
        ]
        return {
            "current_execution_model": "complete_upload_then_single_gpu_worker_then_sequential_artifact_download",
            "actual_parallelism": [
                {
                    "name": "status_polling",
                    "status": "present" if has_polling else "not_observed",
                    "scope": "control-plane observation only",
                    "affects_gpu_critical_path": False,
                },
            ],
            "serialized_critical_path": [
                "client_upload",
                "gpu_queue_and_analysis",
                "artifact_download_and_local_persist",
            ],
            "nodes": nodes,
            "optimization_candidates": [
                {
                    "candidate": "overlap ingest with segment-level inference",
                    "current_status": "not_implemented",
                    "reason": "The current API only enqueues after a complete MP4 multipart body is received.",
                },
                {
                    "candidate": "pipeline TrackNet and pose by independent segments",
                    "current_status": "not_implemented",
                    "reason": "Current top-level stages are recorded as one ordered GPU-worker sequence.",
                },
                {
                    "candidate": "parallel artifact downloads",
                    "current_status": "not_implemented",
                    "reason": "The result client currently downloads the manifest artifacts in a deterministic loop.",
                },
            ],
        }
