"""Durable business-side records for remote GPU analysis tasks.

The GPU API owns inference artifacts. The caller (currently the WebUI backend,
later the business service) owns the submission audit trail. One JSON document
per task ensures a restarted local process never loses a confirmed remote job.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


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
        elif phase in {"queued", "running", "succeeded", "failed"}:
            remote["last_poll_at"] = utc_now()
            remote["poll_count"] = int(remote.get("poll_count", 0)) + 1
            event_name = "remote_status_polled"
            task["status"] = phase
        elif phase == "downloading":
            event_name = "result_download_started"
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

    def record_event(self, task_id, event, *, details=None):
        """Record a business-side recovery or local-fallback event."""
        task = self.get(task_id)
        if task is None:
            return
        self._append(task, event, status=task["status"], details=details)
        self._write(task)

    def pending_tasks(self):
        terminal = {"succeeded", "failed"}
        for path in sorted(self.root.glob("*.json")):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if task.get("status") not in terminal:
                yield task

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
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps(task, ensure_ascii=False, indent=2)
        with self._lock:
            temporary.write_text(payload + "\n", encoding="utf-8")
            os.replace(temporary, path)
