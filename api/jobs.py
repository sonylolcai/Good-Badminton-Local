"""Persistent single-worker jobs for GPU video analysis.

The compute service deliberately owns only uploaded media, model execution, and
result artifacts.  Authentication, users, billing, and business records belong
to the separate application service described in the deployment design.
"""

import json
import os
import queue
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class AnalysisJobManager:
    """Store jobs on disk and process one GPU job at a time.

    A single worker prevents concurrent videos from exhausting the 24 GB GPU.
    JSON manifests remain readable after a service restart; queued/running jobs
    are marked interrupted rather than silently reported as successful.
    """

    def __init__(self, data_dir, start_worker=True):
        self.data_dir = Path(data_dir).resolve()
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._worker = None
        self._recover_interrupted_jobs()
        if start_worker:
            self.start()

    def start(self):
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="good-badminton-gpu-worker",
            daemon=True,
        )
        self._worker.start()

    @property
    def worker_running(self):
        return self._worker is not None and self._worker.is_alive()

    def create_job(self, video_path, template_path, corners, options, idempotency_key=None):
        job_id = uuid.uuid4().hex
        job_dir = self.jobs_dir / job_id
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=False)
        accepted_at = utc_now()
        job = {
            "job_id": job_id,
            "status": "queued",
            "created_at": accepted_at,
            "started_at": None,
            "finished_at": None,
            "progress": {"processed_frames": 0, "total_frames": None, "ratio": 0.0},
            "tracking": {
                "phase": "waiting_for_analysis",
                "match_roster": None,
                "track_candidates": [],
            },
            "input": {
                "video_filename": Path(video_path).name,
                "template_filename": Path(template_path).name,
                "court_corners": corners,
                "match_session_ref": options.get("match_session_ref"),
            },
            "options": options,
            "request": {"idempotency_key": idempotency_key, "accepted_at": accepted_at},
            "state_history": [
                {"at": accepted_at, "status": "accepted", "event": "durably_stored"},
                {"at": accepted_at, "status": "queued", "event": "enqueued"},
            ],
            "execution": {
                "mode": "remote_gpu",
                "fallback_used": False,
            },
            "error": None,
            "result": None,
        }
        self._write_job(job)
        return job

    def get_by_idempotency_key(self, idempotency_key):
        """Find a prior accepted request so retrying a timed-out POST is safe."""
        if not idempotency_key:
            return None
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (job.get("request") or {}).get("idempotency_key") == idempotency_key:
                return job
        return None

    def enqueue(self, job_id, video_path, template_path, corners, options, output_dir):
        """Queue a job only after uploads have moved to their durable paths."""
        self._queue.put((job_id, str(video_path), str(template_path), corners, options, str(output_dir)))

    def get_job(self, job_id):
        path = self._job_path(job_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_path(self, job_id, artifact_name):
        job = self.get_job(job_id)
        if not job or job.get("status") != "succeeded":
            return None
        artifact = (job.get("result") or {}).get("artifacts", {}).get(artifact_name)
        if not artifact:
            return None
        path = (self.jobs_dir / job_id / "output" / artifact["relative_path"]).resolve()
        output_dir = (self.jobs_dir / job_id / "output").resolve()
        try:
            path.relative_to(output_dir)
        except ValueError:
            return None
        return path if path.is_file() else None

    def _worker_loop(self):
        while True:
            job_id, video_path, template_path, corners, options, output_dir = self._queue.get()
            try:
                self._run_job(job_id, video_path, template_path, corners, options, output_dir)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id, video_path, template_path, corners, options, output_dir):
        job = self.get_job(job_id)
        if job is None:
            return
        self._set_status(job, "running", started_at=utc_now(), error=None)
        self._write_job(job)

        def progress(processed_frames, total_frames):
            current = self.get_job(job_id)
            if current is None:
                return
            ratio = processed_frames / total_frames if total_frames else 0.0
            current["progress"] = {
                "processed_frames": int(processed_frames),
                "total_frames": int(total_frames) if total_frames else None,
                "ratio": round(float(ratio), 4),
            }
            self._write_job(current)

        def analysis_state(update):
            current = self.get_job(job_id)
            if current is None:
                return
            current["tracking"] = {
                **(current.get("tracking") or {}),
                **dict(update or {}),
            }
            self._write_job(current)

        try:
            # Lazy import keeps /health inexpensive and avoids model imports
            # before the worker actually receives a GPU task.
            from webui.pipeline import run_analysis

            result = run_analysis(
                video_path,
                template_path,
                corners,
                options,
                progress_cb=progress,
                state_cb=analysis_state,
                output_dir=output_dir,
                cleanup_outputs=False,
            )
            job = self.get_job(job_id)
            self._set_status(
                job,
                "succeeded",
                finished_at=utc_now(),
                progress={**job["progress"], "ratio": 1.0},
                result=self._result_manifest(result, output_dir),
            )
            self._write_job(job)
        except Exception as exc:
            job = self.get_job(job_id) or {"job_id": job_id}
            self._set_status(
                job,
                "failed",
                finished_at=utc_now(),
                error={
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "traceback": traceback.format_exc(limit=20),
                },
            )
            self._write_job(job)

    def _result_manifest(self, result, output_dir):
        output_path = Path(output_dir).resolve()
        artifacts = {}
        candidates = {
            "annotated_video": result.get("video"),
            "metadata": result.get("metadata"),
            "detections": result.get("detections"),
            "tracknet_raw_csv": result.get("tracknet_raw_csv"),
            "performance_report": result.get("performance_report"),
        }
        spatial_summary = output_path / "spatial_match_summary.json"
        if spatial_summary.is_file():
            candidates["spatial_match_summary"] = str(spatial_summary)

        for name, candidate in candidates.items():
            if not candidate:
                continue
            path = Path(candidate).resolve()
            try:
                relative_path = path.relative_to(output_path)
            except ValueError:
                continue
            if path.is_file():
                artifacts[name] = {
                    "relative_path": relative_path.as_posix(),
                    "media_type": self._media_type(path),
                    "size_bytes": path.stat().st_size,
                }

        images = []
        for index, candidate in enumerate(result.get("visualizations", [])):
            path = Path(candidate).resolve()
            try:
                relative_path = path.relative_to(output_path)
            except ValueError:
                continue
            if path.is_file():
                name = f"visualization_{index}"
                artifacts[name] = {
                    "relative_path": relative_path.as_posix(),
                    "media_type": self._media_type(path),
                    "size_bytes": path.stat().st_size,
                }
                images.append(name)
        return {
            "warnings": list(result.get("warnings", [])),
            "artifacts": artifacts,
            "visualizations": images,
        }

    @staticmethod
    def _media_type(path):
        return {
            ".mp4": "video/mp4",
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
            ".csv": "text/csv",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(path.suffix.lower(), "application/octet-stream")

    def _recover_interrupted_jobs(self):
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") in {"queued", "running"}:
                self._set_status(
                    job,
                    "failed",
                    finished_at=utc_now(),
                    error={
                        "type": "ServiceRestarted",
                        "message": "The GPU API restarted before this job completed; submit it again.",
                    },
                )
                self._write_job(job)

    def _job_path(self, job_id):
        return self.jobs_dir / job_id / "job.json"

    @staticmethod
    def _set_status(job, status, **fields):
        previous = job.get("status")
        job.update(fields)
        job["status"] = status
        if previous != status:
            job.setdefault("state_history", []).append(
                {"at": utc_now(), "status": status, "event": "state_changed"}
            )

    def _write_job(self, job):
        path = self._job_path(job["job_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        payload = json.dumps(job, ensure_ascii=False, indent=2)
        with self._lock:
            temporary.write_text(payload + "\n", encoding="utf-8")
            os.replace(temporary, path)
