"""Persistent single-worker jobs for GPU video analysis.

The compute service deliberately owns only uploaded media, model execution, and
result artifacts.  Authentication, users, billing, and business records belong
to the separate application service described in the deployment design.
"""

import hashlib
import json
import os
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from badminton_analysis.cancellation import AnalysisCancelled


TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _elapsed_seconds(started_at, finished_at):
    """Return a non-negative wall duration for persisted ISO timestamps."""
    try:
        started = datetime.fromisoformat(str(started_at))
        finished = datetime.fromisoformat(str(finished_at))
        return round(max(0.0, (finished - started).total_seconds()), 3)
    except (TypeError, ValueError):
        return None


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
        self._lock = threading.RLock()
        self._cancel_events = {}
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
            # ``timing`` is the durable performance trace for this one job.
            # It deliberately records stage boundaries instead of writing one
            # JSON file per frame, which would itself distort throughput.
            "timing": {
                "schema_version": "1.0",
                "current_stage": "queue_wait",
                "last_heartbeat_at": accepted_at,
                "stages": [
                    {
                        "name": "queue_wait",
                        "started_at": accepted_at,
                        "last_heartbeat_at": accepted_at,
                        "details": {},
                    }
                ],
            },
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

    def cancel_job(self, job_id):
        """Request cooperative cancellation without deleting inputs or artifacts."""
        with self._lock:
            job = self.get_job(job_id)
            if job is None or job.get("status") in TERMINAL_JOB_STATUSES:
                return job

            cancel_event = self._cancel_events.setdefault(job_id, threading.Event())
            cancel_event.set()
            requested_at = utc_now()
            job.setdefault("state_history", []).append({
                "at": requested_at,
                "status": "cancelling",
                "event": "cancellation_requested",
            })
            job["tracking"] = {
                **(job.get("tracking") or {}),
                "phase": "cancellation_requested",
            }
            if job.get("status") == "queued":
                self._set_status(
                    job,
                    "cancelled",
                    finished_at=requested_at,
                    error={"type": "TaskCancelled", "message": "任务已由操作员中断。"},
                )
                self._finish_timing(job, "cancelled")
                self._persist_performance_trace(job)
            else:
                self._set_status(job, "cancelling")
            self._write_job(job)
            return job

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

    def performance_trace_path(self, job_id):
        """Return the terminal performance trace even for failed jobs."""
        job = self.get_job(job_id)
        if not job or job.get("status") not in TERMINAL_JOB_STATUSES:
            return None
        trace = job.get("performance_trace") or {}
        relative_path = trace.get("relative_path")
        if relative_path != "performance_trace.json":
            return None
        output_dir = (self.jobs_dir / job_id / "output").resolve()
        path = (output_dir / relative_path).resolve()
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
        if job.get("status") == "cancelled":
            self._cancel_events.pop(job_id, None)
            return
        cancel_event = self._cancel_events.setdefault(job_id, threading.Event())
        if cancel_event.is_set():
            self._finish_cancelled(job_id)
            return
        self._set_status(job, "running", started_at=utc_now(), error=None)
        self._transition_stage(job, "analysis_bootstrap")
        self._write_job(job)

        last_progress_write = {"at": 0.0, "processed": -1}

        def progress(processed_frames, total_frames):
            # The client polls every two seconds.  Persisting a complete job
            # manifest for all 30/60 source frames per second creates avoidable
            # disk contention and makes the measured pipeline look slower.
            now_monotonic = time.monotonic()
            is_final = bool(total_frames) and int(processed_frames) >= int(total_frames)
            if (
                not is_final
                and last_progress_write["processed"] >= 0
                and now_monotonic - last_progress_write["at"] < 0.5
            ):
                return
            current = self.get_job(job_id)
            if current is None:
                return
            ratio = processed_frames / total_frames if total_frames else 0.0
            current["progress"] = {
                "processed_frames": int(processed_frames),
                "total_frames": int(total_frames) if total_frames else None,
                "ratio": round(float(ratio), 4),
            }
            self._heartbeat_stage(current)
            self._write_job(current)
            last_progress_write.update({"at": now_monotonic, "processed": int(processed_frames)})

        def analysis_state(update):
            current = self.get_job(job_id)
            if current is None:
                return
            current["tracking"] = {
                **(current.get("tracking") or {}),
                **dict(update or {}),
            }
            stage_name = (update or {}).get("stage") or (update or {}).get("phase")
            if stage_name:
                self._transition_stage(current, str(stage_name), details=(update or {}).get("stage_detail"))
            else:
                self._heartbeat_stage(current)
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
                cancel_cb=cancel_event.is_set,
                output_dir=output_dir,
                cleanup_outputs=False,
            )
            if cancel_event.is_set():
                raise AnalysisCancelled("任务已由操作员中断。")
            job = self.get_job(job_id)
            self._set_status(
                job,
                "succeeded",
                finished_at=utc_now(),
                progress={**job["progress"], "ratio": 1.0},
            )
            # Component-level measurements are produced by the analysis
            # worker.  Persist them in the job before freezing the terminal
            # performance trace, so client-side end_to_end_trace.json can
            # expose where serial processing time was spent.
            job["execution"] = {
                **(job.get("execution") or {}),
                "analysis_metrics": result.get("execution_metrics"),
            }
            self._finish_timing(job, "succeeded")
            self._persist_performance_trace(job)
            job["result"] = self._result_manifest(result, output_dir)
            self._write_job(job)
        except AnalysisCancelled:
            self._finish_cancelled(job_id)
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
            self._finish_timing(job, "failed")
            self._persist_performance_trace(job)
            self._write_job(job)
        finally:
            self._cancel_events.pop(job_id, None)

    def _finish_cancelled(self, job_id):
        job = self.get_job(job_id)
        if job is None or job.get("status") == "cancelled":
            return
        self._set_status(
            job,
            "cancelled",
            finished_at=utc_now(),
            error={"type": "TaskCancelled", "message": "任务已由操作员中断。"},
        )
        job["tracking"] = {
            **(job.get("tracking") or {}),
            "phase": "cancelled",
        }
        self._finish_timing(job, "cancelled")
        self._persist_performance_trace(job)
        self._write_job(job)

    @staticmethod
    def _transition_stage(job, stage_name, details=None):
        """Close the previous stage and start/heartbeat *stage_name*.

        Stage data is intentionally small, JSON-safe, and persisted with the
        job.  It can therefore be inspected after a restart without relying on
        transient server logs.
        """
        now = utc_now()
        timing = job.setdefault("timing", {"schema_version": "1.0", "stages": []})
        stages = timing.setdefault("stages", [])
        current = stages[-1] if stages and not stages[-1].get("finished_at") else None
        if current is not None and current.get("name") == stage_name:
            current["last_heartbeat_at"] = now
            if details:
                current.setdefault("details", {}).update(details)
        else:
            if current is not None:
                current["finished_at"] = now
                current["elapsed_seconds"] = _elapsed_seconds(current.get("started_at"), now)
            next_stage = {
                "name": stage_name,
                "started_at": now,
                "last_heartbeat_at": now,
                "details": dict(details or {}),
            }
            stages.append(next_stage)
        timing["current_stage"] = stage_name
        timing["last_heartbeat_at"] = now

    @staticmethod
    def _heartbeat_stage(job):
        now = utc_now()
        timing = job.setdefault("timing", {"schema_version": "1.0", "stages": []})
        stages = timing.setdefault("stages", [])
        if stages and not stages[-1].get("finished_at"):
            stages[-1]["last_heartbeat_at"] = now
        timing["last_heartbeat_at"] = now

    @staticmethod
    def _finish_timing(job, outcome):
        now = utc_now()
        timing = job.setdefault("timing", {"schema_version": "1.0", "stages": []})
        stages = timing.setdefault("stages", [])
        if stages and not stages[-1].get("finished_at"):
            stages[-1]["finished_at"] = now
            stages[-1]["elapsed_seconds"] = _elapsed_seconds(stages[-1].get("started_at"), now)
        timing["current_stage"] = outcome
        timing["completed_at"] = now
        timing["last_heartbeat_at"] = now

    def _persist_performance_trace(self, job):
        """Write a terminal, immutable timing artifact beside analysis output.

        ``job.json`` remains the mutable status record while a task is
        running.  This separate trace is intentionally written only at a
        terminal state, so it is suitable for long-term comparison and is not
        overwritten by later polling heartbeats.
        """
        job_id = str(job.get("job_id") or "")
        if not job_id:
            return None
        output_dir = self.jobs_dir / job_id / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "performance_trace.json"
        input_info = job.get("input") or {}
        trace = {
            "schema_version": "1.0",
            "kind": "good_badminton_analysis_performance_trace",
            "recorded_at": utc_now(),
            "task": {
                "job_id": job_id,
                "business_task_id": (job.get("request") or {}).get("idempotency_key"),
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
            },
            "input": {
                "video_filename": input_info.get("video_filename"),
                "template_filename": input_info.get("template_filename"),
            },
            "options": job.get("options") or {},
            "execution": job.get("execution") or {},
            "progress": job.get("progress") or {},
            "timing": job.get("timing") or {},
            "error": job.get("error"),
        }
        temporary = trace_path.with_suffix(".json.tmp")
        encoded = (json.dumps(trace, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        temporary.write_bytes(encoded)
        os.replace(temporary, trace_path)
        job["performance_trace"] = {
            "relative_path": "performance_trace.json",
            "media_type": "application/json",
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "recorded_at": trace["recorded_at"],
        }
        return trace_path

    def _result_manifest(self, result, output_dir):
        output_path = Path(output_dir).resolve()
        artifacts = {}
        candidates = {
            "annotated_video": result.get("video"),
            "metadata": result.get("metadata"),
            "detections": result.get("detections"),
            "tracknet_raw_csv": result.get("tracknet_raw_csv"),
            "performance_report": result.get("performance_report"),
            "movement_metrics": result.get("movement_metrics"),
            "movement_rallies": result.get("movement_rallies"),
            "movement_rally_window_sweep": result.get("movement_rally_window_sweep"),
            "position_evidence_summary": result.get("position_evidence_summary"),
        }
        trace_path = output_path / "performance_trace.json"
        if trace_path.is_file():
            candidates["performance_trace"] = str(trace_path)
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

        portraits = result.get("player_portraits") or {}
        if isinstance(portraits, dict):
            for track_id, candidate in sorted(portraits.items()):
                safe_track_id = "".join(
                    char if char.isalnum() or char in "_-" else "_"
                    for char in str(track_id)
                )
                if not safe_track_id or not candidate:
                    continue
                path = Path(candidate).resolve()
                try:
                    relative_path = path.relative_to(output_path)
                except ValueError:
                    continue
                if path.is_file():
                    artifacts[f"player_portrait_{safe_track_id}"] = {
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
            if job.get("status") in {"queued", "running", "cancelling"}:
                self._set_status(
                    job,
                    "failed",
                    finished_at=utc_now(),
                    error={
                        "type": "ServiceRestarted",
                        "message": "The GPU API restarted before this job completed; submit it again.",
                    },
                )
                self._finish_timing(job, "interrupted")
                self._persist_performance_trace(job)
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
