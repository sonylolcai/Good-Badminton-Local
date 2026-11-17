import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from webui.app import _local_fallback_policy, _task_history_detail, _task_history_rows
from webui.reconcile_remote_tasks import reconcile_once
from webui.remote_gpu import RemoteAnalysisError
from webui.task_ledger import BusinessTaskLedger
from webui.task_control import AnalysisTaskController


class BusinessTaskLedgerTests(unittest.TestCase):
    def test_remote_receipt_and_polls_are_durable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = BusinessTaskLedger(Path(temp_dir))
            task_id = ledger.start_task(output_dir="outputs/remote_jobs/test", remote_base_url="http://gpu")
            ledger.record_remote_event(task_id, {
                "phase": "accepted", "job_id": "a" * 32, "accepted_at": "2026-01-01T00:00:00+00:00",
            })
            ledger.record_remote_event(task_id, {
                "phase": "running", "job_id": "a" * 32,
                "processed_frames": 25, "total_frames": 100, "ratio": 0.25,
            })
            task = ledger.get(task_id)
            self.assertTrue(task["remote"]["accepted"])
            self.assertEqual(task["remote"]["job_id"], "a" * 32)
            self.assertEqual(task["remote"]["poll_count"], 1)
            self.assertEqual(task["status"], "running")
            self.assertEqual([item["event"] for item in task["history"]], [
                "submission_started", "remote_receipt_confirmed", "remote_status_polled",
            ])

    def test_webui_interrupt_is_scoped_to_the_active_task(self):
        controller = AnalysisTaskController()
        first = controller.start()

        snapshot = controller.request_cancel()

        self.assertTrue(first.is_cancelled())
        self.assertTrue(snapshot["cancel_requested"])
        controller.finish(first)
        self.assertIsNone(controller.snapshot())

        second = controller.start()
        self.assertFalse(second.is_cancelled())
        controller.finish(second)

    def test_history_lists_all_tasks_and_keeps_remote_progress_after_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = BusinessTaskLedger(Path(temp_dir))
            task_id = ledger.start_task(output_dir="outputs/remote_jobs/test", remote_base_url="http://gpu")
            ledger.record_remote_event(task_id, {
                "phase": "accepted", "job_id": "job-1", "accepted_at": "2026-01-01T00:00:00+00:00",
            })
            ledger.record_remote_event(task_id, {
                "phase": "running", "job_id": "job-1", "processed_frames": 40,
                "total_frames": 100, "ratio": 0.4,
                "stage": "tracknet.inference", "tracking": {"phase": "tracknet_inference"},
            })

            rows = _task_history_rows(ledger)
            detail = _task_history_detail(task_id, ledger)

            self.assertEqual(1, len(rows))
            self.assertEqual("远端运行中", rows[0][2])
            self.assertEqual("40.0% (40/100)", rows[0][4])
            self.assertEqual("tracknet.inference", rows[0][5])
            self.assertEqual("job-1", detail["remote"]["job_id"])
            self.assertEqual(0.4, detail["last_remote_status"]["ratio"])

    def test_missing_remote_receipt_stops_automatic_404_retries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = BusinessTaskLedger(Path(temp_dir))
            task_id = ledger.start_task(output_dir="outputs/remote_jobs/no-receipt", remote_base_url="http://gpu")

            with patch(
                "webui.reconcile_remote_tasks.recover_remote_task",
                side_effect=RemoteAnalysisError("remote GPU request failed: HTTP Error 404: Not Found"),
            ):
                summary = reconcile_once(ledger)

            self.assertEqual("submission_unconfirmed", summary[0]["status"])
            self.assertEqual("submission_unconfirmed", ledger.get(task_id)["status"])
            self.assertEqual([], list(ledger.pending_tasks()))

    def test_remote_failure_after_confirmed_receipt_never_restarts_full_local_run(self):
        """A completed/failed GPU job must not silently duplicate work locally."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = BusinessTaskLedger(Path(temp_dir))
            task_id = ledger.start_task(output_dir="outputs/remote_jobs/accepted", remote_base_url="http://gpu")
            ledger.record_remote_event(task_id, {
                "phase": "accepted", "job_id": "job-already-ran",
            })

            allowed, reason = _local_fallback_policy(
                ledger, task_id, {"shuttle_detector": "yolo"},
            )

            self.assertFalse(allowed)
            self.assertIn("已确认接收", reason)

    def test_pre_receipt_transport_failure_can_still_use_explicit_local_continuity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = BusinessTaskLedger(Path(temp_dir))
            task_id = ledger.start_task(output_dir="outputs/remote_jobs/no-receipt", remote_base_url="http://gpu")

            allowed, reason = _local_fallback_policy(
                ledger, task_id, {"shuttle_detector": "yolo"},
            )

            self.assertTrue(allowed)
            self.assertIsNone(reason)

    def test_performance_trace_is_copied_outside_the_prunable_result_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = BusinessTaskLedger(root / "business_tasks")
            task_id = ledger.start_task(output_dir=str(root / "temporary_result"), remote_base_url="http://gpu")
            source = root / "temporary_result" / "performance_trace.json"
            source.parent.mkdir()
            source.write_text(
                '{"schema_version":"1.0","task":{"job_id":"job-1","status":"succeeded"}}\n',
                encoding="utf-8",
            )

            record = ledger.archive_performance_trace(task_id, source)
            source.unlink()

            task = ledger.get(task_id)
            archived = ledger.root / record["relative_path"]
            self.assertTrue(archived.is_file())
            self.assertEqual(task["performance_trace"]["source_job_id"], "job-1")
            self.assertEqual(task["history"][-1]["event"], "performance_trace_archived")

    def test_end_to_end_trace_joins_client_and_gpu_evidence_and_marks_serial_work(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "remote_result"
            ledger = BusinessTaskLedger(root / "business_tasks")
            task_id = ledger.start_task(output_dir=str(output), remote_base_url="http://gpu")

            # These events model the complete client-side path: upload, a
            # receipt, non-blocking polling, then deterministic artifact I/O.
            ledger.record_remote_event(task_id, {
                "phase": "uploading", "uploaded_bytes": 0, "total_upload_bytes": 100,
            })
            ledger.record_remote_event(task_id, {
                "phase": "uploading", "uploaded_bytes": 100, "total_upload_bytes": 100,
            })
            ledger.record_remote_event(task_id, {
                "phase": "accepted", "job_id": "job-1", "accepted_at": "2026-01-01T00:00:01+00:00",
            })
            ledger.record_remote_event(task_id, {"phase": "running", "job_id": "job-1"})
            ledger.record_remote_event(task_id, {"phase": "succeeded", "job_id": "job-1"})
            ledger.record_remote_event(task_id, {
                "phase": "downloading", "job_id": "job-1",
            })
            ledger.record_remote_event(task_id, {
                "phase": "artifact_downloading", "job_id": "job-1",
                "artifact": "detections", "relative_path": "detections.jsonl",
            })
            ledger.record_remote_event(task_id, {
                "phase": "artifact_downloaded", "job_id": "job-1",
                "artifact": "detections", "relative_path": "detections.jsonl", "size_bytes": 123,
            })
            ledger.record_remote_event(task_id, {
                "phase": "local_metadata_persisted", "job_id": "job-1",
                "metadata_path": str(output / "metadata.json"),
            })
            ledger.record_remote_event(task_id, {"phase": "downloaded", "job_id": "job-1"})

            source = output / "performance_trace.json"
            output.mkdir(parents=True, exist_ok=True)
            source.write_text(
                """{
                  "schema_version": "1.0",
                  "task": {
                    "job_id": "job-1",
                    "status": "succeeded",
                    "created_at": "2026-01-01T00:00:01+00:00",
                    "started_at": "2026-01-01T00:00:02+00:00",
                    "finished_at": "2026-01-01T00:00:12+00:00"
                  },
                  "timing": {
                    "stages": [
                      {"name": "queue_wait", "started_at": "2026-01-01T00:00:01+00:00", "finished_at": "2026-01-01T00:00:02+00:00"},
                      {"name": "tracknet.inference", "started_at": "2026-01-01T00:00:02+00:00", "finished_at": "2026-01-01T00:00:06+00:00"},
                      {"name": "human_frame_processing", "started_at": "2026-01-01T00:00:06+00:00", "finished_at": "2026-01-01T00:00:12+00:00"}
                    ]
                  },
                  "execution": {
                    "analysis_metrics": {
                      "components": {
                        "pose_inference": {"calls": 100, "elapsed_seconds": 3.2}
                      }
                    }
                  }
                }\n""",
                encoding="utf-8",
            )
            ledger.archive_performance_trace(task_id, source)
            ledger.record_terminal(task_id, status="succeeded")

            record = ledger.finalize_end_to_end_trace(task_id)
            trace_path = ledger.root / record["relative_path"]
            trace = json.loads(trace_path.read_text(encoding="utf-8"))

            self.assertTrue((output / "end_to_end_trace.json").is_file())
            self.assertEqual(trace["timeline"]["upload"]["media_bytes"], 100)
            self.assertEqual(trace["timeline"]["result_transfer"]["artifacts"][0]["artifact"], "detections")
            self.assertEqual(trace["execution_topology"]["current_execution_model"],
                             "complete_upload_then_single_gpu_worker_then_sequential_artifact_download")
            self.assertEqual(trace["execution_topology"]["actual_parallelism"][0]["scope"],
                             "control-plane observation only")
            self.assertEqual(trace["execution_topology"]["nodes"][1]["serialized_stages"], [
                "queue_wait", "tracknet.inference", "human_frame_processing",
            ])
            self.assertEqual(
                trace["timeline"]["remote_gpu"]["component_metrics"]["components"]["pose_inference"]["calls"],
                100,
            )
            self.assertEqual(ledger.finalize_end_to_end_trace(task_id), record)
