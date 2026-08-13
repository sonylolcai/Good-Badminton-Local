import tempfile
import unittest
from pathlib import Path

from webui.task_ledger import BusinessTaskLedger


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
