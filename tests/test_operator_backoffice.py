import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from operator_api.services.operator_backoffice import BackofficeError, BusinessDatabase, OperatorBackoffice, _validate_base_url
from operator_api.services.task_ledger import BusinessTaskLedger


class _Controller:
    def __init__(self):
        self.cancelled = False

    def snapshot(self):
        return {"webui_task_id": "local-task", "cancel_requested": self.cancelled}

    def request_cancel(self):
        self.cancelled = True
        return self.snapshot()


class _CaptureDatabase(BusinessDatabase):
    """Exercises business writes without an actual PostgreSQL instance."""

    def __init__(self):
        super().__init__(database_url="postgresql://not-used")
        self.calls = []

    def _execute(self, query, params):
        self.calls.append((query, params))
        return 1

    def _audit(self, action, resource_type, resource_id, after_summary):
        self.calls.append(("audit", (action, resource_type, resource_id, after_summary)))


class OperatorBackofficeTests(unittest.TestCase):
    def test_gpu_base_url_rejects_credentials_and_query(self):
        self.assertEqual(_validate_base_url("https://gpu.example.com/"), "https://gpu.example.com")
        with self.assertRaises(BackofficeError):
            _validate_base_url("https://key@gpu.example.com")
        with self.assertRaises(BackofficeError):
            _validate_base_url("https://gpu.example.com/?token=bad")

    def test_save_gpu_config_never_returns_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = OperatorBackoffice(
                _Controller(),
                config_path=root / ".webui-remote-gpu.env",
                operations_path=root / "operations.jsonl",
            )
            previous_url = os.environ.get("GOOD_BADMINTON_GPU_API_URL")
            previous_key = os.environ.get("GOOD_BADMINTON_GPU_API_KEY")
            try:
                result = service.save_gpu_config("http://127.0.0.1:9000", "test-secret-value")
            finally:
                if previous_url is None:
                    os.environ.pop("GOOD_BADMINTON_GPU_API_URL", None)
                else:
                    os.environ["GOOD_BADMINTON_GPU_API_URL"] = previous_url
                if previous_key is None:
                    os.environ.pop("GOOD_BADMINTON_GPU_API_KEY", None)
                else:
                    os.environ["GOOD_BADMINTON_GPU_API_KEY"] = previous_key
            self.assertNotIn("test-secret-value", str(result))
            self.assertIn("GOOD_BADMINTON_GPU_API_KEY=test-secret-value", (root / ".webui-remote-gpu.env").read_text(encoding="utf-8"))
            self.assertNotIn("test-secret-value", (root / "operations.jsonl").read_text(encoding="utf-8"))

    def test_automation_is_guarded_when_no_server_command_exists(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"GOOD_BADMINTON_GPU_CONTROL_COMMAND": "", "GOOD_BADMINTON_ENABLE_GPU_AUTOMATION": ""}, clear=False):
            service = OperatorBackoffice(_Controller(), operations_path=Path(temporary) / "operations.jsonl")
            result = service.request_gpu_operation("stop")
        self.assertEqual(result["status"], "not_available")

    def test_remote_cancel_marks_unconfirmed_when_confirmation_fails(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"GOOD_BADMINTON_TASK_LEDGER_DIR": temporary}, clear=False):
            ledger = BusinessTaskLedger()
            task_id = ledger.start_task(output_dir=temporary, remote_base_url="http://127.0.0.1:9")
            ledger.record_remote_event(task_id, {"phase": "accepted", "job_id": "gpu-job"})
            service = OperatorBackoffice(_Controller(), operations_path=Path(temporary) / "operations.jsonl")
            with patch("operator_api.services.operator_backoffice._json_request", side_effect=BackofficeError("offline")):
                result = service.cancel_remote_task(task_id)
            self.assertEqual(result["status"], "interrupted_unconfirmed")
            self.assertEqual(BusinessTaskLedger().get(task_id)["status"], "interrupted_unconfirmed")

    def test_qr_generation_persists_only_a_digest(self):
        database = _CaptureDatabase()
        result = database.create_qr_token("venue-id", "court-id", "court", "1 号场", "")
        insert_parameters = database.calls[0][1]
        self.assertTrue(result["payload"].startswith("badminton://v1/scan/"))
        self.assertNotIn(result["payload"], insert_parameters)
        self.assertEqual(len(insert_parameters[5]), 64)
        self.assertEqual(insert_parameters[3], "court")

    def test_track_claim_review_cannot_stay_pending(self):
        database = _CaptureDatabase()
        with self.assertRaises(BackofficeError):
            database.review_track_claim("claim-id", "pending")
        database.review_track_claim("claim-id", "confirmed")
        self.assertIn("update business.track_claims", database.calls[0][0])


if __name__ == "__main__":
    unittest.main()
