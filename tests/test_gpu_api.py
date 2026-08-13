import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app


class GpuApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_key = os.environ.get("GOOD_BADMINTON_API_KEY")
        os.environ["GOOD_BADMINTON_API_KEY"] = "test-api-key"
        self.app = create_app(data_dir=Path(self.temp_dir.name), start_worker=False)
        self.client = TestClient(self.app)

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("GOOD_BADMINTON_API_KEY", None)
        else:
            os.environ["GOOD_BADMINTON_API_KEY"] = self.previous_key
        self.temp_dir.cleanup()

    def test_health_is_available_without_secret(self):
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_job_submission_requires_api_key(self):
        response = self.client.post("/api/v1/jobs")

        self.assertEqual(response.status_code, 401)

    def test_invalid_upload_is_rejected_before_creating_job(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-0001"},
            files={
                "video": ("notes.txt", b"not a video", "text/plain"),
                "template": ("court.png", b"fake", "image/png"),
            },
            data={"court_corners": "[[1,1],[2,1],[2,2],[1,2]]"},
        )

        self.assertEqual(response.status_code, 422)

    def test_valid_upload_creates_a_queued_job_with_stable_status_url(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-0001"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={"court_corners": "[[1,1],[2,1],[2,2],[1,2]]"},
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["execution"]["mode"], "remote_gpu")
        self.assertTrue(payload["receipt"]["accepted"])
        self.assertFalse(payload["receipt"]["reused"])
        job_id = payload["job_id"]

        status_response = self.client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-api-key"}
        )
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["job_id"], job_id)

        repeated = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-0001"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={"court_corners": "[[1,1],[2,1],[2,2],[1,2]]"},
        )
        self.assertEqual(repeated.status_code, 202)
        self.assertEqual(repeated.json()["job_id"], job_id)
        self.assertTrue(repeated.json()["receipt"]["reused"])

        recovered = self.client.get(
            "/api/v1/jobs/by-idempotency/business-task-0001",
            headers={"X-API-Key": "test-api-key"},
        )
        self.assertEqual(recovered.status_code, 200)
        self.assertEqual(recovered.json()["job_id"], job_id)
        self.assertTrue(recovered.json()["receipt"]["reused"])


if __name__ == "__main__":
    unittest.main()
