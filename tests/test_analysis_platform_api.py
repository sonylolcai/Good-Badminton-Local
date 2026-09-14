import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from analysis_platform.api import app


class AnalysisPlatformApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(
            os.environ,
            {"GOOD_BADMINTON_ANALYSIS_STORE": self.temporary.name},
        )
        self.environment.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_health_exposes_only_evaluation_service(self):
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "evaluation-platform-api")
        self.assertNotIn("venue", response.json())

    def test_manifest_draft_round_trip(self):
        manifest = {
            "schema_version": "analysis-platform.v1",
            "kind": "dataset",
            "dataset_id": "dataset-api",
            "name": "API fixture",
            "purpose": "API boundary test",
            "created_at": "2026-09-14T00:00:00Z",
        }
        saved = self.client.post(
            "/api/v1/manifests/drafts",
            json={"kind": "dataset", "manifest": manifest},
        )
        listed = self.client.get("/api/v1/manifests/dataset?draft=true")

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json(), [manifest])

    def test_file_endpoint_rejects_outside_paths(self):
        outside = Path(self.temporary.name).parent / "outside.txt"
        response = self.client.get("/api/v1/files", params={"path": str(outside)})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_failed")


if __name__ == "__main__":
    unittest.main()
