"""Task F: production FastAPI route composition and legacy compatibility."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from tests.stream_test_utils import (
    counting_processor_factory,
    create_request,
    segment_metadata,
    write_video_segment_bytes,
)


class StreamProductionRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.previous_key = os.environ.get("GOOD_BADMINTON_API_KEY")
        os.environ["GOOD_BADMINTON_API_KEY"] = "task-f-test-key"
        self.app = create_app(
            Path(self.temporary.name),
            start_worker=False,
            stream_processor_factory=counting_processor_factory(),
        )
        self.client = TestClient(self.app)
        self.headers = {"X-API-Key": "task-f-test-key"}
        self.segment = write_video_segment_bytes()

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("GOOD_BADMINTON_API_KEY", None)
        else:
            os.environ["GOOD_BADMINTON_API_KEY"] = self.previous_key
        self.temporary.cleanup()

    def test_formal_routes_execute_a_complete_stream_session(self):
        response = self.client.post(
            "/api/v1/stream-sessions",
            headers={**self.headers, "X-Idempotency-Key": "task-f-create-session-0001"},
            json=create_request(),
        )
        self.assertEqual(response.status_code, 202, response.text)
        session_id = response.json()["analysis_session_id"]
        metadata = segment_metadata(0, self.segment)
        received = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/segments/0",
            headers=self.headers,
            files={"segment": ("segment-0.mp4", self.segment, "video/mp4")},
            data={"metadata": json.dumps(metadata)},
        )
        self.assertEqual(received.status_code, 202, received.text)

        self.app.state.stream_manager.drain()
        current = self.client.get(
            f"/api/v1/stream-sessions/{session_id}", headers=self.headers
        )
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["progress"]["processed_segments"], 1)
        completed = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/complete",
            headers=self.headers,
            json={
                "schema_version": "stream-session.v1",
                "expected_last_segment_index": 0,
                "allow_partial": False,
            },
        )
        self.assertEqual(completed.status_code, 202, completed.text)
        self.assertEqual(completed.json()["status"], "finalized")
        trace_response = self.client.get(
            f"/api/v1/stream-sessions/{session_id}/trace",
            headers=self.headers,
        )
        self.assertEqual(trace_response.status_code, 200)
        trace = trace_response.json()
        self.assertEqual(trace["terminal_status"], "finalized")
        self.assertEqual(trace["summary"]["processed_segments"], 1)
        self.assertGreaterEqual(
            trace["stages"]["decode_sample_models_tracking"]["elapsed_seconds"],
            0.0,
        )
        self.assertFalse(trace["streaming_slo_proven"])

    def test_legacy_complete_file_jobs_route_remains_registered(self):
        response = self.client.post("/api/v1/jobs")
        self.assertEqual(response.status_code, 401)
        paths = {route.path for route in self.app.routes}
        self.assertIn("/api/v1/jobs", paths)
        self.assertIn("/api/v1/stream-sessions", paths)

    def test_stream_routes_share_the_existing_api_authentication(self):
        response = self.client.post("/api/v1/stream-sessions", json=create_request())
        self.assertEqual(response.status_code, 401)

    def test_stream_video_resources_and_full_data_have_distinct_deletes(self):
        response = self.client.post(
            "/api/v1/stream-sessions",
            headers={**self.headers, "X-Idempotency-Key": "resource-delete-session-01"},
            json=create_request(),
        )
        session_id = response.json()["analysis_session_id"]
        received = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/segments/0",
            headers=self.headers,
            files={"segment": ("segment-0.mp4", self.segment, "video/mp4")},
            data={"metadata": json.dumps(segment_metadata(0, self.segment))},
        )
        self.assertEqual(received.status_code, 202, received.text)
        self.client.delete(f"/api/v1/stream-sessions/{session_id}", headers=self.headers)
        session_dir = Path(self.temporary.name) / "stream_sessions" / session_id

        resources = self.client.delete(
            f"/api/v1/stream-sessions/{session_id}/resources",
            headers=self.headers,
        )

        self.assertEqual(resources.status_code, 200, resources.text)
        self.assertTrue((session_dir / "manifest.json").is_file())
        self.assertFalse((session_dir / "segments").exists())

        full = self.client.delete(
            f"/api/v1/stream-sessions/{session_id}/data",
            headers=self.headers,
        )
        self.assertEqual(full.status_code, 200, full.text)
        self.assertFalse(session_dir.exists())


if __name__ == "__main__":
    unittest.main()
