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

    def test_gap_route_requires_auth_and_unblocks_later_segment(self):
        created = self.client.post(
            "/api/v1/stream-sessions",
            headers={**self.headers, "X-Idempotency-Key": "task-f-gap-session-0001"},
            json=create_request(),
        )
        session_id = created.json()["analysis_session_id"]
        later = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/segments/1",
            headers=self.headers,
            files={"segment": ("segment-1.mp4", self.segment, "video/mp4")},
            data={"metadata": json.dumps(segment_metadata(1, self.segment, source_start=1.0))},
        )
        self.assertEqual(later.status_code, 202, later.text)
        path = f"/api/v1/stream-sessions/{session_id}/gaps/0"
        declaration = {"schema_version": "stream-session.v1", "source_start_time_sec": 0,
                       "duration_sec": 1, "reason": "source_segment_missing"}
        self.assertEqual(self.client.post(path, json=declaration).status_code, 401)
        declared = self.client.post(path, headers=self.headers, json=declaration)
        self.assertEqual(declared.status_code, 202, declared.text)
        self.app.state.stream_manager.drain()
        status = self.client.get(f"/api/v1/stream-sessions/{session_id}", headers=self.headers).json()
        self.assertEqual(status["progress"]["processed_segments"], 1)
        self.assertEqual(status["current_stage"], "analyzing")


    def test_repair_routes_grant_reconcile_and_accept_a_fragment(self):
        created = self.client.post(
            "/api/v1/stream-sessions",
            headers={**self.headers, "X-Idempotency-Key": "task-f-repair-session-0001"},
            json=create_request(),
        )
        self.assertEqual(created.status_code, 202, created.text)
        session_id = created.json()["analysis_session_id"]
        for index in (0, 2):
            received = self.client.post(
                f"/api/v1/stream-sessions/{session_id}/segments/{index}",
                headers=self.headers,
                files={"segment": (f"segment-{index}.mp4", self.segment, "video/mp4")},
                data={
                    "metadata": json.dumps(
                        segment_metadata(index, self.segment, source_start=float(index)))
                },
            )
            self.assertEqual(received.status_code, 202, received.text)
        self.assertEqual(
            self.client.post(
                f"/api/v1/stream-sessions/{session_id}/gaps/1",
                headers=self.headers,
                json={"schema_version": "stream-session.v1", "source_start_time_sec": 1.0,
                      "duration_sec": 1.0, "reason": "source_segment_missing"},
            ).status_code,
            202,
        )
        self.app.state.stream_manager.drain()

        window_path = f"/api/v1/stream-sessions/{session_id}/repairs"
        body = {
            "schema_version": "stream-session.v1", "reason": "business_gap_inventory",
            "ttl_seconds": 600, "ranges": [{"segment_index": 1, "kind": "missing"}],
        }
        self.assertEqual(self.client.post(window_path, json=body).status_code, 401)
        granted = self.client.post(
            window_path,
            headers={**self.headers, "X-Idempotency-Key": "task-f-repair-window-0001"},
            json=body,
        )
        self.assertEqual(granted.status_code, 202, granted.text)
        payload = granted.json()
        grant_token = payload["ranges"][0]["grant_token"]

        inventory = self.client.get(window_path, headers=self.headers)
        self.assertEqual(inventory.status_code, 200, inventory.text)
        self.assertTrue(inventory.json()["window_open"])
        # Reconciliation reports state only; the cleartext token was delivered once.
        self.assertNotIn(grant_token, json.dumps(inventory.json()))

        sealed = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/complete",
            headers=self.headers,
            json={"schema_version": "stream-session.v1", "expected_last_segment_index": 2,
                  "allow_partial": True},
        )
        self.assertEqual(sealed.status_code, 409, sealed.text)
        self.assertEqual(sealed.json()["error"]["code"], "repair_in_progress")

        metadata = {
            **segment_metadata(1, self.segment, source_start=1.0),
            "repair_request_id": payload["request_id"],
        }
        repaired = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/segments/1/repair",
            headers={**self.headers, "X-Grant-Token": grant_token},
            files={"segment": ("segment-1.mp4", self.segment, "video/mp4")},
            data={"metadata": json.dumps(metadata)},
        )
        self.assertEqual(repaired.status_code, 200, repaired.text)
        self.assertTrue(repaired.json()["receipt"]["repaired"])

        self.app.state.stream_manager.drain()
        status = self.client.get(
            f"/api/v1/stream-sessions/{session_id}", headers=self.headers
        ).json()
        self.assertEqual(status["events_revision"], 2)
        self.assertEqual(status["progress"]["processed_segments"], 3)
        self.assertEqual(status["progress"]["missing_segment_indexes"], [])
        self.assertEqual(
            self.client.post(
                f"{window_path}/rp_does-not-exist/abandon", headers=self.headers
            ).status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
