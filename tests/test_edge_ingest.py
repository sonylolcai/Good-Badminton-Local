import hashlib
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from business_gateway.edge_api import RollingPreviewStore, create_edge_app
from business_gateway.edge_contract import (
    EDGE_NONCE_HEADER,
    EDGE_PAYLOAD_SHA256_HEADER,
    EDGE_SCHEMA_VERSION,
    EDGE_SIGNATURE_HEADER,
    EDGE_TIMESTAMP_HEADER,
    derive_device_secret,
    payload_sha256,
    sign_request,
)


MASTER_KEY = "0123456789abcdef0123456789abcdef"
DEVICE_ID = "3dd6f345-b55e-4a48-b4b6-06f20cf3e217"
CAMERA_ID = "13f59f04-68b7-4545-920f-4dfcfd8524b9"
COURT_ID = "0e1d5699-791c-4224-903f-0fd83df6a587"


class FakeRepository:
    def __init__(self):
        self.nonces = set()
        self.sessions = {}
        self.forwarded = []
        self.heartbeats = []
        self.capture_mode = "preview"
        self.calibration_id = "804ff390-87e9-497c-aae9-e0a56942e7cf"

    def binding(self, device_id, camera_id):
        if device_id != DEVICE_ID or camera_id != CAMERA_ID:
            return None
        return {
            "device_id": DEVICE_ID,
            "camera_id": CAMERA_ID,
            "venue_id": "4dff0e8e-a7d8-4fde-9d7a-c44bf871917d",
            "court_id": COURT_ID,
            "credential_version": "v1",
            "device_status": "offline",
            "camera_status": "offline",
            "calibration_id": self.calibration_id,
            "court_corners": [[0, 0], [1, 0], [1, 1], [0, 1]] if self.calibration_id else None,
        }

    def claim_nonce(self, device_id, nonce, expires_at):
        key = (device_id, nonce)
        if key in self.nonces:
            return False
        self.nonces.add(key)
        return True

    def heartbeat(self, binding, payload):
        self.heartbeats.append((binding, payload))

    def capture_control(self, binding):
        return {"mode": self.capture_mode, "revision": 7, "updated_at": "2026-09-01T00:00:00Z"}

    def create_session(self, binding, configuration):
        session = {
            "id": str(uuid.uuid4()),
            "venue_id": binding["venue_id"],
            "court_id": binding["court_id"],
            "camera_id": binding["camera_id"],
            "device_id": binding["device_id"],
            "calibration_id": binding["calibration_id"],
            "configuration": configuration,
            "court_corners": binding["court_corners"],
            "status": "requested",
        }
        self.sessions[session["id"]] = session
        return session

    def session(self, session_id):
        return self.sessions.get(session_id)

    def receive_segment(self, session, metadata):
        return {"id": str(uuid.uuid4()), "status": "received", "reused": False, "gpu_receipt": {}}

    def mark_forwarded(self, session_id, segment_index, receipt, gpu_session_id):
        self.forwarded.append((session_id, segment_index, receipt, gpu_session_id))

    def mark_segment_failed(self, session_id, segment_index, message):
        raise AssertionError(f"unexpected relay failure: {message}")

    def mark_completed(self, session_id, gpu_status, receipt):
        self.sessions[session_id]["completed"] = (gpu_status, receipt)


class FakeRelay:
    def __init__(self):
        self.calls = []

    def forward(self, session, metadata, segment_bytes):
        self.calls.append((session, metadata, segment_bytes))
        return "ssn_edge_test_0001", {"receipt": {"accepted": True}}

    def complete(self, session, expected_last_segment_index, allow_partial):
        return {"status": "finalized", "expected_last_segment_index": expected_last_segment_index, "allow_partial": allow_partial}


class EdgeIngestApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = FakeRepository()
        self.relay = FakeRelay()
        self.client = TestClient(create_edge_app(self.repository, self.relay, edge_master_key=MASTER_KEY))

    def _headers(self, method, path, body, nonce, segment_bytes=None):
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        body["timestamp"] = timestamp
        body["nonce"] = nonce
        digest = payload_sha256(body, segment_bytes)
        secret = derive_device_secret(MASTER_KEY, DEVICE_ID, "v1")
        return {
            EDGE_TIMESTAMP_HEADER: timestamp,
            EDGE_NONCE_HEADER: nonce,
            EDGE_PAYLOAD_SHA256_HEADER: digest,
            EDGE_SIGNATURE_HEADER: sign_request(secret, method, path, timestamp, nonce, digest),
        }

    def test_signed_heartbeat_updates_bound_terminal(self):
        path = f"/api/v1/edge/devices/{DEVICE_ID}/heartbeats"
        body = {
            "schema_version": EDGE_SCHEMA_VERSION,
            "device_id": DEVICE_ID,
            "camera_id": CAMERA_ID,
            "timestamp": "",
            "nonce": "",
            "agent_version": "camera-agent/0.1.0",
            "disk_free_bytes": 1024,
            "capture_state": "idle",
            "active_session_id": None,
            "last_segment_index": None,
        }
        response = self.client.post(path, json=body, headers=self._headers("POST", path, body, "heartbeat_nonce_0001"))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["court_id"], COURT_ID)
        self.assertEqual(response.json()["capture_control"]["mode"], "preview")
        self.assertEqual(len(self.repository.heartbeats), 1)

    def test_idle_capture_control_rejects_terminal_session_start(self):
        self.repository.capture_mode = "idle"
        path = f"/api/v1/edge/devices/{DEVICE_ID}/sessions"
        body = {
            "schema_version": EDGE_SCHEMA_VERSION,
            "device_id": DEVICE_ID,
            "camera_id": CAMERA_ID,
            "timestamp": "",
            "nonce": "",
            "configuration": {"analysis_sample_hz": 10, "pose_imgsz": 960,
                              "shuttle_detector": "yolo", "generate_annotated_video": False},
        }
        response = self.client.post(path, json=body, headers=self._headers("POST", path, body, "idle_start_nonce_01"))
        self.assertEqual(response.status_code, 409, response.text)

    def test_segment_is_authenticated_then_relayed_without_terminal_court_override(self):
        start_path = f"/api/v1/edge/devices/{DEVICE_ID}/sessions"
        start = {
            "schema_version": EDGE_SCHEMA_VERSION,
            "device_id": DEVICE_ID,
            "camera_id": CAMERA_ID,
            "timestamp": "",
            "nonce": "",
            "configuration": {
                "analysis_sample_hz": 10,
                "pose_imgsz": 960,
                "shuttle_detector": "yolo",
                "generate_annotated_video": False,
            },
        }
        started = self.client.post(start_path, json=start, headers=self._headers("POST", start_path, start, "start_nonce_0000001"))
        self.assertEqual(started.status_code, 202, started.text)
        session_id = started.json()["edge_ingest_session_id"]
        video = b"closed-independent-video-segment"
        segment = {
            "schema_version": "stream-session.v1",
            "segment_index": 0,
            "source_start_time_sec": 0,
            "duration_sec": 2,
            "sha256": hashlib.sha256(video).hexdigest(),
            "idempotency_key": "edge-segment-key-000001",
            "content_type": "video/mp4",
            "content_length_bytes": len(video),
            "court_corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
        }
        envelope = {"schema_version": EDGE_SCHEMA_VERSION, "device_id": DEVICE_ID, "camera_id": CAMERA_ID, "timestamp": "", "nonce": "", "segment": segment}
        path = f"/api/v1/edge/sessions/{session_id}/segments/0"
        headers = self._headers("POST", path, envelope, "segment_nonce_00001", video)
        response = self.client.post(
            path,
            data={"metadata": __import__("json").dumps(envelope)},
            files={"segment": ("00000000.mp4", video, "video/mp4")},
            headers=headers,
        )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["gpu_analysis_session_id"], "ssn_edge_test_0001")
        self.assertEqual(len(self.relay.calls), 1)
        self.assertEqual(self.relay.calls[0][0]["court_id"], COURT_ID)

    def test_reused_nonce_is_rejected_before_relay(self):
        path = f"/api/v1/edge/devices/{DEVICE_ID}/heartbeats"
        body = {"schema_version": EDGE_SCHEMA_VERSION, "device_id": DEVICE_ID, "camera_id": CAMERA_ID, "timestamp": "", "nonce": "", "agent_version": "camera-agent/0.1.0", "disk_free_bytes": 1, "capture_state": "idle", "active_session_id": None, "last_segment_index": None}
        nonce = "reused_nonce_00001"
        first = self.client.post(path, json=body, headers=self._headers("POST", path, body, nonce))
        self.assertEqual(first.status_code, 200)
        second_body = dict(body)
        second = self.client.post(path, json=second_body, headers=self._headers("POST", path, second_body, nonce))
        self.assertEqual(second.status_code, 409, second.text)

    def test_paused_gpu_delivery_still_keeps_business_server_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            self.client = TestClient(create_edge_app(
                self.repository, self.relay, edge_master_key=MASTER_KEY,
                preview_store=RollingPreviewStore(Path(directory)),
            ))
            session = self.repository.create_session(self.repository.binding(DEVICE_ID, CAMERA_ID), {"analysis_sample_hz": 10})
            session["gpu_forwarding_enabled"] = False
            video = b"closed-independent-video-segment"
            segment = {"schema_version": "stream-session.v1", "segment_index": 0, "source_start_time_sec": 0,
                       "duration_sec": 2, "sha256": hashlib.sha256(video).hexdigest(),
                       "idempotency_key": "edge-preview-key-000001", "content_type": "video/mp4",
                       "content_length_bytes": len(video), "court_corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            envelope = {"schema_version": EDGE_SCHEMA_VERSION, "device_id": DEVICE_ID, "camera_id": CAMERA_ID,
                        "timestamp": "", "nonce": "", "segment": segment}
            path = f"/api/v1/edge/sessions/{session['id']}/segments/0"
            headers = self._headers("POST", path, envelope, "segment_nonce_preview_00001", video)
            response = self.client.post(path, data={"metadata": __import__("json").dumps(envelope)},
                                        files={"segment": ("00000000.mp4", video, "video/mp4")},
                                        headers=headers)
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["gpu_forwarding"], "paused")
            self.assertEqual(len(self.relay.calls), 0)
            preview = self.client.get(f"/api/v1/edge/sessions/{session['id']}/preview/latest.mp4")
            self.assertEqual(preview.status_code, 200, preview.text)
            self.assertEqual(preview.content, video)

    def test_unconfigured_camera_can_preview_but_never_relays_to_gpu(self):
        self.repository.calibration_id = None
        with tempfile.TemporaryDirectory() as directory:
            self.client = TestClient(create_edge_app(
                self.repository, self.relay, edge_master_key=MASTER_KEY,
                preview_store=RollingPreviewStore(Path(directory)),
            ))
            start_path = f"/api/v1/edge/devices/{DEVICE_ID}/sessions"
            start = {
                "schema_version": EDGE_SCHEMA_VERSION, "device_id": DEVICE_ID, "camera_id": CAMERA_ID,
                "timestamp": "", "nonce": "", "configuration": {
                    "analysis_sample_hz": 10, "pose_imgsz": 960,
                    "shuttle_detector": "yolo", "generate_annotated_video": False,
                },
            }
            started = self.client.post(start_path, json=start, headers=self._headers(
                "POST", start_path, start, "preview_start_nonce_01",
            ))
            self.assertEqual(started.status_code, 202, started.text)
            self.assertEqual(started.json()["ingest_mode"], "preview_only")
            session_id = started.json()["edge_ingest_session_id"]
            video = b"uncalibrated-preview-segment"
            segment = {
                "schema_version": "stream-session.v1", "segment_index": 0, "source_start_time_sec": 0,
                "duration_sec": 2, "sha256": hashlib.sha256(video).hexdigest(),
                "idempotency_key": "preview-only-segment-0001", "content_type": "video/mp4",
                "content_length_bytes": len(video), "court_corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
            }
            envelope = {
                "schema_version": EDGE_SCHEMA_VERSION, "device_id": DEVICE_ID, "camera_id": CAMERA_ID,
                "timestamp": "", "nonce": "", "segment": segment,
            }
            path = f"/api/v1/edge/sessions/{session_id}/segments/0"
            headers = self._headers("POST", path, envelope, "preview_segment_nonce01", video)
            response = self.client.post(
                path, data={"metadata": __import__("json").dumps(envelope)},
                files={"segment": ("00000000.mp4", video, "video/mp4")},
                headers=headers,
            )
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["gpu_forwarding"], "blocked_pending_calibration")
            self.assertEqual(len(self.relay.calls), 0)
            self.assertEqual(self.client.get(f"/api/v1/edge/sessions/{session_id}/preview/latest.mp4").content, video)

    def test_unconfigured_camera_cannot_start_recording(self):
        self.repository.calibration_id = None
        self.repository.capture_mode = "record"
        path = f"/api/v1/edge/devices/{DEVICE_ID}/sessions"
        body = {
            "schema_version": EDGE_SCHEMA_VERSION, "device_id": DEVICE_ID, "camera_id": CAMERA_ID,
            "timestamp": "", "nonce": "", "configuration": {
                "analysis_sample_hz": 10, "pose_imgsz": 960,
                "shuttle_detector": "yolo", "generate_annotated_video": False,
            },
        }
        response = self.client.post(path, json=body, headers=self._headers(
            "POST", path, body, "record_without_calibration_01",
        ))
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("no validated calibration", response.json()["detail"]["message"])

    def test_signed_completion_records_terminal_gpu_state(self):
        session = self.repository.create_session(self.repository.binding(DEVICE_ID, CAMERA_ID), {"analysis_sample_hz": 10})
        path = f"/api/v1/edge/sessions/{session['id']}/complete"
        body = {"schema_version": EDGE_SCHEMA_VERSION, "device_id": DEVICE_ID, "camera_id": CAMERA_ID, "timestamp": "", "nonce": "", "expected_last_segment_index": 0, "allow_partial": False}
        response = self.client.post(path, json=body, headers=self._headers("POST", path, body, "complete_nonce_001"))
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["gpu_status"], "finalized")
        self.assertEqual(session["completed"][0], "finalized")


if __name__ == "__main__":
    unittest.main()
