import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from business_gateway.streaming.client import StreamAPIError, StreamSessionClient
from business_gateway.streaming.models import DeliveryLedger, SegmentMetadata, StreamClientConfig
from business_gateway.streaming.replay import replay_video


def create_request():
    return {
        "schema_version": "stream-session.v1",
        "camera_id": "court-camera-a",
        "calibration_id": "calibration-v1",
        "court_corners": [[0, 0], [64, 0], [64, 64], [0, 64]],
        "analysis_mode": "person_only",
        "client_reference": "opaque-business-session",
        "configuration": {
            "analysis_sample_hz": 10,
            "pose_imgsz": 960,
            "shuttle_detector": "none",
            "generate_annotated_video": False,
            "preserve_audio": False,
            "court_health_check_hz": 2,
        },
    }


class FakeTransport:
    def __init__(self):
        self.session_id = "ssn_test_001"
        self.create_calls = 0
        self.upload_calls = 0
        self.create_failures = []
        self.upload_failures = []
        self.uploaded = {}

    def request_json(self, method, path, *, body=None, headers=None):
        if method == "POST" and path == "/api/v1/stream-sessions":
            self.create_calls += 1
            if self.create_failures:
                raise self.create_failures.pop(0)
            return {
                "schema_version": "stream-session.v1",
                "analysis_session_id": self.session_id,
                "status": "accepted",
            }
        if method == "POST" and path.endswith("/complete"):
            return {
                "schema_version": "stream-session.v1",
                "analysis_session_id": self.session_id,
                "status": "draining",
                "sealed_at": "2026-08-23T00:00:00Z",
                "missing_segment_indexes": [],
            }
        if method == "GET" and "/events?" in path:
            return {"events": [], "next_cursor": None}
        if method == "GET" and path.endswith("/trace"):
            return {
                "kind": "good_badminton_stream_end_to_end_trace",
                "streaming_slo_proven": False,
            }
        if method == "GET":
            return {"analysis_session_id": self.session_id, "status": "running"}
        if method == "DELETE":
            return {"analysis_session_id": self.session_id, "status": "cancelled"}
        raise AssertionError((method, path, body, headers))

    def upload_segment(self, session_id, segment_path, metadata):
        self.upload_calls += 1
        if self.upload_failures:
            failure = self.upload_failures.pop(0)
            if callable(failure):
                failure(metadata)
            else:
                raise failure
        reused = metadata.segment_index in self.uploaded
        self.uploaded[metadata.segment_index] = metadata.sha256
        return {
            "schema_version": "stream-session.v1",
            "analysis_session_id": session_id,
            "segment_index": metadata.segment_index,
            "session_status": "running",
            "receipt": {
                "accepted": True,
                "reused": reused,
                "received_at": "2026-08-23T00:00:01Z",
                "sha256": metadata.sha256,
                "content_length_bytes": metadata.content_length_bytes,
                "processing_disposition": "already_accepted" if reused else "queued",
            },
            "status_url": f"/api/v1/stream-sessions/{session_id}",
        }


class StreamSessionClientTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.segment_path = self.root / "segment_000000.mp4"
        self.segment_path.write_bytes(b"independently-decodable-test-segment")
        self.metadata = SegmentMetadata.from_file(
            self.segment_path,
            segment_index=0,
            source_start_time_sec=0.0,
            duration_sec=2.0,
            idempotency_prefix="business-session-001",
            court_corners=[[0, 0], [64, 0], [64, 64], [0, 64]],
        )
        self.config = StreamClientConfig(
            base_url="https://gpu.invalid",
            api_key="test-key",
            max_attempts=3,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def client(self, transport, ledger=None):
        return StreamSessionClient(
            self.config,
            ledger or DeliveryLedger(self.root / "delivery-ledger.json"),
            transport=transport,
            sleep_fn=lambda _seconds: None,
        )

    def test_create_retries_with_the_same_idempotency_request(self):
        transport = FakeTransport()
        transport.create_failures.append(StreamAPIError("response lost", retryable=True))
        client = self.client(transport)

        response = client.create_session(create_request(), idempotency_key="create-key-000001")

        self.assertEqual(response["analysis_session_id"], transport.session_id)
        self.assertEqual(transport.create_calls, 2)
        self.assertEqual(client.ledger.analysis_session_id, transport.session_id)

    def test_response_loss_retries_segment_and_persists_reused_receipt(self):
        transport = FakeTransport()

        def accept_then_drop(metadata):
            transport.uploaded[metadata.segment_index] = metadata.sha256
            raise StreamAPIError("connection closed after durable receipt", retryable=True)

        transport.upload_failures.append(accept_then_drop)
        client = self.client(transport)
        client.create_session(create_request(), idempotency_key="create-key-000001")

        receipt = client.submit_segment(self.segment_path, self.metadata)

        self.assertEqual(transport.upload_calls, 2)
        self.assertTrue(receipt["receipt"]["reused"])
        record = client.ledger.segment(0)
        self.assertEqual(record["status"], "accepted")
        self.assertEqual(record["attempts"], 2)

    def test_restart_skips_confirmed_segment(self):
        ledger_path = self.root / "delivery-ledger.json"
        first_transport = FakeTransport()
        first = self.client(first_transport, DeliveryLedger(ledger_path))
        first.create_session(create_request(), idempotency_key="create-key-000001")
        accepted = first.submit_segment(self.segment_path, self.metadata)

        restarted_transport = FakeTransport()
        restarted = self.client(restarted_transport, DeliveryLedger(ledger_path))
        returned = restarted.submit_segment(self.segment_path, self.metadata)

        self.assertEqual(returned, accepted)
        self.assertEqual(restarted_transport.upload_calls, 0)

    def test_ledger_retries_a_transient_windows_replace_lock(self):
        ledger_path = self.root / "delivery-ledger-retry.json"
        ledger = DeliveryLedger(ledger_path)
        real_replace = os.replace
        attempts = []

        def transient_lock(source, destination):
            attempts.append((source, destination))
            if len(attempts) == 1:
                raise PermissionError("simulated Windows sharing violation")
            return real_replace(source, destination)

        with patch("business_gateway.streaming.models.os.replace", side_effect=transient_lock):
            ledger.remember_create(create_request(), "create-key-000001")

        self.assertEqual(len(attempts), 2)
        self.assertTrue(ledger_path.is_file())
        self.assertEqual(json.loads(ledger_path.read_text(encoding="utf-8"))["create"]["idempotency_key"], "create-key-000001")

    def test_restart_resends_an_unconfirmed_segment(self):
        ledger_path = self.root / "delivery-ledger.json"
        one_attempt_config = StreamClientConfig(
            base_url="https://gpu.invalid",
            api_key="test-key",
            max_attempts=1,
            initial_backoff_seconds=0,
            max_backoff_seconds=0,
        )
        first_transport = FakeTransport()
        first_transport.upload_failures.append(StreamAPIError("temporary 503", status_code=503, retryable=True))
        first = StreamSessionClient(
            one_attempt_config,
            DeliveryLedger(ledger_path),
            transport=first_transport,
            sleep_fn=lambda _seconds: None,
        )
        first.create_session(create_request(), idempotency_key="create-key-000001")
        with self.assertRaises(StreamAPIError):
            first.submit_segment(self.segment_path, self.metadata)

        restarted_transport = FakeTransport()
        restarted = self.client(restarted_transport, DeliveryLedger(ledger_path))
        receipts = restarted.deliver_pending()

        self.assertEqual(len(receipts), 1)
        self.assertEqual(restarted_transport.upload_calls, 1)
        self.assertEqual(restarted.ledger.segment(0)["status"], "accepted")

    def test_non_retryable_rejection_is_not_automatically_resent(self):
        transport = FakeTransport()
        transport.upload_failures.append(
            StreamAPIError("hash conflict", status_code=409, code="segment_hash_conflict")
        )
        client = self.client(transport)
        client.create_session(create_request(), idempotency_key="create-key-000001")

        with self.assertRaises(StreamAPIError):
            client.submit_segment(self.segment_path, self.metadata)

        self.assertEqual(client.ledger.segment(0)["status"], "failed")
        self.assertEqual(client.deliver_pending(), [])
        self.assertEqual(transport.upload_calls, 1)

    def test_complete_is_persisted_and_identity_fields_are_rejected(self):
        transport = FakeTransport()
        client = self.client(transport)
        invalid = create_request()
        invalid["user_id"] = "must-not-cross-gpu-boundary"
        with self.assertRaises(ValueError):
            client.create_session(invalid, idempotency_key="invalid-create-key-000001")

        client.create_session(create_request(), idempotency_key="create-key-000001")
        client.submit_segment(self.segment_path, self.metadata)
        response = client.complete(0)
        self.assertEqual(response["status"], "draining")
        self.assertEqual(
            client.ledger.snapshot()["completion"]["request"]["expected_last_segment_index"],
            0,
        )

    def test_create_rejects_key_that_gpu_api_would_reject(self):
        transport = FakeTransport()
        client = self.client(transport)

        with self.assertRaisesRegex(ValueError, "16 to 128"):
            client.create_session(create_request(), idempotency_key="too-short")

        self.assertEqual(transport.create_calls, 0)

    def test_service_neutral_environment_names_take_precedence(self):
        with patch.dict(
            "os.environ",
            {
                "GPU_ANALYSIS_BASE_URL": "https://gpu.example.test/root",
                "GPU_ANALYSIS_API_KEY": "new-key",
                "GOOD_BADMINTON_STREAM_API_URL": "https://legacy.invalid",
                "GOOD_BADMINTON_STREAM_API_KEY": "legacy-key",
            },
            clear=False,
        ):
            configured = StreamClientConfig.from_environment()
        self.assertEqual(configured.base_url, "https://gpu.example.test/root")
        self.assertEqual(configured.api_key, "new-key")

    def test_trace_is_fetched_as_a_separate_durable_resource(self):
        transport = FakeTransport()
        client = self.client(transport)
        client.create_session(create_request(), idempotency_key="create-key-000001")

        trace = client.get_trace()

        self.assertEqual(trace["kind"], "good_badminton_stream_end_to_end_trace")
        self.assertFalse(trace["streaming_slo_proven"])

    def test_file_replay_persists_business_and_gpu_end_to_end_evidence(self):
        transport = FakeTransport()
        client = self.client(transport)

        class OneSegment:
            def iter_input(self, *_args, **_kwargs):
                yield SimpleNamespace(
                    path=self.segment_path,
                    segment_index=0,
                    source_start_time_sec=0.0,
                    duration_sec=2.0,
                    content_type="video/mp4",
                )

            def __init__(self, segment_path):
                self.segment_path = segment_path

        result = replay_video(
            self.segment_path,
            client=client,
            segmenter=OneSegment(self.segment_path),
            create_request=create_request(),
            create_idempotency_key="replay-create-key-000001",
        )

        trace_path = Path(result["end_to_end_trace"])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self.assertEqual(trace["kind"], "good_badminton_business_gpu_end_to_end_trace")
        self.assertEqual(trace["delivery"]["uploaded_segments"], 1)
        self.assertEqual(
            trace["gpu_trace"]["kind"],
            "good_badminton_stream_end_to_end_trace",
        )
        self.assertFalse(trace["streaming_slo_proven"])


if __name__ == "__main__":
    unittest.main()
