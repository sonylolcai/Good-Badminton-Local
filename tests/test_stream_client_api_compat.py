"""Task C client -> task B HTTP contract compatibility test."""

import json
import os
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.stream_sessions import StreamSessionManager
from business_gateway.streaming.client import StreamAPIError, StreamSessionClient
from business_gateway.streaming.models import DeliveryLedger, SegmentMetadata, StreamClientConfig
from tests.stream_test_utils import counting_processor_factory, create_request, write_video_segment_bytes
from tests.test_stream_api import _build_test_app
from api.app import create_app


class _FastApiTransport:
    """Exercise StreamSessionClient against task B's real route handlers."""

    def __init__(self, client):
        self.client = client
        self.headers = {"X-API-Key": "test-api-key"}

    def request_json(self, method, path, *, body=None, headers=None):
        request_headers = dict(self.headers)
        request_headers.update(headers or {})
        response = self.client.request(method, path, headers=request_headers, json=body)
        return self._decode(response)

    def upload_segment(self, session_id, segment_path, metadata):
        with Path(segment_path).open("rb") as segment:
            response = self.client.post(
                f"/api/v1/stream-sessions/{session_id}/segments/{metadata.segment_index}",
                headers=self.headers,
                files={
                    "segment": (
                        Path(segment_path).name,
                        segment,
                        metadata.content_type,
                    )
                },
                data={"metadata": json.dumps(metadata.to_payload())},
            )
        return self._decode(response)

    @staticmethod
    def _decode(response):
        payload = response.json()
        if response.status_code >= 400:
            error = payload.get("error") or {}
            raise StreamAPIError(
                error.get("message") or response.text,
                status_code=response.status_code,
                code=error.get("code") or "http_error",
                retryable=bool(error.get("retryable")),
                response=payload,
            )
        return payload


class StreamClientApiCompatibilityTests(unittest.TestCase):
    def test_business_client_completes_a_gpu_stream_session(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            manager = StreamSessionManager(
                root / "gpu-data",
                processor_factory=counting_processor_factory(),
                start_worker=False,
            )
            api_client = TestClient(_build_test_app(manager))
            segment_path = root / "segment-000000.mp4"
            segment_path.write_bytes(write_video_segment_bytes())
            metadata = SegmentMetadata.from_file(
                segment_path,
                segment_index=0,
                source_start_time_sec=0.0,
                duration_sec=1.0,
                idempotency_prefix="business-compat-000001",
                court_corners=[[0, 0], [64, 0], [64, 64], [0, 64]],
            )
            client = StreamSessionClient(
                StreamClientConfig(
                    base_url="http://gpu.test",
                    api_key="test-api-key",
                    max_attempts=1,
                ),
                DeliveryLedger(root / "delivery-ledger.json"),
                transport=_FastApiTransport(api_client),
            )

            created = client.create_session(
                create_request(),
                idempotency_key="business-create-compat-000001",
            )
            receipt = client.submit_segment(segment_path, metadata)
            manager.drain()
            status = client.get_status()
            events = client.read_events()
            completed = client.complete(0)
            trace = client.get_trace()

            self.assertEqual(receipt["analysis_session_id"], created["analysis_session_id"])
            self.assertEqual(status["progress"]["processed_segments"], 1)
            self.assertTrue(
                any(event["event_type"] == "person_observation" for event in events["events"])
            )
            self.assertEqual(completed["status"], "finalized")
            self.assertEqual(trace["terminal_status"], "finalized")

    def test_real_urllib_transport_matches_the_production_fastapi_routes(self):
        """Exercise actual JSON HTTP plus bounded multipart, not a test transport."""
        import uvicorn

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            previous_key = os.environ.get("GOOD_BADMINTON_API_KEY")
            os.environ["GOOD_BADMINTON_API_KEY"] = "real-http-compat-key"
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = listener.getsockname()[1]
            app = create_app(
                root / "gpu-data",
                start_worker=False,
                stream_processor_factory=counting_processor_factory(),
            )
            server = uvicorn.Server(
                uvicorn.Config(app, log_level="error", lifespan="off")
            )
            server.install_signal_handlers = lambda: None
            worker = threading.Thread(
                target=lambda: server.run(sockets=[listener]),
                daemon=True,
            )
            worker.start()
            try:
                deadline = time.time() + 5
                while not server.started and time.time() < deadline:
                    time.sleep(0.01)
                self.assertTrue(server.started)

                segment_path = root / "segment-000000.mp4"
                segment_path.write_bytes(write_video_segment_bytes())
                metadata = SegmentMetadata.from_file(
                    segment_path,
                    segment_index=0,
                    source_start_time_sec=0.0,
                    duration_sec=1.0,
                    idempotency_prefix="real-http-segment-0001",
                    court_corners=[[0, 0], [64, 0], [64, 64], [0, 64]],
                    source_frame_start_index=0,
                    source_frame_count=30,
                )
                client = StreamSessionClient(
                    StreamClientConfig(
                        base_url=f"http://127.0.0.1:{port}",
                        api_key="real-http-compat-key",
                        max_attempts=1,
                    ),
                    DeliveryLedger(root / "delivery-ledger-http.json"),
                )
                created = client.create_session(
                    create_request(),
                    idempotency_key="real-http-create-0001",
                )
                receipt = client.submit_segment(segment_path, metadata)
                stored = app.state.stream_manager.get_session(created["analysis_session_id"])
                self.assertEqual(stored["segments"]["0"]["source_frame_count"], 30)
                app.state.stream_manager.drain()
                status = client.get_status()
                completed = client.complete(0)
                trace = client.get_trace()

                self.assertEqual(receipt["analysis_session_id"], created["analysis_session_id"])
                self.assertEqual(status["progress"]["processed_segments"], 1)
                self.assertEqual(completed["status"], "finalized")
                self.assertEqual(trace["terminal_status"], "finalized")
            finally:
                server.should_exit = True
                worker.join(timeout=5)
                listener.close()
                if previous_key is None:
                    os.environ.pop("GOOD_BADMINTON_API_KEY", None)
                else:
                    os.environ["GOOD_BADMINTON_API_KEY"] = previous_key


if __name__ == "__main__":
    unittest.main()
