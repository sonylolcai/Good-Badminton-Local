import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import EdgeClient, Settings
from business_gateway.edge_contract import (
    EDGE_NONCE_HEADER,
    EDGE_PAYLOAD_SHA256_HEADER,
    EDGE_SIGNATURE_HEADER,
    EDGE_TIMESTAMP_HEADER,
)


class EdgeClientTransportTests(unittest.TestCase):
    def test_full_signed_control_and_segment_transport(self):
        received = []
        test_case = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                pass

            def do_POST(self):
                content_length = int(self.headers["Content-Length"])
                body = self.rfile.read(content_length)
                for header in (
                    EDGE_TIMESTAMP_HEADER,
                    EDGE_NONCE_HEADER,
                    EDGE_PAYLOAD_SHA256_HEADER,
                    EDGE_SIGNATURE_HEADER,
                ):
                    test_case.assertTrue(self.headers.get(header), header)
                received.append((self.path, self.headers.get("Content-Type", ""), body))
                if self.path.endswith("/heartbeats"):
                    response = {"capture_control": {"mode": "idle", "revision": 1}}
                elif self.path.endswith("/sessions"):
                    response = {"edge_ingest_session_id": "edge_session_smoke_1"}
                elif self.path.endswith("/complete"):
                    response = {"status": "accepted"}
                else:
                    response = {"status": "accepted", "segment_index": 0}
                encoded = json.dumps(response).encode("utf-8")
                self.send_response(202 if "/sessions" in self.path else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                settings = Settings(
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    device_id="device-smoke-1",
                    camera_id="camera-smoke-1",
                    credential_version="v1",
                    device_secret="test-secret",
                    rtsp_url="rtsp://camera/stream",
                    ffmpeg_bin="ffmpeg",
                    spool_dir=Path(temp),
                    segment_seconds=2,
                    heartbeat_seconds=10,
                    corners=[[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
                )
                client = EdgeClient(settings)
                self.assertEqual(
                    client.heartbeat(
                        "error",
                        None,
                        None,
                        {"code": "ffmpeg_exited", "message": "ffmpeg exited with 1", "count": 1},
                    )["capture_control"]["mode"],
                    "idle",
                )
                session_id = client.start_case()
                segment = Path(temp) / "00000000.mp4"
                segment.write_bytes(b"independently-playable-mp4-smoke-segment")
                client.upload(session_id, 0, segment)
                client.complete(session_id, 0)
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(
            [path for path, _, _ in received],
            [
                "/api/v1/edge/devices/device-smoke-1/heartbeats",
                "/api/v1/edge/devices/device-smoke-1/sessions",
                "/api/v1/edge/sessions/edge_session_smoke_1/segments/0",
                "/api/v1/edge/sessions/edge_session_smoke_1/complete",
            ],
        )
        upload_content_type = received[2][1]
        upload_body = received[2][2]
        heartbeat_body = json.loads(received[0][2])
        self.assertEqual(heartbeat_body["error_report"]["code"], "ffmpeg_exited")
        self.assertIn("multipart/form-data", upload_content_type)
        self.assertIn(b'name="metadata"', upload_body)
        self.assertIn(b'name="segment"', upload_body)
        self.assertIn(b"independently-playable-mp4-smoke-segment", upload_body)


if __name__ == "__main__":
    unittest.main()
