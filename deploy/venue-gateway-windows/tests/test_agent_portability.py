import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import (
    EdgeClient,
    Settings,
    ffmpeg_command,
    load_active_session,
    load_env_file,
    save_active_session,
)
from business_gateway.edge_contract import payload_sha256, sign_request


class AgentPortabilityTests(unittest.TestCase):
    def test_env_file_preserves_values_after_first_equals(self):
        with tempfile.TemporaryDirectory() as temp:
            env_file = Path(temp) / "gateway.env"
            env_file.write_text("CAMERA_RTSP_URL=rtsp://user:p=a@camera/stream?x=1&y=2\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                load_env_file(env_file)
                self.assertEqual(os.environ["CAMERA_RTSP_URL"], "rtsp://user:p=a@camera/stream?x=1&y=2")

    def test_env_file_accepts_windows_utf8_bom(self):
        with tempfile.TemporaryDirectory() as temp:
            env_file = Path(temp) / "gateway.env"
            env_file.write_text("EDGE_DEVICE_ID=device-1\n", encoding="utf-8-sig")
            with patch.dict(os.environ, {}, clear=True):
                load_env_file(env_file)
                self.assertEqual(os.environ["EDGE_DEVICE_ID"], "device-1")

    def test_settings_and_ffmpeg_command_accept_windows_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            values = {
                "EDGE_GATEWAY_URL": "https://example.test/badminton-edge",
                "EDGE_DEVICE_ID": "device-1",
                "EDGE_CAMERA_ID": "camera-1",
                "EDGE_DEVICE_SECRET": "secret",
                "CAMERA_RTSP_URL": "rtsp://camera/stream",
                "COURT_CORNERS_JSON": "[[0,0],[1,0],[1,1],[0,1]]",
                "FFMPEG_BIN": r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
                "SPOOL_DIR": temp,
            }
            with patch.dict(os.environ, values, clear=True):
                settings = Settings.from_environment()
            command = ffmpeg_command(settings, Path(temp) / "%08d.mp4")
            self.assertEqual(command[0], values["FFMPEG_BIN"])
            self.assertIn("-rtsp_transport", command)
            self.assertEqual(command[-1], str(Path(temp) / "%08d.mp4"))

    def test_ffmpeg_resume_uses_next_segment_number(self):
        settings = Settings(
            base_url="https://example.test",
            device_id="device-1",
            camera_id="camera-1",
            credential_version="v1",
            device_secret="secret",
            rtsp_url="rtsp://camera/stream",
            ffmpeg_bin="ffmpeg",
            spool_dir=Path("spool"),
            segment_seconds=2,
            heartbeat_seconds=10,
            corners=[[0, 0], [1, 0], [1, 1], [0, 1]],
        )
        command = ffmpeg_command(settings, Path("%08d.mp4"), start_number=17)
        self.assertEqual(command[command.index("-segment_start_number") + 1], "17")

    def test_active_session_state_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            state_path = Path(temp) / "active-session.json"
            save_active_session(state_path, "edge-session-1", 16)
            self.assertEqual(load_active_session(state_path), ("edge-session-1", 16))

    def test_start_case_adopts_server_reported_active_session(self):
        class ConflictResponse:
            status_code = 409
            text = '{"detail":{"active_session_id":"edge-session-existing"}}'

            @staticmethod
            def json():
                return {"detail": {"active_session_id": "edge-session-existing"}}

            @staticmethod
            def raise_for_status():
                raise AssertionError("409 should be handled before raise_for_status")

        settings = Settings(
            base_url="https://example.test",
            device_id="device-1",
            camera_id="camera-1",
            credential_version="v1",
            device_secret="secret",
            rtsp_url="rtsp://camera/stream",
            ffmpeg_bin="ffmpeg",
            spool_dir=Path("spool"),
            segment_seconds=2,
            heartbeat_seconds=10,
            corners=[[0, 0], [1, 0], [1, 1], [0, 1]],
        )
        client = EdgeClient(settings)
        with patch.object(client.session, "post", return_value=ConflictResponse()):
            self.assertEqual(client.start_case(), "edge-session-existing")

    def test_edge_signature_is_deterministic(self):
        body = {"schema_version": "edge-ingest.v1", "device_id": "d"}
        digest = payload_sha256(body)
        first = sign_request("secret", "POST", "/path", "2026-09-01T00:00:00Z", "nonce-value-123456", digest)
        second = sign_request("secret", "POST", "/path", "2026-09-01T00:00:00Z", "nonce-value-123456", digest)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
