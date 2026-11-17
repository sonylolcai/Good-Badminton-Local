"""Session-scoped venue credentials cannot access the GPU control plane."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.gpu_stream_app import create_gpu_stream_app
from tests.stream_test_utils import counting_processor_factory, create_request, segment_metadata, write_video_segment_bytes


class GpuDirectUploadTests(unittest.TestCase):
    def test_upload_token_only_writes_its_own_session(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"GOOD_BADMINTON_API_KEY": "service-secret",
                           "GOOD_BADMINTON_UPLOAD_TOKEN_TTL_SECONDS": "3600",
                           "GOOD_BADMINTON_STREAM_MIN_FREE_BYTES": "1"}
        ):
            app = create_gpu_stream_app(
                data_dir=Path(directory), start_worker=False,
                stream_processor_factory=counting_processor_factory(),
            )
            client = TestClient(app)
            admin = {"X-API-Key": "service-secret"}
            first = client.post("/api/v1/stream-sessions", json=create_request(), headers=admin)
            self.assertEqual(first.status_code, 202)
            first_id = first.json()["analysis_session_id"]
            second_request = create_request()
            second_request["camera_id"] = "court-02-camera-a"
            second = client.post("/api/v1/stream-sessions", json=second_request, headers=admin)
            self.assertEqual(second.status_code, 202)
            second_id = second.json()["analysis_session_id"]
            grant = client.post(f"/api/v1/stream-sessions/{first_id}/upload-authorization", headers=admin)
            self.assertEqual(grant.status_code, 200)
            upload_headers = {"X-Upload-Token": grant.json()["upload_token"]}
            raw = write_video_segment_bytes()
            payload = {"metadata": json.dumps(segment_metadata(0, raw))}
            files = {"segment": ("segment.mp4", raw, "video/mp4")}
            wrong = client.post(f"/api/v1/stream-sessions/{second_id}/segments/0",
                                data=payload, files=files, headers=upload_headers)
            self.assertEqual(wrong.status_code, 401)
            accepted = client.post(f"/api/v1/stream-sessions/{first_id}/segments/0",
                                   data=payload, files=files, headers=upload_headers)
            self.assertEqual(accepted.status_code, 202)
            self.assertEqual(client.get(f"/api/v1/stream-sessions/{first_id}", headers=upload_headers).status_code, 401)
            receipt = client.get(f"/api/v1/stream-sessions/{first_id}/segments/0/receipt", headers=admin)
            self.assertTrue(receipt.json()["gpu_accepted"])
            self.assertEqual(client.delete(f"/api/v1/stream-sessions/{first_id}/upload-authorization", headers=admin).status_code, 200)
            retried = client.post(f"/api/v1/stream-sessions/{first_id}/segments/0",
                                  data=payload, files=files, headers=upload_headers)
            self.assertEqual(retried.status_code, 401)
