import json
import tempfile
import unittest
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, UploadFile
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api.stream_errors import StreamSessionError
from api.stream_sessions import (
    StreamSessionManager,
    cancel_session_handler,
    complete_session_handler,
    create_session_handler,
    get_session_status_handler,
    read_events_handler,
    submit_segment_handler,
)
from tests.stream_test_utils import (
    counting_processor_factory,
    create_request,
    segment_metadata,
    write_video_segment_bytes,
)

def _build_test_app(manager):
    app = FastAPI(title="stream-session test harness")
    app.state.stream_manager = manager

    def require_api_key(x_api_key: str = Header(default=None)):
        if x_api_key != "test-api-key":
            raise StreamSessionError("authentication_required", "Invalid API key", 401)

    @app.exception_handler(StreamSessionError)
    async def handle_stream_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content=exc.to_error_response())

    @app.post("/api/v1/stream-sessions")
    async def create(body: dict, x_idempotency_key: str = Header(default=None), _=Depends(require_api_key)):
        status, payload = create_session_handler(manager, body, x_idempotency_key)
        return JSONResponse(status_code=status, content=payload)

    @app.post("/api/v1/stream-sessions/{session_id}/segments/{segment_index}")
    async def submit(session_id: str, segment_index: int, segment: UploadFile = File(...), metadata: str = Form(...), _=Depends(require_api_key)):
        metadata_body = json.loads(metadata)
        data_bytes = await segment.read()
        status, payload = submit_segment_handler(manager, session_id, segment_index, metadata_body, data_bytes)
        return JSONResponse(status_code=status, content=payload)

    @app.get("/api/v1/stream-sessions/{session_id}")
    async def status(session_id: str, _=Depends(require_api_key)):
        status, payload = get_session_status_handler(manager, session_id)
        return JSONResponse(status_code=status, content=payload)

    @app.get("/api/v1/stream-sessions/{session_id}/events")
    async def events(session_id: str, cursor: str = None, limit: int = 100, _=Depends(require_api_key)):
        status, payload = read_events_handler(manager, session_id, cursor=cursor, limit=limit)
        return JSONResponse(status_code=status, content=payload)

    @app.get("/api/v1/stream-sessions/{session_id}/trace")
    async def trace(session_id: str, _=Depends(require_api_key)):
        # The production route streams the file; this handler-level harness
        # returns the same JSON contract without duplicating response plumbing.
        get_session_status_handler(manager, session_id)
        path = manager.trace_path(session_id)
        if path is None:
            raise StreamSessionError("trace_not_ready", "Trace is not available", 404)
        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))

    @app.post("/api/v1/stream-sessions/{session_id}/complete")
    async def complete(session_id: str, body: dict, _=Depends(require_api_key)):
        status, payload = complete_session_handler(manager, session_id, body)
        return JSONResponse(status_code=status, content=payload)

    @app.delete("/api/v1/stream-sessions/{session_id}")
    async def cancel(session_id: str, _=Depends(require_api_key)):
        status, payload = cancel_session_handler(manager, session_id)
        return JSONResponse(status_code=status, content=payload)

    return app

class StreamApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.manager = StreamSessionManager(
            Path(self.temp_dir.name),
            processor_factory=counting_processor_factory(),
            start_worker=False,
        )
        self.client = TestClient(_build_test_app(self.manager))
        self.segment_bytes = write_video_segment_bytes()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_session(self):
        response = self.client.post(
            "/api/v1/stream-sessions",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-stream-api-01"},
            json=create_request(),
        )
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["analysis_session_id"]

    def _submit(self, session_id, index):
        metadata = segment_metadata(index, self.segment_bytes, source_start=index * 1.0)
        return self.client.post(
            f"/api/v1/stream-sessions/{session_id}/segments/{index}",
            headers={"X-API-Key": "test-api-key"},
            files={"segment": (f"seg{index}.mp4", self.segment_bytes, "video/mp4")},
            data={"metadata": json.dumps(metadata)},
        )

    def test_full_stream_flow_over_http(self):
        session_id = self._create_session()
        for index in range(2):
            response = self._submit(session_id, index)
            self.assertEqual(response.status_code, 202, response.text)
        self.manager.drain()

        status = self.client.get(f"/api/v1/stream-sessions/{session_id}", headers={"X-API-Key": "test-api-key"})
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["progress"]["processed_segments"], 2)

        events = self.client.get(f"/api/v1/stream-sessions/{session_id}/events", headers={"X-API-Key": "test-api-key"})
        self.assertEqual(events.status_code, 200)
        self.assertGreaterEqual(len(events.json()["events"]), 2)

        complete = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/complete",
            headers={"X-API-Key": "test-api-key"},
            json={"schema_version": "stream-session.v1", "expected_last_segment_index": 1, "allow_partial": False},
        )
        self.assertEqual(complete.status_code, 202, complete.text)
        self.assertEqual(complete.json()["status"], "finalized")

    def test_hash_conflict_returns_structured_error(self):
        session_id = self._create_session()
        self._submit(session_id, 0)
        other = b"different-bytes-not-video"
        other_metadata = segment_metadata(0, other)
        response = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/segments/0",
            headers={"X-API-Key": "test-api-key"},
            files={"segment": ("seg0.mp4", other, "video/mp4")},
            data={"metadata": json.dumps(other_metadata)},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "segment_hash_conflict")

    def test_missing_segments_returns_retryable_conflict(self):
        session_id = self._create_session()
        self._submit(session_id, 0)
        response = self.client.post(
            f"/api/v1/stream-sessions/{session_id}/complete",
            headers={"X-API-Key": "test-api-key"},
            json={"schema_version": "stream-session.v1", "expected_last_segment_index": 2, "allow_partial": False},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "missing_segments")
        self.assertTrue(response.json()["error"]["retryable"])

    def test_missing_api_key_returns_401(self):
        response = self.client.post("/api/v1/stream-sessions", json=create_request())
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "authentication_required")

    def test_create_rejects_business_identity_fields_instead_of_dropping_them(self):
        body = create_request()
        body["check_id"] = "business-user-should-not-cross-boundary"
        response = self.client.post(
            "/api/v1/stream-sessions",
            headers={
                "X-API-Key": "test-api-key",
                "X-Idempotency-Key": "business-stream-api-extra-01",
            },
            json=body,
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_error")
        self.assertIn("unsupported keys", response.json()["error"]["message"])

if __name__ == "__main__":
    unittest.main()
