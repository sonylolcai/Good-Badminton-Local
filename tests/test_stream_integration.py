"""End-to-end integration: real video bytes through decoder, engine, restart, finalize.

This test proves the task A -> task B boundary: OpenCVSegmentDecoder decodes the
stored segment, AnalysisEngine processes it, events.jsonl and checkpoint.json are
persisted, a simulated restart restores the engine from checkpoint, and finalize
emits the session_finalized event.
"""

import json
import tempfile
import unittest
from pathlib import Path

from api.stream_sessions import StreamSessionManager
from tests.stream_test_utils import (
    counting_processor_factory,
    create_request,
    segment_metadata,
    write_video_segment_bytes,
)

class StreamIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.data_dir = Path(self.temp_dir.name)
        self.segment_bytes = write_video_segment_bytes(frame_count=30, fps=30.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_and_receive(self, manager, session_id, index, source_start):
        metadata = segment_metadata(index, self.segment_bytes, source_start=source_start)
        manager.receive_segment(session_id, index, metadata, self.segment_bytes)

    def test_full_chain_decode_engine_checkpoint_restart_finalize(self):
        manager = StreamSessionManager(
            self.data_dir,
            processor_factory=counting_processor_factory(),
            start_worker=False,
        )
        _, created = manager.create_session(create_request(), "business-stream-integration-01")
        session_id = created["analysis_session_id"]

        self._create_and_receive(manager, session_id, 0, 0.0)
        manager.drain()

        # Events and checkpoint are durable.
        events_path = manager._events_path(session_id)
        checkpoint_path = manager._checkpoint_path(session_id)
        self.assertTrue(events_path.is_file())
        self.assertTrue(checkpoint_path.is_file())
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        event_ids = [event["event_id"] for event in events]
        self.assertEqual(len(event_ids), len(set(event_ids)))  # no duplicates
        self.assertTrue(any(event["event_type"] == "person_observation" for event in events))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["next_expected_segment_index"], 1)

        # Simulated restart: a fresh manager over the same directory restores
        # the engine from the checkpoint with brand-new processor instances.
        restarted = StreamSessionManager(
            self.data_dir,
            processor_factory=counting_processor_factory(),
            start_worker=False,
        )
        recovered = restarted.get_session(session_id)
        self.assertNotEqual(recovered["status"], "interrupted_needs_rebuild")
        self.assertIn(session_id, restarted._engines)

        self._create_and_receive(restarted, session_id, 1, 1.0)
        restarted.drain()

        # Continuity: the restored processor resumed its counter, so segment 1
        # events continue from segment 0 rather than restarting at 1.
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        observation_counts = [
            event["data"]["processor_count"]
            for event in events
            if event["event_type"] == "person_observation"
        ]
        self.assertGreater(min(observation_counts[10:]), max(observation_counts[:10]))

        status, completed = restarted.complete(
            session_id,
            {"schema_version": "stream-session.v1", "expected_last_segment_index": 1, "allow_partial": False},
        )
        self.assertEqual(completed["status"], "finalized")

        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(any(event["event_type"] == "session_finalized" for event in events))
        status, status_payload = restarted.get_status(session_id)
        self.assertEqual(status_payload["status"], "finalized")

if __name__ == "__main__":
    unittest.main()
