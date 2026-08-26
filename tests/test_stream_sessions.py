import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from api.stream_errors import StreamSessionError
from api.stream_sessions import StreamSessionManager
from tests.stream_test_utils import (
    counting_processor_factory,
    create_request,
    segment_metadata,
    write_video_segment_bytes,
)

class _RaisingProcessor:
    def process_frame(self, frame, context):
        raise RuntimeError("injected processor failure")
    def finalize(self, context):
        return []
    def snapshot_state(self):
        return {}
    def restore_state(self, state):
        pass


class _FailingSegmentProcessor:
    """Fail one segment so the manager's recovery boundary is observable."""

    def __init__(self, failing_segment_index):
        self.failing_segment_index = int(failing_segment_index)
        self.count = 0

    def process_frame(self, _frame, context):
        if context.segment_index == self.failing_segment_index:
            raise RuntimeError(f"injected failure for segment {context.segment_index}")
        self.count += 1
        return []

    def finalize(self, _context):
        return []

    def snapshot_state(self):
        return {"count": self.count}

    def restore_state(self, state):
        self.count = int(state.get("count", 0))


class _BlockingProcessor:
    """Processor used to prove cancellation wins an in-flight commit race."""

    def __init__(self, started, release):
        self.started = started
        self.release = release
        self.count = 0

    def process_frame(self, frame, context):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test did not release the blocking processor")
        self.count += 1
        return []

    def finalize(self, context):
        return []

    def snapshot_state(self):
        return {"count": self.count}

    def restore_state(self, state):
        self.count = int(state.get("count", 0))

class StreamSessionManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.manager = StreamSessionManager(
            Path(self.temp_dir.name),
            processor_factory=counting_processor_factory(),
            start_worker=False,
        )
        self.segment_bytes = write_video_segment_bytes()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create(self, key="business-stream-0001"):
        return self.manager.create_session(create_request(), key)

    def _receive(self, session_id, index, raw=None, source_start=0.0):
        raw = raw if raw is not None else self.segment_bytes
        metadata = segment_metadata(index, raw, source_start=source_start)
        return self.manager.receive_segment(session_id, index, metadata, raw)

    def test_full_flow_create_segments_status_events_complete(self):
        status, created = self._create()
        self.assertEqual(status, 202)
        session_id = created["analysis_session_id"]
        for index in range(2):
            status, receipt = self._receive(session_id, index, source_start=index * 1.0)
            self.assertEqual(status, 202)
        self.manager.drain()
        status, status_payload = self.manager.get_status(session_id)
        self.assertEqual(status_payload["progress"]["received_segments"], 2)
        self.assertEqual(status_payload["progress"]["processed_segments"], 2)
        status, events_payload = self.manager.read_events(session_id)
        event_types = [event["event_type"] for event in events_payload["events"]]
        self.assertIn("person_observation", event_types)
        status, completed = self.manager.complete(
            session_id,
            {"schema_version": "stream-session.v1", "expected_last_segment_index": 1, "allow_partial": False},
        )
        self.assertEqual(completed["status"], "finalized")
        status, final_status = self.manager.get_status(session_id)
        self.assertEqual(final_status["status"], "finalized")

    def test_stream_request_persists_tracking_and_no_substitution_defaults(self):
        _, created = self._create("business-stream-tracker-config")
        session = self.manager.get_session(created["analysis_session_id"])
        configuration = session["configuration"]
        self.assertEqual(configuration["tracker_backend"], "court_association")
        self.assertTrue(configuration["lock_match_roster"])
        self.assertEqual(configuration["roster_stable_frames"], 3)

    def test_default_segment_timeout_is_ten_seconds(self):
        """Two-second stream fragments may not block the GPU queue indefinitely."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("GOOD_BADMINTON_STREAM_SEGMENT_TIMEOUT_SECONDS", None)
            manager = StreamSessionManager(
                Path(self.temp_dir.name) / "default-timeout",
                processor_factory=counting_processor_factory(),
                start_worker=False,
            )
        self.assertEqual(manager.segment_timeout_seconds, 10.0)

    def test_duplicate_segment_is_not_processed_twice(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        metadata = segment_metadata(0, self.segment_bytes)
        self.manager.receive_segment(session_id, 0, metadata, self.segment_bytes)
        status, duplicate = self.manager.receive_segment(session_id, 0, metadata, self.segment_bytes)
        self.assertEqual(status, 200)
        self.assertTrue(duplicate["receipt"]["reused"])
        self.manager.drain()
        status, status_payload = self.manager.get_status(session_id)
        self.assertEqual(status_payload["progress"]["processed_segments"], 1)

    def test_duplicate_segment_requires_the_same_idempotency_key(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        metadata = segment_metadata(0, self.segment_bytes)
        self.manager.receive_segment(session_id, 0, metadata, self.segment_bytes)
        changed_key = dict(metadata)
        changed_key["idempotency_key"] = "different-segment-key-000"

        with self.assertRaises(StreamSessionError) as context:
            self.manager.receive_segment(session_id, 0, changed_key, self.segment_bytes)

        self.assertEqual(context.exception.code, "invalid_state")

    def test_each_segment_must_repeat_the_business_owned_court_corners(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        metadata = segment_metadata(0, self.segment_bytes)
        metadata["court_corners"] = [[1, 1], [63, 1], [63, 63], [1, 63]]

        with self.assertRaises(StreamSessionError) as context:
            self.manager.receive_segment(session_id, 0, metadata, self.segment_bytes)

        self.assertEqual(context.exception.code, "validation_error")

    def test_same_index_different_hash_conflicts(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        self._receive(session_id, 0)
        other = b"not-a-video-but-different-bytes"
        other_metadata = segment_metadata(0, other)
        with self.assertRaises(StreamSessionError) as context:
            self.manager.receive_segment(session_id, 0, other_metadata, other)
        self.assertEqual(context.exception.code, "segment_hash_conflict")

    def test_out_of_order_segment_waits_for_predecessor(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        status, receipt = self._receive(session_id, 2, source_start=2.0)
        self.assertEqual(receipt["receipt"]["processing_disposition"], "waiting_for_predecessor")
        self.manager.drain()
        status, status_payload = self.manager.get_status(session_id)
        self.assertEqual(status_payload["progress"]["processed_segments"], 0)
        self._receive(session_id, 0, source_start=0.0)
        self._receive(session_id, 1, source_start=1.0)
        self.manager.drain()
        status, status_payload = self.manager.get_status(session_id)
        self.assertEqual(status_payload["progress"]["processed_segments"], 3)

    def test_complete_with_gap_requires_partial_then_partial_finalizes(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        self._receive(session_id, 0)
        self._receive(session_id, 1, source_start=1.0)
        self.manager.drain()
        with self.assertRaises(StreamSessionError) as context:
            self.manager.complete(
                session_id,
                {"schema_version": "stream-session.v1", "expected_last_segment_index": 2, "allow_partial": False},
            )
        self.assertEqual(context.exception.code, "missing_segments")
        status, completed = self.manager.complete(
            session_id,
            {"schema_version": "stream-session.v1", "expected_last_segment_index": 2, "allow_partial": True},
        )
        self.assertEqual(completed["status"], "partial")
        self.assertEqual(completed["missing_segment_indexes"], [2])
        checkpoint = self.manager._load_checkpoint(session_id)
        self.assertEqual(checkpoint["status"], "finalized")
        _, events = self.manager.read_events(session_id)
        self.assertTrue(
            any(event["data"].get("processor_finalized") for event in events["events"])
        )
        final_events = [
            event for event in events["events"] if event["event_type"] == "session_finalized"
        ]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0]["data"]["status"], "partial")
        self.assertEqual(final_events[0]["data"]["missing_segment_indexes"], [2])

    def test_backward_source_time_is_a_recorded_gap_without_requeue_loop(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        self._receive(session_id, 0, source_start=0.0)
        self._receive(session_id, 1, source_start=0.0)

        self.manager.drain()

        _, status_payload = self.manager.get_status(session_id)
        self.assertEqual(status_payload["status"], "running")
        self.assertEqual(status_payload["progress"]["processed_segments"], 1)
        self.assertEqual(status_payload["progress"]["failed_segment_indexes"], [1])

    def test_cancel_is_terminal_and_idempotent(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        self._receive(session_id, 0)
        status, cancelled = self.manager.cancel(session_id)
        self.assertEqual(cancelled["status"], "cancelled")
        status, again = self.manager.cancel(session_id)
        self.assertEqual(again["status"], "cancelled")
        with self.assertRaises(StreamSessionError) as context:
            self._receive(session_id, 1)
        self.assertEqual(context.exception.code, "segment_after_seal")

    def test_cancelled_session_is_not_resurrected_by_inflight_processing(self):
        started = threading.Event()
        release = threading.Event()
        manager = StreamSessionManager(
            Path(self.temp_dir.name),
            processor_factory=lambda: (_BlockingProcessor(started, release), None),
            start_worker=False,
        )
        _, created = manager.create_session(create_request(), "business-stream-cancel-race")
        session_id = created["analysis_session_id"]
        manager.receive_segment(
            session_id,
            0,
            segment_metadata(0, self.segment_bytes),
            self.segment_bytes,
        )
        worker = threading.Thread(target=manager.drain, daemon=True)
        worker.start()
        self.assertTrue(started.wait(timeout=5))

        manager.cancel(session_id)
        release.set()
        worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        _, status_payload = manager.get_status(session_id)
        self.assertEqual(status_payload["status"], "cancelled")
        self.assertEqual(status_payload["progress"]["processed_segments"], 0)

    def test_invalid_metadata_is_rejected_without_a_receipt(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        metadata = segment_metadata(0, self.segment_bytes)
        metadata["sha256"] = "0" * 64
        with self.assertRaises(StreamSessionError) as context:
            self.manager.receive_segment(session_id, 0, metadata, self.segment_bytes)
        self.assertEqual(context.exception.code, "validation_error")
        self.assertEqual(self.manager.get_session(session_id)["segments"], {})

    def test_event_cursor_uses_contract_time_order_not_append_order(self):
        _, created = self._create("business-stream-event-order")
        session_id = created["analysis_session_id"]
        base = {
            "schema_version": "stream-session.v1",
            "event_type": "session_status",
            "analysis_session_id": session_id,
            "segment_index": 0,
            "emitted_at": "2026-08-23T00:00:00Z",
            "confidence": 1.0,
            "evidence_state": "derived",
            "data": {},
        }
        self.manager._write_events(
            session_id,
            [
                {**base, "event_id": "evt_late", "source_time_sec": 2.0},
                {**base, "event_id": "evt_early_b", "source_time_sec": 1.0},
                {**base, "event_id": "evt_early_a", "source_time_sec": 1.0},
            ],
        )

        _, all_events = self.manager.read_events(session_id, limit=10)
        ordered_ids = [event["event_id"] for event in all_events["events"]]
        self.assertEqual(ordered_ids[-3:], ["evt_early_a", "evt_early_b", "evt_late"])

        _, first = self.manager.read_events(session_id, limit=2)
        _, second = self.manager.read_events(
            session_id,
            cursor=first["next_cursor"],
            limit=10,
        )
        self.assertEqual(
            [event["event_id"] for event in first["events"] + second["events"]],
            ordered_ids,
        )

    def test_failed_segment_is_recorded_and_later_segments_continue_as_partial(self):
        manager = StreamSessionManager(
            Path(self.temp_dir.name),
            processor_factory=lambda: (_FailingSegmentProcessor(1), None),
            start_worker=False,
        )
        _, created = manager.create_session(create_request(), "business-stream-soft-failure-01")
        session_id = created["analysis_session_id"]
        for index in range(3):
            manager.receive_segment(
                session_id,
                index,
                segment_metadata(index, self.segment_bytes, source_start=float(index)),
                self.segment_bytes,
            )
        manager.drain()

        _, status_payload = manager.get_status(session_id)
        self.assertEqual(status_payload["status"], "running")
        self.assertEqual(status_payload["progress"]["processed_segments"], 2)
        self.assertEqual(status_payload["progress"]["failed_segments"], 1)
        self.assertEqual(status_payload["progress"]["failed_segment_indexes"], [1])
        self.assertEqual(status_payload["progress"]["next_expected_segment_index"], 3)
        self.assertEqual(status_payload["current_stage"], "analyzing_with_errors")

        manager.complete(
            session_id,
            {
                "schema_version": "stream-session.v1",
                "expected_last_segment_index": 2,
                "allow_partial": False,
            },
        )
        _, terminal = manager.get_status(session_id)
        self.assertEqual(terminal["status"], "partial")
        self.assertEqual(terminal["progress"]["failed_segment_indexes"], [1])
        _, events = manager.read_events(session_id, limit=100)
        failure_events = [
            event for event in events["events"]
            if event["event_type"] == "session_status"
            and event.get("data", {}).get("status") == "segment_failed_continue"
        ]
        self.assertEqual(len(failure_events), 1)
        self.assertEqual(failure_events[0]["data"]["segment_failure"]["segment_index"], 1)

    def test_decode_failure_is_recorded_as_a_non_terminal_processing_gap(self):
        _, created = self._create()
        session_id = created["analysis_session_id"]
        bad = b"this-is-not-a-video-file"
        metadata = segment_metadata(0, bad)
        self.manager.receive_segment(session_id, 0, metadata, bad)
        self.manager.drain()
        status, status_payload = self.manager.get_status(session_id)
        self.assertEqual(status_payload["status"], "running")
        self.assertEqual(status_payload["progress"]["failed_segment_indexes"], [0])

    def test_watchdog_timeout_recovers_after_restart_and_continues_later_segments(self):
        restart_requests = []
        manager = StreamSessionManager(
            Path(self.temp_dir.name),
            processor_factory=counting_processor_factory(),
            hard_timeout_handler=lambda session_id, index, failure: restart_requests.append(
                (session_id, index, failure)
            ),
            start_worker=False,
        )
        _, created = manager.create_session(create_request(), "business-stream-timeout-01")
        session_id = created["analysis_session_id"]
        manager.receive_segment(session_id, 0, segment_metadata(0, self.segment_bytes), self.segment_bytes)
        manager.drain()
        for index in (1, 2):
            manager.receive_segment(
                session_id,
                index,
                segment_metadata(index, self.segment_bytes, source_start=float(index)),
                self.segment_bytes,
            )

        manager._set_active_segment(session_id, 1)
        manager._persist_timeout_then_restart(
            {"session_id": session_id, "index": 1},
            elapsed_seconds=61.0,
        )
        self.assertEqual(restart_requests[0][0:2], (session_id, 1))

        recovered = StreamSessionManager(
            Path(self.temp_dir.name),
            processor_factory=counting_processor_factory(),
            start_worker=False,
        )
        recovered.drain()
        _, progress = recovered.get_status(session_id)
        self.assertEqual(progress["progress"]["processed_segments"], 2)
        self.assertEqual(progress["progress"]["failed_segment_indexes"], [1])
        recovered.complete(
            session_id,
            {
                "schema_version": "stream-session.v1",
                "expected_last_segment_index": 2,
                "allow_partial": False,
            },
        )
        _, terminal = recovered.get_status(session_id)
        self.assertEqual(terminal["status"], "partial")

    def test_retention_cleanup_is_terminal_only_and_dry_run_by_default(self):
        manager = StreamSessionManager(
            Path(self.temp_dir.name),
            processor_factory=counting_processor_factory(),
            retention_hours=1,
            start_worker=False,
        )
        _, created = manager.create_session(create_request(), "business-stream-retention-01")
        session_id = created["analysis_session_id"]
        manager.cancel(session_id)
        future = datetime.now(timezone.utc) + timedelta(hours=2)

        candidates = manager.cleanup_expired_terminal_sessions(now=future)

        self.assertEqual([item["analysis_session_id"] for item in candidates], [session_id])
        self.assertFalse(candidates[0]["deleted"])
        self.assertIsNotNone(manager.get_session(session_id))

        deleted = manager.cleanup_expired_terminal_sessions(dry_run=False, now=future)
        self.assertTrue(deleted[0]["deleted"])
        self.assertIsNone(manager.get_session(session_id))

if __name__ == "__main__":
    unittest.main()
