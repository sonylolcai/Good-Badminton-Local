"""Repair-window semantics: bounded grants, idempotent openings, seal blocks.

The window is the only way a venue can hand back bytes for a fragment the GPU has
already counted as lost, so these tests pin down who may open one, what a grant
promises, and what the session must refuse while it is open.
"""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from api.stream_errors import StreamSessionError
from api.stream_sessions import StreamSessionManager
from tests.stream_test_utils import (
    counting_processor_factory,
    create_request,
    segment_metadata,
    write_video_segment_bytes,
)


class SegmentRepairWindowTests(unittest.TestCase):
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

    def _gap_session(self, key="business-stream-repair"):
        """Return a session whose middle fragment is declared lost and unsettled."""
        _, created = self.manager.create_session(create_request(), key)
        session_id = created["analysis_session_id"]
        self.manager.receive_segment(
            session_id, 0, segment_metadata(0, self.segment_bytes, source_start=0.0),
            self.segment_bytes)
        self.manager.receive_segment(
            session_id, 2, segment_metadata(2, self.segment_bytes, source_start=2.0),
            self.segment_bytes)
        self.manager.declare_gap(session_id, 1, {
            "schema_version": "stream-session.v1", "source_start_time_sec": 1.0,
            "duration_sec": 1.0, "reason": "source_segment_missing"})
        return session_id

    @staticmethod
    def _request(index=1, kind="missing", ttl=600):
        return {"schema_version": "stream-session.v1", "reason": "business_gap_inventory",
                "ttl_seconds": ttl, "ranges": [{"segment_index": index, "kind": kind}]}

    def _complete_body(self):
        return {"schema_version": "stream-session.v1", "expected_last_segment_index": 2,
                "allow_partial": True}

    def _expire_grant(self, session_id, request_id, index):
        with self.manager._lock:
            session = self.manager._load(session_id)
            past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace(
                "+00:00", "Z")
            session["repair"]["requests"][request_id]["grants"][str(index)][
                "grant_expires_at"] = past
            self.manager._save(session)

    def test_declared_gap_receives_a_one_time_grant_for_its_interval(self):
        session_id = self._gap_session()
        code, payload = self.manager.request_repairs(
            session_id, self._request(), "repair-window-key-0001")
        self.assertEqual(code, 202)
        self.assertEqual(payload["status"], "open")
        self.assertEqual(len(payload["ranges"]), 1)
        granted = payload["ranges"][0]
        self.assertEqual(granted["state"], "open")
        # The expected interval is copied from the declaration, never invented here.
        self.assertEqual(
            granted["expected"], {"source_start_time_sec": 1.0, "duration_sec": 1.0})
        self.assertTrue(granted["grant_token"])
        self.assertTrue(self.manager.get_session(session_id)["repair"]["window_open"])

    def test_idempotent_retry_reuses_the_window_and_conflicts_on_new_intent(self):
        session_id = self._gap_session()
        first = self.manager.request_repairs(
            session_id, self._request(), "repair-window-key-0002")
        retry = self.manager.request_repairs(
            session_id, self._request(), "repair-window-key-0002")
        self.assertEqual(retry[0], 200)
        self.assertTrue(retry[1]["reused"])
        self.assertEqual(retry[1]["request_id"], first[1]["request_id"])
        with self.assertRaises(StreamSessionError) as context:
            self.manager.request_repairs(
                session_id, self._request(index=2), "repair-window-key-0002")
        self.assertEqual(context.exception.code, "repair_idempotency_conflict")

    def test_corrupt_replacement_needs_a_gpu_integrity_verdict(self):
        session_id = self._gap_session()
        with self.assertRaises(StreamSessionError) as context:
            self.manager.request_repairs(
                session_id, self._request(index=0, kind="corrupt"), "repair-window-key-0003")
        self.assertEqual(context.exception.code, "repair_not_granted")
        self.assertEqual(context.exception.status_code, 403)

    def test_indexes_without_a_declared_gap_are_refused(self):
        _, created = self.manager.create_session(
            create_request(), "business-stream-repair-no-gap")
        session_id = created["analysis_session_id"]
        with self.assertRaises(StreamSessionError) as context:
            self.manager.request_repairs(
                session_id, self._request(index=5), "repair-window-key-0004")
        self.assertEqual(context.exception.code, "invalid_state")
        # A refused request must not leave a half-open window behind.
        self.assertFalse(self.manager.get_session(session_id)["repair"]["window_open"])

    def test_window_size_limits_are_enforced(self):
        session_id = self._gap_session()
        for ttl in (30, 100000):
            with self.assertRaises(StreamSessionError) as context:
                self.manager.request_repairs(
                    session_id, self._request(ttl=ttl), "repair-window-ttl")
            self.assertEqual(context.exception.code, "validation_error")
        too_many = {
            "schema_version": "stream-session.v1", "reason": "business_gap_inventory",
            "ttl_seconds": 600,
            "ranges": [{"segment_index": i, "kind": "missing"} for i in range(65)],
        }
        with self.assertRaises(StreamSessionError) as context:
            self.manager.request_repairs(session_id, too_many, "repair-window-many")
        self.assertEqual(context.exception.code, "validation_error")

    def test_a_second_window_cannot_double_grant_an_index(self):
        session_id = self._gap_session()
        self.manager.request_repairs(session_id, self._request(), "repair-window-key-0005")
        with self.assertRaises(StreamSessionError) as context:
            self.manager.request_repairs(session_id, self._request(), "repair-window-key-0006")
        self.assertEqual(context.exception.code, "repair_already_granted")

    def test_lapsed_grants_expire_without_a_sweeper(self):
        session_id = self._gap_session()
        _, payload = self.manager.request_repairs(
            session_id, self._request(), "repair-window-key-0007")
        self._expire_grant(session_id, payload["request_id"], 1)
        inventory = self.manager.list_repairs(session_id)[1]
        self.assertFalse(inventory["window_open"])
        self.assertEqual(inventory["requests"][0]["status"], "closed")
        self.assertEqual(inventory["requests"][0]["ranges"][0]["state"], "expired")
        # A lapsed grant releases its index; the window does not linger forever.
        self.assertEqual(
            self.manager.request_repairs(
                session_id, self._request(), "repair-window-key-0008")[0],
            202,
        )

    def test_reconciliation_never_reissues_grant_tokens(self):
        session_id = self._gap_session()
        _, payload = self.manager.request_repairs(
            session_id, self._request(), "repair-window-key-0009")
        inventory = self.manager.list_repairs(session_id)[1]
        granted = inventory["requests"][0]["ranges"][0]
        self.assertNotIn("grant_token", granted)
        self.assertTrue(granted["grant_token_issued"])
        self.assertEqual(inventory["window_expires_at"], payload["window_expires_at"])

    def test_abandon_is_idempotent_and_releases_the_index(self):
        session_id = self._gap_session()
        _, payload = self.manager.request_repairs(
            session_id, self._request(), "repair-window-key-0010")
        request_id = payload["request_id"]
        first = self.manager.abandon_repair(session_id, request_id)
        self.assertEqual(first[0], 200)
        self.assertEqual(first[1]["status"], "abandoned")
        self.assertEqual(first[1]["released_segment_indexes"], [1])
        repeat = self.manager.abandon_repair(session_id, request_id)
        self.assertEqual(repeat[1]["status"], "abandoned")
        self.assertEqual(repeat[1]["released_segment_indexes"], [])
        self.assertFalse(self.manager.get_session(session_id)["repair"]["window_open"])
        self.assertEqual(
            self.manager.request_repairs(
                session_id, self._request(), "repair-window-key-0011")[0],
            202,
        )
        with self.assertRaises(StreamSessionError) as context:
            self.manager.abandon_repair(session_id, "rp_unknown")
        self.assertEqual(context.exception.status_code, 404)

    def test_sealing_is_blocked_while_a_repair_window_is_open(self):
        session_id = self._gap_session()
        _, payload = self.manager.request_repairs(
            session_id, self._request(), "repair-window-key-0012")
        with self.assertRaises(StreamSessionError) as context:
            self.manager.complete(session_id, self._complete_body())
        self.assertEqual(context.exception.code, "repair_in_progress")
        self.assertEqual(context.exception.details["repairing_segment_indexes"], [1])
        self.assertFalse(self.manager.get_session(session_id)["sealed"])
        self.manager.abandon_repair(session_id, payload["request_id"])
        code, _ = self.manager.complete(session_id, self._complete_body())
        self.assertEqual(code, 202)
        self.assertTrue(self.manager.get_session(session_id)["sealed"])


if __name__ == "__main__":
    unittest.main()
