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
from badminton_analysis.streaming import ProcessorEvent
from tests.stream_test_utils import (
    counting_processor_factory,
    create_request,
    segment_metadata,
    write_video_segment_bytes,
)


class _TrackEmittingProcessor:
    """Emits the nested track shape real detectors use, so epoch naming is visible."""

    def __init__(self):
        self.count = 0

    def process_frame(self, frame, context):
        self.count += 1
        return [
            ProcessorEvent(
                event_type="person_observation",
                evidence_state="detected",
                confidence=0.9,
                data={"track": {"track_id": "track_001"}, "processor_count": self.count},
            )
        ]

    def finalize(self, context):
        return []

    def snapshot_state(self):
        return {"count": self.count}

    def restore_state(self, state):
        self.count = int(state.get("count", 0))


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


class SegmentRepairDataPlaneTests(unittest.TestCase):
    """Delivering bytes against a grant, and what the session must do with them."""

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

    def _settled_gap_session(self, key="business-stream-repair-dp"):
        """A session that already moved on past a declared gap (epoch 1)."""
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
        self.manager.drain()
        self.assertEqual(self.manager.get_session(session_id)["continuity_epoch"], 1)
        return session_id

    def _grant(self, session_id, key="repair-dp-window-0001"):
        _, payload = self.manager.request_repairs(
            session_id,
            {"schema_version": "stream-session.v1", "reason": "business_gap_inventory",
             "ttl_seconds": 600, "ranges": [{"segment_index": 1, "kind": "missing"}]},
            key)
        return payload["request_id"], payload["ranges"][0]["grant_token"]

    def _repair_metadata(self, request_id, index=1, source_start=1.0):
        return {
            **segment_metadata(index, self.segment_bytes, source_start=source_start),
            "repair_request_id": request_id,
        }

    def test_repaired_gap_is_reanalyzed_in_its_own_epoch(self):
        session_id = self._settled_gap_session()
        head_engine = self.manager._engines[session_id]
        before_page = self.manager.read_events(session_id, limit=10000)[1]
        before_events = len(before_page["events"])
        consumed_cursor = before_page["next_cursor"]
        request_id, grant_token = self._grant(session_id)

        code, response = self.manager.repair_segment(
            session_id, 1, self._repair_metadata(request_id), self.segment_bytes, grant_token)
        self.assertEqual(code, 200)
        receipt = response["receipt"]
        self.assertTrue(receipt["repaired"])
        self.assertTrue(receipt["requeued"])
        self.assertEqual(receipt["generation"], 1)
        self.manager.drain()

        session = self.manager.get_session(session_id)
        entry = session["segments"]["1"]
        self.assertTrue(entry["processed"])
        self.assertEqual(session["repair_reanalysis"], {})
        # The forward timeline never regressed and the head engine was never replaced.
        self.assertIs(self.manager._engines[session_id], head_engine)
        self.assertEqual(session["continuity_epoch"], 1)
        self.assertEqual(self.manager._next_expected_index(session), 3)
        # The repair published a strictly newer epoch, and the log got a new revision.
        self.assertEqual(entry["reanalysis_continuity_epoch"], 2)
        self.assertEqual(session["max_epoch_issued"], 2)
        self.assertEqual(session["events_revision"], 2)
        # The gap stays on record as resolved rather than vanishing.
        self.assertEqual(session["declared_gaps"]["1"]["resolved_by"], request_id)

        events = self.manager.read_events(session_id, limit=10000)[1]["events"]
        self.assertGreater(len(events), before_events)
        repaired = [
            event for event in events
            if event.get("segment_index") == 1 and event["data"].get("repair_kind") == "missing"
        ]
        self.assertTrue(repaired)
        for event in repaired:
            # Evidence time stays the fragment's real source time; only identity moves.
            self.assertTrue(event["data"]["repaired"])
            self.assertTrue(1.0 <= event["source_time_sec"] < 2.0)
            self.assertEqual(event["data"]["continuity_epoch"], 2)
            self.assertEqual(event["data"]["repair_request_id"], request_id)
        track_ids = {
            event["data"].get("track", {}).get("track_id")
            for event in repaired
            if event["data"].get("track")
        }
        # This fixture's processor reports a flat track_id, so identity isolation is
        # asserted through the epoch field; nested track naming has its own test below.
        self.assertEqual(track_ids, set())
        self.assertTrue(all(event["data"]["continuity_epoch"] == 2 for event in repaired))
        # No pre-existing event was overwritten or deduplicated away.
        self.assertEqual(
            len({event["event_id"] for event in events}), len(events))
        status = self.manager.get_status(session_id)[1]
        self.assertEqual(status["progress"]["missing_segment_indexes"], [])
        self.assertEqual(status["events_revision"], 2)
        # Paging on the cursor the business side already consumed can never surface the
        # repaired rows, because the log is ordered by evidence time.  The watermark in
        # events_revision is the only signal that says "re-pull from the beginning".
        tail = self.manager.read_events(session_id, cursor=consumed_cursor, limit=10000)[1]
        self.assertFalse(
            {event["event_id"] for event in tail["events"]}
            & {event["event_id"] for event in repaired}
        )

    def test_wrong_grant_token_is_rejected(self):
        session_id = self._settled_gap_session()
        request_id, _ = self._grant(session_id)
        with self.assertRaises(StreamSessionError) as context:
            self.manager.repair_segment(
                session_id, 1, self._repair_metadata(request_id), self.segment_bytes,
                "not-the-granted-token")
        self.assertEqual(context.exception.code, "invalid_repair_grant")
        self.assertEqual(context.exception.status_code, 401)
        self.assertNotIn("1", self.manager.get_session(session_id)["segments"])

    def test_index_outside_the_grant_is_refused(self):
        session_id = self._settled_gap_session()
        request_id, token = self._grant(session_id)
        with self.assertRaises(StreamSessionError) as context:
            self.manager.repair_segment(
                session_id, 5, self._repair_metadata(request_id, index=5, source_start=5.0),
                self.segment_bytes, token)
        self.assertEqual(context.exception.code, "repair_not_granted")
        self.assertEqual(context.exception.status_code, 403)

    def test_bytes_for_a_different_interval_are_refused(self):
        session_id = self._settled_gap_session()
        request_id, token = self._grant(session_id)
        with self.assertRaises(StreamSessionError) as context:
            self.manager.repair_segment(
                session_id, 1, self._repair_metadata(request_id, source_start=1.5),
                self.segment_bytes, token)
        self.assertEqual(context.exception.code, "repair_time_mismatch")
        self.assertEqual(context.exception.status_code, 422)
        # A refusal must not leave orphan bytes behind.
        self.assertFalse(self.manager._segment_path(session_id, 1).is_file())

    def test_lapsed_grant_cannot_be_redeemed(self):
        session_id = self._settled_gap_session()
        request_id, token = self._grant(session_id)
        with self.manager._lock:
            session = self.manager._load(session_id)
            past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace(
                "+00:00", "Z")
            session["repair"]["requests"][request_id]["grants"]["1"]["grant_expires_at"] = past
            self.manager._save(session)
        with self.assertRaises(StreamSessionError) as context:
            self.manager.repair_segment(
                session_id, 1, self._repair_metadata(request_id), self.segment_bytes, token)
        self.assertEqual(context.exception.code, "repair_window_closed")
        self.assertEqual(context.exception.details["grant_state"], "expired")

    def test_repair_after_seal_is_rejected(self):
        session_id = self._settled_gap_session()
        request_id, token = self._grant(session_id)
        self.manager.abandon_repair(session_id, request_id)
        self.manager.complete(session_id, {
            "schema_version": "stream-session.v1", "expected_last_segment_index": 2,
            "allow_partial": True})
        with self.assertRaises(StreamSessionError) as context:
            self.manager.repair_segment(
                session_id, 1, self._repair_metadata(request_id), self.segment_bytes, token)
        self.assertEqual(context.exception.code, "repair_after_seal")

    def test_a_second_delivery_reuses_the_first_generation(self):
        session_id = self._settled_gap_session()
        request_id, token = self._grant(session_id)
        metadata = self._repair_metadata(request_id)
        first = self.manager.repair_segment(
            session_id, 1, metadata, self.segment_bytes, token)[1]["receipt"]
        self.manager.drain()
        retry = self.manager.repair_segment(
            session_id, 1, metadata, self.segment_bytes, token)[1]["receipt"]
        self.assertTrue(retry["reused"])
        self.assertEqual(retry["generation"], first["generation"])
        self.assertEqual(retry["received_at"], first["received_at"])
        session = self.manager.get_session(session_id)
        self.assertEqual(session["segments"]["1"]["received_at"], first["received_at"])
        self.assertEqual(session["repair_reanalysis"], {})
        # A different digest under the same (now spent) grant earns no third generation.
        other_bytes = write_video_segment_bytes(frame_count=12)
        with self.assertRaises(StreamSessionError) as context:
            self.manager.repair_segment(
                session_id, 1,
                {**segment_metadata(1, other_bytes, source_start=1.0),
                 "repair_request_id": request_id},
                other_bytes, token)
        self.assertEqual(context.exception.code, "repair_window_closed")
        self.assertEqual(self.manager.get_session(session_id)["segments"]["1"]["sha256"],
                         first["sha256"])

    def test_an_open_grant_does_not_loosen_the_ordinary_upload_path(self):
        session_id = self._settled_gap_session()
        self._grant(session_id)
        with self.assertRaises(StreamSessionError) as context:
            self.manager.receive_segment(
                session_id, 1, segment_metadata(1, self.segment_bytes, source_start=1.0),
                self.segment_bytes)
        self.assertEqual(context.exception.code, "invalid_state")

    def test_repaired_tracks_are_not_linked_across_the_repair_boundary(self):
        """Epoch naming must stop a repair from claiming the same people on both sides."""
        manager = StreamSessionManager(
            Path(self.temp_dir.name),
            processor_factory=lambda: (_TrackEmittingProcessor(), None),
            start_worker=False,
        )
        _, created = manager.create_session(
            create_request(), "business-stream-repair-tracks")
        session_id = created["analysis_session_id"]
        manager.receive_segment(
            session_id, 0, segment_metadata(0, self.segment_bytes, source_start=0.0),
            self.segment_bytes)
        manager.receive_segment(
            session_id, 2, segment_metadata(2, self.segment_bytes, source_start=2.0),
            self.segment_bytes)
        manager.declare_gap(session_id, 1, {
            "schema_version": "stream-session.v1", "source_start_time_sec": 1.0,
            "duration_sec": 1.0, "reason": "source_segment_missing"})
        manager.drain()
        _, payload = manager.request_repairs(
            session_id,
            {"schema_version": "stream-session.v1", "reason": "business_gap_inventory",
             "ttl_seconds": 600, "ranges": [{"segment_index": 1, "kind": "missing"}]},
            "repair-tracks-window-0001")
        manager.repair_segment(
            session_id, 1,
            {**segment_metadata(1, self.segment_bytes, source_start=1.0),
             "repair_request_id": payload["request_id"]},
            self.segment_bytes,
            payload["ranges"][0]["grant_token"])
        manager.drain()

        events = manager.read_events(session_id, limit=10000)[1]["events"]

        def track_ids(index):
            return {
                event["data"]["track"]["track_id"]
                for event in events
                if event.get("segment_index") == index and event["data"].get("track")
            }

        self.assertEqual(track_ids(0), {"track_001"})
        self.assertEqual(track_ids(2), {"track_epoch1_001"})
        self.assertEqual(track_ids(1), {"track_epoch2_001"})
        # The repaired fragment shares no tracker identity with the timeline inside
        # which it sits, so nothing downstream can invent a cross-break movement.
        self.assertFalse(track_ids(1) & track_ids(2))


if __name__ == "__main__":
    unittest.main()

