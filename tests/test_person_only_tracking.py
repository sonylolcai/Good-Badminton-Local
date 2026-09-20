import unittest

import numpy as np

from badminton_analysis.streaming import (
    AnalysisEngine,
    FinalizationContext,
    FrameContext,
    FramePacket,
    FrameSegment,
    SegmentDescriptor,
)
from badminton_analysis.tracking.person_only import (
    PersonOnlyFrameProcessor,
    PersonOnlyTracker,
)


class _StableByteTracker:
    def __init__(self, _args):
        pass

    def update(self, batch):
        if not len(batch):
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(
            [
                [
                    *batch.xyxy[index].tolist(),
                    index + 7,
                    float(batch.conf[index]),
                    0,
                    index,
                ]
                for index in range(len(batch))
            ],
            dtype=np.float32,
        )


class PersonOnlyTrackingTests(unittest.TestCase):
    CORNERS = [(0, 0), (610, 0), (610, 1340), (0, 1340)]

    def tracker(self, **overrides):
        return PersonOnlyTracker(
            self.CORNERS,
            fps=10,
            min_confirmed_detections=2,
            max_missed_frames=1,
            max_retained_missing_frames=3,
            **overrides,
        )

    def test_open_set_accepts_late_second_third_and_fourth_people(self):
        tracker = self.tracker()
        first = tracker.update(1, [self.observation((1.0, 1.0), "a")])
        second = tracker.update(
            2,
            [
                self.observation((1.1, 1.0), "a"),
                self.observation((5.0, 12.0), "b"),
            ],
        )
        fourth = tracker.update(
            3,
            [
                self.observation((1.2, 1.0), "a"),
                self.observation((5.0, 11.9), "b"),
                self.observation((2.0, 2.0), "c"),
                self.observation((4.0, 11.0), "d"),
            ],
        )

        self.assertEqual(first["analysis_mode"], "person_only")
        self.assertEqual(len(second["tracks"]), 2)
        self.assertEqual(len(fourth["tracks"]), 4)
        self.assertEqual(len(fourth["track_candidates"]), 4)
        self.assertEqual(len(fourth["track_profiles"]), 4)
        self.assertEqual(
            set(fourth["track_candidates"][0]),
            {
                "track_id",
                "state",
                "first_source_time_sec",
                "last_source_time_sec",
                "detected_coverage",
                "confidence",
            },
        )
        self.assertTrue(all("team_id" not in item for item in fourth["tracks"]))
        self.assertTrue(all("person_id" not in item for item in fourth["tracks"]))

    def test_candidate_becomes_active_without_a_fixed_roster_count(self):
        tracker = self.tracker()
        candidate = tracker.update(1, [self.observation((1.0, 1.0), "a")])
        active = tracker.update(2, [self.observation((1.1, 1.0), "a")])

        self.assertEqual(candidate["tracks"][0]["lifecycle_state"], "candidate")
        self.assertEqual(active["tracks"][0]["lifecycle_state"], "active")
        self.assertEqual(active["tracks"][0]["track_id"], candidate["tracks"][0]["track_id"])

    def test_anonymous_roster_starts_at_two_then_promotes_to_four_before_lock(self):
        """No-mode streams start promptly but retain a bounded expansion window."""
        tracker = self.tracker(
            lock_match_roster=True,
            roster_stable_frames=2,
            roster_discovery_seconds=8.0,
            max_roster_count=4,
        )
        first = tracker.update(
            1,
            [
                self.observation((1.0, 1.0), "a"),
                self.observation((5.0, 12.0), "b"),
            ],
        )
        discovering = tracker.update(
            2,
            [
                self.observation((1.1, 1.0), "a"),
                self.observation((5.0, 11.9), "b"),
            ],
        )
        promoted = tracker.update(
            3,
            [
                self.observation((1.2, 1.0), "a"),
                self.observation((5.0, 11.8), "b"),
                self.observation((3.0, 6.7), "c"),
                self.observation((4.0, 11.0), "d"),
            ],
        )

        self.assertEqual(first["tracks"], [])
        self.assertEqual(discovering["tracking"]["roster"]["status"], "discovering")
        self.assertIsNone(discovering["tracking"]["expected_player_count"])
        self.assertEqual(len(discovering["tracks"]), 2)
        self.assertEqual(promoted["tracking"]["roster"]["status"], "locked")
        self.assertEqual(promoted["tracking"]["expected_player_count"], 4)
        self.assertEqual(len(promoted["tracks"]), 4)
        self.assertEqual(promoted["tracking"]["roster"]["observed_max_candidate_count"], 4)

    def test_predicted_missing_closed_and_quality_are_explicit(self):
        tracker = self.tracker()
        tracker.update(1, [self.observation((1.0, 1.0), "a")])
        tracker.update(2, [self.observation((1.1, 1.0), "a")])
        predicted = tracker.update(3, [])
        missing = tracker.update(5, [])
        closed = tracker.update(7, [])

        self.assertEqual(predicted["tracks"][0]["lifecycle_state"], "predicted")
        self.assertEqual(missing["tracks"][0]["lifecycle_state"], "missing")
        self.assertEqual(closed["tracks"], [])
        candidate = closed["track_profiles"][0]
        self.assertEqual(candidate["lifecycle_state"], "closed")
        self.assertGreater(candidate["quality"]["predicted_ratio"], 0)
        self.assertGreater(candidate["quality"]["missing_ratio"], 0)
        self.assertFalse(candidate["quality"]["reliable_for_person_analytics"])

    def test_two_person_discovery_locks_only_after_the_deadline(self):
        tracker = self.tracker(
            lock_match_roster=True,
            roster_stable_frames=1,
            roster_discovery_seconds=1.0,
            max_roster_count=4,
        )
        start = tracker.update(
            1,
            [self.observation((1.0, 1.0), "a"), self.observation((5.0, 12.0), "b")],
        )
        before_deadline = tracker.update(
            9,
            [self.observation((1.1, 1.0), "a"), self.observation((5.0, 11.9), "b")],
        )
        locked = tracker.update(
            11,
            [self.observation((1.2, 1.0), "a"), self.observation((5.0, 11.8), "b")],
        )

        self.assertEqual(start["tracking"]["roster"]["status"], "discovering")
        self.assertEqual(before_deadline["tracking"]["roster"]["status"], "discovering")
        self.assertEqual(locked["tracking"]["roster"]["status"], "locked")
        self.assertEqual(locked["tracking"]["expected_player_count"], 2)

    def test_expected_four_person_roster_waits_for_the_discovery_window(self):
        """A configured count must not turn three warm-up frames into four IDs."""
        tracker = self.tracker(
            lock_match_roster=True,
            expected_roster_count=4,
            max_roster_count=4,
            roster_stable_frames=2,
            roster_discovery_seconds=1.0,
        )
        observations = [
            self.observation((1.0, 1.0), "a"),
            self.observation((4.8, 1.1), "b"),
            self.observation((1.1, 12.0), "c"),
            self.observation((4.9, 11.9), "d"),
        ]

        first = tracker.update(1, observations)
        stable_but_early = tracker.update(2, observations)
        before_deadline = tracker.update(10, observations)
        locked = tracker.update(11, observations)

        self.assertEqual(first["tracking"]["roster"]["status"], "bootstrapping")
        self.assertEqual(stable_but_early["tracking"]["roster"]["status"], "bootstrapping")
        self.assertEqual(before_deadline["tracking"]["roster"]["status"], "bootstrapping")
        self.assertEqual(locked["tracking"]["roster"]["status"], "locked")
        self.assertEqual(len(locked["tracks"]), 4)

    def test_court_only_gap_recovery_is_marked_uncertain_not_silent(self):
        tracker = self.tracker()
        tracker.update(1, [self.observation((1.0, 1.0), None)])
        tracker.update(2, [self.observation((1.1, 1.0), None)])
        tracker.update(3, [])
        recovered = tracker.update(4, [self.observation((1.2, 1.0), None)])

        self.assertEqual(recovered["tracks"][0]["lifecycle_state"], "reassociation_uncertain")
        self.assertEqual(
            recovered["track_profiles"][0]["quality"]["reassociation_uncertain_count"],
            1,
        )

    def test_same_association_key_keeps_track_id_across_any_zone(self):
        tracker = self.tracker()
        first = tracker.update(1, [self.observation((0.5, 0.5), "player-a")])
        second = tracker.update(2, [self.observation((5.5, 12.5), "player-a")])

        self.assertEqual(first["tracks"][0]["track_id"], second["tracks"][0]["track_id"])
        self.assertNotEqual(first["tracks"][0]["zone_id"], second["tracks"][0]["zone_id"])

    def test_state_checkpoint_restores_open_track_continuity(self):
        tracker = self.tracker()
        initial = tracker.update(1, [self.observation((1.0, 1.0), "a")])
        tracker.update(2, [self.observation((1.1, 1.0), "a")])
        state = tracker.snapshot_state()

        restored = self.tracker()
        restored.restore_state(state)
        continued = restored.update(3, [self.observation((1.2, 1.0), "a")])

        self.assertEqual(continued["tracks"][0]["track_id"], initial["tracks"][0]["track_id"])
        self.assertEqual(continued["tracks"][0]["lifecycle_state"], "active")

    def test_legacy_v1_badminton_checkpoint_restores_open_track_continuity(self):
        tracker = self.tracker()
        initial = tracker.update(1, [self.observation((1.0, 1.0), "a")])
        tracker.update(2, [self.observation((1.1, 1.0), "a")])
        state = tracker.snapshot_state()
        state["state_version"] = "person-only.v1"
        for field in (
            "sport_id",
            "session_mode",
            "calibration_scope",
            "coordinate_system_id",
            "court_dimensions_m",
            "calibration_world_points_m",
            "athlete_observation_region",
            "athlete_observation_margin_m",
            "athlete_observation_lateral_margin_m",
            "athlete_observation_baseline_margin_m",
        ):
            state.pop(field, None)

        restored = self.tracker()
        restored.restore_state(state)
        continued = restored.update(3, [self.observation((1.2, 1.0), "a")])

        self.assertEqual(continued["tracks"][0]["track_id"], initial["tracks"][0]["track_id"])

    def test_bytetrack_key_can_recover_without_court_only_uncertainty(self):
        tracker = self.tracker(
            tracker_backend="bytetrack",
            enable_bytetrack=True,
            byte_tracker_factory=_StableByteTracker,
        )
        tracker.update(1, [self.observation((1.0, 1.0), None)])
        tracker.update(2, [self.observation((1.1, 1.0), None)])
        tracker.update(3, [])
        recovered = tracker.update(4, [self.observation((1.2, 1.0), None)])

        self.assertEqual(recovered["tracks"][0]["lifecycle_state"], "active")
        self.assertEqual(
            recovered["track_profiles"][0]["quality"]["reassociation_uncertain_count"],
            0,
        )
        state = tracker.snapshot_state()
        self.assertFalse(state["association_runtime_checkpointable"])
        with self.assertRaisesRegex(RuntimeError, "cannot be restored"):
            self.tracker(
                tracker_backend="bytetrack",
                enable_bytetrack=True,
                byte_tracker_factory=_StableByteTracker,
            ).restore_state(state)

    def test_detected_box_is_detected_evidence_while_identity_is_candidate(self):
        processor = PersonOnlyFrameProcessor(
            self.tracker(),
            observation_provider=lambda frame, _context: frame,
        )
        context = FrameContext(
            analysis_session_id="ssn_candidate_evidence",
            segment_index=0,
            source_frame_index=1,
            source_time_sec=0.1,
            is_measurement_frame=True,
            measurement_bucket=1,
        )
        event = list(
            processor.process_frame([self.observation((1.0, 1.0), "a")], context)
        )[0]
        self.assertEqual(event.evidence_state, "detected")
        self.assertEqual(event.data["track"]["lifecycle_state"], "candidate")

    def test_unstable_expected_roster_emits_review_candidates_not_empty_output(self):
        """A fail-closed four-person roster must still expose raw evidence."""
        tracker = self.tracker(
            lock_match_roster=True,
            expected_roster_count=4,
            max_roster_count=4,
            roster_stable_frames=2,
            roster_discovery_seconds=1.0,
        )
        processor = PersonOnlyFrameProcessor(
            tracker,
            observation_provider=lambda frame, _context: frame,
        )
        context = FrameContext(
            analysis_session_id="ssn_roster_review",
            segment_index=0,
            source_frame_index=1,
            source_time_sec=0.1,
            is_measurement_frame=True,
            measurement_bucket=1,
        )
        events = list(processor.process_frame([
            self.observation((1.0, 1.0), "a"),
            self.observation((4.8, 1.1), "b"),
            self.observation((1.1, 12.0), "c"),
            self.observation((4.9, 11.9), "d"),
        ], context))

        self.assertEqual(len(events), 4)
        self.assertTrue(all(event.event_type == "roster_candidate_observation" for event in events))
        self.assertTrue(all(event.data["track"]["analytics_eligible"] is False for event in events))
        self.assertTrue(all(event.data["track"]["lifecycle_state"] == "unconfirmed_roster" for event in events))

    def test_stream_processor_emits_anonymous_events_and_final_candidates(self):
        tracker = self.tracker()
        processor = PersonOnlyFrameProcessor(
            tracker,
            observation_provider=lambda frame, _context: frame,
        )
        context = FrameContext(
            analysis_session_id="ssn_person_only_test",
            segment_index=0,
            source_frame_index=1,
            source_time_sec=0.1,
            is_measurement_frame=True,
            measurement_bucket=1,
        )
        events = list(processor.process_frame([self.observation((1.0, 1.0), "a")], context))
        final = list(
            processor.finalize(
                FinalizationContext(
                    analysis_session_id=context.analysis_session_id,
                    last_segment_index=0,
                    last_source_time_sec=0.1,
                    source_frames=1,
                    measurement_frames=1,
                )
            )
        )

        self.assertEqual(events[0].event_type, "person_observation")
        self.assertEqual(events[0].data["analysis_mode"], "person_only")
        self.assertNotIn("team_id", str(events[0].data))
        self.assertNotIn("person_id", str(events[0].data))
        self.assertEqual(final[0].event_type, "session_status")
        self.assertEqual(len(final[0].data["track_candidates"]), 1)

    def test_analysis_engine_checkpoint_preserves_person_only_track_id(self):
        processor = PersonOnlyFrameProcessor(
            self.tracker(),
            observation_provider=lambda frame, _context: frame,
        )
        engine = AnalysisEngine("ssn_person_only_engine", 10, processor)
        first = engine.process_segment(
            self.segment(
                0,
                0.0,
                [
                    [self.observation((1.0, 1.0), "a")],
                    [self.observation((1.1, 1.0), "a")],
                ],
            )
        )
        first_ids = {
            event["data"]["track"]["track_id"]
            for event in first.events
            if event["event_type"] == "person_observation"
        }

        restored_processor = PersonOnlyFrameProcessor(
            self.tracker(),
            observation_provider=lambda frame, _context: frame,
        )
        restored = AnalysisEngine.restore(first.checkpoint, restored_processor)
        second = restored.process_segment(
            self.segment(1, 0.2, [[self.observation((1.2, 1.0), "a")]])
        )
        second_ids = {
            event["data"]["track"]["track_id"]
            for event in second.events
            if event["event_type"] == "person_observation"
        }

        self.assertEqual(first_ids, {"track_001"})
        self.assertEqual(second_ids, first_ids)
        self.assertNotEqual(restored.status, "interrupted_needs_rebuild")

    @staticmethod
    def segment(index, source_start, frames):
        digest = f"{index + 1:064x}"
        descriptor = SegmentDescriptor(
            segment_index=index,
            source_start_time_sec=source_start,
            duration_sec=max(0.1, len(frames) / 10),
            sha256=digest,
            idempotency_key=f"person-only-segment-{index:06d}",
        )
        packets = [
            FramePacket(
                frame=frame,
                source_frame_index=int(round((source_start + offset / 10) * 10)),
                source_time_sec=source_start + offset / 10,
                segment_index=index,
            )
            for offset, frame in enumerate(frames)
        ]
        return FrameSegment(descriptor=descriptor, frames=packets)

    @staticmethod
    def observation(court_xy, association_key):
        return {
            "court_xy": court_xy,
            "image_xy": (court_xy[0] * 100, court_xy[1] * 100),
            "bbox_xyxy": [10, 20, 30, 60],
            "confidence": 0.9,
            "location_method": "ankles_midpoint",
            "location_confidence": 0.9,
            "source": "full_frame",
            "association_key": association_key,
        }


if __name__ == "__main__":
    unittest.main()
