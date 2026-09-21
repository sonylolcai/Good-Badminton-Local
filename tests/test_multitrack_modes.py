import unittest

import numpy as np

from badminton_analysis.analysis.fixed_camera_match import (
    CourtMultiObjectTracker,
    CourtSpace,
    FixedCameraMatchPipeline,
    RallyStateMachine,
)
from badminton_analysis.tracking.bytetrack_adapter import ByteTrackAdapter


class _FakeByteTracker:
    def __init__(self, _args):
        self.received = []

    def update(self, batch):
        self.received.append(batch)
        if not len(batch):
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray([[10, 20, 30, 60, 7, 0.88, 0, 0]], dtype=np.float32)


class _DelayedTwoPersonByteTracker:
    """Emit no identifiers once, then confirm both pose detections."""

    def __init__(self, _args):
        self.calls = 0

    def update(self, batch):
        self.calls += 1
        if self.calls == 1 or not len(batch):
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray([
            [10, 20, 30, 60, 17, 0.90, 0, 0],
            [40, 20, 60, 60, 18, 0.91, 0, 1],
        ], dtype=np.float32)


class MultiTrackModeTests(unittest.TestCase):
    CORNERS = [(0, 0), (610, 0), (610, 1340), (0, 1340)]

    def test_bytetrack_uses_far_court_continuity_match_threshold(self):
        """Fixed-camera far players need a less brittle IoU association gate."""
        adapter = ByteTrackAdapter(fps=10, tracker_factory=_FakeByteTracker)
        self.assertEqual(0.50, adapter.config["match_thresh"])

    def test_doubles_keeps_four_independent_tracks_and_team_is_not_court_side(self):
        pipeline = FixedCameraMatchPipeline(self.CORNERS, fps=10, match_mode="doubles")
        first = pipeline.update(
            1,
            [
                self._observation((0.8, 1.0), "pose_a"),
                self._observation((4.9, 1.3), "pose_b"),
                self._observation((1.2, 12.0), "pose_c"),
                self._observation((4.7, 11.8), "pose_d"),
            ],
            None,
        )
        self.assertEqual(first["match"]["mode"], "doubles")
        self.assertEqual(first["match"]["max_players_per_team"], 2)
        self.assertEqual(len(first["tracks"]), 4)
        track_ids = [item["track_id"] for item in first["tracks"]]
        self.assertTrue(all(item["team_id"] is None for item in first["tracks"]))

        # Pose A crossed the net in court coordinates. Its ByteTrack-like
        # association key remains durable; it must not become another person.
        second = pipeline.update(
            2,
            [
                self._observation((1.0, 11.4), "pose_a"),
                self._observation((4.8, 1.6), "pose_b"),
                self._observation((1.1, 11.7), "pose_c"),
                self._observation((4.6, 11.5), "pose_d"),
            ],
            None,
        )
        by_key = {item["association"]["key"]: item for item in second["tracks"]}
        self.assertEqual(by_key["pose_a"]["track_id"], track_ids[0])
        self.assertEqual(by_key["pose_a"]["court_end"], "end_b")
        self.assertEqual(len(second["tracks"]), 4)

        pipeline.claim_teams({track_ids[0]: "team_a", track_ids[1]: "team_a", track_ids[2]: "team_b", track_ids[3]: "team_b"})
        finished = pipeline.finalize()
        self.assertEqual(finished["team_claims"][track_ids[0]], "team_a")
        self.assertEqual(finished["match"]["max_players_per_team"], 2)

    def test_status_distinguishes_detected_predicted_and_missing(self):
        tracker = CourtMultiObjectTracker(
            CourtSpace(self.CORNERS), fps=10, max_missed_frames=2, max_retained_missing_frames=10
        )
        tracker.update(1, [self._observation((2.0, 2.0), "pose_a")])
        self.assertEqual(tracker.update(2, [])[0]["status"], "predicted")
        missing = tracker.update(4, [])[0]
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["confidence"], 0.0)

    def test_track_current_speed_uses_fresh_track_evidence_and_filters_standing_jitter(self):
        tracker = CourtMultiObjectTracker(CourtSpace(self.CORNERS), fps=10, max_missed_frames=2)
        tracker.update(1, [self._observation((2.0, 2.0), "pose_a")])

        # Two one-centimetre foot-point changes are below the display dead
        # zone. A standing player must display 0 rather than a false walk.
        second = tracker.update(2, [self._observation((2.01, 2.0), "pose_a")])[0]
        third = tracker.update(3, [self._observation((2.02, 2.0), "pose_a")])[0]
        self.assertEqual(second["motion"]["status"], "stationary")
        self.assertEqual(second["motion"]["current_speed_mps"], 0.0)
        self.assertEqual(third["motion"]["current_speed_mps"], 0.0)

        moving = tracker.update(4, [self._observation((2.22, 2.0), "pose_a")])[0]
        self.assertEqual(moving["status"], "detected")
        self.assertEqual(moving["motion"]["status"], "moving")
        self.assertGreater(moving["motion"]["current_speed_mps"], 0.35)

        predicted = tracker.update(5, [])[0]
        self.assertEqual(predicted["status"], "predicted")
        self.assertIsNone(predicted["motion"]["current_speed_mps"])
        self.assertEqual(predicted["motion"]["status"], "not_currently_measured")

    def test_locked_singles_roster_waits_for_two_people_then_rejects_new_tracks(self):
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=10,
            match_mode="singles",
            lock_match_roster=True,
            roster_stable_frames=2,
        )
        only_one = pipeline.update(1, [self._observation((2.0, 2.0), None)], None)
        self.assertEqual(only_one["match_roster"]["status"], "bootstrapping")
        self.assertEqual(only_one["match_roster"]["observed_on_court_candidate_count"], 1)
        self.assertEqual(only_one["tracks"], [])

        first_complete = pipeline.update(
            2,
            [self._observation((2.0, 2.0), None), self._observation((4.0, 11.0), None)],
            None,
        )
        self.assertEqual(first_complete["match_roster"]["status"], "bootstrapping")
        locked = pipeline.update(
            3,
            [self._observation((2.1, 2.0), None), self._observation((4.0, 10.9), None)],
            None,
        )
        track_ids = [item["track_id"] for item in locked["tracks"]]
        self.assertEqual(locked["match_roster"]["status"], "locked")
        self.assertEqual(locked["match_roster"]["track_ids"], track_ids)
        self.assertEqual(len(track_ids), 2)

        # A third person-shaped detection is evidence for review, not a new
        # match participant. The locked roster still has exactly two IDs.
        after_extra_detection = pipeline.update(
            4,
            [
                self._observation((2.2, 2.0), None),
                self._observation((4.0, 10.8), None),
                self._observation((3.0, 6.7), None),
            ],
            None,
        )
        self.assertEqual([item["track_id"] for item in after_extra_detection["tracks"]], track_ids)
        self.assertEqual(after_extra_detection["match_roster"]["unassigned_observation_count"], 1)
        self.assertEqual(
            after_extra_detection["match_roster"]["unassigned_observations"][0]["reason"],
            "unassigned_after_locked_roster_association",
        )

    def test_auto_mode_resolves_to_doubles_when_four_stable_tracks_appear(self):
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=10,
            match_mode="auto",
            lock_match_roster=True,
            roster_stable_frames=1,
        )
        resolved = pipeline.update(
            1,
            [
                self._observation((1.0, 1.0), None),
                self._observation((5.0, 1.0), None),
                self._observation((1.0, 12.0), None),
                self._observation((5.0, 12.0), None),
            ],
            None,
        )

        self.assertEqual(resolved["match"]["requested_mode"], "auto")
        self.assertEqual(resolved["match"]["mode"], "doubles")
        self.assertEqual(resolved["match_roster"]["expected_player_count"], 4)
        self.assertEqual(resolved["match_roster"]["status"], "locked")

    def test_auto_mode_resolves_to_singles_only_after_the_discovery_window(self):
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=1,
            match_mode="auto",
            lock_match_roster=True,
            roster_stable_frames=1,
        )
        observations = [self._observation((1.0, 1.0), None), self._observation((5.0, 12.0), None)]
        pending = pipeline.update(1, observations, None)
        self.assertEqual(pending["match"]["mode"], "auto")
        for frame_index in range(2, 10):
            resolved = pipeline.update(frame_index, observations, None)

        self.assertEqual(resolved["match"]["requested_mode"], "auto")
        self.assertEqual(resolved["match"]["mode"], "singles")
        self.assertEqual(resolved["match_roster"]["expected_player_count"], 2)

    def test_bytetrack_roster_waits_for_confirmed_association_keys(self):
        """Do not bind durable roster IDs before ByteTrack has confirmed them."""
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=10,
            match_mode="singles",
            tracker_backend="bytetrack",
            enable_bytetrack=True,
            byte_tracker_factory=_DelayedTwoPersonByteTracker,
            lock_match_roster=True,
            roster_stable_frames=1,
        )
        observations = [
            self._observation((2.0, 2.0), None),
            self._observation((4.0, 11.0), None),
        ]

        unconfirmed = pipeline.update(1, observations, None)
        self.assertEqual(unconfirmed["match_roster"]["status"], "bootstrapping")
        self.assertEqual(
            unconfirmed["match_roster"]["reason"],
            "awaiting_bytetrack_confirmation",
        )

        locked = pipeline.update(2, observations, None)
        self.assertEqual(locked["match_roster"]["status"], "locked")
        self.assertEqual(
            {track["association"]["key"] for track in locked["tracks"]},
            {"bytetrack_17", "bytetrack_18"},
        )

    def test_locked_bytetrack_roster_does_not_relabel_a_new_key_by_court_position(self):
        """A changed ByteTrack key is review evidence, not a replacement player.

        The fixed roster previously fell through to court-distance matching
        after a key change.  That lets an ROI duplicate overwrite another
        player's slot when the two boxes happen to be near the same court
        position.
        """
        tracker = CourtMultiObjectTracker(
            CourtSpace(self.CORNERS),
            fps=10,
            match_mode="person_only",
            lock_match_roster=True,
            expected_roster_count=2,
            max_roster_count=2,
            roster_stable_frames=1,
            roster_discovery_seconds=0.0,
            require_association_keys=True,
        )
        first = tracker.update(
            1,
            [self._observation((2.0, 2.0), "bytetrack_a"), self._observation((4.0, 11.0), "bytetrack_b")],
        )
        first_roster = tracker.roster_summary()
        locked = tracker.update(
            2,
            [self._observation((2.1, 2.0), "bytetrack_a"), self._observation((4.0, 10.9), "bytetrack_b")],
        )
        locked_roster = tracker.roster_summary()
        replaced = tracker.update(
            3,
            [self._observation((2.2, 2.0), "bytetrack_x"), self._observation((4.0, 10.8), "bytetrack_y")],
        )

        self.assertEqual(first_roster["status"], "bootstrapping")
        self.assertEqual(locked_roster["status"], "locked")
        self.assertEqual({item["status"] for item in replaced}, {"predicted"})
        self.assertEqual(tracker.roster_summary()["unassigned_observation_count"], 2)

    def test_locked_singles_recovers_one_new_bytetrack_key_when_the_other_player_matches(self):
        """A two-person roster has no third-player ambiguity for this frame."""
        tracker = CourtMultiObjectTracker(
            CourtSpace(self.CORNERS),
            fps=10,
            match_mode="singles",
            lock_match_roster=True,
            expected_roster_count=2,
            max_roster_count=2,
            roster_stable_frames=1,
            roster_discovery_seconds=0.0,
            require_association_keys=True,
        )
        initial = [
            self._observation((2.0, 2.0), "bytetrack_a"),
            self._observation((4.0, 11.0), "bytetrack_b"),
        ]
        tracker.update(1, initial)
        tracker.update(2, initial)

        recovered = tracker.update(
            3,
            [
                self._observation((2.1, 2.0), "bytetrack_a"),
                self._observation((4.1, 10.9), "bytetrack_restarted"),
            ],
        )

        self.assertEqual({item["status"] for item in recovered}, {"detected"})
        self.assertIn(
            "singles_roster_complement",
            {item["association"]["source"] for item in recovered},
        )
        self.assertEqual(0, tracker.roster_summary()["unassigned_observation_count"])

    def test_locked_singles_recovers_both_new_bytetrack_keys_by_unique_court_end(self):
        """A two-player roster survives a simultaneous ByteTrack restart."""
        tracker = CourtMultiObjectTracker(
            CourtSpace(self.CORNERS),
            fps=10,
            match_mode="singles",
            lock_match_roster=True,
            expected_roster_count=2,
            max_roster_count=2,
            roster_stable_frames=1,
            roster_discovery_seconds=0.0,
            require_association_keys=True,
        )
        initial = [
            self._observation((2.0, 2.0), "bytetrack_a"),
            self._observation((4.0, 11.0), "bytetrack_b"),
        ]
        tracker.update(1, initial)
        tracker.update(2, initial)

        recovered = tracker.update(
            3,
            [
                self._observation((2.2, 2.1), "bytetrack_restarted_upper"),
                self._observation((3.8, 10.8), "bytetrack_restarted_lower"),
            ],
        )

        self.assertEqual({item["status"] for item in recovered}, {"detected"})
        self.assertEqual(
            {item["association"]["source"] for item in recovered},
            {"singles_roster_full_reacquisition"},
        )
        self.assertEqual(0, tracker.roster_summary()["unassigned_observation_count"])

    def test_locked_doubles_recovers_both_new_bytetrack_pairs_by_court_end(self):
        """Two simultaneous teammate ID fragments retain the four slots."""
        tracker = CourtMultiObjectTracker(
            CourtSpace(self.CORNERS),
            fps=10,
            match_mode="doubles",
            lock_match_roster=True,
            expected_roster_count=4,
            max_roster_count=4,
            roster_stable_frames=1,
            roster_discovery_seconds=0.0,
            require_association_keys=True,
        )
        initial = [
            self._observation((1.0, 2.0), "bytetrack_a"),
            self._observation((5.0, 2.4), "bytetrack_b"),
            self._observation((1.2, 11.0), "bytetrack_c"),
            self._observation((4.8, 10.6), "bytetrack_d"),
        ]
        tracker.update(1, initial)
        tracker.update(2, initial)
        recovered = tracker.update(
            3,
            [
                self._observation((1.2, 2.1), "bytetrack_upper_a"),
                self._observation((4.8, 2.3), "bytetrack_upper_b"),
                self._observation((1.4, 10.9), "bytetrack_lower_a"),
                self._observation((4.6, 10.7), "bytetrack_lower_b"),
            ],
        )

        self.assertEqual({item["status"] for item in recovered}, {"detected"})
        self.assertEqual(
            {item["association"]["source"] for item in recovered},
            {"doubles_team_pair_reacquisition"},
        )
        self.assertEqual(0, tracker.roster_summary()["unassigned_observation_count"])

    def test_locked_doubles_recovers_visible_teammate_then_its_new_keyed_partner(self):
        """One temporarily occluded teammate must not deadlock the whole roster."""
        tracker = CourtMultiObjectTracker(
            CourtSpace(self.CORNERS),
            fps=10,
            match_mode="doubles",
            lock_match_roster=True,
            expected_roster_count=4,
            max_roster_count=4,
            roster_stable_frames=1,
            roster_discovery_seconds=0.0,
            require_association_keys=True,
        )
        initial = [
            self._observation((1.0, 2.0), "bytetrack_a"),
            self._observation((5.0, 2.4), "bytetrack_b"),
            self._observation((1.2, 11.0), "bytetrack_c"),
            self._observation((4.8, 10.6), "bytetrack_d"),
        ]
        tracker.update(1, initial)
        tracker.update(2, initial)

        partial = tracker.update(
            3,
            [
                self._observation((1.2, 2.1), "bytetrack_upper_a"),
                self._observation((1.4, 10.9), "bytetrack_lower_a"),
                self._observation((4.6, 10.7), "bytetrack_lower_b"),
            ],
        )
        self.assertEqual(3, sum(item["status"] == "detected" for item in partial))
        self.assertIn(
            "doubles_team_side_reacquisition",
            {item["association"]["source"] for item in partial},
        )

        complete = tracker.update(
            4,
            [
                self._observation((1.3, 2.2), "bytetrack_upper_a"),
                self._observation((4.7, 2.2), "bytetrack_upper_b"),
                self._observation((1.5, 10.8), "bytetrack_lower_a"),
                self._observation((4.5, 10.8), "bytetrack_lower_b"),
            ],
        )
        self.assertEqual({item["status"] for item in complete}, {"detected"})
        self.assertEqual(0, tracker.roster_summary()["unassigned_observation_count"])

    def test_expected_bytetrack_roster_restarts_discovery_when_key_set_changes(self):
        """Four transient tracker fragments cannot consume the full lock window."""
        tracker = CourtMultiObjectTracker(
            CourtSpace(self.CORNERS),
            fps=10,
            match_mode="person_only",
            lock_match_roster=True,
            expected_roster_count=4,
            max_roster_count=4,
            roster_stable_frames=1,
            roster_discovery_seconds=1.0,
            require_association_keys=True,
        )
        first_keys = ["bytetrack_a", "bytetrack_b", "bytetrack_c", "bytetrack_d"]
        replacement_keys = ["bytetrack_a", "bytetrack_b", "bytetrack_c", "bytetrack_e"]
        positions = [(1.0, 1.0), (4.8, 1.1), (1.1, 12.0), (4.9, 11.9)]

        tracker.update(1, [self._observation(position, key) for position, key in zip(positions, first_keys)])
        tracker.update(2, [self._observation(position, key) for position, key in zip(positions, replacement_keys)])
        tracker.update(11, [self._observation(position, key) for position, key in zip(positions, replacement_keys)])
        self.assertEqual(tracker.roster_summary()["status"], "bootstrapping")

        tracker.update(12, [self._observation(position, key) for position, key in zip(positions, replacement_keys)])
        self.assertEqual(tracker.roster_summary()["status"], "locked")

    def test_roster_bootstrap_ignores_frames_without_a_fresh_pose_measurement(self):
        """Sampling gaps must not be mistaken for a zero-person detection.

        A 10 Hz pose detector on a 30 fps source has two source timestamps
        between every real measurement.  Those timestamps have no new pose
        evidence, so they must neither advance nor reset roster bootstrap.
        """
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=30,
            match_mode="singles",
            lock_match_roster=True,
            roster_stable_frames=2,
        )
        observations = [
            self._observation((2.0, 2.0), None),
            self._observation((4.0, 11.0), None),
        ]

        first = pipeline.update(1, observations, None, has_fresh_observations=True)
        self.assertEqual(first["match_roster"]["stable_observation_frames"], 1)

        skipped_one = pipeline.update(2, [], None, has_fresh_observations=False)
        skipped_two = pipeline.update(3, [], None, has_fresh_observations=False)
        self.assertEqual(skipped_one["match_roster"]["stable_observation_frames"], 1)
        self.assertEqual(skipped_two["match_roster"]["stable_observation_frames"], 1)
        self.assertEqual(skipped_two["match_roster"]["status"], "bootstrapping")

        locked = pipeline.update(4, observations, None, has_fresh_observations=True)
        self.assertEqual(locked["match_roster"]["status"], "locked")
        self.assertEqual(len(locked["tracks"]), 2)

    def test_roster_bootstrap_still_resets_on_a_fresh_incomplete_measurement(self):
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=30,
            match_mode="singles",
            lock_match_roster=True,
            roster_stable_frames=2,
        )
        complete = [
            self._observation((2.0, 2.0), None),
            self._observation((4.0, 11.0), None),
        ]
        pipeline.update(1, complete, None, has_fresh_observations=True)
        incomplete = pipeline.update(
            4, [self._observation((2.0, 2.0), None)], None, has_fresh_observations=True
        )
        self.assertEqual(incomplete["match_roster"]["stable_observation_frames"], 0)

    def test_missing_shuttle_falls_back_to_person_stillness_not_a_gap_timer(self):
        rallies = RallyStateMachine(
            fps=10,
            min_active_frames=4,
            shuttle_enabled=True,
            expected_player_count=2,
            settle_window_seconds=0.5,
        )

        def tracks(left, right):
            return [
                {"track_id": "track_001", "status": "detected", "court_xy_m": left},
                {"track_id": "track_002", "status": "detected", "court_xy_m": right},
            ]

        shuttle = {"status": "approximate", "xyz_m": [3.0, 6.7, 1.0]}

        # A detected shuttle plus two players opens a candidate rally, which
        # turns active after the minimum active-frame count.
        rallies.update(1, tracks((1.0, 2.0), (5.0, 11.0)), shuttle, [])
        self.assertEqual(rallies.update(4, tracks((1.4, 2.0), (5.0, 11.0)), shuttle, [])["state"], "active")

        # A missing-ball gap must not itself end the rally.  Only the two
        # players settling (fresh detection, low speed) closes it.
        self.assertEqual(
            rallies.update(5, tracks((1.4, 2.0), (5.0, 11.0)), {"status": "missing"}, [])["state"],
            "active",
        )
        self.assertEqual(len(rallies.completed), 0)

        # Establish stability, then hold it for the 0.5 s window.
        rallies.update(6, tracks((1.4, 2.0), (5.0, 11.0)), {"status": "missing"}, [])
        for frame in range(7, 11):
            rallies.update(frame, tracks((1.4, 2.0), (5.0, 11.0)), {"status": "missing"}, [])

        self.assertEqual(len(rallies.completed), 1)
        self.assertEqual(rallies.completed[-1]["end_reason"], "all_players_stable_for_0.5s")

    def test_detected_shuttle_landing_ends_rally(self):
        rallies = RallyStateMachine(
            fps=10,
            min_active_frames=4,
            shuttle_enabled=True,
            expected_player_count=2,
        )

        def tracks():
            return [
                {"track_id": "track_001", "status": "detected", "court_xy_m": [1.0, 2.0]},
                {"track_id": "track_002", "status": "detected", "court_xy_m": [5.0, 11.0]},
            ]

        def shuttle_at(x, y):
            return {"status": "approximate", "xyz_m": [x, y, 1.0]}

        # A shuttle that travels, then stops, is a landing.
        for frame, x in [(1, 1.0), (2, 3.0), (3, 5.0), (4, 6.0), (5, 6.3), (6, 6.5), (7, 6.5)]:
            rallies.update(frame, tracks(), shuttle_at(x, 6.7), [])
        for frame in range(8, 14):
            rallies.update(frame, tracks(), shuttle_at(6.5, 6.7), [])

        self.assertEqual(len(rallies.completed), 1)
        self.assertEqual(rallies.completed[-1]["end_reason"], "shuttle_landed")

    def test_person_only_rally_requires_all_players_stable_for_selected_window(self):
        rallies = RallyStateMachine(
            fps=10,
            min_active_frames=1,
            shuttle_enabled=False,
            expected_player_count=2,
            settle_window_seconds=0.5,
        )

        def tracks(left, right):
            return [
                {"track_id": "track_001", "status": "detected", "court_xy_m": left},
                {"track_id": "track_002", "status": "detected", "court_xy_m": right},
            ]

        # The first observation cannot establish stability. The second is
        # measured motion, so it opens a person-only candidate rally.
        rallies.update(1, tracks((1.0, 2.0), (5.0, 11.0)), None, [])
        self.assertEqual(rallies.update(2, tracks((1.1, 2.0), (5.0, 11.0)), None, [])["state"], "active")

        # Five 10 Hz stable observations satisfy the explicit 0.5 second
        # window. The end is the beginning of that stable window (frame 3).
        for frame in range(3, 8):
            snapshot = rallies.update(frame, tracks((1.1, 2.0), (5.0, 11.0)), None, [])
        self.assertEqual(snapshot["state"], "awaiting_serve")
        self.assertEqual(len(rallies.completed), 1)
        self.assertEqual(rallies.completed[0]["end_frame"], 3)
        self.assertEqual(rallies.completed[0]["end_reason"], "all_players_stable_for_0.5s")
        self.assertEqual(rallies.completed[0]["evidence_mode"], "player_stability_only")

        # Missing detector data must not become a terminal or a new rally.
        after_gap = rallies.update(8, tracks((1.1, 2.0), (5.0, 11.0))[:1], None, [])
        self.assertEqual(after_gap["state"], "awaiting_serve")
        self.assertEqual(len(rallies.completed), 1)

    def test_locked_roster_reassociates_brief_gap_without_creating_a_new_id(self):
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=10,
            match_mode="singles",
            lock_match_roster=True,
            roster_stable_frames=1,
        )
        initial = pipeline.update(
            1,
            [self._observation((2.0, 2.0), None), self._observation((4.0, 11.0), None)],
            None,
        )
        track_ids = [item["track_id"] for item in initial["tracks"]]
        for frame_index in range(2, 15):
            pipeline.update(frame_index, [], None)
        recovered = pipeline.update(
            15,
            [self._observation((2.3, 2.0), None), self._observation((4.1, 10.8), None)],
            None,
        )
        self.assertEqual([item["track_id"] for item in recovered["tracks"]], track_ids)
        self.assertEqual(len(recovered["tracks"]), 2)
        self.assertIn(
            "roster_reassociation",
            {item["association"]["source"] for item in recovered["tracks"]},
        )

    def test_locked_singles_assigns_the_only_remaining_pose_to_the_only_missing_track(self):
        """A two-person match may use roster complement after a large pose jump."""
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=10,
            match_mode="singles",
            lock_match_roster=True,
            roster_stable_frames=1,
        )
        initial = pipeline.update(
            1,
            [self._observation((5.5, 0.2), None), self._observation((4.0, 11.0), None)],
            None,
        )
        missing_id = next(
            item["track_id"] for item in initial["tracks"] if item["court_xy_m"] == [5.5, 0.2]
        )

        # Keep the lower player continuously associated while the upper
        # player's Pose is absent long enough to exceed the metric recovery
        # gate. The sole newly observed pose is still the only remaining
        # singles participant, not a third roster member.
        for frame_index in range(2, 32):
            pipeline.update(frame_index, [self._observation((4.0, 11.0), None)], None)
        recovered = pipeline.update(
            32,
            [self._observation((0.1, 0.2), None), self._observation((4.0, 11.0), None)],
            None,
        )

        restored = next(item for item in recovered["tracks"] if item["track_id"] == missing_id)
        self.assertEqual(restored["status"], "detected")
        self.assertEqual(restored["association"]["source"], "singles_roster_complement")
        self.assertGreaterEqual(restored["association"]["identity_confidence"], 0.7)
        self.assertEqual(recovered["match_roster"]["unassigned_observation_count"], 0)

    def test_locked_doubles_roster_recovers_one_unambiguous_long_occlusion(self):
        """A real returning detection must reclaim its locked ID after overlap.

        The recovery deliberately relies only on the current court end and a
        one-to-one candidate set.  It never claims that the missing player was
        observed during the occlusion, and it must not use image-side labels as
        identity.
        """
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=10,
            match_mode="doubles",
            lock_match_roster=True,
            roster_stable_frames=1,
        )
        initial = pipeline.update(
            1,
            [
                self._observation((1.0, 1.0), None),
                self._observation((4.8, 1.2), None),
                self._observation((1.1, 12.0), None),
                self._observation((4.9, 11.8), None),
            ],
            None,
        )
        first_end_a_id = next(
            item["track_id"]
            for item in initial["tracks"]
            if item["court_xy_m"] == [1.0, 1.0]
        )

        # The player at (1, 1) is invisible longer than the normal short-gap
        # reassociation window while the other three people remain detected.
        for frame_index in range(2, 32):
            pipeline.update(
                frame_index,
                [
                    self._observation((4.8, 1.2), None),
                    self._observation((1.1, 12.0), None),
                    self._observation((4.9, 11.8), None),
                ],
                None,
            )

        recovered = pipeline.update(
            32,
            [
                self._observation((1.2, 1.1), None),
                self._observation((4.8, 1.2), None),
                self._observation((1.1, 12.0), None),
                self._observation((4.9, 11.8), None),
            ],
            None,
        )
        restored = next(item for item in recovered["tracks"] if item["track_id"] == first_end_a_id)
        self.assertEqual(restored["status"], "detected")
        self.assertEqual(restored["association"]["source"], "doubles_team_side_complement")
        self.assertGreaterEqual(restored["association"]["identity_confidence"], 0.7)
        self.assertEqual(recovered["match_roster"]["unassigned_observation_count"], 0)

    def test_locked_doubles_long_pair_recovery_reclaims_the_fixed_court_end(self):
        """Two long-missing teammates remain the known pair for that court end."""
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=10,
            match_mode="doubles",
            lock_match_roster=True,
            roster_stable_frames=1,
        )
        pipeline.update(
            1,
            [
                self._observation((1.0, 1.0), None),
                self._observation((4.8, 1.2), None),
                self._observation((1.1, 12.0), None),
                self._observation((4.9, 11.8), None),
            ],
            None,
        )
        for frame_index in range(2, 32):
            pipeline.update(
                frame_index,
                [
                    self._observation((1.1, 12.0), None),
                    self._observation((4.9, 11.8), None),
                ],
                None,
            )
        recovered = pipeline.update(
            32,
            [
                self._observation((1.0, 1.0), None),
                self._observation((4.8, 1.2), None),
                self._observation((1.1, 12.0), None),
                self._observation((4.9, 11.8), None),
            ],
            None,
        )
        self.assertEqual(recovered["match_roster"]["unassigned_observation_count"], 0)
        recovered_end_a = [
            item for item in recovered["tracks"]
            if item["court_end"] == "end_a" and item["status"] == "detected"
        ]
        self.assertEqual(len(recovered_end_a), 2)
        self.assertEqual(
            {item["association"]["source"] for item in recovered_end_a},
            {"doubles_team_pair_reacquisition"},
        )

    def test_bytetrack_is_not_enabled_without_recorded_gate(self):
        with self.assertRaisesRegex(ValueError, "evaluation-gated"):
            FixedCameraMatchPipeline(self.CORNERS, fps=10, tracker_backend="bytetrack")

    def test_bytetrack_adapter_uses_official_output_index_contract(self):
        adapter = ByteTrackAdapter(fps=30, tracker_factory=_FakeByteTracker)
        keys = adapter.update([self._observation((2.0, 2.0), "pose_a")])
        self.assertEqual(keys, {0: "bytetrack_7"})

    def test_official_bytetrack_can_supply_a_persistent_association_key(self):
        available, reason = ByteTrackAdapter.availability()
        self.assertTrue(available, reason)
        pipeline = FixedCameraMatchPipeline(
            self.CORNERS,
            fps=30,
            tracker_backend="bytetrack",
            enable_bytetrack=True,
        )
        first = pipeline.update(1, [self._observation((2.0, 2.0), None)], None)
        second = pipeline.update(2, [self._observation((2.1, 2.0), None)], None)
        self.assertEqual(first["tracks"][0]["association"]["source"], "bytetrack")
        self.assertEqual(first["tracks"][0]["track_id"], second["tracks"][0]["track_id"])

    @staticmethod
    def _observation(court_xy, association_key):
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
