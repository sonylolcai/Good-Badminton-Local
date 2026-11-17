"""Slot semantics use controlled observations, not identity-accuracy claims."""
import unittest

from badminton_analysis.tracking.half_court_slots import HALF_COURT_SLOTS, HalfCourtSlotTracker
from badminton_analysis.analysis.fixed_camera_match import CourtSpace
from badminton_analysis.tracking.person_only import PersonOnlyTracker


CORNERS = [(0, 0), (610, 0), (610, 1340), (0, 1340)]


def person(x, y, key=None, **kwargs):
    return dict(court_xy=[x, y], image_xy=[x * 100, y * 100], confidence=.9,
                location_confidence=.9, association_key=key, **kwargs)


def four():
    return [person(1, 3, 'a'), person(5, 3, 'b'), person(1, 11, 'c'), person(5, 11, 'd')]


class HalfCourtSlotsTests(unittest.TestCase):
    def tracker(self, count=4):
        return HalfCourtSlotTracker(CourtSpace(CORNERS), fps=10, match_mode='doubles' if count == 4 else 'singles',
            lock_match_roster=True, expected_roster_count=count, roster_stable_frames=3,
            roster_discovery_seconds=1, require_association_keys=True)

    def opened(self):
        tracker = self.tracker()
        for frame in range(11):
            tracker.update(frame, four())
        self.assertEqual(len(tracker.tracks), 4)
        return tracker

    def test_disordered_four_people_do_not_get_formal_ids(self):
        tracker = self.tracker()
        for frame in range(30):
            self.assertEqual(tracker.update(frame, [person(x, 11) for x in [1, 2, 3, 4]]), [])
        self.assertEqual(tracker._next_id, 1)

    def test_balanced_opening_waits_for_source_time_not_just_three_frames(self):
        tracker = self.tracker()
        for frame in range(10):
            self.assertEqual(tracker.update(frame, four()), [])
        self.assertEqual(len(tracker.update(10, four())), 4)

    def test_readiness_resets_when_sides_become_unbalanced(self):
        tracker = self.tracker()
        for frame in range(9):
            tracker.update(frame, four())
        for frame in range(9, 21):
            tracker.update(frame, four()[:3])
        for frame in range(21, 31):
            self.assertEqual(tracker.update(frame, four()), [])
        self.assertEqual(len(tracker.update(31, four())), 4)

    def test_brief_missed_detection_does_not_restart_entire_readiness_window(self):
        tracker = self.tracker()
        for frame in range(11):
            rows = tracker.update(frame, four()[:3] if frame == 5 else four())
        self.assertEqual(len(rows), 4)

    def test_singles_has_one_slot_per_half(self):
        tracker = self.tracker(2)
        for frame in range(11):
            tracker.update(frame, [person(2, 3), person(3, 11)])
        self.assertEqual(sorted(tracker.slot_ends.values()), ['end_a', 'end_b'])

    def test_new_person_after_long_absence_automatically_inherits_same_half_slot(self):
        tracker = self.opened()
        for frame in range(11, 50):
            tracker.update(frame, four()[1:])
        before = tracker.track_metrics['track_001']['distance_m']
        rows = tracker.update(1000, [person(2, 4, 'entirely-new-person')] + four()[1:])
        returned = next(r for r in rows if r['track_id'] == 'track_001')
        self.assertEqual(returned['status'], 'detected')
        self.assertEqual(returned['association']['source'], 'half_court_slot_replacement')
        self.assertEqual(returned['occupancy_segment'], 2)
        self.assertGreaterEqual(returned['association']['identity_confidence'], .7)
        self.assertEqual(returned['motion']['current_speed_mps'], None)
        self.assertEqual(tracker.track_metrics['track_001']['distance_m'], before)
        self.assertEqual(len(tracker.tracks), 4)

    def test_two_returning_people_are_assigned_one_to_one_within_their_half(self):
        tracker = self.opened()
        tracker.update(11, four()[2:])
        rows = tracker.update(12, [person(4.8, 4, 'new-right'), person(1.2, 4, 'new-left')] + four()[2:])
        self.assertEqual({r['association']['key'] for r in rows}, {'new-right', 'new-left', 'c', 'd'})
        self.assertEqual(tracker.tracks['track_001'].association_key, 'new-left')
        self.assertEqual(tracker.tracks['track_002'].association_key, 'new-right')

    def test_same_half_movement_preserves_visual_key_instead_of_left_right_resorting(self):
        tracker = self.opened()
        tracker.update(11, [person(5, 3, 'a'), person(1, 3, 'b')] + four()[2:])
        self.assertEqual(tracker.tracks['track_001'].association_key, 'a')

    def test_temporarily_keyless_measurements_do_not_repeatedly_split_a_slot(self):
        tracker = self.opened()
        for frame in range(11, 21):
            rows = four()
            rows[0]['association_key'] = None
            tracker.update(frame, rows)
        self.assertEqual(tracker.occupancy_segments['track_001'], 1)
        tracker.update(21, four()[1:])
        rows = four()
        rows[0]['association_key'] = None
        tracker.update(22, rows)
        tracker.update(23, rows)
        self.assertEqual(tracker.occupancy_segments['track_001'], 2)

    def test_crossing_net_does_not_steal_an_opposite_half_slot(self):
        tracker = self.opened()
        tracker.update(11, [person(1, 11, 'a')] + four()[1:])
        self.assertEqual(tracker.slot_ends['track_001'], 'end_a')
        self.assertGreater(tracker.tracks['track_001'].missed_frames, 0)
        self.assertEqual(len(tracker.tracks), 4)

    def test_duplicate_key_or_low_quality_observation_cannot_open_roster(self):
        tracker = self.tracker()
        rows = four()
        rows[1]['association_key'] = 'a'
        for frame in range(15):
            self.assertEqual(tracker.update(frame, rows), [])
        rows = four()
        rows[0]['location_confidence'] = .2
        for frame in range(15, 30):
            self.assertEqual(tracker.update(frame, rows), [])

    def test_checkpoint_retains_slot_scope_and_segment_and_rejects_legacy(self):
        tracker = self.opened()
        tracker.update(11, [])
        state = tracker.snapshot_state()
        restored = self.tracker()
        restored.restore_state(state)
        self.assertEqual(restored.slot_ends, tracker.slot_ends)
        self.assertEqual(restored.occupancy_segments, tracker.occupancy_segments)
        rows = restored.update(12, four())
        self.assertTrue(all(r['identity_scope'] == HALF_COURT_SLOTS for r in rows))
        state.pop('half_court_slots')
        with self.assertRaisesRegex(ValueError, 'policy mismatch'):
            restored.restore_state(state)

    def test_public_registry_does_not_require_human_confirmation_on_return(self):
        tracker = PersonOnlyTracker(CORNERS, fps=10, lock_match_roster=True, expected_roster_count=4,
            match_mode='doubles', roster_discovery_seconds=1, roster_identity_policy=HALF_COURT_SLOTS)
        for frame in range(11):
            tracker.update(frame, four())
        tracker.update(11, four()[1:])
        snapshot = tracker.update(12, [person(2, 4, 'replacement')] + four()[1:])
        returned = next(r for r in snapshot['tracks'] if r['track_id'] == 'track_001')
        self.assertEqual(returned['lifecycle_state'], 'active')
        self.assertEqual(returned['quality']['reassociation_uncertain_count'], 0)
        self.assertEqual(snapshot['tracking']['identity_policy'], HALF_COURT_SLOTS)

    def test_api_config_carries_explicit_policy_and_rejects_unlocked_slots(self):
        from api.stream_models import validate_configuration
        config = dict(analysis_sample_hz=10, pose_imgsz=960, shuttle_detector='tracknet_v3',
                      generate_annotated_video=False, expected_player_count=4,
                      roster_identity_policy=HALF_COURT_SLOTS)
        self.assertEqual(validate_configuration(config)['roster_identity_policy'], HALF_COURT_SLOTS)
        with self.assertRaises(ValueError):
            validate_configuration(dict(config, lock_match_roster=False))

    def test_identity_checkpoint_cannot_be_read_as_person_tracking(self):
        tracker = PersonOnlyTracker(CORNERS, fps=10, lock_match_roster=True, expected_roster_count=4,
            match_mode='doubles', roster_discovery_seconds=1, roster_identity_policy=HALF_COURT_SLOTS)
        state = tracker.snapshot_state()
        individual = PersonOnlyTracker(CORNERS, fps=10, lock_match_roster=True, expected_roster_count=4,
            match_mode='doubles', roster_discovery_seconds=1)
        with self.assertRaisesRegex(ValueError, 'identity policy'):
            individual.restore_state(state)

    def test_whole_video_pipeline_uses_same_slots_on_sampled_source_clock(self):
        from badminton_analysis.analysis.fixed_camera_match import FixedCameraMatchPipeline
        from api.app import _parse_options
        import json
        options = _parse_options(json.dumps(dict(match_mode='doubles', roster_identity_policy=HALF_COURT_SLOTS)))
        self.assertEqual(options['roster_identity_policy'], HALF_COURT_SLOTS)
        pipeline = FixedCameraMatchPipeline(CORNERS, fps=30, match_mode='doubles', lock_match_roster=True,
            roster_discovery_seconds=1, roster_identity_policy=HALF_COURT_SLOTS)
        for frame in range(0, 30, 3):
            self.assertEqual(pipeline.update(frame, four(), None)['tracks'], [])
        self.assertEqual(len(pipeline.update(30, four(), None)['tracks']), 4)
        pipeline.update(33, four()[1:], None)
        returned = pipeline.update(36, [person(2, 4, 'another-person')] + four()[1:], None)
        self.assertEqual(returned['tracks'][0]['track_id'], 'track_001')
        self.assertEqual(returned['tracks'][0]['occupancy_segment'], 2)


if __name__ == '__main__':
    unittest.main()
