import copy
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.analysis.offline_shot_reconstruction import (
    build_rallies_from_manual_terminals,
    build_shot_events,
    build_rallies,
    evaluate_rally_terminal_predictions,
    generate_offline_artifacts,
    infer_missing_shuttle_events,
    reconstruct_shuttle_track,
    write_jsonl,
)


class OfflineShotReconstructionTests(unittest.TestCase):
    def test_reconstructs_only_a_short_bounded_gap_and_marks_long_gap_unknown(self):
        rows = [
            self._row(1, [10, 10], accepted=True, confidence=0.50),
            self._row(2, [10, 10], accepted=True, confidence=0.50),
            self._row(3, [10, 10], accepted=True, confidence=0.50),
            self._row(4, [100, 100], accepted=True, confidence=0.90),
            self._row(5, None),
            self._row(6, [140, 120], accepted=True, confidence=0.90),
            self._row(7, None),
            self._row(8, None),
            self._row(9, None),
            self._row(10, None),
            self._row(11, [500, 300], accepted=True, confidence=0.90),
        ]
        original = copy.deepcopy(rows)
        tracks = reconstruct_shuttle_track(rows, fps=10, width=640, height=480, max_gap_sec=0.20)

        self.assertEqual([item["status"] for item in tracks[:3]], ["rejected_artifact"] * 3)
        self.assertEqual(tracks[4]["status"], "reconstructed")
        self.assertEqual(tracks[4]["image_xy"], [120.0, 110.0])
        self.assertEqual(tracks[7]["status"], "unknown_gap")
        self.assertEqual(rows, original)

    def test_event_artifacts_keep_machine_candidates_out_of_statistics(self):
        rows = [
            self._row(1, [100, 100], accepted=True, confidence=0.90, hit="track_001", zone="front_center"),
            self._row(2, [120, 120], accepted=True, confidence=0.90),
            self._row(3, [180, 180], accepted=True, confidence=0.90),
            self._row(5, [250, 240], accepted=True, confidence=0.90, hit="track_002", zone="rear_center"),
        ]
        tracks = reconstruct_shuttle_track(rows, fps=10)
        events = build_shot_events(rows, tracks, fps=10)

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["hitter"]["track_id"], "track_001")
        self.assertEqual(events[0]["receiver"]["track_id"], "track_002")
        self.assertEqual(events[0]["proposal"]["label"], "lift")
        self.assertEqual(events[0]["trajectory"]["outbound_speed"]["basis"], "first_two_post_hit_trajectory_points")
        self.assertEqual(events[0]["trajectory"]["outbound_speed"]["unit"], "px/s")
        self.assertFalse(events[0]["decision"]["eligible_for_statistics"])

    def test_generator_writes_separate_derived_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "detections.jsonl"
            write_jsonl(source, [self._row(1, [10, 20], accepted=True, confidence=0.8)])
            result = generate_offline_artifacts(source, fps=10)

            self.assertTrue(Path(result["tracks_path"]).is_file())
            self.assertTrue(Path(result["events_path"]).is_file())
            self.assertTrue(Path(result["rallies_path"]).is_file())
            self.assertEqual(result["frame_count"], 1)

    def test_stationary_shuttle_boundary_counts_shots_in_one_candidate_rally(self):
        rows = [
            self._row(1, [100, 100], accepted=True, confidence=0.9, hit="track_001", zone="front_center"),
            self._row(2, [130, 120], accepted=True, confidence=0.9),
            self._row(3, [165, 145], accepted=True, confidence=0.9),
            self._row(5, [220, 210], accepted=True, confidence=0.9, hit="track_002", zone="rear_center"),
            self._row(6, [230, 220], accepted=True, confidence=0.9),
            self._row(7, [232, 220], accepted=True, confidence=0.9),
            self._row(8, [233, 220], accepted=True, confidence=0.9),
            self._row(9, [233, 220], accepted=True, confidence=0.9),
            self._row(10, [233, 220], accepted=True, confidence=0.9),
        ]
        tracks = reconstruct_shuttle_track(rows, fps=10)
        events = build_shot_events(rows, tracks, fps=10)
        rallies = build_rallies(
            tracks,
            events,
            stationary_speed_px_s=30.0,
            stationary_min_duration_sec=0.25,
            court_polygon=[[0, 0], [640, 0], [640, 480], [0, 480]],
        )

        self.assertEqual(len(rallies), 1)
        self.assertEqual(rallies[0]["shot_count"], 2)
        self.assertEqual(rallies[0]["end_reason"], "shuttle_stationary_or_slow")
        self.assertEqual(events[0]["rally_id"], "rally_0001")
        self.assertEqual(events[1]["shot_index_in_rally"], 2)

    def test_outside_static_artifact_and_missing_gap_do_not_split_a_rally(self):
        events = [
            {"event_id": "shot_0001", "hit_time_sec": 1.0, "event_origin": "observed"},
            {"event_id": "shot_0002", "hit_time_sec": 9.0, "event_origin": "observed"},
        ]
        tracks = [
            {"time_sec": 2.0, "status": "detected", "confidence": 0.9, "image_xy": [900, 20]},
            {"time_sec": 2.4, "status": "detected", "confidence": 0.9, "image_xy": [900, 20]},
            {"time_sec": 2.8, "status": "detected", "confidence": 0.9, "image_xy": [900, 20]},
            {"time_sec": 3.2, "status": "unknown_gap", "confidence": 0.0, "image_xy": None},
            {"time_sec": 7.0, "status": "unknown_gap", "confidence": 0.0, "image_xy": None},
        ]

        rallies = build_rallies(
            tracks,
            events,
            stationary_min_duration_sec=0.5,
            court_polygon=[[0, 0], [640, 0], [640, 480], [0, 480]],
        )

        self.assertEqual(1, len(rallies))
        self.assertEqual("video_end_without_confirmed_terminal_event", rallies[0]["end_reason"])
        self.assertEqual("rally_0001", events[1]["rally_id"])

    def test_manual_terminal_facts_create_eight_style_reviewed_segments_without_mutating_events(self):
        events = [
            {"event_id": "shot_0001", "hit_time_sec": 1.0, "event_origin": "observed"},
            {"event_id": "shot_0002", "hit_time_sec": 4.0, "event_origin": "motion_constraint_candidate"},
            {"event_id": "shot_0003", "hit_time_sec": 6.0, "event_origin": "observed"},
        ]
        original = copy.deepcopy(events)

        reviewed = build_rallies_from_manual_terminals(
            events,
            [
                {"terminal_id": "terminal_0001", "time_sec": 5.0, "outcome": "out_of_bounds"},
                {"terminal_id": "terminal_0002", "time_sec": 10.0, "outcome": "landed_in_bounds"},
            ],
        )

        self.assertEqual(2, len(reviewed))
        self.assertEqual((0.0, 5.0, 2), (
            reviewed[0]["start_time_sec"], reviewed[0]["end_time_sec"], reviewed[0]["shot_count"],
        ))
        self.assertEqual((5.0, 10.0, 1), (
            reviewed[1]["start_time_sec"], reviewed[1]["end_time_sec"], reviewed[1]["shot_count"],
        ))
        self.assertEqual("out_of_bounds", reviewed[0]["terminal"]["outcome"])
        self.assertEqual(events, original)

    def test_terminal_evaluation_compares_auto_candidates_without_using_references_as_predictions(self):
        report = evaluate_rally_terminal_predictions(
            [
                {"rally_id": "rally_0001", "end_time_sec": 10.2, "end_reason": "shuttle_stationary_or_slow"},
                {"rally_id": "rally_0002", "end_time_sec": 20.0, "end_reason": "shuttle_stationary_or_slow"},
                {"rally_id": "rally_0003", "end_time_sec": 30.0, "end_reason": "video_end_without_confirmed_terminal_event"},
                {"rally_id": "reviewed_rally_0001", "end_time_sec": 40.0, "end_reason": "human_confirmed_out"},
            ],
            [
                {"terminal_id": "reference_01", "time_sec": 10.0, "outcome": "out_of_bounds"},
                {"terminal_id": "reference_02", "time_sec": 40.0, "outcome": "landed_in_bounds"},
            ],
            tolerance_sec=0.5,
        )

        self.assertEqual(2, report["prediction_count"])
        self.assertEqual(2, report["reference_count"])
        self.assertEqual(1, report["matched_count"])
        self.assertEqual(0.5, report["precision"])
        self.assertEqual(0.5, report["recall"])
        self.assertEqual(20.0, report["false_positives"][0]["time_sec"])
        self.assertEqual("reference_02", report["missed_references"][0]["terminal_id"])

    def test_same_singles_hitter_can_create_low_confidence_missing_return_candidate(self):
        rows = [
            self._row(
                1, None, hit="track_001", zone="front_center", match_mode="singles",
                visible_tracks=["track_001", "track_002"],
            ),
            self._row(
                11, None, hit="track_001", zone="rear_center", match_mode="singles",
                visible_tracks=["track_001", "track_002"],
            ),
        ]
        events = build_shot_events(rows, reconstruct_shuttle_track(rows, fps=10), fps=10)
        inferred = infer_missing_shuttle_events(rows, events)

        self.assertEqual(len(inferred), 3)
        missing_return = inferred[1]
        self.assertEqual(missing_return["candidate_source"], "motion_inferred_missing_shuttle")
        self.assertEqual(missing_return["event_origin"], "motion_constraint_candidate")
        self.assertEqual(missing_return["hitter"]["track_id"], "track_002")
        self.assertIsNone(missing_return["trajectory"]["outbound_speed"]["value"])
        self.assertFalse(missing_return["decision"]["eligible_for_statistics"])

    def test_missing_shuttle_rule_does_not_assign_a_doubles_hitter(self):
        rows = [
            self._row(
                1, None, hit="track_001", zone="front_center", match_mode="doubles",
                visible_tracks=["track_001", "track_002", "track_003", "track_004"],
            ),
            self._row(
                11, None, hit="track_001", zone="rear_center", match_mode="doubles",
                visible_tracks=["track_001", "track_002", "track_003", "track_004"],
            ),
        ]
        events = build_shot_events(rows, reconstruct_shuttle_track(rows, fps=10), fps=10)

        self.assertEqual(len(infer_missing_shuttle_events(rows, events)), 2)

    @staticmethod
    def _row(frame, point, accepted=False, confidence=0.0, hit=None, zone=None, match_mode=None, visible_tracks=None):
        tracks = []
        for track_id in visible_tracks or ([hit] if hit else []):
            tracks.append(
                {
                    "track_id": track_id,
                    "status": "detected",
                    "confidence": 0.9,
                    "zone_id": zone if track_id == hit else "rear_center",
                    "court_xy_m": [3.0, 3.0],
                    "location_evidence": {"hands_image": {"left": [point[0] if point else 0, point[1] if point else 0], "right": None}},
                }
            )
        return {
            "frame": frame,
            "time_sec": frame / 10,
            "shuttlecock": {
                "image": point,
                "status": "detected" if accepted else "missing",
                "accepted": accepted,
                "confidence": confidence,
            },
            "spatial": {
                "match": {"mode": match_mode} if match_mode else {},
                "tracks": tracks,
                "hit_events": ([{"status": "candidate", "hitter_track_id": hit, "confidence": 0.4, "reason": "synthetic"}] if hit else []),
            },
        }


if __name__ == "__main__":
    unittest.main()
