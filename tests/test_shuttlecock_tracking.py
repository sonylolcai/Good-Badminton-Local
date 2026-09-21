import unittest

from badminton_analysis.detection.shuttlecock import ShuttlecockTracker
from badminton_analysis.analysis.fixed_camera_match import FixedCameraMatchPipeline


class _NoopBallModel:
    pass


class ShuttlecockPredictionTests(unittest.TestCase):
    def test_detector_opt_out_is_not_reported_as_a_missing_or_predicted_ball(self):
        tracker = ShuttlecockTracker(_NoopBallModel())

        self.assertIsNone(tracker.mark_not_requested())
        state = tracker.get_last_detection()
        self.assertEqual(state["status"], "not_requested")
        self.assertEqual(state["source"], "disabled_by_option")
        self.assertFalse(state["visible"])
        self.assertFalse(state["accepted"])
        self.assertEqual(state["rejection_reason"], "detector_disabled")

    def test_short_missing_gap_is_predicted_and_long_gap_remains_missing(self):
        tracker = ShuttlecockTracker(
            _NoopBallModel(),
            max_prediction_frames=3,
            prediction_confidence_decay=0.6,
        )

        tracker.last_detection.update({"visible": True, "confidence": 0.8})
        self.assertEqual(tracker.update_trajectory([10, 10]), [10, 10])
        tracker.last_detection.update({"visible": True, "confidence": 0.8})
        self.assertEqual(tracker.update_trajectory([12, 11]), [12, 11])

        self.assertEqual(tracker.update_trajectory([0, 0]), [14, 12])
        predicted = tracker.get_last_detection()
        self.assertEqual(predicted["status"], "predicted")
        self.assertFalse(predicted["accepted"])
        self.assertEqual(predicted["gap_frames"], 1)
        self.assertAlmostEqual(predicted["confidence"], 0.48)

        tracker.update_trajectory([0, 0])
        tracker.update_trajectory([0, 0])
        self.assertEqual(tracker.update_trajectory([0, 0]), [0, 0])
        self.assertEqual(tracker.get_last_detection()["status"], "missing")

    def test_speed_uses_a_one_second_window_of_observed_points_only(self):
        tracker = ShuttlecockTracker(_NoopBallModel(), measurement_fps=10)
        for x in range(0, 101, 10):
            tracker.update_trajectory([x, 100])
        measured = tracker.get_last_detection()
        self.assertEqual(measured["speed_status"], "measured")
        self.assertAlmostEqual(measured["speed_px_s"], 100.0)

        tracker.update_trajectory([0, 0])
        self.assertIsNone(tracker.get_last_detection()["speed_px_s"])

    def test_contact_candidate_uses_visible_hand_distance_not_court_metres(self):
        pipeline = FixedCameraMatchPipeline(
            [[0, 0], [600, 0], [600, 1200], [0, 1200]], fps=10,
        )
        tracks = [{
            "track_id": "track_001",
            "status": "detected",
            "confidence": 0.9,
            "court_xy_m": [2.0, 2.0],
            "location_evidence": {
                "is_current_measurement": True,
                "bbox_xyxy": [80, 30, 200, 330],
                "hands_image": {"left": None, "right": [150, 160]},
            },
        }]
        shuttle = {
            "status": "approximate",
            "confidence": 0.9,
            "image_xy": [165, 160],
        }

        events = pipeline._detect_hit_events(tracks, shuttle, 10)
        self.assertEqual(events[0]["hitter_track_id"], "track_001")
        self.assertEqual(events[0]["half_court_zone_id"], "rear_left")
        self.assertEqual(pipeline._detect_hit_events(tracks, shuttle, 11), [])


if __name__ == "__main__":
    unittest.main()
