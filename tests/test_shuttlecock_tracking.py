import unittest

from badminton_analysis.detection.shuttlecock import ShuttlecockTracker


class _NoopBallModel:
    pass


class ShuttlecockPredictionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
