import csv
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.detection.shuttlecock import ShuttlecockTracker
from badminton_analysis.detection.tracknet_v3 import TrackNetV3RawMeasurements


class TrackNetPrimaryFlowTests(unittest.TestCase):
    def _csv(self, root):
        path = Path(root) / "match_ball.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Frame", "Visibility", "X", "Y"])
            writer.writeheader()
            writer.writerows([
                {"Frame": 0, "Visibility": 1, "X": 100.5, "Y": 200.25},
                {"Frame": 1, "Visibility": 0, "X": 0, "Y": 0},
                {"Frame": 2, "Visibility": 1, "X": 140, "Y": 180},
            ])
        return path

    def test_raw_measurements_keep_binary_visibility_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            source = TrackNetV3RawMeasurements(self._csv(root))
            visible = source.measurement_for_frame(0)
            missing = source.measurement_for_frame(1)

        self.assertEqual(source.frame_count, 3)
        self.assertTrue(visible["visible"])
        self.assertEqual(visible["image"], [100.5, 200.25])
        self.assertEqual(visible["source"], "tracknet_v3_raw")
        self.assertEqual(visible["confidence_status"], "uncalibrated_binary_visibility_threshold_0.5")
        self.assertFalse(missing["visible"])
        self.assertIsNone(missing["image"])

    def test_tracker_accepts_raw_tracknet_measurement_without_yolo_motion_gate(self):
        tracker = ShuttlecockTracker(yolo_ball_model=None, max_jump_pixels=1)
        first = {
            "visible": True,
            "image": [100, 100],
            "source": "tracknet_v3_raw",
            "measurement_kind": "temporal_heatmap",
            "confidence": 0.5,
            "confidence_status": "uncalibrated_binary_visibility_threshold_0.5",
        }
        fast_next = {**first, "image": [800, 600]}

        self.assertEqual(tracker.update_external_measurement(first), [100.0, 100.0])
        self.assertEqual(tracker.update_external_measurement(fast_next), [800.0, 600.0])
        state = tracker.get_last_detection()

        self.assertEqual(state["status"], "detected")
        self.assertTrue(state["accepted"])
        self.assertEqual(state["source"], "tracknet_v3_raw")
        self.assertEqual(state["measurement_kind"], "temporal_heatmap")


if __name__ == "__main__":
    unittest.main()
