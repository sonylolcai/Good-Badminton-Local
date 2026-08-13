import unittest

import numpy as np

from badminton_analysis.tracking.player import PlayerTracker


class _Writer:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


class PlayerTrackingEvidenceTests(unittest.TestCase):
    def test_writes_degraded_location_and_inference_provenance(self):
        writer = _Writer()
        tracker = PlayerTracker(
            corners=[(0, 0), (100, 0), (100, 200), (0, 200)],
            threshold=100,
            detection_writer=writer,
            fps=30,
        )
        location = (50.0, 50.0)
        tracker.update(
            frame_index=3,
            centroids=[location],
            ball_image_position=None,
            left_hand_positions={},
            right_hand_positions={},
            detect_frame_count=1,
            pose_detections=[
                {
                    "location": list(location),
                    "location_method": "bbox_bottom_center",
                    "location_confidence": 0.21,
                    "location_degraded": True,
                    "source": "far_roi",
                    "merged_sources": ["far_roi", "full_frame"],
                    "confidence": 0.6,
                    "bbox": np.asarray([40, 20, 60, 50]),
                    "inference": {
                        "model": "weights/yolo11n-pose.pt",
                        "imgsz": 640,
                        "conf": 0.25,
                        "device": "cpu",
                        "roi": [0, 0, 100, 100],
                        "input_shape": [100, 100],
                    },
                }
            ],
        )

        evidence = writer.records[0]["players"]["upper"]["position_evidence"]
        self.assertEqual(evidence["status"], "detected_degraded")
        self.assertEqual(evidence["method"], "bbox_bottom_center")
        self.assertEqual(evidence["source"], "far_roi")
        self.assertEqual(evidence["inference"]["imgsz"], 640)
        self.assertEqual(evidence["bbox_xyxy"], [40.0, 20.0, 60.0, 50.0])

    def test_prefers_candidate_nearest_previous_player_position(self):
        tracker = PlayerTracker(
            corners=[(0, 0), (100, 0), (100, 200), (0, 200)],
            threshold=100,
            fps=30,
        )
        tracker.update(1, [(50.0, 50.0)], None, {}, {}, 1)

        tracker.update(2, [(52.0, 51.0), (80.0, 90.0)], None, {}, {}, 2)

        self.assertEqual(tracker.players["upper"], (52.0, 51.0))

    def test_selects_one_stable_candidate_for_each_legacy_region(self):
        """Both legacy slots must use the same deterministic candidate policy.

        This guards the transition period before the slot-based tracker is
        replaced by persistent track IDs: an extra person must not overwrite
        the lower player simply because it appears later in YOLO's result list.
        """
        tracker = PlayerTracker(
            corners=[(0, 0), (100, 0), (100, 200), (0, 200)],
            threshold=100,
            fps=30,
        )

        tracker.update(
            1,
            [(40.0, 40.0), (50.0, 50.0), (80.0, 120.0), (50.0, 160.0)],
            None,
            {},
            {},
            1,
        )
        self.assertEqual(tracker.players["upper"], (50.0, 50.0))
        self.assertEqual(tracker.players["lower"], (50.0, 160.0))

        tracker.update(
            2,
            [(52.0, 51.0), (80.0, 90.0), (55.0, 159.0), (90.0, 190.0)],
            None,
            {},
            {},
            2,
        )
        self.assertEqual(tracker.players["upper"], (52.0, 51.0))
        self.assertEqual(tracker.players["lower"], (55.0, 159.0))

    def test_writes_shuttlecock_prediction_evidence(self):
        writer = _Writer()
        tracker = PlayerTracker(
            corners=[(0, 0), (100, 0), (100, 200), (0, 200)],
            threshold=100,
            detection_writer=writer,
            fps=30,
        )

        tracker.update(
            2,
            [],
            [14, 12],
            {},
            {},
            2,
            ball_detection={
                "status": "predicted",
                "confidence": 0.48,
                "gap_frames": 1,
                "source": "constant_velocity",
                "accepted": False,
            },
        )

        shuttle = writer.records[0]["shuttlecock"]
        self.assertEqual(shuttle["image"], [14.0, 12.0])
        self.assertEqual(shuttle["status"], "predicted")
        self.assertAlmostEqual(shuttle["confidence"], 0.48)
        self.assertEqual(shuttle["gap_frames"], 1)


if __name__ == "__main__":
    unittest.main()
