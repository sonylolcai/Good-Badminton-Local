import unittest

import numpy as np

from badminton_analysis.court.mapper import CourtMapper
from badminton_analysis.visualization.player_pose import PlayerPoseVisualizer


class FakePoseProcessor:
    inference_name = "Fake-Pose"

    def __init__(self, people, scores, detections):
        self.people = people
        self.scores = scores
        self.detections = detections

    def process_frame(self, _frame):
        return self.people, self.scores

    def get_last_detections(self):
        return self.detections


def person_with_ankles(left, right):
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[5:15] = (20, 20)
    keypoints[15] = left
    keypoints[16] = right
    return keypoints


class PlayerGroundPointTests(unittest.TestCase):
    def test_uses_two_ankles_single_ankle_then_bbox_bottom(self):
        people = np.stack(
            [
                person_with_ankles((10, 60), (30, 62)),
                person_with_ankles((40, 55), (0, 0)),
                person_with_ankles((0, 0), (0, 0)),
            ]
        )
        scores = np.full((3, 17), 0.9, dtype=float)
        scores[1, 16] = 0.0
        scores[2, 15:17] = 0.0
        detections = [
            {"bbox": np.asarray([5, 5, 35, 65]), "confidence": 0.8, "source": "full_frame", "keypoint_scores": scores[0]},
            {"bbox": np.asarray([35, 5, 55, 60]), "confidence": 0.8, "source": "far_roi", "keypoint_scores": scores[1]},
            {"bbox": np.asarray([60, 5, 90, 58]), "confidence": 0.8, "source": "far_roi", "keypoint_scores": scores[2]},
        ]
        visualizer = PlayerPoseVisualizer(
            rtmpose_processor=FakePoseProcessor(people, scores, detections)
        )

        centroids, _, _ = visualizer.detect_players(np.zeros((80, 100, 3), dtype=np.uint8), 100, 200)
        pose_data = visualizer.get_current_pose_data()

        self.assertEqual(len(centroids), 3)
        np.testing.assert_allclose(centroids[0], [120, 261])
        np.testing.assert_allclose(centroids[1], [140, 255])
        np.testing.assert_allclose(centroids[2], [175, 258])
        self.assertEqual(
            [location["method"] for location in pose_data["locations"]],
            ["ankles_midpoint", "single_ankle", "bbox_bottom_center"],
        )
        self.assertEqual(
            [location["degraded"] for location in pose_data["locations"]],
            [False, True, True],
        )
        self.assertAlmostEqual(pose_data["locations"][0]["confidence"], 0.72)
        self.assertAlmostEqual(pose_data["locations"][1]["confidence"], 0.54)
        self.assertAlmostEqual(pose_data["locations"][2]["confidence"], 0.28)
        self.assertEqual(pose_data["detections"][1]["source"], "far_roi")
        np.testing.assert_allclose(pose_data["detections"][2]["bbox"], [160, 205, 190, 258])

    def test_low_confidence_ankles_fall_back_to_bbox(self):
        person = person_with_ankles((10, 60), (30, 62))
        scores = np.full((1, 17), 0.9, dtype=float)
        scores[0, 15:17] = 0.1
        detections = [
            {"bbox": np.asarray([5, 5, 35, 65]), "confidence": 0.6, "keypoint_scores": scores[0]}
        ]
        visualizer = PlayerPoseVisualizer(
            rtmpose_processor=FakePoseProcessor(np.stack([person]), scores, detections),
            keypoint_conf_threshold=0.25,
        )

        visualizer.detect_players(np.zeros((80, 100, 3), dtype=np.uint8), 0, 0)
        location = visualizer.get_current_pose_data()["locations"][0]

        self.assertEqual(location["method"], "bbox_bottom_center")
        self.assertAlmostEqual(location["confidence"], 0.21)

    def test_far_baseline_margin_does_not_expand_lateral_boundary(self):
        mapper = CourtMapper([(0, 0), (100, 0), (100, 200), (0, 200)])
        visualizer = PlayerPoseVisualizer(
            rtmpose_processor=FakePoseProcessor(None, None, []),
            court_filter_margin=0.75,
            far_baseline_margin=3.0,
        )

        self.assertTrue(visualizer._is_on_court((50, -30), mapper))
        self.assertFalse(visualizer._is_on_court((125, 20), mapper))


if __name__ == "__main__":
    unittest.main()
