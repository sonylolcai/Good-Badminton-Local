import unittest
from unittest.mock import Mock

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


def person_with_ankles(left, right, hips=((18, 30), (22, 30))):
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[5:15] = (20, 20)
    keypoints[11] = hips[0]
    keypoints[12] = hips[1]
    keypoints[15] = left
    keypoints[16] = right
    return keypoints


class PlayerGroundPointTests(unittest.TestCase):
    def test_uses_pelvis_projection_then_degraded_pelvis_fallbacks(self):
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
        np.testing.assert_allclose(centroids[1], [120, 255])
        np.testing.assert_allclose(centroids[2], [120, 258])
        self.assertEqual(
            [location["method"] for location in pose_data["locations"]],
            [
                "pelvis_ground_projection",
                "pelvis_single_ankle_ground_projection",
                "pelvis_bbox_ground_projection",
            ],
        )
        self.assertEqual(
            [location["degraded"] for location in pose_data["locations"]],
            [False, True, True],
        )
        self.assertAlmostEqual(pose_data["locations"][0]["confidence"], 0.72)
        self.assertAlmostEqual(pose_data["locations"][1]["confidence"], 0.54)
        self.assertAlmostEqual(pose_data["locations"][2]["confidence"], 0.252)
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

        self.assertEqual(location["method"], "pelvis_bbox_ground_projection")
        self.assertAlmostEqual(location["confidence"], 0.189)

    def test_alternating_foot_positions_keep_body_centre_stable(self):
        visualizer = PlayerPoseVisualizer(
            rtmpose_processor=FakePoseProcessor(None, None, [])
        )
        left_step = person_with_ankles((5, 60), (35, 62))
        right_step = person_with_ankles((15, 62), (45, 60))
        scores = np.full(17, 0.9, dtype=float)

        first = visualizer._select_ground_point(left_step, scores, [0, 0, 50, 65], 0.9)
        second = visualizer._select_ground_point(right_step, scores, [0, 0, 50, 65], 0.9)

        self.assertEqual(first["method"], "pelvis_ground_projection")
        self.assertEqual(second["method"], "pelvis_ground_projection")
        self.assertEqual(first["point"][0], second["point"][0])

    def test_far_baseline_margin_does_not_expand_lateral_boundary(self):
        mapper = CourtMapper([(0, 0), (100, 0), (100, 200), (0, 200)])
        visualizer = PlayerPoseVisualizer(
            rtmpose_processor=FakePoseProcessor(None, None, []),
            court_filter_margin=0.75,
            far_baseline_margin=3.0,
        )

        self.assertTrue(visualizer._is_on_court((50, -30), mapper))
        self.assertFalse(visualizer._is_on_court((125, 20), mapper))

    def test_spatial_tracks_draw_detected_box_and_keep_missing_box_absent(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        tracks = [
            {
                "track_id": "track_001",
                "image_xy": [30, 60],
                "status": "detected",
                "trajectory_image": [[20, 70], [30, 60]],
                "location_evidence": {"bbox_xyxy": [10, 20, 50, 80]},
            },
            {
                "track_id": "track_002",
                "image_xy": [90, 60],
                "status": "missing",
                "trajectory_image": [[80, 70], [90, 60]],
                "location_evidence": {"bbox_xyxy": [70, 20, 110, 80]},
            },
        ]

        PlayerPoseVisualizer._draw_spatial_tracks(frame, tracks)

        # Green BGR box for a measured player. The missing track only has a
        # red last-known point/label, never a human-shaped rectangle.
        np.testing.assert_array_equal(frame[20, 10], np.asarray([0, 255, 0], dtype=np.uint8))
        np.testing.assert_array_equal(frame[20, 70], np.asarray([0, 0, 0], dtype=np.uint8))

    def test_spatial_track_id_is_black_at_the_foot_label(self):
        frame = np.full((100, 160, 3), (0, 180, 0), dtype=np.uint8)
        tracks = [{
            "track_id": "track_001",
            "image_xy": [30, 60],
            "status": "detected",
            "trajectory_image": [],
            "location_evidence": {"bbox_xyxy": [10, 20, 50, 80]},
        }]

        PlayerPoseVisualizer._draw_spatial_tracks(frame, tracks)

        # The label is below the foot point, not green like the box.  The
        # white outline is allowed, but the glyph itself must contain black.
        foot_label_region = frame[63:82, 35:130]
        self.assertGreater(np.count_nonzero(np.all(foot_label_region == 0, axis=2)), 0)

    def test_spatial_track_stats_are_used_instead_of_legacy_upper_lower_stats(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        visualizer = PlayerPoseVisualizer(
            rtmpose_processor=FakePoseProcessor(None, None, []),
            show_skeletons=False,
        )
        stats_visualizer = Mock()
        tracks = [{
            "track_id": "track_001",
            "status": "missing",
            "image_xy": [30, 60],
            "location_evidence": {},
            "motion": {"current_speed_mps": None},
        }]

        visualizer.draw_players(
            frame,
            player_tracker=Mock(),
            cached_movement_stats={"upper": {"current_speed": 99.0}},
            stats_visualizer=stats_visualizer,
            rally_count=None,
            spatial_tracks=tracks,
        )

        stats_visualizer.draw_spatial_track_stats.assert_called_once_with(frame, tracks, None)
        stats_visualizer.draw_player_stats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
