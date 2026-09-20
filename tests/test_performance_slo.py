import json
import os
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.analysis.fixed_camera_match import CourtMultiObjectTracker, CourtSpace
from business_gateway.report.performance import generate_performance_report
from badminton_analysis.system import BadmintonAnalysisSystem


class PerformanceSloTests(unittest.TestCase):
    CORNERS = [(0, 0), (610, 0), (610, 1340), (0, 1340)]

    def test_pose_sampling_is_timestamp_based_for_30_and_60_fps_sources(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.pose_sample_hz = 10.0

        system.fps = 30.0
        sampled_30 = [index for index in range(1, 13) if system._should_sample_pose(index)]
        self.assertEqual(sampled_30, [1, 4, 7, 10])

        system.fps = 60.0
        sampled_60 = [index for index in range(1, 20) if system._should_sample_pose(index)]
        self.assertEqual(sampled_60, [1, 7, 13, 19])

    def test_zero_pose_sample_hz_processes_every_source_frame(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.pose_sample_hz = 0.0
        system.fps = 50.0

        self.assertTrue(all(system._should_sample_pose(index) for index in range(1, 31)))

    def test_shared_analysis_cadence_controls_primary_measurement_timestamps(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.analysis_sample_hz = 15.0
        system.pose_sample_hz = 10.0  # Legacy field must not override the shared one.
        system.court_health_check_hz = 2.0
        system.fps = 30.0

        analysis_frames = [index for index in range(1, 12) if system._should_sample_analysis(index)]
        health_frames = [index for index in range(1, 32) if system._should_sample_court_health(index)]

        self.assertEqual(analysis_frames, [1, 3, 5, 7, 9, 11])
        self.assertEqual(health_frames, [1, 16, 31])

    def test_shuttle_cadence_can_exceed_player_cadence(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.analysis_sample_hz = 10.0
        system.pose_sample_hz = 10.0
        system.shuttle_sample_hz = 30.0
        system.fps = 30.0

        player_frames = [index for index in range(1, 10) if system._should_sample_analysis(index)]
        shuttle_frames = [index for index in range(1, 10) if system._should_sample_shuttle(index)]

        self.assertEqual(player_frames, [1, 4, 7])
        self.assertEqual(shuttle_frames, list(range(1, 10)))

    def test_pose_keypoints_remain_tied_to_their_measurement_frame(self):
        tracker = CourtMultiObjectTracker(CourtSpace(self.CORNERS), fps=10)
        observation = {
            "court_xy": (2.0, 2.0),
            "image_xy": (200.0, 200.0),
            "bbox_xyxy": [180, 100, 220, 200],
            "confidence": 0.9,
            "location_method": "ankles_midpoint",
            "location_confidence": 0.8,
            "source": "full_frame",
            "keypoints_image": [[float(index), float(index + 1)] for index in range(17)],
            "keypoint_scores": [0.8] * 17,
        }
        detected = tracker.update(1, [observation])[0]
        self.assertTrue(detected["location_evidence"]["is_current_measurement"])
        self.assertTrue(detected["pose"]["is_current_measurement"])
        self.assertEqual(detected["pose"]["measurement_frame"], 1)
        self.assertEqual(len(detected["pose"]["keypoints_image"]), 17)
        self.assertEqual(len(detected["pose"]["keypoint_scores"]), 17)

        predicted = tracker.update(2, [])[0]
        self.assertEqual(predicted["status"], "predicted")
        self.assertFalse(predicted["location_evidence"]["is_current_measurement"])
        self.assertEqual(predicted["location_evidence"]["measurement_frame"], 1)
        self.assertFalse(predicted["pose"]["is_current_measurement"])
        self.assertIsNone(predicted["pose"]["measurement_frame"])
        self.assertEqual(predicted["pose"]["last_measurement_frame"], 1)
        self.assertIsNone(predicted["pose"]["keypoints_image"])
        self.assertIsNone(predicted["pose"]["keypoint_scores"])

    def test_pose_contract_preserves_invisible_joint_slots(self):
        tracker = CourtMultiObjectTracker(CourtSpace(self.CORNERS), fps=10)
        observation = {
            "court_xy": (2.0, 2.0),
            "image_xy": (200.0, 200.0),
            "confidence": 0.9,
            "keypoints_image": [[float(index), float(index + 1)] if index != 5 else None for index in range(17)],
            "keypoint_scores": [0.8] * 17,
        }

        record = tracker.update(1, [observation])[0]["pose"]

        self.assertEqual(record["format"], "coco17_image_v1")
        self.assertEqual(len(record["keypoints_image"]), 17)
        self.assertIsNone(record["keypoints_image"][5])
        self.assertEqual(record["keypoint_scores"][5], 0.8)

    def test_report_evidence_is_available_without_claiming_an_llm_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.json"
            summary_path = root / "spatial_match_summary.json"
            metadata_path.write_text(json.dumps({
                "video": {"duration_sec": 900, "fps": 30},
                "models": {
                    "pose": {"imgsz": 960, "sample_hz": 10, "processed_frame_count": 9000},
                    "shuttlecock_detection": {"primary_source": "tracknet_v3", "measurement_kind": "temporal_heatmap"},
                },
            }), encoding="utf-8")
            summary_path.write_text(json.dumps({
                "match": {"mode": "singles"},
                "score_policy": "unknown scores excluded",
                "player_style_inputs": [{
                    "track_id": "track_001", "detected_frames": 8000,
                    "predicted_frames": 700, "missing_frames": 300,
                    "distance_m": 125.2, "zone_frames": {"rear_left": 320},
                }],
            }), encoding="utf-8")
            previous = {key: os.environ.pop(key, None) for key in (
                "GOOD_BADMINTON_LLM_BASE_URL", "GOOD_BADMINTON_LLM_API_KEY", "GOOD_BADMINTON_LLM_MODEL"
            )}
            try:
                report = generate_performance_report(root, metadata_path, summary_path)
            finally:
                for key, value in previous.items():
                    if value is not None:
                        os.environ[key] = value
            self.assertEqual(report["status"], "not_configured")
            self.assertTrue(Path(report["report_path"]).is_file())
            self.assertTrue(Path(report["evidence_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
