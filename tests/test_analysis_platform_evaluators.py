import copy
import json
import tempfile
import unittest
from pathlib import Path

from analysis_platform.evaluators import execute_evaluator, normalize_report


FIXTURES = Path(__file__).parent / "fixtures"


class EvaluationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.reports = json.loads(
            (FIXTURES / "evaluator_reports.json").read_text(encoding="utf-8")
        )

    def test_four_existing_report_shapes_are_normalized_without_mutation(self):
        for name, report in self.reports.items():
            original = copy.deepcopy(report)
            with self.subTest(name=name):
                normalized = normalize_report(name, report)
                self.assertGreater(normalized["metric_count"], 0)
                self.assertEqual(original, report)

    def test_rectified_shuttle_points_remain_preliminary(self):
        normalized = normalize_report(
            "shuttle_tracknet_ab", self.reports["shuttle_tracknet_ab"]
        )
        rectified = [
            row
            for row in normalized["metrics"]
            if row["scope_id"] == "method:b_star_tracknet_v3_rectified"
        ]
        raw = [
            row
            for row in normalized["metrics"]
            if row["scope_id"] == "method:b_tracknet_v3_raw"
        ]
        self.assertTrue(rectified)
        self.assertTrue(all(row["status"] == "preliminary" for row in rectified))
        self.assertTrue(all(row["status"] == "valid" for row in raw))

    def test_doubles_adapter_executes_the_existing_cli(self):
        annotations = [
            {
                "frame": 1,
                "view_id": "rear",
                "players": [
                    {"person_id": "p1", "court_xy_m": [1.0, 2.0]},
                    {"person_id": "p2", "court_xy_m": [5.0, 2.0]},
                    {"person_id": "p3", "court_xy_m": [1.0, 11.0]},
                    {"person_id": "p4", "court_xy_m": [5.0, 11.0]}
                ],
                "occluded_person_ids": [],
                "hits": []
            }
        ]
        detections = [
            {
                "frame": 1,
                "spatial": {
                    "tracks": [
                        {"track_id": "t1", "status": "detected", "court_xy_m": [1.0, 2.0]},
                        {"track_id": "t2", "status": "detected", "court_xy_m": [5.0, 2.0]},
                        {"track_id": "t3", "status": "detected", "court_xy_m": [1.0, 11.0]},
                        {"track_id": "t4", "status": "detected", "court_xy_m": [5.0, 11.0]}
                    ],
                    "hit_events": []
                }
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation_path = root / "annotations.jsonl"
            detection_path = root / "detections.jsonl"
            annotation_path.write_text(json.dumps(annotations[0]) + "\n", encoding="utf-8")
            detection_path.write_text(json.dumps(detections[0]) + "\n", encoding="utf-8")
            normalized = execute_evaluator(
                "doubles",
                ["--annotations", str(annotation_path), "--detections", str(detection_path)],
                root / "report.json",
            )

        recall = next(
            row for row in normalized["metrics"] if row["metric_key"] == "tracking.player_recall"
        )
        self.assertEqual(recall["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
