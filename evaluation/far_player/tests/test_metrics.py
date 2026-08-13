import unittest

from evaluation.far_player.annotations import validate_annotations
from evaluation.far_player.compare_baselines import build_summary
from evaluation.far_player.detector_runner import nms_merge, roi_to_pixels
from evaluation.far_player.metrics import evaluate_predictions, iou_xyxy


def annotation(frame_index, people, complete=True):
    return {
        "frame_index": frame_index,
        "time_sec": frame_index * 0.2,
        "label_status": "complete",
        "people_annotation_complete": complete,
        "people": people,
        "ignore_regions": [],
    }


def far_person(x=10):
    return {"id": "far", "role": "far_player", "visibility": "visible", "bbox_xyxy": [x, 10, x + 20, 50]}


class MetricsTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_recall_longest_miss_and_false_positive(self):
        rows = [annotation(index, [far_person()]) for index in range(5)]
        hit = {"bbox_xyxy": [10, 10, 30, 50], "confidence": 0.9, "source": "full_frame"}
        false_positive = {"bbox_xyxy": [70, 10, 90, 50], "confidence": 0.8, "source": "full_frame"}
        predictions = {0: [hit], 1: [hit, false_positive], 4: [hit]}
        result = evaluate_predictions(rows, predictions, iou_threshold=0.3)
        self.assertAlmostEqual(result.metrics["far_player_recall"], 0.6)
        self.assertAlmostEqual(result.metrics["far_player_miss_rate"], 0.4)
        self.assertEqual(result.metrics["longest_consecutive_miss_samples"], 2)
        self.assertAlmostEqual(result.metrics["longest_consecutive_miss_seconds"], 0.4)
        self.assertEqual(result.metrics["false_positive_count"], 1)

    def test_false_positive_is_null_for_incomplete_people_annotation(self):
        rows = [annotation(0, [far_person()], complete=False)]
        result = evaluate_predictions(rows, {0: []})
        self.assertIsNone(result.metrics["false_positive_count"])

    def test_annotation_validation_rejects_unreviewed_rows(self):
        row = annotation(0, [far_person()])
        row["label_status"] = "unlabeled"
        errors = validate_annotations([row], 100, 100)
        self.assertTrue(any("label_status" in error for error in errors))

    def test_roi_and_cross_source_nms(self):
        self.assertEqual(roi_to_pixels((0.0, 0.0, 1.0, 0.5), 852, 480), (0, 0, 852, 240))
        merged = nms_merge(
            [
                {"bbox_xyxy": [10, 10, 30, 50], "confidence": 0.7, "source": "full_frame"},
                {"bbox_xyxy": [11, 10, 31, 50], "confidence": 0.9, "source": "far_roi"},
            ],
            iou_threshold=0.5,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "far_roi")
        self.assertEqual(merged[0]["sources"], ["far_roi", "full_frame"])

    def test_cross_video_summary_and_bytetrack_gate(self):
        def report(width, height, sha, method_recall, timings, review_tier="human_reviewed"):
            methods = {}
            for method, recall in method_recall.items():
                methods[method] = {
                    "metrics": {
                        "far_player_ground_truth_count": 20,
                        "far_player_detected_count": round(20 * recall),
                        "far_player_recall": recall,
                        "longest_consecutive_miss_seconds": 0.1 if recall >= 0.95 else 0.4,
                        "false_positive_count": 0,
                    },
                    "timing": {"evaluated_frame_count": 20, "mean_ms_per_frame": timings[method]},
                }
            return {
                "status": "complete",
                "video": {"width": width, "height": height, "sha256": sha},
                "annotations": {"ground_truth_source": "human_manual", "review_tier": review_tier},
                "methods": methods,
            }

        recalls = {"full_640": 0.90, "full_1280": 1.0, "full_640+far_roi_640": 1.0}
        timings = {"full_640": 10.0, "full_1280": 30.0, "full_640+far_roi_640": 18.0}
        summary = build_summary(
            [report(852, 480, "low", recalls, timings), report(1280, 720, "high", recalls, timings)]
        )
        self.assertEqual(summary["recommended_default_method"], "full_640+far_roi_640")
        self.assertTrue(summary["bytetrack_entry_gate"]["ready"])
        preliminary = build_summary(
            [
                report(852, 480, "low", recalls, timings, "preliminary_reviewed"),
                report(1280, 720, "high", recalls, timings, "preliminary_reviewed"),
            ]
        )
        self.assertFalse(preliminary["bytetrack_entry_gate"]["ready"])
        self.assertFalse(preliminary["bytetrack_entry_gate"]["checks"]["human_ground_truth_reviewed"])


if __name__ == "__main__":
    unittest.main()
