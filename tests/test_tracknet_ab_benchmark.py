import tempfile
import unittest
from argparse import Namespace
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.shuttle_tracknet_ab.annotations import validate_annotations
from evaluation.shuttle_tracknet_ab.fast_predict_tracknet_v3 import _iter_batches
from evaluation.shuttle_tracknet_ab.fast_predict_tracknet_v3 import _iter_stream_batches
from evaluation.shuttle_tracknet_ab.run_tracknet_v3 import _run_predict
from evaluation.shuttle_tracknet_ab.metrics import evaluate_shuttle_predictions
from evaluation.shuttle_tracknet_ab.prediction_io import (
    load_tracknet_csv,
    materialize_tracknet_detections,
)
from evaluation.shuttle_tracknet_ab.run_ab_benchmark import _write_isolated_derived_artifacts


class TrackNetABBenchmarkTests(unittest.TestCase):
    def test_fast_predictor_receives_hyphenated_cli_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "match.mp4"
            checkpoint = root / "TrackNet_best.pt"
            predictor = root / "fast_predict_tracknet_v3.py"
            for path in (video, checkpoint, predictor):
                path.write_bytes(b"fixture")
            args = Namespace(
                tracknet_python="python3",
                fast_predictor=predictor,
                tracknet_root=root,
                video=video,
                tracknet_checkpoint=checkpoint,
                batch_size=16,
                eval_mode="weight",
                background_sample_count=120,
            )
            with patch("evaluation.shuttle_tracknet_ab.run_tracknet_v3.subprocess.run") as process_run:
                _run_predict(args, root / "output", inpaint_checkpoint=None)

        command = process_run.call_args.args[0]
        self.assertIn("--batch-size", command)
        self.assertIn("--eval-mode", command)
        self.assertNotIn("--batch_size", command)
        self.assertNotIn("--eval_mode", command)
        python_path = process_run.call_args.kwargs["env"]["PYTHONPATH"]
        self.assertEqual(root.name, Path(python_path.split(os.pathsep)[0]).name)

    def test_fast_predictor_uses_only_valid_sliding_windows_then_leaves_tail_to_ensemble(self):
        processed = np.zeros((10, 3, 2, 2), dtype=np.uint8)
        batches = list(
            _iter_batches(
                np,
                processed,
                median_channels=None,
                sequence_length=8,
                batch_size=2,
            )
        )

        # Ten frames with a length-eight window has exactly three valid model
        # windows.  The ensemble path emits the final seven frame positions;
        # passing ten padded windows here would duplicate output frames.
        flattened_indexes = np.concatenate([indexes for indexes, _ in batches], axis=0)
        self.assertEqual((3, 8, 2), flattened_indexes.shape)
        self.assertEqual(list(range(8)), flattened_indexes[0, :, 1].tolist())
        self.assertEqual(list(range(2, 10)), flattened_indexes[-1, :, 1].tolist())

    def test_bounded_streaming_batches_match_full_buffer_window_indexes(self):
        processed = np.zeros((10, 3, 2, 2), dtype=np.uint8)
        chunks = iter(((0, processed[:3]), (3, processed[3:7]), (7, processed[7:])))
        batches = list(
            _iter_stream_batches(
                np,
                chunks,
                median_channels=None,
                sequence_length=8,
                batch_size=2,
                frame_count=10,
            )
        )
        flattened_indexes = np.concatenate([indexes for indexes, _ in batches], axis=0)
        self.assertEqual((3, 8, 2), flattened_indexes.shape)
        self.assertEqual(list(range(8)), flattened_indexes[0, :, 1].tolist())
        self.assertEqual(list(range(2, 10)), flattened_indexes[-1, :, 1].tolist())
        full_batches = list(_iter_batches(np, processed, None, sequence_length=8, batch_size=2))
        self.assertTrue(
            np.array_equal(
                np.concatenate([inputs for _, inputs in batches], axis=0),
                np.concatenate([inputs for _, inputs in full_batches], axis=0),
            )
        )

    def test_metrics_credit_only_raw_measurements_and_keep_inference_separate(self):
        annotations = [
            self._annotation(1, 0.0, "visible", [10, 10]),
            self._annotation(2, 0.1, "visible", [20, 20]),
            self._annotation(3, 0.2, "not_visible", None),
            self._annotation(4, 0.3, "ambiguous", None),
        ]
        predictions = {
            1: {"status": "detected", "image_xy": [12, 11], "source": "yolo"},
            2: {"status": "rectified_tracknet", "image_xy": [20, 20], "source": "tracknet"},
            3: {"status": "detected_tracknet", "image_xy": [30, 30], "source": "tracknet"},
        }

        result = evaluate_shuttle_predictions(annotations, predictions, tolerance_px=4.0)

        self.assertEqual(2, result.metrics["visible_ground_truth_count"])
        self.assertEqual(1, result.metrics["true_positive_count"])
        self.assertEqual(0.5, result.metrics["raw_detection_recall"])
        self.assertEqual(1, result.metrics["inferred_points_on_visible_ground_truth"])
        self.assertEqual(1, result.metrics["false_positive_count"])
        self.assertEqual(1, result.metrics["longest_consecutive_raw_miss_samples"])

    def test_tracknet_csv_and_candidate_artifacts_preserve_raw_vs_rectified_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "prediction.csv"
            csv_path.write_text("Frame,Visibility,X,Y\n5,1,120,80\n6,0,0,0\n", encoding="utf-8")
            predictions = load_tracknet_csv(csv_path, status="detected_tracknet")
            baseline = [
                {"frame": 5, "time_sec": 0.2, "shuttlecock": {"status": "missing"}},
                {"frame": 6, "time_sec": 0.24, "shuttlecock": {"status": "missing"}},
            ]

            raw = materialize_tracknet_detections(baseline, predictions, kind="raw")
            rectified = materialize_tracknet_detections(baseline, predictions, kind="rectified")

        self.assertEqual("detected", raw[0]["shuttlecock"]["status"])
        self.assertTrue(raw[0]["shuttlecock"]["accepted"])
        self.assertEqual("uncalibrated_binary_visibility_threshold_0.5", raw[0]["shuttlecock"]["confidence_status"])
        self.assertEqual("missing", raw[1]["shuttlecock"]["status"])
        self.assertEqual("rectified", rectified[0]["shuttlecock"]["status"])
        self.assertFalse(rectified[0]["shuttlecock"]["accepted"])
        self.assertEqual("missing", baseline[0]["shuttlecock"]["status"])

    def test_annotation_validation_requires_visible_center_and_complete_labels(self):
        rows = [
            self._annotation(2, 0.1, "visible", [10, 10]),
            self._annotation(3, 0.2, "not_visible", None),
        ]
        self.assertEqual([], validate_annotations(rows, 100, 60, require_complete=True))
        rows[0]["shuttle"]["image_xy"] = None
        errors = validate_annotations(rows, 100, 60, require_complete=True)
        self.assertTrue(any("visible shuttle" in error for error in errors))

    def test_candidate_derivation_is_isolated_from_existing_analysis_output(self):
        baseline = [
            {
                "frame": 1,
                "time_sec": 0.0,
                "shuttlecock": {"image": [10, 10], "status": "detected", "accepted": True, "confidence": 0.9},
                "spatial": {"hit_events": []},
            },
            {
                "frame": 2,
                "time_sec": 0.1,
                "shuttlecock": {"image": [20, 15], "status": "detected", "accepted": True, "confidence": 0.9},
                "spatial": {"hit_events": []},
            },
            {
                "frame": 3,
                "time_sec": 0.2,
                "shuttlecock": {"image": [30, 20], "status": "detected", "accepted": True, "confidence": 0.9},
                "spatial": {"hit_events": []},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "detections.jsonl"
            source.parent.mkdir()
            source.write_text("{}\n", encoding="utf-8")
            (source.parent / "metadata.json").write_text("{}", encoding="utf-8")
            artifacts = _write_isolated_derived_artifacts(
                root / "benchmark",
                source,
                baseline,
                {1: {"status": "detected_tracknet", "image_xy": [11, 10]}},
                None,
            )
            self.assertTrue(Path(artifacts["a_current_yolo"]["tracks_path"]).is_file())
            self.assertTrue(Path(artifacts["b_tracknet_v3_raw"]["tracks_path"]).is_file())
        self.assertEqual("detected", baseline[0]["shuttlecock"]["status"])

    @staticmethod
    def _annotation(frame_index, time_sec, visibility, image_xy):
        return {
            "frame_index": frame_index,
            "time_sec": time_sec,
            "label_status": "complete",
            "shuttle": {"visibility": visibility, "image_xy": image_xy},
        }


if __name__ == "__main__":
    unittest.main()
