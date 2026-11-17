import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.system import BadmintonAnalysisSystem
from business_gateway.metrics.movement import write_body_profiles
from business_gateway.post_match import generate_business_interpretation


class BusinessInterpretationTests(unittest.TestCase):
    def test_legacy_public_imports_delegate_to_the_same_business_implementation(self):
        from badminton_analysis.analysis.movement_metrics import (
            generate_movement_metrics as legacy_movement,
        )
        from badminton_analysis.analysis.performance_report import (
            generate_performance_report as legacy_report,
        )
        from business_gateway.metrics.movement import generate_movement_metrics
        from business_gateway.report.performance import generate_performance_report

        self.assertIs(legacy_movement, generate_movement_metrics)
        self.assertIs(legacy_report, generate_performance_report)

    def test_three_analysis_artifacts_are_sufficient_and_refresh_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self._write_analysis_contract(run_dir)
            body_path = write_body_profiles(
                run_dir,
                [{"track_id": "track_001", "weight_kg": 70, "height_cm": 175}],
                consent=True,
            )
            previous = {
                key: os.environ.pop(key, None)
                for key in (
                    "GOOD_BADMINTON_LLM_BASE_URL",
                    "GOOD_BADMINTON_LLM_API_KEY",
                    "GOOD_BADMINTON_LLM_MODEL",
                )
            }
            try:
                first = generate_business_interpretation(
                    run_dir,
                    body_profiles_path=body_path,
                )
                second = generate_business_interpretation(
                    run_dir,
                    body_profiles_path=body_path,
                )
            finally:
                for key, value in previous.items():
                    if value is not None:
                        os.environ[key] = value

            self.assertEqual(first["movement_metrics_path"], second["movement_metrics_path"])
            self.assertEqual(first["performance_report_path"], second["performance_report_path"])
            self.assertTrue(Path(first["manifest_path"]).is_file())
            self.assertEqual(first["movement_metrics"]["track_count"], 1)
            self.assertEqual(
                first["movement_metrics"]["players"][0]["energy_estimate"]["status"],
                "estimated",
            )
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["business_interpretation"]["owner"], "business_gateway")

    def test_missing_public_artifact_fails_before_derivation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "metadata.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "detections.jsonl"):
                generate_business_interpretation(run_dir)

    def test_gpu_cleanup_contains_no_business_metric_or_llm_generation(self):
        cleanup_source = inspect.getsource(BadmintonAnalysisSystem._cleanup)

        self.assertNotIn("generate_movement_metrics", cleanup_source)
        self.assertNotIn("generate_performance_report", cleanup_source)
        self.assertIn("deferred_to_business_service", cleanup_source)

    @staticmethod
    def _write_analysis_contract(run_dir):
        rows = []
        for frame in range(1, 7):
            rows.append(
                {
                    "frame": frame,
                    "time_sec": frame / 10,
                    "spatial": {
                        "match": {"mode": "person_only"},
                        "tracks": [
                            {
                                "track_id": "track_001",
                                "status": "detected",
                                "court_xy_m": [1.0 + frame * 0.1, 2.0],
                                "confidence": 0.95,
                                "location_evidence": {"confidence": 0.9},
                                "association": {"identity_confidence": 0.9},
                            }
                        ],
                    },
                }
            )
        (run_dir / "detections.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text(
            json.dumps({"video": {"fps": 10, "duration_sec": 0.6}}),
            encoding="utf-8",
        )
        (run_dir / "spatial_match_summary.json").write_text(
            json.dumps(
                {
                    "match": {"mode": "person_only"},
                    "score_policy": "unknown",
                    "player_style_inputs": [
                        {
                            "track_id": "track_001",
                            "detected_frames": 6,
                            "predicted_frames": 0,
                            "missing_frames": 0,
                            "distance_m": 0.5,
                            "zone_frames": {"mid_center": 6},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
