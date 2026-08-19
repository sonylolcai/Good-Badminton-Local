import json
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.analysis.offline_shot_reconstruction import write_jsonl
from webui.app import _rally_summary_from_result


class RallySummaryTests(unittest.TestCase):
    def test_explicit_no_shuttle_run_does_not_rebuild_shot_or_rally_artifacts(self):
        summary, rows = _rally_summary_from_result(
            {"derived": {"status": "not_requested"}},
            {
                "models": {
                    "shuttlecock_detection": {"primary_source": "none"},
                },
                "derived": {"status": "not_requested"},
            },
        )

        self.assertIn("不检测羽毛球", summary)
        self.assertEqual(rows, [])

    def test_result_summary_rebuilds_missing_local_artifacts_and_keeps_metadata_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            detections_path = run_dir / "detections.jsonl"
            metadata_path = run_dir / "metadata.json"
            write_jsonl(detections_path, [
                {
                    "frame": 1,
                    "time_sec": 0.04,
                    "shuttlecock": {"image": None, "status": "missing", "accepted": False, "confidence": 0.0},
                    "spatial": {"tracks": [], "hit_events": []},
                }
            ])
            metadata_path.write_text("{}\n", encoding="utf-8")

            summary, rows = _rally_summary_from_result(
                {"detections": str(detections_path), "metadata": str(metadata_path)},
                {},
            )

            self.assertIn("0** 个候选回合", summary)
            self.assertEqual(rows, [])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["derived"]["rally_count"], 0)


if __name__ == "__main__":
    unittest.main()
