import json
import tempfile
import unittest
from pathlib import Path

from webui.app import _match_identity_table, _save_match_identity_table


class MatchIdentityReviewTests(unittest.TestCase):
    def test_post_match_binding_is_separate_from_raw_detections(self):
        with tempfile.TemporaryDirectory() as temporary:
            analysis = Path(temporary)
            (analysis / "spatial_match_summary.json").write_text(
                json.dumps({
                    "player_style_inputs": [
                        {"track_id": "track_001", "detected_frames": 10, "predicted_frames": 1, "missing_frames": 2, "distance_m": 4.1},
                        {"track_id": "track_002", "detected_frames": 11, "predicted_frames": 0, "missing_frames": 1, "distance_m": 4.2},
                    ]
                }),
                encoding="utf-8",
            )
            (analysis / "metadata.json").write_text(
                json.dumps({"temporal_tracking": {"players": {"match_mode": "doubles"}}}),
                encoding="utf-8",
            )
            detections = analysis / "detections.jsonl"
            detections.write_text('{"raw":true}\n', encoding="utf-8")

            table, _notice = _match_identity_table(str(analysis))
            table[0][1:3] = ["alice", "team_a"]
            table[1][1:3] = ["bob", "team_a"]
            message = _save_match_identity_table(str(analysis), table)

            bindings = json.loads((analysis / "match_identity_claims.json").read_text(encoding="utf-8"))
            self.assertIn("已保存 2 条", message)
            self.assertEqual(bindings["bindings"][0]["person_id"], "alice")
            self.assertEqual(detections.read_text(encoding="utf-8"), '{"raw":true}\n')


if __name__ == "__main__":
    unittest.main()
