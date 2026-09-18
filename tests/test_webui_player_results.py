import json
import tempfile
import unittest
from pathlib import Path

from webui.player_results import _best_detected_observations, build_player_result_display


class PlayerResultDisplayTests(unittest.TestCase):
    def test_full_video_portrait_requires_high_detection_location_and_identity_confidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            detections = Path(temporary) / "detections.jsonl"
            detections.write_text("\n".join([
                json.dumps({"frame": 1, "time_sec": 0.1, "spatial": {"tracks": [
                    {"track_id": "track_low", "status": "detected", "confidence": 0.91, "association": {"identity_confidence": 0.79}, "location_evidence": {"confidence": 0.91, "bbox_xyxy": [1, 1, 90, 180]}},
                    {"track_id": "track_high", "status": "detected", "confidence": 0.91, "association": {"identity_confidence": 0.92}, "location_evidence": {"confidence": 0.89, "bbox_xyxy": [1, 1, 90, 180]}},
                ]}}),
                "",
            ]), encoding="utf-8")
            observations = _best_detected_observations(detections)

        self.assertEqual(list(observations), ["track_high"])

    def test_merges_stream_photo_status_with_real_movement_evidence(self):
        gallery, rows, detail = build_player_result_display(
            {
                "players": [
                    {
                        "track_id": "track_001",
                        "movement": {"distance_m": 12.345, "mean_speed_mps": 1.2, "peak_speed_mps": 3.4},
                        "measurement_coverage": {"usable_measurement_ratio": 0.875},
                        "quality": {"status": "reviewable"},
                    }
                ],
                "limitations": ["visual track only"],
            },
            track_candidates=[
                {
                    "track_id": "track_001",
                    "state": "closed",
                    "confidence": 0.91,
                    "detected_coverage": 0.9,
                    "candidate_photo": {"source_time_sec": 4.2, "view_label": "front"},
                },
                {"track_id": "track_002", "state": "closed", "detected_coverage": 0.2},
            ],
            photo_records=[
                {"track_id": "track_001", "status": "unavailable", "source_time_sec": 4.2},
            ],
        )

        self.assertEqual(gallery, [])
        self.assertEqual([row[0] for row in rows], ["track_001", "track_002"])
        self.assertEqual(rows[0][1], "closed")
        self.assertEqual(rows[0][2], "unavailable")
        self.assertEqual(rows[0][4], 87.5)
        self.assertEqual(rows[0][6], 12.345)
        self.assertEqual(rows[1][4], 20.0)
        self.assertEqual(detail["limitations"], ["visual track only"])

    def test_unconfirmed_roster_candidate_is_visible_but_not_presented_as_player_metrics(self):
        _gallery, rows, _detail = build_player_result_display(
            {"players": []},
            track_candidates=[
                {
                    "track_id": "candidate_bytetrack_12",
                    "state": "unconfirmed_roster",
                    "confidence": 0.88,
                    "analytics_eligible": False,
                }
            ],
        )

        self.assertEqual(rows[0][1], "名单待确认")
        self.assertIsNone(rows[0][4])
        self.assertEqual(rows[0][-1], "名单未确认；不生成速度、距离等正式指标")

    def test_tennis_rows_contain_only_visual_speed_evidence(self):
        _gallery, rows, _detail = build_player_result_display(
            {
                "sport_id": "tennis",
                "players": [{
                    "track_id": "track_001",
                    "movement": {
                        "distance_m": 12.3,
                        "mean_speed_mps": 1.2,
                        "peak_speed_mps": 3.4,
                        "moving_time_sec": 8.0,
                    },
                    "measurement_coverage": {"usable_measurement_ratio": 0.625},
                    "quality": {"status": "measured"},
                }],
            }
        )

        self.assertEqual(len(rows[0]), 10)
        self.assertEqual(rows[0][5:9], [12.3, 1.2, 3.4, 8.0])
        self.assertEqual(rows[0][-1], "measured")


if __name__ == "__main__":
    unittest.main()
