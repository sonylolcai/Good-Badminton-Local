import unittest

from webui.player_results import build_player_result_display


class PlayerResultDisplayTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
