"""Regression coverage for in-page movement-metric presentation."""

import unittest

from webui.app import _movement_metric_summary_rows


class WebUIMovementDisplayTests(unittest.TestCase):
    def test_summary_rows_keep_metrics_energy_and_coverage_visible(self):
        rows = _movement_metric_summary_rows({
            "players": [{
                "track_id": "track_001",
                "movement": {
                    "distance_m": 12.3,
                    "mean_speed_mps": 1.2,
                    "peak_speed_mps": 3.4,
                    "moving_time_sec": 8.0,
                    "high_intensity_movement_time_sec": 2.0,
                    "acceleration_event_count": 4,
                    "deceleration_event_count": 5,
                    "direction_change_count": 6,
                    "peak_direction_changes_30s": 3,
                },
                "measurement_coverage": {"usable_measurement_ratio": 0.625},
                "energy_estimate": {
                    "estimated_kcal_rounded": 4,
                },
                "quality": {"status": "reviewable"},
            }],
        })

        self.assertEqual(rows, [[
            "track_001", 12.3, 1.2, 3.4, 8.0, 2.0, 4, 5, "6 / 3",
            62.5, "4", "reviewable",
        ]])

    def test_summary_marks_energy_as_pending_without_a_weight(self):
        rows = _movement_metric_summary_rows({
            "players": [{"track_id": "track_001", "movement": {}}],
        })
        self.assertEqual(rows[0][10], "待填写体重")


if __name__ == "__main__":
    unittest.main()
