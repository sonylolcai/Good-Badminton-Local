import json
import tempfile
import unittest
from pathlib import Path

from business_gateway.metrics.movement import (
    _acceleration_event_counts,
    _direction_change_events,
    _peak_direction_changes_in_window,
    _speed_statistic_segments,
    _energy_estimate,
    generate_movement_metrics,
    write_body_profiles,
)
from badminton_analysis.analysis.movement_rally_evaluation import evaluate_movement_rally_windows


class MovementMetricsTests(unittest.TestCase):
    def test_metrics_exclude_non_detected_rows_and_estimate_only_after_weight_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            detections = directory / "detections.jsonl"
            rows = []
            for frame, point, status in [
                (1, [1.0, 2.0], "detected"),
                (2, [1.1, 2.0], "detected"),
                (3, [1.2, 2.0], "detected"),
                (4, [1.3, 2.0], "detected"),
                (5, [1.4, 2.0], "detected"),
                (6, [1.5, 2.0], "detected"),
                (7, [9.0, 9.0], "predicted"),
            ]:
                rows.append({
                    "frame": frame,
                    "time_sec": frame / 10,
                    "spatial": {
                        "match": {"mode": "singles"},
                        "tracks": [{
                            "track_id": "track_001",
                            "status": status,
                            "court_xy_m": point,
                            "confidence": 0.9,
                            "location_evidence": {"confidence": 0.9},
                            "association": {"identity_confidence": 0.9},
                        }],
                    },
                })
            detections.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            metadata = directory / "metadata.json"
            metadata.write_text(json.dumps({"video": {"fps": 10, "duration_sec": 0.7}}), encoding="utf-8")
            spatial = directory / "spatial_match_summary.json"
            spatial.write_text(json.dumps({"player_style_inputs": [{"track_id": "track_001", "zone_frames": {"mid_center": 3}}]}), encoding="utf-8")

            initial = generate_movement_metrics(directory, detections, spatial, metadata)
            player = initial["players"][0]
            self.assertEqual(player["movement"]["distance_m"], 0.5)
            self.assertEqual(player["movement"]["accepted_segment_count"], 5)
            self.assertEqual(player["movement"]["speed_statistic_segment_count"], 1)
            self.assertEqual(player["movement"]["mean_speed_mps"], 1.0)
            self.assertEqual(player["energy_estimate"]["status"], "requires_weight")

            profile_path = write_body_profiles(
                directory,
                [{"track_id": "track_001", "weight_kg": 70, "height_cm": 175}],
                consent=True,
            )
            refreshed = generate_movement_metrics(
                directory, detections, spatial, metadata, profile_path,
            )
            energy = refreshed["players"][0]["energy_estimate"]
            self.assertEqual(energy["status"], "estimated")
            self.assertEqual(energy["weight_kg"], 70.0)
            # This fixture contains only 0.5 seconds of usable movement, so
            # the displayed whole-kcal value can validly round down to zero.
            self.assertGreater(energy["estimated_kcal"], 0)
            self.assertIn("estimated_kcal_rounded", energy)

    def test_body_profile_refuses_write_without_consent(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Explicit consent"):
                write_body_profiles(temporary, [{"track_id": "track_001", "weight_kg": 70}], consent=False)

    def test_energy_uses_a_single_intensity_weighted_estimate(self):
        energy = _energy_estimate(
            {"weight_kg": 70, "height_cm": 175},
            active_seconds=120,
            high_intensity_seconds=60,
            coverage_ratio=0.9,
            mean_speed_mps=1.5,
            direction_change_count=60,
        )

        self.assertEqual(energy["met_estimate"], 7.316)
        self.assertIn("estimated_kcal_rounded", energy)
        self.assertNotIn("estimated_kcal_range", energy)
        self.assertEqual(energy["confidence"], "high")

    def test_acceleration_requires_two_metres_inside_half_a_second(self):
        confirmed_sprint = [
            {"series_id": 0, "seconds": 0.5, "distance_m": 0.3, "speed_mps": 1.0},
            {"series_id": 0, "seconds": 0.5, "distance_m": 0.4, "speed_mps": 1.9},
            {"series_id": 0, "seconds": 0.5, "distance_m": 2.0, "speed_mps": 2.8},
            {"series_id": 0, "seconds": 0.5, "distance_m": 0.3, "speed_mps": 2.8},
            {"series_id": 0, "seconds": 0.5, "distance_m": 0.4, "speed_mps": 1.8},
        ]
        short_burst = [
            {"series_id": 0, "seconds": 0.5, "distance_m": 0.2, "speed_mps": 1.0},
            {"series_id": 0, "seconds": 0.5, "distance_m": 0.2, "speed_mps": 1.9},
            {"series_id": 0, "seconds": 0.5, "distance_m": 1.9, "speed_mps": 2.8},
            {"series_id": 0, "seconds": 0.5, "distance_m": 0.2, "speed_mps": 2.8},
        ]

        acceleration_count, deceleration_count = _acceleration_event_counts(confirmed_sprint)

        self.assertEqual(acceleration_count, 1)
        self.assertEqual(deceleration_count, 1)
        self.assertEqual(_acceleration_event_counts(short_burst)[0], 0)

    def test_speed_statistics_use_net_motion_in_half_second_windows(self):
        # The alternating raw steps imitate short localisation jitter.  They
        # must not become five separate 3m/s user-facing speed samples.
        raw_segments = [
            {
                "start_time_sec": index / 10,
                "end_time_sec": (index + 1) / 10,
                "seconds": 0.1,
                "distance_m": 0.3,
                "speed_mps": 3.0,
                "vector": (0.3 if index % 2 == 0 else -0.3, 0.0),
            }
            for index in range(5)
        ]

        samples = _speed_statistic_segments(raw_segments)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["seconds"], 0.5)
        self.assertEqual(samples[0]["speed_mps"], 0.6)

    def test_direction_change_requires_one_metre_on_both_legs_and_has_peak_window(self):
        confirmed_turn_segments = [
            {"seconds": 0.5, "distance_m": 1.0, "speed_mps": 2.0, "vector": (1.0, 0.0), "end_time_sec": 0.5},
            {"seconds": 0.5, "distance_m": 1.0, "speed_mps": 2.0, "vector": (0.0, 1.0), "end_time_sec": 1.0},
        ]
        jitter_only_segments = [
            {"seconds": 0.5, "distance_m": 0.9, "speed_mps": 2.0, "vector": (0.9, 0.0), "end_time_sec": 0.5},
            {"seconds": 0.5, "distance_m": 1.0, "speed_mps": 2.0, "vector": (0.0, 1.0), "end_time_sec": 1.0},
            {"seconds": 0.6, "distance_m": 1.2, "speed_mps": 2.0, "vector": (1.2, 0.0), "end_time_sec": 1.6},
        ]

        events = _direction_change_events(confirmed_turn_segments)

        self.assertEqual(len(events), 1)
        self.assertEqual(_peak_direction_changes_in_window(events, window_seconds=30.0), 1)
        self.assertEqual(_direction_change_events(jitter_only_segments), [])

    def test_person_only_rally_window_sweep_writes_all_three_approved_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            detections = directory / "detections.jsonl"
            records = []
            for frame in range(1, 16):
                moving = frame == 2
                records.append({
                    "frame": frame,
                    "spatial": {"tracks": [
                        {"track_id": "track_001", "status": "detected", "court_xy_m": [1.0 + (0.2 if moving else 0), 2.0]},
                        {"track_id": "track_002", "status": "detected", "court_xy_m": [5.0, 11.0]},
                    ]},
                })
            detections.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
            result = evaluate_movement_rally_windows(
                detections, directory / "derived" / "sweep.json", fps=10, match_mode="singles",
            )
            self.assertEqual(result["windows_seconds"], [0.5, 0.7, 1.0])
            self.assertTrue(Path(result["path"]).is_file())
            self.assertIn("0.5", result["results"])
            self.assertIn("0.7", result["results"])
            self.assertIn("1.0", result["results"])
