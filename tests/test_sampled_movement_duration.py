"""Regression coverage for sampled detections versus source video time."""
import json
import tempfile
import unittest
from pathlib import Path

from business_gateway.metrics.movement import generate_movement_metrics
from good_badminton_contracts.detections_reader import collect_track_position_evidence


class SampledMovementDurationTests(unittest.TestCase):
    def rows(self, fps):
        rows = []
        for index in range(1, 301):
            seconds = index / 10
            # Ten seconds at 1 m/s, ten stationary, ten at 0.5 m/s.
            distance = min(seconds, 10) + max(0, seconds - 20) * 0.5
            x = 1 + abs(distance % 8 - 4)
            rows.append({
                "frame": round(seconds * fps),
                "time_sec": seconds,
                "spatial": {"tracks": [{
                    "track_id": "track_001", "status": "detected",
                    "court_xy_m": [x, 2], "confidence": 0.9,
                    "location_evidence": {"confidence": 0.9},
                    "association": {"source": "bytetrack", "identity_confidence": 0.9},
                }]},
            })
        return rows

    def derive(self, rows, video):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detections = root / "detections.jsonl"
            detections.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"video": video}))
            result = generate_movement_metrics(root, detections, root / "spatial.json", metadata)
            return result["players"][0], collect_track_position_evidence(detections)

    def test_10hz_sampling_preserves_30_seconds_at_30_and_60fps(self):
        players = []
        for fps in (30, 60):
            with self.subTest(fps=fps):
                player, evidence = self.derive(self.rows(fps), {"fps": fps, "duration_sec": 30})
                effective = player["ability_scores"]["effective_running"]
                self.assertEqual(effective["match_duration_sec"], 30)
                self.assertAlmostEqual(player["match_load"]["duration_factor"], 30 / 1800, places=4)
                self.assertEqual(player["measurement_coverage"]["usable_measurement_ratio"], 1)
                self.assertGreater(player["movement"]["moving_time_sec"], 15)
                self.assertLessEqual(player["movement"]["moving_time_sec"], 30)
                self.assertLess(effective["score"], 80)
                endurance = player["ability_scores"]["endurance"]
                self.assertEqual(endurance["status"], "measured")
                self.assertAlmostEqual(endurance["early_mean_speed_mps"], 1, delta=0.15)
                self.assertAlmostEqual(endurance["late_mean_speed_mps"], 0.5, delta=0.1)
                self.assertEqual(evidence["sampled_frames"], 300)
                self.assertEqual(evidence["max_source_frame"], fps * 30)
                self.assertEqual(evidence["max_time_sec"], 30)
                players.append(player)
        self.assertEqual(players[0], players[1])

    def test_metadata_includes_unsampled_tail_and_takes_precedence(self):
        player, _ = self.derive(self.rows(30), {"fps": 30, "duration_sec": 45, "total_frames": 1350})
        self.assertEqual(player["ability_scores"]["effective_running"]["match_duration_sec"], 45)
        self.assertEqual(player["match_load"]["duration_factor"], 0.025)

    def test_missing_duration_uses_total_video_frames(self):
        player, _ = self.derive(self.rows(60), {"fps": 60, "total_frames": 2700})
        self.assertEqual(player["ability_scores"]["effective_running"]["match_duration_sec"], 45)

    def test_missing_metadata_duration_uses_source_timeline_not_row_count(self):
        for fps in (30, 60):
            with self.subTest(fps=fps):
                player, _ = self.derive(self.rows(fps), {"fps": fps})
                self.assertEqual(player["ability_scores"]["effective_running"]["match_duration_sec"], 30)

    def test_frame_only_fallback_uses_source_frame_numbers(self):
        rows = self.rows(60)
        for row in rows:
            del row["time_sec"]
        player, _ = self.derive(rows, {"fps": 60})
        self.assertEqual(player["ability_scores"]["effective_running"]["match_duration_sec"], 30)

    def test_time_only_fallback_needs_no_fps(self):
        rows = self.rows(30)
        for row in rows:
            del row["frame"]
        player, _ = self.derive(rows, {})
        self.assertEqual(player["ability_scores"]["effective_running"]["match_duration_sec"], 30)

    def test_missing_players_still_extend_timeline_and_reduce_coverage(self):
        rows = self.rows(30)
        for row in rows[100:]:
            row["spatial"]["tracks"] = []
        player, evidence = self.derive(rows, {"fps": 30})
        self.assertEqual(player["ability_scores"]["effective_running"]["match_duration_sec"], 30)
        self.assertEqual(player["measurement_coverage"]["usable_measurement_ratio"], 0.3333)
        self.assertEqual(evidence["sampled_frames"], 300)

    def test_invalid_metadata_duration_falls_back_to_finite_source_time(self):
        for invalid in (float("nan"), float("inf"), -1, 0, "invalid", True):
            with self.subTest(duration=invalid):
                player, _ = self.derive(self.rows(30), {"fps": 30, "duration_sec": invalid})
                self.assertEqual(player["ability_scores"]["effective_running"]["match_duration_sec"], 30)

    def test_sparse_stream_events_use_terminal_session_duration(self):
        from business_gateway.streaming.derivation import derive_stream_movement_metrics

        events = [
            {
                "event_type": "person_observation",
                "source_time_sec": seconds,
                "data": {"track": {
                    **self.rows(10)[0]["spatial"]["tracks"][0],
                    "measurement_bucket": bucket,
                    "court_xy_m": [1 + seconds, 2],
                }},
            }
            for seconds, bucket in ((0, 0), (0.5, 5))
        ]

        class Client:
            def read_events(self, **kwargs):
                return {"events": events, "next_cursor": None}

        with tempfile.TemporaryDirectory() as temporary:
            result = derive_stream_movement_metrics(
                temporary, client=Client(),
                terminal_status={
                    "status": "finalized",
                    "analysis_session_id": "ssn_sampled_duration",
                    "progress": {"processed_source_time_sec": 30},
                },
                create_request={"configuration": {"analysis_sample_hz": 10}},
            )
            metrics = json.loads(Path(result["movement_metrics_path"]).read_text())
        player = metrics["players"][0]
        self.assertEqual(player["ability_scores"]["effective_running"]["match_duration_sec"], 30)
        self.assertEqual(player["movement"]["moving_time_sec"], 0.5)
        self.assertEqual(player["measurement_coverage"]["usable_measurement_ratio"], 1)

    def test_missing_timeline_does_not_invent_duration_from_sample_count(self):
        rows = self.rows(30)
        for row in rows:
            del row["frame"]
            del row["time_sec"]
        player, _ = self.derive(rows, {"fps": 30})
        effective = player["ability_scores"]["effective_running"]
        self.assertEqual(effective["match_duration_sec"], 0)
        self.assertEqual(effective["status"], "insufficient_measurements")


if __name__ == "__main__":
    unittest.main()
