import json
import tempfile
import unittest
from pathlib import Path

from webui.stream_speed_summary import summarize_stream_player_speeds


class _EventClient:
    def __init__(self, events):
        self.events = events

    def read_events(self, *, cursor=None, limit=500):
        self.assert_cursor(cursor)
        return {"events": self.events, "next_cursor": None}

    @staticmethod
    def assert_cursor(cursor):
        if cursor is not None:
            raise AssertionError("unexpected extra event page")


class StreamSpeedSummaryTests(unittest.TestCase):
    def test_tennis_speed_uses_only_detected_contiguous_court_positions(self):
        events = [
            self._event(0.0, "detected", [1.0, 10.0], bucket=0),
            self._event(0.25, "detected", [1.5, 10.0], bucket=1),
            self._event(0.5, "detected", [2.0, 10.0], bucket=2),
            self._event(0.75, "predicted", [2.5, 10.0], bucket=3),
            self._event(1.0, "detected", [3.0, 10.0], bucket=4),
            # A one-second gap may not be bridged into a speed measurement.
            self._event(2.0, "detected", [4.0, 10.0], bucket=8),
            {
                "event_type": "ball_observation",
                "evidence_state": "detected",
                "data": {
                    "detector_mode": "experimental_badminton_yolo",
                    "experimental": True,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_stream_player_speeds(
                directory,
                client=_EventClient(events),
                terminal_status={"analysis_session_id": "ssn_tennis"},
                create_request={
                    "configuration": {"sport_id": "tennis", "session_mode": "singles_match"}
                },
            )
            payload = json.loads(Path(result["movement_metrics_path"]).read_text(encoding="utf-8"))

        player = payload["players"][0]
        self.assertEqual(payload["sport_id"], "tennis")
        self.assertEqual(player["track_id"], "track_001")
        self.assertAlmostEqual(player["movement"]["distance_m"], 1.0, places=4)
        self.assertAlmostEqual(player["movement"]["mean_speed_mps"], 2.0, places=4)
        self.assertAlmostEqual(player["movement"]["peak_speed_mps"], 2.0, places=4)
        self.assertEqual(player["measurement_coverage"]["accepted_segment_count"], 2)
        # The predicted sample ends one detected series and the later one-second
        # gap is independently rejected, so neither gap becomes inferred motion.
        self.assertEqual(player["measurement_coverage"]["excluded_segment_count"], 2)
        self.assertEqual(player["quality"]["status"], "measured")
        self.assertEqual(payload["ball_detection"]["detected_event_count"], 1)
        self.assertTrue(payload["ball_detection"]["experimental"])
        self.assertEqual(
            payload["ball_detection"]["accuracy_status"],
            "requires_same_video_ground_truth",
        )

    @staticmethod
    def _event(source_time, status, point, *, bucket):
        return {
            "event_type": "person_observation",
            "source_time_sec": source_time,
            "data": {
                "track": {
                    "track_id": "track_001",
                    "status": status,
                    "court_xy_m": point,
                    "measurement_bucket": bucket,
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
