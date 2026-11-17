import json
import tempfile
import unittest
from pathlib import Path

from business_gateway.streaming.derivation import derive_stream_movement_metrics


class _PagedEventClient:
    def __init__(self, events):
        self.events = events

    def read_events(self, *, cursor=None, limit=500):
        if cursor is None:
            return {"events": self.events[:2], "next_cursor": "page-two"}
        if cursor == "page-two":
            return {"events": self.events[2:], "next_cursor": None}
        raise AssertionError(f"unexpected cursor {cursor}")


def _person_event(time_sec, bucket, point):
    return {
        "event_type": "person_observation",
        "source_time_sec": time_sec,
        "data": {
            "track": {
                "track_id": "track_001",
                "status": "detected",
                "confidence": 0.95,
                "court_xy_m": point,
                "zone_id": "mid_center",
                "measurement_bucket": bucket,
                "location_evidence": {"confidence": 0.9},
                "association": {"source": "bytetrack", "identity_confidence": 0.95},
            }
        },
    }


class StreamMovementDerivationTests(unittest.TestCase):
    def test_terminal_stream_events_materialize_metrics_without_video_reanalysis(self):
        events = [
            {"event_type": "session_status", "source_time_sec": 0.0, "data": {}},
            _person_event(0.0, 0, [1.0, 2.0]),
            _person_event(0.5, 5, [1.5, 2.0]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = derive_stream_movement_metrics(
                temporary,
                client=_PagedEventClient(events),
                terminal_status={
                    "status": "finalized",
                    "analysis_session_id": "ssn_stream_derivation",
                    "progress": {"processed_source_time_sec": 0.5},
                },
                create_request={"configuration": {"analysis_sample_hz": 10}},
            )
            metrics_path = Path(result["movement_metrics_path"])
            self.assertTrue(metrics_path.is_file())
            self.assertEqual(result["event_count"], 3)
            self.assertEqual(result["person_observation_count"], 2)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["players"][0]["track_id"], "track_001")
            self.assertEqual(metrics["players"][0]["movement"]["distance_m"], 0.5)
            self.assertTrue(Path(result["raw_events_path"]).is_file())
            self.assertTrue(Path(result["materialized_detections_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
