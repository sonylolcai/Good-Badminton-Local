import unittest

from business_gateway.visual_observation import from_stream_event


class VisualObservationAdapterTests(unittest.TestCase):
    def test_normalizes_badminton_person_without_requiring_pose(self):
        result = from_stream_event({
            "event_type": "person_observation",
            "source_time_sec": 1.25,
            "evidence_state": "detected",
            "confidence": 0.91,
            "data": {
                "track": {
                    "track_id": "track_001",
                    "source_frame_index": 38,
                    "image_xy": [320, 540],
                    "court_xy_m": [3.1, 11.2],
                    "court_end": "near",
                    "location_evidence": {"bbox_xyxy": [280, 200, 360, 540]},
                    "pose": {"is_current_measurement": False},
                }
            },
        })

        self.assertEqual(result["schema_version"], "visual-observation.v1")
        self.assertEqual(result["sport_id"], "badminton")
        self.assertEqual(result["source_frame_index"], 38)
        self.assertEqual(result["data"]["track_id"], "track_001")
        self.assertEqual(result["data"]["pose"]["evidence_state"], "missing")

    def test_normalizes_badminton_ball_and_keeps_missing_court_projection(self):
        result = from_stream_event({
            "event_type": "shuttle_observation",
            "source_time_sec": 2.0,
            "evidence_state": "detected",
            "confidence": 0.72,
            "data": {
                "source_frame_index": 60,
                "measurement": {"image": [410, 220]},
            },
        })

        self.assertEqual(result["observation_type"], "ball")
        self.assertEqual(result["data"]["kind"], "shuttlecock")
        self.assertEqual(result["data"]["image_xy"], [410.0, 220.0])
        self.assertIsNone(result["data"]["court_xy_m"])

    def test_ignores_transport_events(self):
        self.assertIsNone(from_stream_event({"event_type": "session_status"}))


if __name__ == "__main__":
    unittest.main()
