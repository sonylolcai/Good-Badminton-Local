import unittest
from pathlib import Path

from business_gateway.streaming.continuity_replay_cli import (
    build_business_session_manifest,
    build_gpu_create_request,
)


class ContinuityReplayCliTests(unittest.TestCase):
    def test_business_ids_are_local_but_gpu_request_is_anonymous(self):
        request = build_gpu_create_request(
            camera_id="camera_test_01",
            calibration_id="calibration_test_01",
            court_corners=[[0, 0], [100, 0], [100, 100], [0, 100]],
            opaque_client_reference="continuity_test_opaque",
            analysis_sample_hz=10,
            pose_imgsz=960,
            shuttle_detector="yolo",
            tracker_backend="court_association",
        )
        manifest = build_business_session_manifest(
            venue_id="venue_test_01",
            court_id="court_test_01",
            match_id="match_test_01",
            camera_id="camera_test_01",
            calibration_id="calibration_test_01",
            video_path=Path("S:/example/match.mp4"),
            video_sha256="a" * 64,
            opaque_client_reference="continuity_test_opaque",
            create_idempotency_key="continuity_test_0001",
            segment_seconds=2.0,
        )

        self.assertEqual(manifest["business_identity"]["court_id"], "court_test_01")
        self.assertEqual(manifest["business_identity"]["match_id"], "match_test_01")
        self.assertEqual(manifest["transport"]["gpu_analysis_session_id"], None)
        self.assertEqual(manifest["stream_order_contract"]["segment_index_start"], 0)
        self.assertTrue(manifest["stream_order_contract"]["requires_contiguous_indexes"])
        self.assertNotIn("venue_id", request)
        self.assertNotIn("court_id", request)
        self.assertNotIn("match_id", request)
        self.assertEqual(request["client_reference"], "continuity_test_opaque")


if __name__ == "__main__":
    unittest.main()
