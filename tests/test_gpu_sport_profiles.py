"""GPU-only sport/mode tests; no tennis ball model or WebUI is involved."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import _configured_vision_profile, create_app
from api.gpu_stream_app import create_gpu_stream_app
from api.mode_sync import VisionModeSynchronizer
from api.stream_runtime import StreamProcessorFactory
from api.vision_profiles import (
    FULL_COURT,
    NEAR_HALF_COURT,
    TENNIS_PROFILE,
)
from badminton_analysis.analysis.fixed_camera_match import CourtSpace
from badminton_analysis.tracking.person_only import PersonOnlyTracker
from tests.stream_test_utils import create_request


class GpuSportProfileTests(unittest.TestCase):
    def setUp(self):
        self.tennis_modes = VisionModeSynchronizer(TENNIS_PROFILE)

    @staticmethod
    def configuration(**overrides):
        return {
            "analysis_sample_hz": 10,
            "pose_imgsz": 960,
            "shuttle_detector": "none",
            "generate_annotated_video": False,
            **overrides,
        }

    def test_tennis_requires_a_mode_and_derives_fixed_singles_roster(self):
        with self.assertRaisesRegex(ValueError, "session_mode is required"):
            self.tennis_modes.synchronize(self.configuration())

        resolved = self.tennis_modes.synchronize(
            self.configuration(session_mode="singles_match", calibration_scope=FULL_COURT)
        )

        self.assertEqual(resolved["sport_id"], "tennis")
        self.assertEqual(resolved["expected_player_count"], 2)
        self.assertEqual(resolved["max_roster_count"], 2)
        self.assertEqual(resolved["calibration_scope"], FULL_COURT)
        self.assertEqual(
            resolved["calibration_world_points_m"],
            [[0.0, 0.0], [8.23, 0.0], [8.23, 23.77], [0.0, 23.77]],
        )

    def test_training_derives_one_near_athlete_and_near_half_world_points(self):
        resolved = self.tennis_modes.synchronize(
            self.configuration(session_mode="single_player_training")
        )

        self.assertEqual(resolved["expected_player_count"], 1)
        self.assertEqual(resolved["max_roster_count"], 1)
        self.assertEqual(resolved["calibration_scope"], NEAR_HALF_COURT)
        self.assertEqual(resolved["athlete_observation_region"], "near_court_athlete")
        self.assertEqual(
            resolved["calibration_world_points_m"],
            [[0.0, 11.885], [8.23, 11.885], [8.23, 23.77], [0.0, 23.77]],
        )

    def test_tennis_rejects_wrong_sport_mode_roster_and_ball_detector(self):
        cases = (
            (self.configuration(sport_id="badminton", session_mode="singles_match"), "does not match"),
            (self.configuration(session_mode="doubles"), "session_mode must be one of"),
            (
                self.configuration(
                    session_mode="single_player_training", expected_player_count=2
                ),
                "conflicts",
            ),
            (self.configuration(session_mode="singles_match", shuttle_detector="yolo"), "must be one of"),
        )
        for configuration, message in cases:
            with self.subTest(configuration=configuration):
                with self.assertRaisesRegex(ValueError, message):
                    self.tennis_modes.synchronize(configuration)

    def test_near_half_maps_to_global_near_side_and_excludes_far_people(self):
        resolved = self.tennis_modes.synchronize(
            self.configuration(session_mode="single_player_training")
        )
        court = CourtSpace(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            court_dimensions=resolved["court_dimensions_m"],
            world_points_m=resolved["calibration_world_points_m"],
            athlete_observation_region=resolved["athlete_observation_region"],
        )

        self.assertAlmostEqual(court.image_to_court((0, 0))[0], 0.0, places=4)
        self.assertAlmostEqual(court.image_to_court((0, 0))[1], 11.88, places=4)
        self.assertAlmostEqual(court.image_to_court((100, 100))[0], 8.23, places=4)
        self.assertAlmostEqual(court.image_to_court((100, 100))[1], 23.77, places=4)
        self.assertTrue(court.contains_athlete((4.0, 18.0), margin_m=0.35))
        self.assertFalse(court.contains_athlete((4.0, 5.0), margin_m=0.35))

    def test_training_tracker_locks_one_person_and_checkpoint_rejects_other_mode(self):
        resolved = self.tennis_modes.synchronize(
            self.configuration(session_mode="single_player_training")
        )
        tracker = PersonOnlyTracker(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            fps=10,
            lock_match_roster=True,
            expected_roster_count=resolved["expected_player_count"],
            max_roster_count=resolved["max_roster_count"],
            roster_stable_frames=1,
            roster_discovery_seconds=resolved["roster_discovery_seconds"],
            court_dimensions_m=resolved["court_dimensions_m"],
            calibration_world_points_m=resolved["calibration_world_points_m"],
            athlete_observation_region=resolved["athlete_observation_region"],
            sport_id=resolved["sport_id"],
            session_mode=resolved["session_mode"],
            calibration_scope=resolved["calibration_scope"],
            coordinate_system_id=TENNIS_PROFILE.coordinate_system_id,
        )
        observation = {"court_xy": [4.0, 18.0], "image_xy": [50.0, 50.0], "confidence": 0.9}
        far_person = {"court_xy": [4.0, 5.0], "image_xy": [50.0, 10.0], "confidence": 0.9}
        tracker.update(1, [observation, far_person])
        locked = tracker.update(2, [observation, far_person])

        self.assertEqual(locked["tracking"]["roster"]["status"], "locked")
        self.assertEqual(locked["tracking"]["expected_player_count"], 1)
        self.assertEqual(len(locked["tracks"]), 1)
        state = tracker.snapshot_state()
        incompatible = PersonOnlyTracker(
            [(0, 0), (100, 0), (100, 100), (0, 100)],
            fps=10,
            lock_match_roster=True,
            expected_roster_count=2,
            max_roster_count=2,
            roster_stable_frames=1,
            court_dimensions_m=(8.23, 23.77),
            calibration_world_points_m=[[0, 0], [8.23, 0], [8.23, 23.77], [0, 23.77]],
            sport_id="tennis",
            session_mode="singles_match",
            coordinate_system_id=TENNIS_PROFILE.coordinate_system_id,
        )
        with self.assertRaisesRegex(ValueError, "vision profile does not match"):
            incompatible.restore_state(state)

    def test_tennis_app_health_and_runtime_are_profile_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(
                data_dir=Path(directory),
                start_worker=False,
                vision_profile=TENNIS_PROFILE,
            )
            response = TestClient(app).get("/api/v1/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["sport_id"], "tennis")
            self.assertEqual(
                response.json()["supported_session_modes"],
                ["singles_match", "single_player_training"],
            )

            request = create_request()
            request["configuration"].update(
                {"sport_id": "tennis", "session_mode": "single_player_training"}
            )
            factory = StreamProcessorFactory(Path(directory), vision_profile=TENNIS_PROFILE)
            factory.validate_session_request(request)
            self.assertEqual(request["configuration"]["expected_player_count"], 1)
            self.assertEqual(request["configuration"]["shuttle_detector"], "none")

    def test_pure_gpu_app_exposes_stream_contract_without_job_or_business_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"GOOD_BADMINTON_API_KEY": "test-key"}, clear=False
        ):
            app = create_gpu_stream_app(
                data_dir=Path(directory),
                start_worker=False,
                vision_profile=TENNIS_PROFILE,
            )
            client = TestClient(app)

            health = client.get("/api/v1/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["sport_id"], "tennis")
            self.assertEqual(health.json()["service_kind"], "pure_gpu_visual_observation")
            self.assertFalse(hasattr(app.state, "job_manager"))

            # A pure GPU deployment has no endpoint that can run the legacy
            # whole-video pipeline (hit/rally/report/rendering derivation).
            response = client.post("/api/v1/jobs", headers={"X-API-Key": "test-key"})
            self.assertEqual(response.status_code, 404)

    def test_process_profile_is_selected_only_at_startup(self):
        with patch.dict(
            "os.environ", {"GOOD_SPORT_VISION_PROFILE": "tennis"}, clear=False
        ):
            self.assertEqual(_configured_vision_profile().sport_id, "tennis")

    def test_tennis_session_persists_profile_derived_training_configuration(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"GOOD_BADMINTON_API_KEY": "test-key"}, clear=False
        ):
            app = create_app(
                data_dir=Path(directory),
                start_worker=False,
                vision_profile=TENNIS_PROFILE,
            )
            request = create_request()
            request["configuration"].update(
                {"sport_id": "tennis", "session_mode": "single_player_training"}
            )
            response = TestClient(app).post(
                "/api/v1/stream-sessions",
                headers={"X-API-Key": "test-key"},
                json=request,
            )

            self.assertEqual(response.status_code, 202)
            session = app.state.stream_manager.get_session(
                response.json()["analysis_session_id"]
            )
            self.assertEqual(session["configuration"]["expected_player_count"], 1)
            self.assertEqual(session["configuration"]["calibration_scope"], NEAR_HALF_COURT)

    def test_tennis_endpoint_rejects_a_wrong_sport_or_conflicting_roster(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"GOOD_BADMINTON_API_KEY": "test-key"}, clear=False
        ):
            app = create_app(
                data_dir=Path(directory),
                start_worker=False,
                vision_profile=TENNIS_PROFILE,
            )
            client = TestClient(app)
            cases = (
                ({"sport_id": "badminton", "session_mode": "singles_match"}, "does not match"),
                (
                    {
                        "sport_id": "tennis",
                        "session_mode": "single_player_training",
                        "expected_player_count": 2,
                    },
                    "conflicts",
                ),
            )
            for configuration, message in cases:
                request = create_request()
                request["configuration"].update(configuration)
                with self.subTest(configuration=configuration):
                    response = client.post(
                        "/api/v1/stream-sessions",
                        headers={"X-API-Key": "test-key"},
                        json=request,
                    )
                    self.assertEqual(response.status_code, 422)
                    self.assertIn(message, response.json()["error"]["message"])


if __name__ == "__main__":
    unittest.main()
