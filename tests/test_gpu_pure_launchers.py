"""Guard the fixed-sport launch boundary without starting a GPU process."""

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PureGpuLauncherTests(unittest.TestCase):
    def test_sport_wrappers_can_start_only_the_matching_fixed_entrypoint(self):
        deploy = REPOSITORY_ROOT / "deploy"
        common = (deploy / "start_sport_gpu_container.sh").read_text(encoding="utf-8")
        badminton = (deploy / "start_badminton_gpu_container.sh").read_text(encoding="utf-8")
        tennis = (deploy / "start_tennis_gpu_container.sh").read_text(encoding="utf-8")

        self.assertIn("apps.badminton_gpu.app:app", common)
        self.assertIn("apps.tennis_gpu.app:app", common)
        self.assertNotIn("api.app:app", common)
        self.assertIn('POSE_MODEL_VAR="${SPORT_ENV_PREFIX}_POSE_MODEL"', common)
        self.assertIn('Configured ${POSE_MODEL_VAR} does not exist', common)
        self.assertIn('"apps.badminton_gpu.app:app" "badminton"', badminton)
        self.assertIn('"apps.tennis_gpu.app:app" "tennis"', tennis)

    def test_sport_entries_compose_full_video_and_stream_routes_with_fixed_profiles(self):
        apps = REPOSITORY_ROOT / "apps"
        for path in (apps / "badminton_gpu" / "app.py", apps / "tennis_gpu" / "app.py"):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("from api.app import create_app", source)
                self.assertIn("create_app(vision_profile=", source)
                self.assertNotIn("from api.app import app", source)

    def test_pure_stream_only_mode_unmounts_whole_video_endpoints(self):
        import os
        import tempfile
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from api.app import create_app

        deploy = REPOSITORY_ROOT / "deploy"
        self.assertIn("GOOD_BADMINTON_GPU_PURE_STREAM_ONLY", (deploy / "start_sport_gpu_container.sh").read_text(encoding="utf-8"))
        self.assertIn("GOOD_BADMINTON_GPU_PURE_STREAM_ONLY=1", (deploy / "good-badminton-gpu-api.service").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"GOOD_BADMINTON_GPU_PURE_STREAM_ONLY": "1", "GOOD_BADMINTON_API_KEY": "test-key"}):
            app = create_app(data_dir=tmpdir, start_worker=False)
            client = TestClient(app)
            health = client.get("/api/v1/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json().get("service_kind"), "pure_gpu_visual_observation")
            self.assertEqual(client.post("/api/v1/jobs", headers={"X-API-Key": "test-key"}).status_code, 404)
            self.assertEqual(client.post("/api/v1/court/detect", headers={"X-API-Key": "test-key"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
