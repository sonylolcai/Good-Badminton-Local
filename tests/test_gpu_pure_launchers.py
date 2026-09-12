"""Guard the fixed-sport launch boundary without starting a GPU process."""

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PureGpuLauncherTests(unittest.TestCase):
    def test_sport_wrappers_can_start_only_the_matching_pure_entrypoint(self):
        deploy = REPOSITORY_ROOT / "deploy"
        common = (deploy / "start_sport_gpu_container.sh").read_text(encoding="utf-8")
        badminton = (deploy / "start_badminton_gpu_container.sh").read_text(encoding="utf-8")
        tennis = (deploy / "start_tennis_gpu_container.sh").read_text(encoding="utf-8")

        self.assertIn("apps.badminton_gpu.app:app", common)
        self.assertIn("apps.tennis_gpu.app:app", common)
        self.assertNotIn("api.app:app", common)
        self.assertIn('"apps.badminton_gpu.app:app" "badminton"', badminton)
        self.assertIn('"apps.tennis_gpu.app:app" "tennis"', tennis)

    def test_sport_entries_do_not_import_the_legacy_whole_video_api(self):
        apps = REPOSITORY_ROOT / "apps"
        for path in (apps / "badminton_gpu" / "app.py", apps / "tennis_gpu" / "app.py"):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("from api.gpu_stream_app import create_gpu_stream_app", source)
                self.assertNotIn("from api.app import app", source)


if __name__ == "__main__":
    unittest.main()
