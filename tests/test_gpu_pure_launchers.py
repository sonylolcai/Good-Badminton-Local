"""Guard the one-package GPU launch boundary without starting a GPU process."""

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PureGpuLauncherTests(unittest.TestCase):
    def test_fixed_sport_entrypoints_are_not_shipped_as_runtime_sources(self):
        self.assertFalse((REPOSITORY_ROOT / "api" / "gpu_stream_app.py").exists())
        self.assertFalse((REPOSITORY_ROOT / "apps" / "badminton_gpu" / "app.py").exists())
        self.assertFalse((REPOSITORY_ROOT / "apps" / "tennis_gpu" / "app.py").exists())


if __name__ == "__main__":
    unittest.main()
