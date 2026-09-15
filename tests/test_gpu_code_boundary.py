import unittest
from pathlib import Path


class GpuCodeBoundaryTests(unittest.TestCase):
    def test_gpu_service_has_no_evaluation_or_ui_dependency(self):
        root = Path(__file__).parents[1]
        self.assertFalse((root / "evaluation").exists())
        self.assertFalse((root / "analysis_platform").exists())
        self.assertEqual([], list((root / "badminton_analysis" / "streaming_validation").glob("*.py")))
        imports = []
        for directory in (root / "api", root / "badminton_analysis"):
            for path in directory.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for forbidden in (
                    "from webui", "import webui", "from analysis_platform", "import analysis_platform",
                    "from business_gateway", "import business_gateway",
                ):
                    if forbidden in text:
                        imports.append(f"{path.relative_to(root)}: {forbidden}")
        self.assertEqual([], imports)


if __name__ == "__main__":
    unittest.main()
