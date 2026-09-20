import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
REPOSITORIES = {
    "bdteach-venue-gateway": ("agent.py", "README.md"),
    "bdteach-business-api": ("business_gateway", "README.md"),
    "bdteach-clients": ("miniprogram", "operator-web"),
    "Good-Badminton-GPU-Service": ("api", "badminton_analysis", "README.md"),
}


class ServiceRepositoryLayoutTests(unittest.TestCase):
    def test_required_service_repositories_exist_with_minimum_layout(self):
        for repository, required_paths in REPOSITORIES.items():
            root = WORKSPACE / repository
            self.assertTrue((root / ".git").exists(), repository)
            for required_path in required_paths:
                self.assertTrue((root / required_path).exists(), f"{repository}/{required_path}")

    def test_gpu_service_has_no_ui_business_or_evaluation_source_import(self):
        root = WORKSPACE / "Good-Badminton-GPU-Service"
        self.assertTrue(root.is_dir(), root)
        imports = []
        for directory in (root / "api", root / "badminton_analysis"):
            for path in directory.rglob("*.py"):
                text = path.read_text(encoding="utf-8")
                for forbidden in ("from webui", "import webui", "from business_gateway", "import business_gateway", "evaluation_service"):
                    if forbidden in text:
                        imports.append(f"{path.relative_to(root)}: {forbidden}")
        self.assertEqual([], imports)

    def test_venue_gateway_does_not_connect_to_gpu(self):
        root = WORKSPACE / "bdteach-venue-gateway"
        self.assertTrue(root.is_dir(), root)
        forbidden = ("GPU_ANALYSIS_BASE_URL", "GOOD_BADMINTON_GPU_API_URL", "StreamSessionClient")
        matches = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            matches.extend(f"{path.relative_to(root)}: {needle}" for needle in forbidden if needle in text)
        self.assertEqual([], matches)

    def test_clients_do_not_contain_gpu_rtsp_or_database_configuration(self):
        root = WORKSPACE / "bdteach-clients"
        self.assertTrue(root.is_dir(), root)
        forbidden = ("GPU_ANALYSIS_API_KEY", "GOOD_BADMINTON_GPU_API_KEY", "CAMERA_RTSP_URL", "POSTGRES_PASSWORD")
        matches = []
        for path in root.rglob("*"):
            if path.is_file() and ".git" not in path.parts and "node_modules" not in path.parts:
                text = path.read_text(encoding="utf-8", errors="ignore")
                matches.extend(f"{path.relative_to(root)}: {needle}" for needle in forbidden if needle in text)
        self.assertEqual([], matches)


if __name__ == "__main__":
    unittest.main()
