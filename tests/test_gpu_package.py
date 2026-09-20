"""Black-box checks for the single multi-sport GPU source package."""

import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = "good-badminton-gpu-api/"
RELEASE_WEIGHT_FILES = (
    "yolo11n-pose.pt",
    "yolo11s-ball.pt",
    "tennis-ball.pt",
)


class GpuPackageTests(unittest.TestCase):
    def test_packaging_produces_one_linux_safe_gpu_only_archive(self):
        script = REPOSITORY_ROOT / "deploy" / "package_gpu_api.ps1"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "good-badminton-gpu-api-upload.zip"
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-OutputPath",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            self.assertTrue(output.is_file())
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()

        self.assertTrue(names)
        self.assertTrue(all(name.startswith(PACKAGE_ROOT) for name in names))
        self.assertTrue(all("\\" not in name for name in names))
        self.assertIn(f"{PACKAGE_ROOT}api/app.py", names)
        self.assertIn(f"{PACKAGE_ROOT}deploy/install_gpu_api.sh", names)
        self.assertIn(f"{PACKAGE_ROOT}deploy/good-badminton-gpu-api.service", names)
        self.assertIn(f"{PACKAGE_ROOT}deploy/start_gpu_api_container.sh", names)
        forbidden_prefixes = (
            "evaluation/",
            "business_gateway/",
            "webui/",
            "tests/",
            "weights/",
            "api_data/",
        )
        relative_names = [name.removeprefix(PACKAGE_ROOT) for name in names]
        self.assertFalse(
            [
                name
                for name in relative_names
                if name.startswith(forbidden_prefixes)
                or name.endswith(".env")
                or name.endswith(".env.local")
            ],
            "archive contains a non-GPU runtime path",
        )

    def test_complete_release_contains_only_required_models_with_hash_manifest(self):
        script = REPOSITORY_ROOT / "deploy" / "package_gpu_api.ps1"
        missing = [name for name in RELEASE_WEIGHT_FILES if not (REPOSITORY_ROOT / "weights" / name).is_file()]
        if missing:
            self.skipTest(f"complete-release model inputs are not present: {', '.join(missing)}")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "good-badminton-gpu-api-full-release.zip"
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-IncludeWeights",
                    "-OutputPath",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                relative_names = [name.removeprefix(PACKAGE_ROOT) for name in names]
                manifest = json.loads(archive.read(f"{PACKAGE_ROOT}weights/manifest.json"))
                weight_entries = {
                    entry["path"]: entry
                    for entry in manifest["weights"]
                }
                packaged_weight_files = {
                    name.removeprefix("weights/")
                    for name in relative_names
                    if name.startswith("weights/") and name != "weights/manifest.json"
                }

                self.assertEqual(manifest["schema_version"], "complete-model-release.v1")
                self.assertEqual(packaged_weight_files, set(RELEASE_WEIGHT_FILES))
                self.assertEqual(set(weight_entries), set(RELEASE_WEIGHT_FILES))
                for filename in RELEASE_WEIGHT_FILES:
                    payload = archive.read(f"{PACKAGE_ROOT}weights/{filename}")
                    entry = weight_entries[filename]
                    self.assertEqual(entry["bytes"], len(payload))
                    self.assertEqual(
                        entry["sha256"],
                        hashlib.sha256(payload).hexdigest().upper(),
                    )

    def test_package_and_launch_scripts_have_no_fixed_sport_dispatch(self):
        package_script = (REPOSITORY_ROOT / "deploy" / "package_gpu_api.ps1").read_text(encoding="utf-8")
        launcher = (REPOSITORY_ROOT / "deploy" / "start_gpu_api_container.sh").read_text(encoding="utf-8")
        refresh = (REPOSITORY_ROOT / "deploy" / "refresh_gpu_api_from_zip.sh").read_text(encoding="utf-8")
        installer = (REPOSITORY_ROOT / "deploy" / "install_gpu_api.sh").read_text(encoding="utf-8")
        service = (REPOSITORY_ROOT / "deploy" / "good-badminton-gpu-api.service").read_text(encoding="utf-8")

        self.assertNotIn("[string]$Sport", package_script)
        self.assertNotIn("start_${SPORT_ID}", package_script)
        self.assertIn("api.app:app", launcher)
        self.assertIn('"$APP_DIR/deploy/start_gpu_api_container.sh"', refresh)
        self.assertIn("api.app:app", installer)
        self.assertIn("api.app:app", service)
        self.assertIn("$STATE_DIR/.venv/bin/python", refresh)
        self.assertIn("$STATE_DIR/.venv", installer)
        self.assertNotIn("fixed-camera-singles-spatial-tracking", installer)
        self.assertIn("Extract the uploaded complete GPU package", installer)
        self.assertNotIn("git clone", installer)
        self.assertNotIn("GOOD_BADMINTON_SKIP_GIT_SYNC", installer)
        self.assertIn("EnvironmentFile=__ENV_FILE__", service)
        self.assertIn('PREVIOUS_APP_DIR="$STATE_DIR/previous-app"', refresh)
        self.assertIn("restore_previous_api", refresh)
        self.assertIn("yolo11n-pose.pt", refresh)
        self.assertIn("supported_sport_ids", refresh)
        self.assertIn("complete-model-release.v1", package_script)
        self.assertIn("complete-model-release.v1", refresh)
        self.assertIn('FULL_RELEASE_ARCHIVE_PATH="/root/${DEPLOY_NAME}-full-release.zip"', refresh)
        self.assertIn('PREVIOUS_WEIGHTS_DIR="$STATE_DIR/previous-weights"', refresh)
        self.assertIn("restore_previous_weights", refresh)
        self.assertIn("tennis-ball.pt", refresh)
        for text in (package_script, launcher, refresh, installer, service):
            self.assertNotIn("apps.badminton_gpu.app:app", text)
            self.assertNotIn("apps.tennis_gpu.app:app", text)
            self.assertNotIn("start_${SPORT_ID}", text)
        for legacy_launcher in (
            "start_sport_gpu_container.sh",
            "start_badminton_gpu_container.sh",
            "start_tennis_gpu_container.sh",
        ):
            self.assertFalse((REPOSITORY_ROOT / "deploy" / legacy_launcher).exists())


if __name__ == "__main__":
    unittest.main()
