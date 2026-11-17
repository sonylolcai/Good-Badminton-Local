from pathlib import Path
import os
import tempfile
import unittest

from operator_api.services.operator_backoffice import _write_env_file


class PrivateConfigTests(unittest.TestCase):
    def test_operator_save_preserves_private_target_and_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".secrets").mkdir(mode=0o700)
            link = root / ".webui-remote-gpu.env"
            link.symlink_to(".secrets/gpu.env")
            _write_env_file(link, {"GOOD_BADMINTON_GPU_API_KEY": "fixture-only"})
            self.assertTrue(link.is_symlink())
            target = root / ".secrets/gpu.env"
            self.assertIn("fixture-only", target.read_text())
            if os.name == "posix":
                self.assertEqual(target.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
