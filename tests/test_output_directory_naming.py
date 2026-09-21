import unittest
from pathlib import Path

from webui.pipeline import _default_analysis_output_dir


class OutputDirectoryNamingTests(unittest.TestCase):
    def test_local_analysis_output_starts_with_timestamp(self):
        output_dir = _default_analysis_output_dir(
            r"C:\videos\match:demo?.mp4",
            "20260815_010203",
        )

        self.assertEqual(Path(output_dir).name, "20260815_010203_webui_match_demo")


if __name__ == "__main__":
    unittest.main()
