import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2

from business_gateway.streaming.segmenter import GrowingVideoSegmenter, _find_ffmpeg


class GrowingVideoSegmenterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.ffmpeg = _find_ffmpeg()
        if not self.ffmpeg:
            self.skipTest("FFmpeg is unavailable")
        self.video = self.root / "source.mp4"
        command = [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=4.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "30",
            "-keyint_min",
            "30",
            "-sc_threshold",
            "0",
            str(self.video),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            self.skipTest(f"test FFmpeg cannot encode H.264: {completed.stderr[-300:]}")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_copy_mode_outputs_independently_decodable_continuous_segments(self):
        output = self.root / "segments"
        segmenter = GrowingVideoSegmenter(
            output,
            segment_duration_sec=1.0,
            encoding_mode="copy",
            ffmpeg_path=self.ffmpeg,
            poll_interval_sec=0.01,
        )

        artifacts = segmenter.segment_file(self.video)

        self.assertGreaterEqual(len(artifacts), 4)
        previous_end = 0.0
        for index, artifact in enumerate(artifacts):
            self.assertEqual(artifact.segment_index, index)
            self.assertAlmostEqual(artifact.source_start_time_sec, previous_end, places=4)
            self.assertGreater(artifact.duration_sec, 0)
            self.assertLessEqual(artifact.duration_sec, 10)
            capture = cv2.VideoCapture(str(artifact.path))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            previous_end += artifact.duration_sec

        self.assertGreater(previous_end, 4.0)
        self.assertTrue((output / "segment_manifest.json").is_file())

        # A completed manifest is replayable without invoking FFmpeg again.
        reused = segmenter.segment_file(self.video)
        self.assertEqual(artifacts, reused)

        incompatible = GrowingVideoSegmenter(
            output,
            segment_duration_sec=2.0,
            encoding_mode="copy",
            ffmpeg_path=self.ffmpeg,
        )
        with self.assertRaisesRegex(RuntimeError, "different source or configuration"):
            incompatible.segment_file(self.video)

    def test_refuses_an_unfinished_output_directory_without_explicit_overwrite(self):
        output = self.root / "unfinished"
        output.mkdir()
        (output / "segment_000000.mp4").write_bytes(b"partial")
        segmenter = GrowingVideoSegmenter(output, ffmpeg_path=self.ffmpeg)
        with self.assertRaisesRegex(RuntimeError, "unfinished prior run"):
            segmenter.segment_file(self.video)


if __name__ == "__main__":
    unittest.main()
