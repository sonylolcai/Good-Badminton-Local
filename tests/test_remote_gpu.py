import tempfile
import unittest
from pathlib import Path

from api.jobs import AnalysisJobManager
from webui.remote_gpu import _multipart_length, _remote_options


class RemoteGpuTests(unittest.TestCase):
    def test_remote_options_do_not_send_user_controlled_model_paths(self):
        options = _remote_options(
            {
                "pose_imgsz": 1280,
                "yolo_pose_model": "untrusted.pt",
                "ball_model": "another-untrusted.pt",
            }
        )
        self.assertEqual(options, {"pose_imgsz": 1280})

    def test_multipart_length_includes_streamed_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "match.mp4"
            template = root / "court.png"
            video.write_bytes(b"video")
            template.write_bytes(b"template")
            self.assertGreater(
                _multipart_length("boundary", {"court_corners": "[]"}, [("video", video), ("template", template)]),
                video.stat().st_size + template.stat().st_size,
            )

    def test_visualizations_are_downloadable_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = AnalysisJobManager(root, start_worker=False)
            output = root / "output"
            output.mkdir()
            image = output / "position_visualizations" / "heatmap.png"
            image.parent.mkdir()
            image.write_bytes(b"png")
            manifest = manager._result_manifest({"visualizations": [str(image)]}, output)
            artifact = manifest["artifacts"]["visualization_0"]
            self.assertEqual(artifact["relative_path"], "position_visualizations/heatmap.png")
            self.assertEqual(artifact["media_type"], "image/png")


if __name__ == "__main__":
    unittest.main()
