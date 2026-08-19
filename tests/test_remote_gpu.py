import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api.jobs import AnalysisJobManager
from badminton_analysis.cancellation import AnalysisCancelled
from webui.pipeline import run_analysis
from webui.remote_gpu import (
    RemoteAnalysisError,
    _load_local_config_file,
    _multipart_length,
    _remote_options,
    _submit_multipart,
)


class RemoteGpuTests(unittest.TestCase):
    def test_cancelled_local_pipeline_exits_before_loading_models(self):
        with self.assertRaises(AnalysisCancelled):
            run_analysis(
                "missing.mp4", "missing.png", [], {}, cancel_cb=lambda: True,
            )

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

    def test_upload_timeout_explains_that_no_remote_receipt_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "match.mp4"
            template = root / "court.png"
            video.write_bytes(b"video")
            template.write_bytes(b"template")
            config = {"base_url": "http://127.0.0.1:9", "api_key": "test", "timeout_seconds": 1}
            with patch("webui.remote_gpu.http.client.HTTPConnection") as connection_type:
                connection = connection_type.return_value
                connection.putrequest.side_effect = TimeoutError("timed out")
                with self.assertRaisesRegex(RemoteAnalysisError, "未收到接收回执"):
                    _submit_multipart(
                        config,
                        video,
                        template,
                        [[1, 1], [2, 1], [2, 2], [1, 2]],
                        {},
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
            summary = image.parent / "position_evidence_summary.json"
            summary.write_text("{}\n", encoding="utf-8")
            trace = output / "performance_trace.json"
            trace.write_text("{}\n", encoding="utf-8")
            manifest = manager._result_manifest(
                {
                    "visualizations": [str(image)],
                    "position_evidence_summary": str(summary),
                },
                output,
            )
            artifact = manifest["artifacts"]["visualization_0"]
            self.assertEqual(artifact["relative_path"], "position_visualizations/heatmap.png")
            self.assertEqual(artifact["media_type"], "image/png")
            self.assertEqual(
                manifest["artifacts"]["position_evidence_summary"]["relative_path"],
                "position_visualizations/position_evidence_summary.json",
            )
            self.assertEqual(
                manifest["artifacts"]["performance_trace"]["relative_path"],
                "performance_trace.json",
            )

    def test_local_config_does_not_overwrite_explicit_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "remote.env"
            config.write_text("GOOD_BADMINTON_GPU_API_KEY=file-key\n", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {"GOOD_BADMINTON_WEBUI_CONFIG": str(config), "GOOD_BADMINTON_GPU_API_KEY": "explicit-key"},
                clear=False,
            ):
                _load_local_config_file()
                import os
                self.assertEqual(os.environ["GOOD_BADMINTON_GPU_API_KEY"], "explicit-key")


if __name__ == "__main__":
    unittest.main()
