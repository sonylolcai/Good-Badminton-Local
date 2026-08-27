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
    remote_gpu_config,
)
from business_gateway.dev_api import LocalReplayManager


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

    def test_operator_can_override_one_remote_gpu_destination(self):
        with patch.dict(
            "os.environ",
            {"GOOD_BADMINTON_GPU_API_URL": "http://configured.example:8080"},
            clear=False,
        ):
            config = remote_gpu_config("http://xn-g.suanjiayun.com:55606")
        self.assertEqual(config["base_url"], "http://xn-g.suanjiayun.com:55606")

    def test_stream_replay_endpoint_override_is_task_scoped_and_validated(self):
        with patch.dict(
            "os.environ",
            {
                "GPU_ANALYSIS_BASE_URL": "http://configured.example:8080",
                "GPU_ANALYSIS_API_KEY": "development-key",
            },
            clear=False,
        ):
            config = LocalReplayManager._stream_config_for_task(
                {}, gpu_base_url="http://xn-g.suanjiayun.com:55606"
            )
            self.assertEqual(config.base_url, "http://xn-g.suanjiayun.com:55606")
            with self.assertRaisesRegex(ValueError, "base URL"):
                LocalReplayManager._stream_config_for_task(
                    {}, gpu_base_url="http://user:password@gpu.example:8080"
                )

    def test_candidate_photo_proxies_a_known_track_from_the_gpu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalReplayManager(Path(temp_dir))
            task_id = "bstr_test"
            task_dir = Path(temp_dir) / task_id
            task_dir.mkdir()
            manager._write_task(
                task_dir,
                {
                    "analysis_session_id": "ssn_test",
                    "result": {
                        "status": {
                            "track_candidates": [{"track_id": "track_001"}],
                        }
                    },
                },
            )
            with patch.dict(
                "os.environ",
                {
                    "GPU_ANALYSIS_BASE_URL": "http://gpu.example:8080",
                    "GPU_ANALYSIS_API_KEY": "test-key",
                },
                clear=False,
            ), patch("business_gateway.dev_api.urlopen") as urlopen:
                remote = urlopen.return_value.__enter__.return_value
                remote.headers.get_content_type.return_value = "image/jpeg"
                remote.read.return_value = b"jpeg-bytes"
                content, media_type = manager.candidate_photo(task_id, "track_001")

            self.assertEqual((content, media_type), (b"jpeg-bytes", "image/jpeg"))
            self.assertEqual(
                urlopen.call_args.args[0].full_url,
                "http://gpu.example:8080/api/v1/stream-sessions/ssn_test/candidate-photos/track_001",
            )

    def test_claimed_track_summary_uses_business_side_body_profile_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = LocalReplayManager(root)
            task_id = "bstr_test"
            task_dir = root / task_id
            task_dir.mkdir()
            manager._write_task(
                task_dir,
                {
                    "business_task_id": task_id,
                    "analysis_session_id": "ssn_test",
                    "result": {
                        "status": {
                            "status": "finalized",
                            "progress": {"processed_source_time_sec": 120},
                            "track_candidates": [{"track_id": "track_001"}],
                        },
                        "business_derivation": {
                            "materialized_detections_path": "detections.jsonl",
                            "spatial_summary_path": "spatial.json",
                            "metadata_path": "metadata.json",
                            "metrics": {},
                        },
                    },
                },
            )
            derived_metrics = {
                "players": [
                    {
                        "track_id": "track_001",
                        "measurement_coverage": {"usable_measurement_ratio": 0.9},
                        "movement": {"distance_m": 120.0},
                        "energy_estimate": {"status": "estimated", "estimated_kcal_rounded": 88},
                        "quality": {"status": "reviewable"},
                    }
                ]
            }
            with patch("business_gateway.dev_api.write_body_profiles", return_value="profiles.json") as write_profiles, patch(
                "business_gateway.dev_api.generate_movement_metrics", return_value=derived_metrics
            ) as generate_metrics:
                summary = manager.claim_track(
                    task_id,
                    "track_001",
                    {"heightCm": 175, "weightKg": 70},
                )

            self.assertEqual(summary["energy_estimate"]["estimated_kcal_rounded"], 88)
            self.assertEqual(summary["movement"]["distance_m"], 120.0)
            self.assertEqual(write_profiles.call_args.kwargs["consent"], True)
            self.assertEqual(Path(generate_metrics.call_args.kwargs["output_dir"]).resolve(), task_dir.resolve())


if __name__ == "__main__":
    unittest.main()
