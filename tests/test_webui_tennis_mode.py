"""Tennis-mode guards for the WebUI before a browser submits a video."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from webui.app import (
    _APP_CSS,
    _SPORT_MODE_CLIENT_SYNC,
    configure_processing_target,
    configure_sport_mode,
    configure_sport_presentation,
    configure_stream_transport,
    ensure_court_for_analysis,
    run_analysis_with_upload_mode,
)


class TennisWebUiModeTests(unittest.TestCase):
    def test_tennis_browser_mode_sync_hides_badminton_only_controls(self):
        self.assertIn("dataset.goodSportMode", _SPORT_MODE_CLIENT_SYNC)
        for element_id in (
            "badminton-legacy-stream-button",
            "badminton-business-results",
            "badminton-rally-review-tab-button",
        ):
            self.assertIn(f"#{element_id}", _APP_CSS)

    def test_tennis_mode_exposes_visual_only_fixed_singles_defaults(self):
        updates = configure_sport_mode("tennis")

        self.assertEqual(len(updates), 15)
        self.assertIn("网球单打视觉分析", updates[0])
        self.assertIn("当前 YOLO-ball 羽毛球权重", updates[0])
        self.assertIn("非专用网球模型证据", updates[0])
        self.assertEqual(updates[3]["value"], "none")
        self.assertTrue(updates[3]["visible"])
        self.assertEqual([item[1] for item in updates[3]["choices"]], ["none", "yolo"])
        self.assertFalse(updates[4]["value"])
        self.assertTrue(updates[4]["interactive"])
        self.assertEqual(updates[5]["value"], 2)
        # Streaming-only controls stay hidden, while complete-video export is
        # available because the default transport is not segmented.
        self.assertTrue(all(update["visible"] is False for update in updates[7:12]))
        self.assertTrue(updates[12]["visible"])
        self.assertTrue(updates[13]["visible"])
        self.assertEqual(updates[14]["value"], "local_cpu")
        self.assertTrue(updates[14]["visible"])

        local_update = configure_processing_target("local_cpu", "tennis")
        self.assertEqual(local_update["value"], "http://127.0.0.1:18080")
        self.assertIn("本地 CPU", local_update["label"])
        self.assertFalse(local_update["visible"])
        remote_update = configure_processing_target("remote_gpu", "tennis")
        self.assertTrue(remote_update["visible"])

        stream_updates = configure_stream_transport(True, "tennis")
        self.assertFalse(stream_updates[0]["visible"])
        self.assertFalse(stream_updates[1]["visible"])
        full_updates = configure_stream_transport(False, "tennis")
        self.assertTrue(full_updates[0]["visible"])
        self.assertTrue(full_updates[1]["visible"])

        presentation = configure_sport_presentation("tennis")
        self.assertIn("手动", presentation[0]["value"])
        self.assertEqual(presentation[3]["headers"][-1], "数据质量")
        self.assertEqual(presentation[4]["headers"], [
            "匿名视觉 Track ID", "距离(m)", "平均速度(m/s)", "峰值速度(m/s)",
            "有效移动(s)", "可用覆盖率(%)", "数据质量",
        ])

    def test_tennis_preflight_requires_explicit_four_point_calibration(self):
        corners = [(10, 10), (110, 10), (110, 210), (10, 210)]
        result = ensure_court_for_analysis(
            "tennis-match.mp4", "court.jpg", corners, corners, "zh", "tennis"
        )
        self.assertTrue(result[-1])

        with self.assertRaises(gr.Error):
            ensure_court_for_analysis(
                "tennis-match.mp4", "court.jpg", None, [], "zh", "tennis"
            )

    def test_tennis_upload_uses_streaming_only_when_operator_enables_it(self):
        corners = [(10, 10), (110, 10), (110, 210), (10, 210)]
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "tennis-match.mp4"
            video.write_bytes(b"test-video")
            with patch(
                "webui.app.iter_remote_two_second_stream",
                return_value=iter([{"phase": "finalized", "analysis_session_id": "ssn_tennis"}]),
            ) as stream:
                updates = list(
                    run_analysis_with_upload_mode(
                        True, str(video), "court.jpg", corners,
                        "yolo", "whole_body", "zh", False, "singles",
                        "standard", "yolo", "bytetrack", 0.7, False,
                        640, 10, 0.5, False, "0.1,0.2,0.8,0.9",
                        False, False, True, True, True, True, True, True,
                        True, "weights/yolo11n-pose.pt", "weights/yolo11s-ball.pt",
                        "http://tennis.example:8080", True, 4, "tennis",
                    )
                )

        self.assertEqual(len(updates), 1)
        self.assertEqual(stream.call_args.kwargs["sport_id"], "tennis")
        self.assertEqual(stream.call_args.kwargs["session_mode"], "singles_match")
        options = stream.call_args.args[2]
        self.assertEqual(options["shuttle_detector"], "yolo")
        self.assertEqual(options["expected_player_count"], 2)

    def test_tennis_pose_only_selection_does_not_enable_ball_detection(self):
        corners = [(10, 10), (110, 10), (110, 210), (10, 210)]
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "tennis-match.mp4"
            video.write_bytes(b"test-video")
            with patch(
                "webui.app.iter_remote_two_second_stream",
                return_value=iter([{"phase": "finalized", "analysis_session_id": "ssn_tennis"}]),
            ) as stream:
                list(run_analysis_with_upload_mode(
                    True, str(video), "court.jpg", corners,
                    "yolo", "whole_body", "zh", False, "singles",
                    "standard", "none", "bytetrack", 0.7, False,
                    640, 10, 0.5, False, "0.1,0.2,0.8,0.9",
                    False, False, True, True, True, True, True, True,
                    True, "weights/yolo11n-pose.pt", "weights/yolo11s-ball.pt",
                    "http://tennis.example:8080", True, 4, "tennis",
                ))
        self.assertEqual(stream.call_args.args[2]["shuttle_detector"], "none")

    def test_tennis_local_cpu_target_forces_loopback_transport(self):
        corners = [(10, 10), (110, 10), (110, 210), (10, 210)]
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "tennis-match.mp4"
            video.write_bytes(b"test-video")
            with patch(
                "webui.app.iter_remote_two_second_stream",
                return_value=iter([{"phase": "finalized", "analysis_session_id": "ssn_tennis"}]),
            ) as stream:
                list(run_analysis_with_upload_mode(
                    True, str(video), "court.jpg", corners,
                    "yolo", "whole_body", "zh", False, "singles",
                    "standard", "none", "bytetrack", 0.7, False,
                    640, 10, 0.5, False, "0.1,0.2,0.8,0.9",
                    False, False, True, True, True, True, True, True,
                    True, "weights/yolo11n-pose.pt", "weights/yolo11s-ball.pt",
                    "http://remote.example:8080", True, 4, "tennis",
                    processing_target="local_cpu",
                ))
        self.assertEqual(stream.call_args.kwargs["gpu_base_url"], "http://127.0.0.1:18080")
        self.assertTrue(stream.call_args.kwargs["local_cpu"])
        self.assertEqual(stream.call_args.args[2]["tracker_backend"], "court_association")

    def test_tennis_without_segment_checkbox_uses_full_local_video_path(self):
        corners = [(10, 10), (110, 10), (110, 210), (10, 210)]
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "tennis-match.mp4"
            video.write_bytes(b"test-video")
            with patch(
                "webui.app.iter_remote_two_second_stream",
            ) as stream, patch(
                "webui.app.run_full_analysis",
                return_value=iter([tuple([None] * 18)]),
            ) as full:
                list(run_analysis_with_upload_mode(
                    True, str(video), "court.jpg", corners,
                    "yolo", "whole_body", "zh", False, "singles",
                    "standard", "none", "court_association", 0.7, False,
                    640, 10, 0.5, False, "0.1,0.2,0.8,0.9",
                    True, False, True, True, True, True, True, True,
                    True, "weights/yolo11n-pose.pt", "weights/yolo11s-ball.pt",
                    "http://remote.example:8080", False, 2, "tennis",
                    generate_promotion_video=True, processing_target="local_cpu",
                ))
        stream.assert_not_called()
        self.assertEqual(full.call_args.kwargs["sport_id"], "tennis")
        self.assertTrue(full.call_args.kwargs["force_local"])


if __name__ == "__main__":
    unittest.main()
