"""Tennis-mode guards for the WebUI before a browser submits a video."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from webui.app import (
    _APP_CSS,
    _SPORT_MODE_CLIENT_SYNC,
    configure_sport_mode,
    configure_sport_presentation,
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

        self.assertEqual(len(updates), 14)
        self.assertIn("网球单打视觉分析", updates[0])
        self.assertIn("当前 YOLO-ball 羽毛球权重", updates[0])
        self.assertIn("非专用网球模型证据", updates[0])
        self.assertEqual(updates[3]["value"], "none")
        self.assertTrue(updates[3]["visible"])
        self.assertEqual([item[1] for item in updates[3]["choices"]], ["none", "yolo"])
        self.assertTrue(updates[4]["value"])
        self.assertFalse(updates[4]["interactive"])
        self.assertEqual(updates[5]["value"], 2)
        # The remaining five updates hide legacy business-only controls.
        self.assertTrue(all(update["visible"] is False for update in updates[7:]))
        self.assertFalse(updates[12]["value"])
        self.assertFalse(updates[13]["value"])
        self.assertFalse(updates[13]["visible"])

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

    def test_tennis_upload_forces_the_pure_streaming_transport_and_two_tracks(self):
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
                        "http://tennis.example:8080", False, 4, "tennis",
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
                    "http://tennis.example:8080", False, 4, "tennis",
                ))
        self.assertEqual(stream.call_args.args[2]["shuttle_detector"], "none")


if __name__ == "__main__":
    unittest.main()
