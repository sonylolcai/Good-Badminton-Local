"""Contract tests for the post-analysis AI promotion-video renderer."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from business_gateway.promotion_video import (
    _build_commentary_timeline,
    _build_player_profiles,
    _current_speed,
    _load_track_observations,
    _parse_llm_segments,
    generate_promotion_video,
)
from webui.app import generate_promotion_video_for_webui


class PromotionVideoTests(unittest.TestCase):
    def test_profiles_and_current_speed_use_only_detected_spatial_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row) for row in [
                        self._row(0.0, "track_001", "detected", [1.0, 2.0]),
                        self._row(0.2, "track_001", "detected", [1.4, 2.0]),
                        self._row(0.3, "track_001", "predicted", [2.0, 2.0]),
                        self._row(0.4, "track_002", "detected", [3.0, 8.0]),
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            observations = _load_track_observations(path)

        self.assertEqual(len(observations["track_001"]), 2)
        self.assertEqual(_current_speed(observations["track_001"], 0.25), 2.0)
        profiles = _build_player_profiles(
            {
                "players": [{
                    "track_id": "track_001",
                    "movement": {"distance_m": 0.4, "peak_speed_mps": 2.0},
                    "measurement_coverage": {"usable_measurement_ratio": 1.0},
                }]
            },
            observations,
        )
        self.assertEqual(profiles[0]["panel_label"], "上半场运动员")
        self.assertEqual(profiles[1]["panel_label"], "下半场运动员")

    def test_llm_response_must_cover_every_requested_segment(self):
        baseline = _build_commentary_timeline(
            [{"track_id": "track_001", "observations": [], "static_metrics": {}}],
            duration_sec=12.0,
            sport_id="badminton",
        )
        complete = json.dumps({"segments": [{
            "segment_id": baseline[0]["segment_id"],
            "left_positive": "可观察到连续移动，建议结合完整回放复核。",
            "left_improvement": "建议结合完整回放复核击球时机。",
            "right_positive": "可观察到该侧保持有效移动。",
            "right_improvement": "当前证据不足，建议结合完整回放复核。",
        }]})
        parsed = _parse_llm_segments(complete, baseline)
        self.assertEqual(parsed[0]["commentary_source"], "llm_evidence_bounded")
        self.assertIn("连续移动", parsed[0]["left_positive"])
        self.assertIn("完整回放", parsed[0]["right_improvement"])

        with self.assertRaisesRegex(ValueError, "omitted"):
            _parse_llm_segments('{"segments": []}', baseline)

        with self.assertRaisesRegex(ValueError, "positive and improvement"):
            _parse_llm_segments(json.dumps({"segments": [{
                "segment_id": baseline[0]["segment_id"],
                "left_positive": "可观察到连续移动。",
            }]}), baseline)

    def test_webui_promotion_switch_does_not_rerun_or_block_analysis_when_off(self):
        video, timeline, status = generate_promotion_video_for_webui(False, "unused", "unused.mp4")
        self.assertIsNone(video)
        self.assertIsNone(timeline)
        self.assertEqual(status["status"], "not_requested")

        with patch("webui.app.generate_promotion_video", return_value={
            "video_path": "promotion.mp4", "timeline_path": "timeline.json", "status": "succeeded",
        }) as renderer:
            video, timeline, status = generate_promotion_video_for_webui(True, "output", "annotated.mp4")
        self.assertEqual((video, timeline), ("promotion.mp4", "timeline.json"))
        self.assertEqual(status["status"], "succeeded")
        renderer.assert_called_once()

    def test_renderer_composes_a_timeline_from_completed_evidence_without_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "annotated.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180)
            )
            for index in range(3):
                writer.write(np.full((180, 320, 3), index * 30, dtype=np.uint8))
            writer.release()
            (root / "derived").mkdir()
            (root / "metadata.json").write_text("{}", encoding="utf-8")
            (root / "derived" / "player_movement_metrics_v1.json").write_text(json.dumps({
                "players": [{
                    "track_id": "track_001",
                    "movement": {"distance_m": 1.2, "peak_speed_mps": 2.0},
                    "measurement_coverage": {"usable_measurement_ratio": 1.0},
                }]
            }), encoding="utf-8")
            (root / "detections.jsonl").write_text(
                "\n".join(json.dumps(self._row(time_sec, "track_001", "detected", [1 + time_sec, 2]))
                          for time_sec in (0.0, 0.1, 0.2)) + "\n",
                encoding="utf-8",
            )

            def fake_transcode(temporary, final, _audio_source):
                shutil.copyfile(temporary, final)

            with patch("business_gateway.promotion_video._transcode_with_source_audio", fake_transcode):
                result = generate_promotion_video(root, source)

            self.assertTrue(Path(result["video_path"]).is_file())
            timeline = json.loads(Path(result["timeline_path"]).read_text(encoding="utf-8"))
            self.assertEqual(timeline["policy"]["gpu_inference_rerun"], False)
            self.assertEqual(timeline["llm"]["status"], "not_configured")

    @staticmethod
    def _row(time_sec, track_id, status, xy):
        return {
            "time_sec": time_sec,
            "spatial": {"tracks": [{
                "track_id": track_id,
                "status": status,
                "court_xy_m": xy,
                "zone_id": "mid_center",
                "court_end": "end_a",
            }]},
        }


if __name__ == "__main__":
    unittest.main()
