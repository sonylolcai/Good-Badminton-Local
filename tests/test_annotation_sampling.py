"""Regression coverage for stable annotation rendering between pose samples."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from badminton_analysis.system import BadmintonAnalysisSystem


class AnnotationSamplingTests(TestCase):
    def test_skeleton_video_keeps_the_source_video_background(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.output_video_style = "skeleton"
        source = np.full((4, 6, 3), 123, dtype=np.uint8)

        output = system._create_output_frame(source)

        self.assertTrue(np.array_equal(output, source))
        self.assertIsNot(output, source)

    def test_skipped_analysis_frame_reuses_only_the_cached_display_overlay(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.execution_metrics = {
            "sampling": {
                "analysis_skipped_source_frames": 0,
                "shuttle_skipped_source_frames": 0,
                "court_health_skipped_source_frames": 0,
            }
        }
        system.shuttle_detector = "yolo"
        system._last_is_court_view = True
        system._last_spatial_state = {"tracks": [{"track_id": "track_001", "status": "detected"}]}
        system._should_sample_court_health = lambda _frame: False
        system._should_sample_analysis = lambda _frame: False
        system._should_sample_shuttle = lambda _frame: False
        system._needs_visual_output = lambda: True
        system._create_output_frame = lambda frame: f"canvas:{frame}"
        system._draw_player_overlay = Mock()
        system._record_execution_metric = Mock()
        system._write_output_frame = Mock()

        frame, detect_count = system._process_frame(
            "source", None, None, None, frame_count=2, out=None, detect_frame_count=5,
        )

        self.assertEqual(frame, "canvas:source")
        self.assertEqual(detect_count, 5)
        system._draw_player_overlay.assert_called_once_with(
            "canvas:source", system._last_spatial_state,
        )
        system._write_output_frame.assert_called_once_with("canvas:source", 2, None)
        self.assertEqual(system.execution_metrics["sampling"]["analysis_skipped_source_frames"], 1)
        self.assertEqual(system.execution_metrics["sampling"]["shuttle_skipped_source_frames"], 1)

    def test_shuttle_only_sample_is_written_without_a_player_measurement(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.execution_metrics = {
            "sampling": {
                "analysis_skipped_source_frames": 0,
                "shuttle_measurement_frames": 0,
                "shuttle_skipped_source_frames": 0,
                "court_health_skipped_source_frames": 0,
            }
        }
        system.shuttle_detector = "yolo"
        system._last_is_court_view = True
        system._should_sample_court_health = lambda _frame: False
        system._should_sample_analysis = lambda _frame: False
        system._should_sample_shuttle = lambda _frame: True
        system._sample_shuttle = Mock(return_value=([12, 34], {"status": "detected"}, 0.01))
        system._write_shuttle_measurement_record = Mock()
        system._needs_visual_output = lambda: False
        system._write_output_frame = Mock()

        frame, detect_count = system._process_frame(
            "source", None, None, None, frame_count=2, out=None, detect_frame_count=5,
        )

        self.assertEqual(frame, "source")
        self.assertEqual(detect_count, 5)
        system._write_shuttle_measurement_record.assert_called_once_with(
            2, [12, 34], {"status": "detected"},
        )
        self.assertEqual(system.execution_metrics["sampling"]["shuttle_measurement_frames"], 1)
