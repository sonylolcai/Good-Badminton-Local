"""Regression coverage for stable annotation rendering between pose samples."""

from unittest import TestCase
from unittest.mock import Mock

from badminton_analysis.system import BadmintonAnalysisSystem


class AnnotationSamplingTests(TestCase):
    def test_skipped_analysis_frame_reuses_only_the_cached_display_overlay(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.execution_metrics = {
            "sampling": {
                "analysis_skipped_source_frames": 0,
                "court_health_skipped_source_frames": 0,
            }
        }
        system._last_is_court_view = True
        system._last_spatial_state = {"tracks": [{"track_id": "track_001", "status": "detected"}]}
        system._should_sample_court_health = lambda _frame: False
        system._should_sample_analysis = lambda _frame: False
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
