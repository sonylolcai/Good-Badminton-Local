import unittest

from badminton_analysis.system import BadmintonAnalysisSystem


class VideoOutputOptionsTests(unittest.TestCase):
    def test_data_only_mode_does_not_require_an_annotated_frame_sink(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.generate_annotated_video = False
        system.show_display = False
        system.save_images = False

        self.assertFalse(system._needs_visual_output())
        self.assertIsNone(system._setup_video_writer(1920, 1080, 30.0))

    def test_debug_display_remains_a_visual_sink_without_video_export(self):
        system = object.__new__(BadmintonAnalysisSystem)
        system.generate_annotated_video = False
        system.show_display = True
        system.save_images = False

        self.assertTrue(system._needs_visual_output())


if __name__ == "__main__":
    unittest.main()
