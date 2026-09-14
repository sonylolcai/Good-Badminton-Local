"""Regression coverage for automatic court-corner application wiring."""

import unittest

from webui.app import _build_legacy_analysis_ui, build_ui


class CourtCornerWiringTests(unittest.TestCase):
    def test_apply_button_uses_the_detected_corner_state(self):
        """Auto-preview corners must be the same state consumed by Apply.

        The image can visibly contain four red auto-detected markers while the
        manual-click accumulator is empty.  Inspecting the real Gradio event
        graph protects that UI-specific boundary rather than testing the
        callback in isolation with an artificial list of points.
        """
        demo = _build_legacy_analysis_ui()
        handlers = {
            function.fn.__name__: function
            for function in demo.fns.values()
            if getattr(function, "fn", None) is not None
        }
        detected_corner_state = handlers["detect_court"].outputs[1]
        applied_corner_state = handlers["apply_manual_corners"].inputs[1]

        self.assertEqual(applied_corner_state._id, detected_corner_state._id)

    def test_default_business_ui_does_not_wire_analysis_or_review(self):
        demo = build_ui()
        tabs = {
            component.get("props", {}).get("label")
            for component in demo.get_config_file()["components"]
            if component.get("type") == "tabitem"
        }
        handlers = {
            getattr(function.fn, "__name__", "")
            for function in demo.fns.values()
            if getattr(function, "fn", None) is not None
        }

        self.assertTrue({"赛后业务", "教练档案", "业务任务历史"}.issubset(tabs))
        self.assertTrue({"分析工作台", "球路复核", "分析任务历史"}.isdisjoint(tabs))
        self.assertNotIn("run_analysis_with_upload_mode", handlers)
        self.assertNotIn("_review_open_timeline", handlers)
        self.assertIn("save_body_profiles_and_refresh_metrics", handlers)
        self.assertIn("save_coach_athlete_profile", handlers)
        self.assertIn("generate_promotion_video_for_webui", handlers)


if __name__ == "__main__":
    unittest.main()
