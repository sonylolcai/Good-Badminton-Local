"""Regression coverage for automatic court-corner application wiring."""

import unittest

from webui.app import build_ui


class CourtCornerWiringTests(unittest.TestCase):
    def test_apply_button_uses_the_detected_corner_state(self):
        """Auto-preview corners must be the same state consumed by Apply.

        The image can visibly contain four red auto-detected markers while the
        manual-click accumulator is empty.  Inspecting the real Gradio event
        graph protects that UI-specific boundary rather than testing the
        callback in isolation with an artificial list of points.
        """
        demo = build_ui()
        handlers = {
            function.fn.__name__: function
            for function in demo.fns.values()
            if getattr(function, "fn", None) is not None
        }
        detected_corner_state = handlers["detect_court"].outputs[1]
        applied_corner_state = handlers["apply_manual_corners"].inputs[1]

        self.assertEqual(applied_corner_state._id, detected_corner_state._id)


if __name__ == "__main__":
    unittest.main()
