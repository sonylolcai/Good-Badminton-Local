"""Tennis-only YOLO ball measurement adapter.

This class reuses only the low-level, frame-local candidate filtering shared
by the existing tracker.  It deliberately requires a tennis-labelled model,
uses tennis-specific gates, and disables synthetic motion prediction so a
missed ball is recorded as missing rather than presented as a ball path.
"""

from __future__ import annotations

from .shuttlecock import ShuttlecockTracker


class TennisBallTracker(ShuttlecockTracker):
    """Track raw tennis-ball image measurements from a dedicated YOLO model."""

    REQUIRED_CLASS_NAME = "tennis_ball"

    def __init__(self, yolo_ball_model, **overrides):
        settings = {
            # These are conservative initial deployment settings, not a
            # quality claim.  They must be tuned against fixed-camera tennis
            # validation footage packaged with the selected model.
            "trajectory_length": 60,
            "show_trajectory": False,
            "max_jump_pixels": 420,
            "prediction_gate_pixels": 500,
            "max_missing_frames": 3,
            "roi_padding_ratio": 0.04,
            "max_box_area_ratio": 0.012,
            "max_aspect_ratio": 3.5,
            # GPU output must remain observed evidence.  Do not interpolate a
            # fast tennis ball across missed frames.
            "max_prediction_frames": 0,
            "prediction_confidence_decay": 0.0,
            "required_class_names": (self.REQUIRED_CLASS_NAME,),
        }
        settings.update(overrides)
        super().__init__(yolo_ball_model, **settings)


class ExperimentalBadmintonBallTracker(ShuttlecockTracker):
    """Run the existing badminton checkpoint as explicitly experimental evidence.

    This is intentionally not a fallback inside :class:`TennisBallTracker`.
    A model trained with the ``badminton`` class can be useful for an initial
    fixed-camera smoke test, but its observations must remain distinguishable
    from validated ``tennis_ball`` measurements in every persisted event.
    """

    REQUIRED_CLASS_NAME = "badminton"

    def __init__(self, yolo_ball_model, **overrides):
        settings = {
            "trajectory_length": 60,
            "show_trajectory": False,
            "max_jump_pixels": 420,
            "prediction_gate_pixels": 500,
            "max_missing_frames": 3,
            "roi_padding_ratio": 0.04,
            "max_box_area_ratio": 0.012,
            "max_aspect_ratio": 3.5,
            # A cross-sport experiment must never manufacture a tennis ball
            # path through a missed detection.
            "max_prediction_frames": 0,
            "prediction_confidence_decay": 0.0,
            "required_class_names": (self.REQUIRED_CLASS_NAME,),
        }
        settings.update(overrides)
        super().__init__(yolo_ball_model, **settings)
