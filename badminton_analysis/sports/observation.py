"""Configurable observation thresholds, independent of video identity or court position."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ObservationConfig:
    motion_min_speed_diagonals_s: float
    motion_min_duration_s: float
    stationary_duration_s: float
    stationary_radius_diagonals: float
    stationary_max_speed_diagonals_s: float
    max_observation_gap_s: float
    turn_angle_degrees: float

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")
        if self.turn_angle_degrees > 180:
            raise ValueError("turn_angle_degrees must be at most 180")

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else Path(__file__).with_name("observation_defaults.json")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def image_diagonal(image_size):
    if image_size is None:
        return None
    width, height = image_size
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("image_size must contain positive width and height")
    return math.hypot(width, height)
