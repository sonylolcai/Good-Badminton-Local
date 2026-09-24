"""Image-space motion evidence; no sample-specific geometry or assumed metric scale."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .observation import ObservationConfig, image_diagonal


@dataclass
class TrajectoryPoint:
    frame_index: int
    timestamp_s: float
    x: float
    y: float
    speed_kmh: Optional[float] = None
    vx: float = 0.0
    vy: float = 0.0
    visible: bool = True
    in_court_roi: bool = True
    ground_contact: bool = False


@dataclass
class FlightArc:
    """Observed motion segment, not proof of a racket stroke or player identity."""
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    peak_speed_kmh: Optional[float]
    terminal_speed_kmh: Optional[float]
    dx_px: float
    dy_px: float
    flight_direction: str
    tactical_line: str
    points: List[TrajectoryPoint] = field(default_factory=list)


class PhysicalTrajectoryAnalyzer:
    def __init__(self, fps: float, image_size: Tuple[int, int], config: Optional[ObservationConfig] = None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.fps = fps
        self.diagonal = image_diagonal(image_size)
        if self.diagonal is None:
            raise ValueError("image_size is required for normalized motion thresholds")
        self.config = config or ObservationConfig.load()

    def extract_trajectory_points(self, raw_detections: List[Dict]) -> List[TrajectoryPoint]:
        points = []
        for d in raw_detections:
            x, y = d.get("x"), d.get("y")
            visible = bool(d.get("visible")) and d.get("status", "detected") == "detected"
            visible = visible and x is not None and y is not None
            visible = visible and math.isfinite(float(x)) and math.isfinite(float(y))
            frame = int(d["frame_index"])
            point = TrajectoryPoint(
                frame_index=frame,
                timestamp_s=float(d.get("timestamp_s", frame / self.fps)),
                x=float(x) if visible else 0.0,
                y=float(y) if visible else 0.0,
                visible=visible,
                ground_contact=visible and d.get("ground_contact") is True,
            )
            if not math.isfinite(point.timestamp_s):
                raise ValueError("timestamps must be finite")
            if points:
                previous = points[-1]
                dt = point.timestamp_s - previous.timestamp_s
                if point.frame_index <= previous.frame_index or dt <= 0:
                    raise ValueError("detections must be ordered by increasing frame and timestamp")
                if visible and previous.visible and dt <= self.config.max_observation_gap_s:
                    point.vx = (point.x - previous.x) / dt
                    point.vy = (point.y - previous.y) / dt
            # A 2D image displacement does not establish a physical km/h speed.
            points.append(point)
        return points

    def segment_flight_arcs(self, points: List[TrajectoryPoint]) -> List[FlightArc]:
        arcs = []
        current = []
        previous = None
        previous_velocity = None
        min_speed = self.config.motion_min_speed_diagonals_s * self.diagonal
        turn_cos = math.cos(math.radians(self.config.turn_angle_degrees))

        def flush():
            if len(current) >= 2 and current[-1].timestamp_s - current[0].timestamp_s >= self.config.motion_min_duration_s:
                arcs.append(self._build_arc(current))
            current.clear()

        for point in points:
            if not point.visible:
                flush()
                previous = previous_velocity = None
                continue
            if previous is None:
                previous = point
                continue
            dt = point.timestamp_s - previous.timestamp_s
            if dt <= 0 or dt > self.config.max_observation_gap_s:
                flush()
                previous, previous_velocity = point, None
                continue
            dx, dy = point.x - previous.x, point.y - previous.y
            speed = math.hypot(dx, dy) / dt
            if speed < min_speed:
                flush()
                previous_velocity = None
            else:
                if previous_velocity is not None and current:
                    vx, vy = previous_velocity
                    cosine = (dx * vx + dy * vy) / (math.hypot(dx, dy) * math.hypot(vx, vy))
                    if cosine <= turn_cos:
                        flush()
                if not current:
                    current.append(previous)
                current.append(point)
                previous_velocity = (dx, dy)
            if point.ground_contact:
                flush()
                previous = previous_velocity = None
                continue
            previous = point
        flush()
        return arcs

    @staticmethod
    def _build_arc(points: List[TrajectoryPoint]) -> FlightArc:
        first, last = points[0], points[-1]
        speeds = [p.speed_kmh for p in points if p.speed_kmh is not None]
        return FlightArc(
            start_frame=first.frame_index, end_frame=last.frame_index,
            start_time_s=first.timestamp_s, end_time_s=last.timestamp_s,
            start_xy=(first.x, first.y), end_xy=(last.x, last.y),
            peak_speed_kmh=max(speeds) if speeds else None,
            terminal_speed_kmh=last.speed_kmh,
            dx_px=last.x - first.x, dy_px=last.y - first.y,
            flight_direction="unknown", tactical_line="unknown", points=list(points),
        )
