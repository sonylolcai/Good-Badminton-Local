"""Decoupled Physical Trajectory Analysis Layer.

This module is strictly Newtonian physics, coordinate geometry, and kinematics.
It does NOT contain any sport-specific scoring or game rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrajectoryPoint:
    frame_index: int
    timestamp_s: float
    x: float
    y: float
    speed_kmh: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    visible: bool = True
    in_court_roi: bool = True


@dataclass
class FlightArc:
    """A continuous single-flight segment between two physical events (hit, net collision, bounce, or floor)."""
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    peak_speed_kmh: float
    terminal_speed_kmh: float
    dx_px: float
    dy_px: float
    flight_direction: str  # "far_to_near", "near_to_far", "lateral"
    tactical_line: str     # "straight", "cross_court"
    points: List[TrajectoryPoint] = field(default_factory=list)


class PhysicalTrajectoryAnalyzer:
    """Extracts physical dynamics, flight arcs, and turning points from raw ball detections."""

    def __init__(
        self,
        fps: float = 29.57,
        court_roi_y: Tuple[float, float] = (700.0, 2100.0),
        court_roi_x: Tuple[float, float] = (500.0, 3500.0),
        dead_speed_threshold_kmh: float = 5.0,
        min_in_flight_speed_kmh: float = 15.0,
    ):
        self.fps = fps
        self.court_roi_y = court_roi_y
        self.court_roi_x = court_roi_x
        self.dead_speed_threshold_kmh = dead_speed_threshold_kmh
        self.min_in_flight_speed_kmh = min_in_flight_speed_kmh

    def get_perspective_m_per_px(self, y: float) -> float:
        """Dynamic depth-aware scaling from far court to near baseline."""
        y_clamped = max(self.court_roi_y[0], min(self.court_roi_y[1], float(y)))
        # Far court (0.0052 m/px) to near baseline (0.0024 m/px)
        ratio = (y_clamped - self.court_roi_y[0]) / (self.court_roi_y[1] - self.court_roi_y[0])
        return 0.0052 - ratio * (0.0052 - 0.0024)

    def extract_trajectory_points(self, raw_detections: List[Dict]) -> List[TrajectoryPoint]:
        """Convert raw detection dicts into physically validated, speed-calibrated trajectory points."""
        points: List[TrajectoryPoint] = []
        for d in raw_detections:
            vis = bool(d.get("visible", False))
            x = float(d["x"]) if d.get("x") is not None else 0.0
            y = float(d["y"]) if d.get("y") is not None else 0.0
            f_idx = int(d.get("frame_index", 0))
            t_sec = float(d.get("timestamp_s", f_idx / self.fps))
            in_roi = (
                self.court_roi_x[0] <= x <= self.court_roi_x[1]
                and self.court_roi_y[0] <= y <= self.court_roi_y[1]
            )
            points.append(
                TrajectoryPoint(
                    frame_index=f_idx,
                    timestamp_s=t_sec,
                    x=x,
                    y=y,
                    visible=vis,
                    in_court_roi=in_roi,
                )
            )

        # Compute velocities and physical speeds
        n = len(points)
        for i in range(1, n):
            cur = points[i]
            prev = points[i - 1]
            if cur.visible and prev.visible and (cur.frame_index - prev.frame_index == 1):
                dx = cur.x - prev.x
                dy = cur.y - prev.y
                mid_y = (cur.y + prev.y) / 2.0
                m_px = self.get_perspective_m_per_px(mid_y)
                dist_m = math.hypot(dx, dy) * m_px
                speed_kmh = (dist_m * self.fps) * 3.6
                cur.vx = dx * self.fps
                cur.vy = dy * self.fps
                # Clamp physical max at 450 km/h
                cur.speed_kmh = min(speed_kmh, 450.0)
            else:
                cur.speed_kmh = 0.0
                cur.vx = 0.0
                cur.vy = 0.0

        return points

    def segment_flight_arcs(self, points: List[TrajectoryPoint]) -> List[FlightArc]:
        """Segment continuous flight arcs using velocity vector inflection and net crossings."""
        vis_pts = [p for p in points if p.visible and p.in_court_roi]
        if len(vis_pts) < 4:
            return []

        arcs: List[FlightArc] = []
        cur_arc_pts: List[TrajectoryPoint] = []
        slow_count = 0

        for p in vis_pts:
            # Dead-ball check: if ball is stationary/rolling on ground (< 6 km/h)
            if p.speed_kmh <= self.dead_speed_threshold_kmh:
                slow_count += 1
                if slow_count >= 5:
                    if len(cur_arc_pts) >= 4:
                        arcs.append(self._build_arc(cur_arc_pts))
                    cur_arc_pts = []
                    continue
            else:
                slow_count = 0

            if not cur_arc_pts:
                cur_arc_pts.append(p)
                continue

            last = cur_arc_pts[-1]
            # Frame gap break (>4 frames without ball)
            if p.frame_index - last.frame_index > 4:
                if len(cur_arc_pts) >= 4:
                    arcs.append(self._build_arc(cur_arc_pts))
                cur_arc_pts = [p]
                continue

            # Check directional inflection (turning point / hit)
            if len(cur_arc_pts) >= 3:
                dy_prev = cur_arc_pts[-1].y - cur_arc_pts[-3].y
                dy_next = p.y - cur_arc_pts[-1].y
                if (dy_prev < -15 and dy_next > 15) or (dy_prev > 15 and dy_next < -15):
                    # Physical apex check: if inflection is high up near ceiling (y < 550),
                    # it is an aerodynamic arc apex (rising then descending), not a player hit!
                    if p.y < 550 or cur_arc_pts[-1].y < 550:
                        cur_arc_pts.append(p)
                        continue
                    cur_arc_pts.append(p)
                    arc = self._build_arc(cur_arc_pts)
                    if self._is_valid_arc(arc):
                        arcs.append(arc)
                    cur_arc_pts = [p]
                    continue

            cur_arc_pts.append(p)

        if len(cur_arc_pts) >= 4:
            arc = self._build_arc(cur_arc_pts)
            if self._is_valid_arc(arc):
                arcs.append(arc)

        return arcs

    def _is_valid_arc(self, arc: FlightArc) -> bool:
        """Filter out tiny jitter, rolling noise, or micro-movements."""
        disp = math.hypot(arc.dx_px, arc.dy_px)
        # Valid shot must either have peak speed >= 12 km/h or displacement >= 80px
        return arc.peak_speed_kmh >= 12.0 or disp >= 80.0

    def _build_arc(self, pts: List[TrajectoryPoint]) -> FlightArc:
        start = pts[0]
        end = pts[-1]
        dx = end.x - start.x
        dy = end.y - start.y

        vert_dir = "far_to_near" if dy > 0 else "near_to_far"
        # Cross court threshold: horizontal displacement > 350px
        tactical_line = "cross_court" if abs(dx) > 350 else "straight"

        peak_spd = max(p.speed_kmh for p in pts)
        term_spd = pts[-1].speed_kmh

        return FlightArc(
            start_frame=start.frame_index,
            end_frame=end.frame_index,
            start_time_s=start.timestamp_s,
            end_time_s=end.timestamp_s,
            start_xy=(start.x, start.y),
            end_xy=(end.x, end.y),
            peak_speed_kmh=round(peak_spd, 1),
            terminal_speed_kmh=round(term_spd, 1),
            dx_px=round(dx, 1),
            dy_px=round(dy, 1),
            flight_direction=vert_dir,
            tactical_line=tactical_line,
            points=pts,
        )
