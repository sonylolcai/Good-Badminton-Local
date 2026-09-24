"""Evidence-based badminton rally boundaries. No score or sample-specific overrides."""

from __future__ import annotations

from bisect import bisect_left
import math
from typing import Any, List, Optional

import cv2
import numpy as np

from .base import BaseSportRuleEngine, RallyEvent, TerminalEvent
from .observation import ObservationConfig, image_diagonal
from .physics import FlightArc, TrajectoryPoint
from .tactics import CourtZone, DepthZone, LateralZone, HandSide, HeightZone, TacticalNarrativeEngine, TacticalShot


class BadmintonRuleEngine(BaseSportRuleEngine):
    @property
    def sport_id(self):
        return "badminton"

    @property
    def max_allowed_bounces(self):
        return 0

    def __init__(self, *, court_polygon=None, image_size=None, config=None,
                 dominant_hand="unknown", pose_analyzer=None):
        self.config = config or ObservationConfig.load()
        self.diagonal = image_diagonal(image_size)
        self.court_polygon = None
        if court_polygon is not None:
            polygon = np.asarray(court_polygon, dtype=np.float32)
            if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                    or not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon)
                    or cv2.contourArea(polygon) <= 0):
                raise ValueError("court_polygon must be an ordered, finite, convex court boundary")
            self.court_polygon = polygon
        if dominant_hand not in ("left", "right", "unknown"):
            raise ValueError("dominant_hand must be left, right or unknown")
        self.dominant_hand = dominant_hand
        self.pose_analyzer = pose_analyzer

    def evaluate_serve_condition(self, arc, player_stillness_observed=False):
        """Motion or stillness alone cannot establish a legal serve."""
        return None

    def evaluate_dead_ball(self, recent_points, last_arc):
        """Require ground contact or sustained low speed AND low displacement.

        Disappearance, a slow sample and the end of a video are not terminal
        evidence. The motion criterion is an inferred landing, not measured
        ground contact; retain its provenance on the terminal event.
        """
        stationary = []
        radius = self.config.stationary_radius_diagonals * self.diagonal if self.diagonal else None
        max_speed = self.config.stationary_max_speed_diagonals_s * self.diagonal if self.diagonal else None
        for point in recent_points:
            if last_arc and point.frame_index < last_arc.end_frame:
                continue
            if not point.visible or not all(math.isfinite(v) for v in (point.x, point.y, point.timestamp_s)):
                stationary.clear()
                continue
            if point.ground_contact:
                return self._landing_event(point, (point.x, point.y), "ground_contact",
                                           "Observed ground contact.")
            if radius is None:
                continue
            if stationary:
                gap = point.timestamp_s - stationary[-1].timestamp_s
                previous = stationary[-1]
                distance = math.hypot(point.x - previous.x, point.y - previous.y)
                # Pairwise extent rejects oscillation or drift even when the
                # first and last positions happen to coincide.
                exceeds_displacement = any(math.hypot(point.x - p.x, point.y - p.y) > radius for p in stationary)
                if (gap <= 0 or gap > self.config.max_observation_gap_s
                        or distance / gap > max_speed or exceeds_displacement):
                    stationary.clear()
            stationary.append(point)
            if point.timestamp_s - stationary[0].timestamp_s >= self.config.stationary_duration_s:
                landing_xy = tuple(float(v) for v in np.median([(p.x, p.y) for p in stationary], axis=0))
                return self._landing_event(
                    point, landing_xy, "low_speed_low_displacement",
                    "Inferred landing from sustained low image speed and low window displacement.",
                )
        return None

    def _landing_event(self, confirmation, landing_xy, evidence, reason):
        kind = "dead_ball"
        if self.court_polygon is not None:
            inside = cv2.pointPolygonTest(self.court_polygon, landing_xy, False) >= 0
            kind = "in_court_landing" if inside else "out_of_bounds"
        return TerminalEvent(confirmation.frame_index, confirmation.timestamp_s, kind,
                             landing_xy, reason, evidence)

    def coalesce_rally_shots(self, raw_arcs):
        """Preserve observed segments; do not invent or merge racket contacts."""
        return list(raw_arcs)

    def segment_rallies(self, points: List[TrajectoryPoint], arcs: List[FlightArc],
                        video_cap: Optional[Any] = None) -> List[RallyEvent]:
        if not arcs:
            return []
        frames = [point.frame_index for point in points]
        if any(a >= b for a, b in zip(frames, frames[1:])):
            raise ValueError("points must be ordered by increasing frame")
        rallies = []
        current = []
        for index, arc in enumerate(arcs):
            current.append(arc)
            lo = bisect_left(frames, arc.end_frame)
            hi = bisect_left(frames, arcs[index + 1].start_frame) if index + 1 < len(arcs) else len(points)
            terminal = self.evaluate_dead_ball(points[lo:hi], arc)
            if terminal is None and index + 1 < len(arcs):
                continue
            # No time-gap-based split: unfinished footage remains unconfirmed.
            start = current[0]
            end_frame = terminal.frame_index if terminal else arc.end_frame
            end_time = terminal.timestamp_s if terminal else arc.end_time_s
            shots = self.coalesce_rally_shots(current)
            tactical = self._build_tactical_shots(len(rallies) + 1, shots, terminal, video_cap)
            rallies.append(RallyEvent(
                rally_id=len(rallies) + 1, start_frame=start.start_frame,
                end_frame=end_frame, start_time_s=start.start_time_s,
                end_time_s=end_time, duration_s=end_time - start.start_time_s,
                shot_count=None, is_valid_rally=None,
                terminal=terminal, shots=shots, tactical_shots=tactical,
                narrative=TacticalNarrativeEngine.generate_rally_narrative(
                    len(rallies) + 1, tactical,
                    {"terminal_type": terminal.terminal_type} if terminal else None),
                interruption_reason=None if terminal else "end_unconfirmed",
                start_signal="observed_motion", start_confirmed=False,
            ))
            current = []
        return rallies

    def _build_tactical_shots(self, rally_id, shots, terminal_event, video_cap=None):
        result = []
        for index, shot in enumerate(shots):
            hand = HandSide.UNKNOWN
            # Image flight direction does not establish hitter identity or orientation.
            # Only analyze a hand when a future player-evidence adapter supplies both.
            unknown_zone = CourtZone("unknown", DepthZone.UNKNOWN, LateralZone.UNKNOWN,
                                     HeightZone.UNKNOWN, "unknown")
            is_terminal = index == len(shots) - 1 and terminal_event is not None
            result.append(TacticalShot(
                shot_index=index + 1, hitter="unknown", dominant_hand=self.dominant_hand,
                hand_side=hand, shot_type="unknown", shot_type_cn="击球类型未知",
                start_time_s=shot.start_time_s, end_time_s=shot.end_time_s,
                duration_s=shot.end_time_s - shot.start_time_s,
                peak_speed_kmh=shot.peak_speed_kmh,
                start_xy=shot.start_xy, end_xy=shot.end_xy,
                start_zone=unknown_zone, target_zone=unknown_zone,
                flight_direction=shot.flight_direction, tactical_line=shot.tactical_line,
                is_terminal=is_terminal,
                terminal_result=terminal_event.terminal_type if is_terminal else None,
                description="Observed motion segment; racket contact and player attribution unconfirmed.",
            ))
        return result
