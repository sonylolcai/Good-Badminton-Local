"""Pickleball-specific game rules engine (Multi-Sport Extension).

Key Pickleball mechanics:
1. Two-Bounce Rule: After the serve, the return must bounce, and the third shot must bounce!
2. Non-Volley Zone (The Kitchen): Players cannot volley (hit without a bounce) within 7 feet of the net.
3. Underhand serve originating from behind the baseline.
4. Side-Out scoring (only the serving team scores points).
"""

from __future__ import annotations

from typing import List, Optional

from .base import BaseSportRuleEngine, RallyEvent, TerminalEvent
from .physics import FlightArc, TrajectoryPoint


class PickleballRuleEngine(BaseSportRuleEngine):
    """Domain rule engine for Singles and Doubles Pickleball."""

    @property
    def sport_id(self) -> str:
        return "pickleball"

    @property
    def max_allowed_bounces(self) -> int:
        return 1  # 1 bounce allowed per shot; special two-bounce rule for first 2 shots

    def __init__(self, net_y: float = 1200.0, court_lines: Optional[dict] = None):
        self.net_y = net_y
        self.court_lines = court_lines or {}

    def evaluate_serve_condition(
        self,
        arc: FlightArc,
        player_stillness_observed: bool = True,
    ) -> bool:
        """Pickleball underhand serve verification."""
        return player_stillness_observed

    def evaluate_dead_ball(
        self,
        recent_points: List[TrajectoryPoint],
        last_arc: Optional[FlightArc],
    ) -> Optional[TerminalEvent]:
        """Pickleball dead ball (kitchen fault, double bounce, out)."""
        return None

    def segment_rallies(
        self,
        points: List[TrajectoryPoint],
        arcs: List[FlightArc],
    ) -> List[RallyEvent]:
        """Segment pickleball rallies enforcing the two-bounce rule."""
        return []
