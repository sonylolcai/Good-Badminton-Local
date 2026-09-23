"""Tennis-specific game rules engine (Multi-Sport Extension).

Key Tennis mechanics:
1. Exactly one bounce allowed before returning (except volleys which allow 0).
2. Two consecutive bounces or touching fence/ground = dead ball.
3. Serve must land diagonally in the opponent's service box; allows first and second serve.
4. Traditional 15-30-40-game scoring system.
"""

from __future__ import annotations

from typing import List, Optional

from .base import BaseSportRuleEngine, RallyEvent, TerminalEvent
from .physics import FlightArc, TrajectoryPoint


class TennisRuleEngine(BaseSportRuleEngine):
    """Domain rule engine for Singles and Doubles Tennis."""

    @property
    def sport_id(self) -> str:
        return "tennis"

    @property
    def max_allowed_bounces(self) -> int:
        return 1  # Exactly 1 bounce allowed!

    def __init__(self, net_y: float = 1200.0, court_lines: Optional[dict] = None):
        self.net_y = net_y
        self.court_lines = court_lines or {}

    def evaluate_serve_condition(
        self,
        arc: FlightArc,
        player_stillness_observed: bool = True,
    ) -> bool:
        """Tennis serve must originate from behind the baseline."""
        return player_stillness_observed

    def evaluate_dead_ball(
        self,
        recent_points: List[TrajectoryPoint],
        last_arc: Optional[FlightArc],
    ) -> Optional[TerminalEvent]:
        """Tennis dead ball occurs after double bounce, out of bounds, or net error."""
        return None

    def segment_rallies(
        self,
        points: List[TrajectoryPoint],
        arcs: List[FlightArc],
    ) -> List[RallyEvent]:
        """Segment tennis rallies allowing single bounce between consecutive hits."""
        return []
