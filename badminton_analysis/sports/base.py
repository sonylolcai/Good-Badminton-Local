"""Multi-Sport Rules Abstraction Interface.

Defines the contract for sport-specific rule engines (Badminton, Tennis, Pickleball).
Allows complete decoupling between visual physics and game rules / business scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .physics import FlightArc, TrajectoryPoint


@dataclass
class TerminalEvent:
    """Observed rally-ending signal, with no score attribution."""
    frame_index: int
    timestamp_s: float
    terminal_type: str  # "dead_ball", "in_court_landing", "out_of_bounds"
    landing_xy: Optional[Tuple[float, float]]
    reason: str
    evidence: str = "unknown"


@dataclass
class RallyEvent:
    """Observed motion group with independently confirmed start/end evidence."""
    rally_id: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    shot_count: Optional[int]
    is_valid_rally: Optional[bool]
    terminal: Optional[TerminalEvent] = None
    shots: List[FlightArc] = field(default_factory=list)
    tactical_shots: List[Any] = field(default_factory=list)
    narrative: str = ""
    interruption_reason: Optional[str] = None
    start_signal: str = "observed_motion"
    start_confirmed: bool = False


class BaseSportRuleEngine(ABC):
    """Abstract base class for all racquet/paddle sport rule engines."""

    @property
    @abstractmethod
    def sport_id(self) -> str:
        """Sport identifier e.g. 'badminton', 'tennis', 'pickleball'."""
        pass

    @property
    @abstractmethod
    def max_allowed_bounces(self) -> int:
        """Number of bounces allowed before a ball is dead (0 for badminton, 1 for tennis)."""
        pass

    @abstractmethod
    def evaluate_serve_condition(
        self,
        arc: FlightArc,
        player_stillness_observed: bool = False,
    ) -> Optional[bool]:
        """Determine whether a flight arc satisfies the sport's formal service rules."""
        pass

    @abstractmethod
    def evaluate_dead_ball(
        self,
        recent_points: List[TrajectoryPoint],
        last_arc: Optional[FlightArc],
    ) -> Optional[TerminalEvent]:
        """Determine whether the ball has officially died according to the sport's rules."""
        pass

    @abstractmethod
    def segment_rallies(
        self,
        points: List[TrajectoryPoint],
        arcs: List[FlightArc],
    ) -> List[RallyEvent]:
        """Segment the full match trajectory into distinct, rule-compliant rallies."""
        pass
