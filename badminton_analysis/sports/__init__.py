"""Multi-Sport Rules & Physics Layer.

Provides clean decoupling between raw visual detection, Newtonian trajectory kinematics,
and sport-specific rule engines (Badminton, Tennis, Pickleball).
"""

from .base import BaseSportRuleEngine, RallyEvent, TerminalEvent
from .badminton import BadmintonRuleEngine
from .physics import FlightArc, PhysicalTrajectoryAnalyzer, TrajectoryPoint
from .pickleball import PickleballRuleEngine
from .tennis import TennisRuleEngine

__all__ = [
    "BaseSportRuleEngine",
    "BadmintonRuleEngine",
    "TennisRuleEngine",
    "PickleballRuleEngine",
    "PhysicalTrajectoryAnalyzer",
    "FlightArc",
    "TrajectoryPoint",
    "RallyEvent",
    "TerminalEvent",
]
