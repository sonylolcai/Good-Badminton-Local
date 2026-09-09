"""Fixed sport and session-mode profiles for the GPU vision runtime.

The process entry point selects exactly one :class:`SportVisionProfile` at
startup.  Requests may select only a session mode exposed by that profile; they
cannot switch the process to another sport.  This keeps sport-specific court
geometry and roster rules behind one small interface while the streaming,
pose, tracking and checkpoint implementation remains shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


Point = Tuple[float, float]

FULL_COURT = "full_court"
NEAR_HALF_COURT = "near_half_court"
FULL_COURT_ATHLETES = "full_court_athletes"
NEAR_COURT_ATHLETE = "near_court_athlete"


@dataclass(frozen=True)
class CalibrationScopeProfile:
    """Map four ordered image points to four known world-coordinate points."""

    scope_id: str
    world_points_m: Tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class SessionModeProfile:
    """Visual constraints that differ between modes of the same sport."""

    session_mode: str
    allowed_player_counts: Tuple[int, ...]
    expected_player_count: Optional[int]
    calibration_scopes: Tuple[CalibrationScopeProfile, ...]
    default_calibration_scope: str
    athlete_observation_region: str
    athlete_observation_margin_m: float

    def calibration_scope(self, scope_id: str) -> CalibrationScopeProfile:
        for scope in self.calibration_scopes:
            if scope.scope_id == scope_id:
                return scope
        allowed = ", ".join(scope.scope_id for scope in self.calibration_scopes)
        raise ValueError(
            f"calibration_scope must be one of [{allowed}] for session_mode={self.session_mode}"
        )


@dataclass(frozen=True)
class SportVisionProfile:
    """The fixed sport identity and its allowed GPU vision modes."""

    sport_id: str
    service_name: str
    coordinate_system_id: str
    court_dimensions_m: Point
    session_modes: Tuple[SessionModeProfile, ...]
    default_session_mode: Optional[str]
    allowed_ball_detectors: Tuple[str, ...]

    @property
    def supported_session_modes(self) -> Tuple[str, ...]:
        return tuple(mode.session_mode for mode in self.session_modes)

    def mode(self, session_mode: Optional[str]) -> SessionModeProfile:
        requested = session_mode or self.default_session_mode
        if not requested:
            allowed = ", ".join(self.supported_session_modes)
            raise ValueError(
                f"session_mode is required for sport_id={self.sport_id}; allowed values: [{allowed}]"
            )
        for mode in self.session_modes:
            if mode.session_mode == requested:
                return mode
        allowed = ", ".join(self.supported_session_modes)
        raise ValueError(
            f"session_mode must be one of [{allowed}] for sport_id={self.sport_id}"
        )

def _full_court_scope(width: float, length: float) -> CalibrationScopeProfile:
    return CalibrationScopeProfile(
        FULL_COURT,
        ((0.0, 0.0), (width, 0.0), (width, length), (0.0, length)),
    )


BADMINTON_WIDTH_M = 6.10
BADMINTON_LENGTH_M = 13.40
TENNIS_SINGLES_WIDTH_M = 8.23
TENNIS_COURT_LENGTH_M = 23.77

BADMINTON_PROFILE = SportVisionProfile(
    sport_id="badminton",
    service_name="good-badminton-gpu-api",
    coordinate_system_id="standard_badminton_court_m",
    court_dimensions_m=(BADMINTON_WIDTH_M, BADMINTON_LENGTH_M),
    session_modes=(
        SessionModeProfile(
            session_mode="match",
            allowed_player_counts=(2, 4),
            expected_player_count=None,
            calibration_scopes=(
                _full_court_scope(BADMINTON_WIDTH_M, BADMINTON_LENGTH_M),
            ),
            default_calibration_scope=FULL_COURT,
            athlete_observation_region=FULL_COURT_ATHLETES,
            athlete_observation_margin_m=0.35,
        ),
    ),
    default_session_mode="match",
    allowed_ball_detectors=("none", "yolo", "tracknet_v3"),
)

TENNIS_PROFILE = SportVisionProfile(
    sport_id="tennis",
    service_name="good-tennis-gpu-api",
    coordinate_system_id="tennis_singles_court_m_v1",
    court_dimensions_m=(TENNIS_SINGLES_WIDTH_M, TENNIS_COURT_LENGTH_M),
    session_modes=(
        SessionModeProfile(
            session_mode="singles_match",
            allowed_player_counts=(2,),
            expected_player_count=2,
            calibration_scopes=(
                _full_court_scope(TENNIS_SINGLES_WIDTH_M, TENNIS_COURT_LENGTH_M),
            ),
            default_calibration_scope=FULL_COURT,
            athlete_observation_region=FULL_COURT_ATHLETES,
            athlete_observation_margin_m=3.0,
        ),
        SessionModeProfile(
            session_mode="single_player_training",
            allowed_player_counts=(1,),
            expected_player_count=1,
            calibration_scopes=(
                _full_court_scope(TENNIS_SINGLES_WIDTH_M, TENNIS_COURT_LENGTH_M),
                CalibrationScopeProfile(
                    NEAR_HALF_COURT,
                    (
                        (0.0, TENNIS_COURT_LENGTH_M / 2.0),
                        (TENNIS_SINGLES_WIDTH_M, TENNIS_COURT_LENGTH_M / 2.0),
                        (TENNIS_SINGLES_WIDTH_M, TENNIS_COURT_LENGTH_M),
                        (0.0, TENNIS_COURT_LENGTH_M),
                    ),
                ),
            ),
            default_calibration_scope=NEAR_HALF_COURT,
            athlete_observation_region=NEAR_COURT_ATHLETE,
            athlete_observation_margin_m=3.0,
        ),
    ),
    # Tennis must be explicit because silently treating a training video as a
    # two-player match leaves the roster waiting forever.
    default_session_mode=None,
    # ``shuttle_detector`` is the stable stream-session.v1 wire field.  In a
    # tennis process its ``yolo`` value means the tennis-only model adapter;
    # it never selects the badminton shuttle checkpoint.
    allowed_ball_detectors=("none", "yolo"),
)


_PROFILES = {
    BADMINTON_PROFILE.sport_id: BADMINTON_PROFILE,
    TENNIS_PROFILE.sport_id: TENNIS_PROFILE,
}


def get_vision_profile(sport_id: str) -> SportVisionProfile:
    try:
        return _PROFILES[str(sport_id).strip().lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(_PROFILES))
        raise ValueError(f"GPU sport profile must be one of [{allowed}]") from exc
