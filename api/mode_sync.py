"""Synchronize an external session request with one fixed GPU sport profile.

This module intentionally contains no model invocation, frame decoding, pose
logic, tracking logic, or business interpretation.  It is the only seam where
the deployment-selected sport and a client-selected *visual* session mode meet.
The resulting mapping is persisted with the stream session and is then consumed
as immutable visual input by :mod:`api.stream_runtime`.
"""

from __future__ import annotations

from typing import Mapping

from .vision_profiles import SportVisionProfile


class VisionModeSynchronizer:
    """Resolve request configuration without permitting runtime sport switches.

    A business service may request a mode by opaque name, but it cannot supply
    rule semantics, roster size, calibration geometry, or a different sport.
    Those values are derived from the profile selected by the process entry
    point and returned as visual-runtime configuration only.
    """

    def __init__(self, vision_profile: SportVisionProfile):
        self.vision_profile = vision_profile

    def synchronize(self, configuration: Mapping) -> dict:
        """Validate and return the profile-derived visual configuration."""
        normalized = dict(configuration)
        profile = self.vision_profile
        requested_sport = normalized.get("sport_id")
        if requested_sport is not None and requested_sport != profile.sport_id:
            raise ValueError(
                f"sport_id={requested_sport} does not match this {profile.sport_id} GPU process"
            )

        mode = profile.mode(normalized.get("session_mode"))
        expected = normalized.get("expected_player_count")
        if mode.expected_player_count is not None:
            if expected is not None and int(expected) != mode.expected_player_count:
                raise ValueError(
                    "expected_player_count conflicts with "
                    f"session_mode={mode.session_mode}; expected {mode.expected_player_count}"
                )
            expected = mode.expected_player_count
        elif expected is not None and int(expected) not in mode.allowed_player_counts:
            allowed = ", ".join(str(value) for value in mode.allowed_player_counts)
            raise ValueError(
                f"expected_player_count must be one of [{allowed}] for session_mode={mode.session_mode}"
            )

        maximum = normalized.get("max_roster_count")
        if mode.expected_player_count is not None:
            # The old v1 normalizer supplies a default maximum.  A fixed visual
            # mode owns that value so callers cannot turn a one-player training
            # session into a two-person roster by retaining an old default.
            maximum = mode.expected_player_count
        else:
            maximum = int(maximum or max(mode.allowed_player_counts))
            if maximum not in mode.allowed_player_counts:
                allowed = ", ".join(str(value) for value in mode.allowed_player_counts)
                raise ValueError(
                    f"max_roster_count must be one of [{allowed}] for session_mode={mode.session_mode}"
                )
            if expected is not None and int(expected) > maximum:
                raise ValueError("expected_player_count cannot exceed max_roster_count")

        detector = str(normalized.get("shuttle_detector") or "none")
        if detector not in profile.allowed_ball_detectors:
            allowed = ", ".join(profile.allowed_ball_detectors)
            raise ValueError(
                f"shuttle_detector must be one of [{allowed}] for sport_id={profile.sport_id}"
            )

        scope_id = str(
            normalized.get("calibration_scope") or mode.default_calibration_scope
        )
        scope = mode.calibration_scope(scope_id)
        normalized.update(
            {
                "sport_id": profile.sport_id,
                "session_mode": mode.session_mode,
                "calibration_scope": scope.scope_id,
                "court_dimensions_m": list(profile.court_dimensions_m),
                "calibration_world_points_m": [
                    list(point) for point in scope.world_points_m
                ],
                "athlete_observation_region": mode.athlete_observation_region,
                "athlete_observation_margin_m": float(
                    mode.athlete_observation_margin_m
                ),
                "athlete_observation_lateral_margin_m": float(
                    mode.athlete_observation_lateral_margin_m
                    if mode.athlete_observation_lateral_margin_m is not None
                    else mode.athlete_observation_margin_m
                ),
                "athlete_observation_baseline_margin_m": float(
                    mode.athlete_observation_baseline_margin_m
                    if mode.athlete_observation_baseline_margin_m is not None
                    else mode.athlete_observation_margin_m
                ),
                "expected_player_count": (
                    None if expected is None else int(expected)
                ),
                "max_roster_count": int(maximum),
            }
        )
        if mode.expected_player_count == 1:
            # Stable pose frames still protect bootstrap, but training does not
            # enter the legacy two/four-player discovery wait.
            normalized["roster_discovery_seconds"] = 0.0
        return normalized
