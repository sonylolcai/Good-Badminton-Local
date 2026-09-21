"""Business-owned longitudinal coaching artifacts.

The GPU service never reads or writes these profiles.  They combine a
post-match human identity confirmation with already-derived visual movement
evidence, so the coaching layer can compare a player to their own historical
baseline without treating an anonymous visual track as a person.
"""

from .longitudinal import (
    build_confirmed_session_observation,
    create_coach_profile,
    load_or_create_athlete_profile,
    profile_path_for_person,
    record_coach_followup_from_analysis,
    save_athlete_profile,
)

__all__ = [
    "build_confirmed_session_observation",
    "create_coach_profile",
    "load_or_create_athlete_profile",
    "profile_path_for_person",
    "record_coach_followup_from_analysis",
    "save_athlete_profile",
]
