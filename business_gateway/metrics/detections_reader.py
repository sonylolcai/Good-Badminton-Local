"""Compatibility import for the service-neutral detections reader."""

from good_badminton_contracts.detections_reader import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_IDENTITY_CONFIDENCE,
    DEFAULT_MIN_LOCATION_CONFIDENCE,
    collect_track_position_evidence,
)

__all__ = [
    "COURT_WIDTH_M",
    "COURT_LENGTH_M",
    "DEFAULT_MIN_DETECTION_CONFIDENCE",
    "DEFAULT_MIN_LOCATION_CONFIDENCE",
    "DEFAULT_MIN_IDENTITY_CONFIDENCE",
    "collect_track_position_evidence",
]
