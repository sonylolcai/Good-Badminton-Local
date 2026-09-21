"""Service-neutral readers for frozen Good-Badminton analysis contracts."""

from .detections_reader import collect_track_position_evidence

__all__ = ["collect_track_position_evidence"]
