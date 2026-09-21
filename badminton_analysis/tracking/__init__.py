
"""Tracking adapters and policies.

Exports are lazy because ``person_only`` builds on the court tracker, while the
court tracker imports the ByteTrack adapter from this package.  Eager imports
would create a circular dependency for existing fixed-camera callers.
"""

__all__ = ["ByteTrackAdapter", "PersonOnlyFrameProcessor", "PersonOnlyTracker"]


def __getattr__(name):
    if name == "ByteTrackAdapter":
        from .bytetrack_adapter import ByteTrackAdapter

        return ByteTrackAdapter
    if name in {"PersonOnlyFrameProcessor", "PersonOnlyTracker"}:
        from .person_only import PersonOnlyFrameProcessor, PersonOnlyTracker

        return {
            "PersonOnlyFrameProcessor": PersonOnlyFrameProcessor,
            "PersonOnlyTracker": PersonOnlyTracker,
        }[name]
    raise AttributeError(name)
