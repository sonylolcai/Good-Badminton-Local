"""Business-issued media identity and deterministic, credential-free object keys."""
from datetime import date
import re
from uuid import UUID

FIELDS = {"schema_version", "media_id", "tenant_id", "venue_id", "court_id", "camera_id",
          "edge_device_id", "calibration_id", "edge_session_id", "match_id", "recording_session_id",
          "environment", "venue_slug", "capture_date", "daily_sequence"}
UUID_FIELDS = {"media_id", "tenant_id", "venue_id", "court_id", "camera_id", "edge_device_id",
               "calibration_id", "edge_session_id", "match_id", "recording_session_id"}


def validate_media_identity(value, *, camera_id=None, client_reference=None, calibration_id=None):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("media identity fields do not match media-identity.v1")
    result = dict(value)
    if result["schema_version"] != "media-identity.v1" or result["environment"] not in {"test", "production"}:
        raise ValueError("unsupported media identity version or environment")
    for field in UUID_FIELDS:
        item = result[field]
        if item is None and field in {"match_id", "recording_session_id"}:
            continue
        if not isinstance(item, str) or str(UUID(item)) != item:
            raise ValueError("media identity requires canonical UUIDs")
    slug = result["venue_slug"]
    if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", slug):
        raise ValueError("invalid venue storage slug")
    day = result["capture_date"]
    if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
        raise ValueError("capture date must be YYYY-MM-DD")
    number = result["daily_sequence"]
    if type(number) is not int or not 1 <= number <= 999999:
        raise ValueError("daily sequence out of range")
    for supplied, field in ((camera_id, "camera_id"), (client_reference, "edge_session_id"),
                            (calibration_id, "calibration_id")):
        if supplied is not None and supplied != result[field]:
            raise ValueError("media identity does not match its stream session")
    return result


def media_prefix(identity):
    identity = validate_media_identity(identity)
    root = "testVideo" if identity["environment"] == "test" else "productionVideo"
    return (f"{root}/{identity['venue_slug']}/{identity['camera_id']}/{identity['capture_date']}/"
            f"{identity['daily_sequence']:06d}_{identity['media_id']}")
