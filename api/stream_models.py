"""Frozen stream-session.v1 payload constants, validation and factories.

The authoritative wire format is the JSON Schema under
docs/contracts/stream_session_v1/.  This module mirrors that contract in plain
Python (matching api.jobs) so the API boundary needs no extra schema runtime.
Validation here is intentionally semantic: it protects durable state from bad
input and enforces the enums, id shapes and hash/length consistency the state
machine depends on, without re-implementing every schema keyword.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

SCHEMA_VERSION = "stream-session.v1"

SESSION_ID_PATTERN = re.compile(r"^ssn_[a-zA-Z0-9_-]{12,80}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
OPAQUE_ID_MAX = 160

SESSION_STATES = {
    "accepted", "queued", "running", "draining", "finalized", "partial",
    "failed", "cancelled", "interrupted_needs_rebuild",
}
TERMINAL_SESSION_STATES = {
    "finalized", "partial", "failed", "cancelled", "interrupted_needs_rebuild",
}
CURRENT_STAGES = {
    "awaiting_segments", "waiting_for_predecessor", "analyzing",
    "analyzing_with_errors", "draining_backlog", "draining_with_errors",
    "finalizing", "complete", "failed", "cancelled",
}
EVIDENCE_STATES = {
    "detected", "predicted", "missing", "derived", "candidate", "finalized",
}
EVENT_TYPES = {
    "person_observation", "roster_candidate_observation", "shuttle_observation", "ball_observation", "interaction_candidate",
    "session_status", "session_finalized",
}
PROCESSING_DISPOSITIONS = {
    "queued", "waiting_for_predecessor", "already_accepted",
}
TRACK_CANDIDATE_STATES = {
    "candidate", "active", "predicted", "missing", "closed", "reassociation_uncertain",
}
SEGMENT_CONTENT_TYPES = {"video/mp4", "video/iso.segment"}
ANALYSIS_SAMPLE_HZ = {10, 15, 30}
POSE_IMGSZ = {640, 960, 1280}
SHUTTLE_DETECTORS = {"none", "yolo", "tracknet_v3"}
TRACKER_BACKENDS = {"court_association", "bytetrack"}
ANALYSIS_MODE = "person_only"
MAX_SEGMENT_DURATION_SEC = 10.0
MAX_SEGMENT_BYTES = 256 * 1024 * 1024

# -- segment repair window -------------------------------------------------
# A repair grants the venue a bounded chance to deliver bytes for one segment
# index.  Every limit below is sized from the venue-side archive, not from the
# GPU: a window longer than the local rolling retention can only ever come back
# reporting the bytes are gone, and a request with too many indexes would let a
# single caller pin two generations of media for a large part of a session.
REPAIR_KINDS = {"missing", "corrupt"}
REPAIR_REASONS = {"business_gap_inventory", "gpu_integrity"}
REPAIR_TTL_DEFAULT_SECONDS = 21600
REPAIR_TTL_MIN_SECONDS = 60
REPAIR_TTL_MAX_SECONDS = 86400
REPAIR_MAX_INDEXES_PER_REQUEST = 64
REPAIR_GRANT_STATES = {"open", "fulfilled", "expired", "released"}


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp with the Z suffix used by examples."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def new_session_id() -> str:
    return f"ssn_{uuid.uuid4().hex}"


def _opaque_id(value, field):
    if not isinstance(value, str) or not 1 <= len(value) <= OPAQUE_ID_MAX:
        raise ValueError(f"{field} must be a 1-{OPAQUE_ID_MAX} character string")
    return value


def _bool(value, field):
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a JSON boolean")
    return value


def _int(value, field, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _number(value, field, minimum=None, exclusive_minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise ValueError(f"{field} must be greater than {exclusive_minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return float(value)


def _reject_extra_keys(value, allowed, field):
    """Mirror the frozen schemas' ``additionalProperties: false`` rule.

    Silently dropping unknown fields at this boundary is unsafe: it can make a
    business service believe that user identity or scoring data was accepted by
    the anonymous analysis service.  Rejecting it keeps the durable request and
    the caller's intent identical.
    """
    extras = set(value).difference(allowed)
    if extras:
        raise ValueError(f"{field} contains unsupported keys: {sorted(extras)}")


def _court_corners(value, field="court_corners"):
    """Validate four business-supplied image corners for one stream session."""
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must contain exactly four [x, y] points")
    normalized = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{field}[{index}] must be [x, y]")
        normalized.append([
            _number(point[0], f"{field}[{index}][0]"),
            _number(point[1], f"{field}[{index}][1]"),
        ])
    if len({tuple(point) for point in normalized}) != 4:
        raise ValueError(f"{field} must contain four distinct points")
    return normalized


def validate_create_request(body):
    """Validate and normalise a createSessionRequest body (raises ValueError)."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    _reject_extra_keys(
        body,
        {
            "schema_version",
            "camera_id",
            "calibration_id",
            "court_corners",
            "analysis_mode",
            "client_reference",
            "configuration",
        },
        "request body",
    )
    if body.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be stream-session.v1")
    if body.get("analysis_mode") != ANALYSIS_MODE:
        raise ValueError(f"analysis_mode must be {ANALYSIS_MODE}")
    camera_id = _opaque_id(body.get("camera_id"), "camera_id")
    calibration_id = _opaque_id(body.get("calibration_id"), "calibration_id")
    court_corners = _court_corners(body.get("court_corners"))
    client_reference = body.get("client_reference")
    if client_reference is not None:
        client_reference = _opaque_id(client_reference, "client_reference")
    configuration = validate_configuration(body.get("configuration"))
    return {
        "schema_version": SCHEMA_VERSION,
        "camera_id": camera_id,
        "calibration_id": calibration_id,
        "court_corners": court_corners,
        "analysis_mode": ANALYSIS_MODE,
        "client_reference": client_reference,
        "configuration": configuration,
    }


def validate_configuration(configuration):
    """Validate terminal analysis settings independently from calibration.

    A signed terminal may use these settings while requesting a short preview
    before an operator has saved the court corners.  That preview is never an
    anonymous or GPU-analysis request.
    """
    if not isinstance(configuration, dict):
        raise ValueError("configuration must be a JSON object")
    required = {"analysis_sample_hz", "pose_imgsz", "shuttle_detector", "generate_annotated_video"}
    missing = required.difference(configuration)
    if missing:
        raise ValueError(f"configuration is missing required keys: {sorted(missing)}")
    allowed = required | {
        "preserve_audio",
        "court_health_check_hz",
        "tracknet_overlap_frames",
        "tracker_backend",
        "lock_match_roster",
        "roster_stable_frames",
        "max_roster_count",
        "expected_player_count",
        "roster_discovery_seconds",
        "match_mode",
        "far_player_enhancement",
        "far_pose_roi",
        # These fields describe an anonymous visual mode.  They never carry
        # business identity, scoring or participant claims.  A fixed process
        # profile validates their allowed values and derives the real roster.
        "sport_id",
        "session_mode",
        "calibration_scope",
    }
    extras = set(configuration).difference(allowed)
    if extras:
        raise ValueError(f"configuration contains unsupported keys: {sorted(extras)}")
    sample_hz = configuration["analysis_sample_hz"]
    if isinstance(sample_hz, bool) or sample_hz not in ANALYSIS_SAMPLE_HZ:
        raise ValueError("analysis_sample_hz must be 10, 15, or 30")
    pose_imgsz = configuration["pose_imgsz"]
    if isinstance(pose_imgsz, bool) or pose_imgsz not in POSE_IMGSZ:
        raise ValueError("pose_imgsz must be 640, 960, or 1280")
    shuttle_detector = configuration["shuttle_detector"]
    if shuttle_detector not in SHUTTLE_DETECTORS:
        raise ValueError("shuttle_detector must be none, yolo, or tracknet_v3")
    generate_annotated_video = _bool(
        configuration["generate_annotated_video"], "generate_annotated_video"
    )
    normalized = {
        "analysis_sample_hz": int(sample_hz),
        "pose_imgsz": int(pose_imgsz),
        "shuttle_detector": shuttle_detector,
        "generate_annotated_video": generate_annotated_video,
        # The shared contract carries roster inputs but does not decide which
        # sport/mode may use them.  The process profile owns that decision.
        "tracker_backend": configuration.get("tracker_backend", "court_association"),
        "lock_match_roster": configuration.get("lock_match_roster", True),
        "roster_stable_frames": configuration.get("roster_stable_frames", 3),
        # Generic validation only protects the tracker from impossible counts.
        # A SportVisionProfile later determines which count belongs to the
        # fixed process and selected visual session mode.
        "max_roster_count": configuration.get("max_roster_count", 4),
        # The shared tracker can represent a one-person training roster.  The
        # per-sport profile rejects counts its mode does not support.
        "expected_player_count": configuration.get("expected_player_count"),
        "roster_discovery_seconds": configuration.get("roster_discovery_seconds", 3.0),
        "match_mode": configuration.get("match_mode", "person_only"),
        # Fixed-camera far-half inference is an optional detection aid, not a
        # business identity rule.  It is opt-in because it changes the actual
        # inference plan; a camera profile may provide a tighter ROI.
        "far_player_enhancement": configuration.get("far_player_enhancement", False),
    }
    if normalized["tracker_backend"] not in TRACKER_BACKENDS:
        raise ValueError("tracker_backend must be court_association or bytetrack")
    normalized["lock_match_roster"] = _bool(
        normalized["lock_match_roster"], "lock_match_roster"
    )
    if normalized["match_mode"] not in {"auto", "singles", "doubles", "person_only"}:
        raise ValueError("match_mode must be auto, singles, doubles, or person_only")
    stable_frames = normalized["roster_stable_frames"]
    if isinstance(stable_frames, bool) or not 1 <= stable_frames <= 10:
        raise ValueError("roster_stable_frames must be an integer from 1 to 10")
    normalized["roster_stable_frames"] = int(stable_frames)
    max_roster_count = normalized["max_roster_count"]
    if isinstance(max_roster_count, bool) or not 1 <= max_roster_count <= 4:
        raise ValueError("max_roster_count must be an integer from 1 to 4")
    normalized["max_roster_count"] = int(max_roster_count)
    expected_player_count = normalized["expected_player_count"]
    if expected_player_count is not None:
        if isinstance(expected_player_count, bool) or expected_player_count not in {1, 2, 4}:
            raise ValueError("expected_player_count must be 1, 2, 4, or null")
        normalized["expected_player_count"] = int(expected_player_count)
    discovery_seconds = normalized["roster_discovery_seconds"]
    if isinstance(discovery_seconds, bool) or not 1.0 <= float(discovery_seconds) <= 15.0:
        raise ValueError("roster_discovery_seconds must be from 1 to 15")
    normalized["roster_discovery_seconds"] = float(discovery_seconds)
    normalized["far_player_enhancement"] = _bool(
        normalized["far_player_enhancement"], "far_player_enhancement"
    )
    if "far_pose_roi" in configuration:
        roi = configuration["far_pose_roi"]
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError("far_pose_roi must be [x1, y1, x2, y2]")
        values = [float(value) for value in roi]
        if not all(0.0 <= value <= 1.0 for value in values) or values[0] >= values[2] or values[1] >= values[3]:
            raise ValueError("far_pose_roi must be an ordered normalized rectangle")
        normalized["far_pose_roi"] = values
    for field in ("sport_id", "session_mode", "calibration_scope"):
        if field in configuration:
            normalized[field] = _opaque_id(configuration[field], field)
    if "preserve_audio" in configuration:
        normalized["preserve_audio"] = _bool(configuration["preserve_audio"], "preserve_audio")
    if "court_health_check_hz" in configuration:
        normalized["court_health_check_hz"] = _number(
            configuration["court_health_check_hz"],
            "court_health_check_hz",
            exclusive_minimum=0.0,
            maximum=2.0,
        )
    if "tracknet_overlap_frames" in configuration:
        overlap = configuration["tracknet_overlap_frames"]
        if isinstance(overlap, bool) or overlap != 7:
            raise ValueError("tracknet_overlap_frames must be 7 when present")
        normalized["tracknet_overlap_frames"] = 7
    return normalized


def validate_segment_metadata(body):
    """Validate and normalise segmentMetadata (raises ValueError)."""
    if not isinstance(body, dict):
        raise ValueError("segment metadata must be a JSON object")
    _reject_extra_keys(
        body,
        {
            "schema_version",
            "segment_index",
            "source_start_time_sec",
            "duration_sec",
            "sha256",
            "idempotency_key",
            "content_type",
            "content_length_bytes",
            "court_corners",
            "source_frame_start_index",
            "source_frame_count",
            "source_frame_time_offsets_sec",
            "repair_request_id",
        },
        "segment metadata",
    )
    if body.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be stream-session.v1")
    segment_index = _int(body.get("segment_index"), "segment_index", minimum=0)
    source_start_time_sec = _number(
        body.get("source_start_time_sec"), "source_start_time_sec", minimum=0.0
    )
    duration_sec = _number(
        body.get("duration_sec"),
        "duration_sec",
        exclusive_minimum=0.0,
        maximum=MAX_SEGMENT_DURATION_SEC,
    )
    sha256 = body.get("sha256")
    if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
        raise ValueError("sha256 must be a 64-character lowercase hex digest")
    idempotency_key = _opaque_id(body.get("idempotency_key"), "idempotency_key")
    content_type = body.get("content_type")
    if content_type not in SEGMENT_CONTENT_TYPES:
        raise ValueError("content_type must be video/mp4 or video/iso.segment")
    content_length_bytes = _int(
        body.get("content_length_bytes"), "content_length_bytes", minimum=1
    )
    raw_corners = body.get("court_corners")
    # The session already owns the calibrated corners. A per-segment copy stays optional
    # so a stale venue value cannot permanently reject every later segment of a match.
    court_corners = None if raw_corners is None else _court_corners(raw_corners)
    source_frame_start_index = body.get("source_frame_start_index")
    source_frame_count = body.get("source_frame_count")
    if (source_frame_start_index is None) != (source_frame_count is None):
        raise ValueError("source_frame_start_index and source_frame_count must be provided together")
    if source_frame_start_index is not None:
        source_frame_start_index = _int(source_frame_start_index, "source_frame_start_index", minimum=0)
        source_frame_count = _int(source_frame_count, "source_frame_count", minimum=1)
    offsets = body.get("source_frame_time_offsets_sec")
    if offsets is not None:
        if source_frame_count is None or not isinstance(offsets, list) or len(offsets) != source_frame_count:
            raise ValueError("source_frame_time_offsets_sec must match source_frame_count")
        offsets = [_number(value, "source frame time offset", minimum=0) for value in offsets]
        if offsets[0] != 0 or offsets[-1] >= duration_sec or any(
            later <= earlier for earlier, later in zip(offsets, offsets[1:])
        ):
            raise ValueError("source frame time offsets must start at zero and increase within duration_sec")
    raw_repair_request_id = body.get("repair_request_id")
    # Present only on the repair data plane; the ordinary path never carries it.
    repair_request_id = (
        None if raw_repair_request_id is None
        else _opaque_id(raw_repair_request_id, "repair_request_id")
    )
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "segment_index": segment_index,
        "source_start_time_sec": source_start_time_sec,
        "duration_sec": duration_sec,
        "sha256": sha256,
        "idempotency_key": idempotency_key,
        "content_type": content_type,
        "content_length_bytes": content_length_bytes,
        "court_corners": court_corners,
        "repair_request_id": repair_request_id,
    }
    if source_frame_start_index is not None:
        normalized["source_frame_start_index"] = source_frame_start_index
        normalized["source_frame_count"] = source_frame_count
        if offsets is not None:
            normalized["source_frame_time_offsets_sec"] = offsets
    return normalized


def validate_gap_declaration(body):
    """Validate an explicit capture-side declaration that media was lost."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    _reject_extra_keys(body, {"schema_version", "source_start_time_sec", "duration_sec", "reason"}, "gap declaration")
    if body.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be stream-session.v1")
    if body.get("reason") != "source_segment_missing":
        raise ValueError("gap reason must be source_segment_missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_start_time_sec": _number(body.get("source_start_time_sec"), "source_start_time_sec", minimum=0),
        "duration_sec": _number(body.get("duration_sec"), "duration_sec", exclusive_minimum=0, maximum=MAX_SEGMENT_DURATION_SEC),
        "reason": "source_segment_missing",
    }


def validate_complete_request(body):
    """Validate and normalise completeSessionRequest (raises ValueError)."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    _reject_extra_keys(
        body,
        {"schema_version", "expected_last_segment_index", "allow_partial"},
        "request body",
    )
    if body.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be stream-session.v1")
    expected_last_segment_index = _int(
        body.get("expected_last_segment_index"), "expected_last_segment_index", minimum=0
    )
    allow_partial = _bool(body.get("allow_partial"), "allow_partial")
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_last_segment_index": expected_last_segment_index,
        "allow_partial": allow_partial,
    }


def validate_repair_request(body):
    """Validate a repair window request (raises ValueError).

    ``kind`` is per index because a lost fragment and a corrupt fragment are
    authorized differently: only the GPU's own integrity verdict may order a
    replacement of media that was already accepted.
    """
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    _reject_extra_keys(
        body,
        {"schema_version", "reason", "ttl_seconds", "ranges"},
        "repair request",
    )
    if body.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be stream-session.v1")
    reason = body.get("reason")
    if reason not in REPAIR_REASONS:
        raise ValueError("reason must be business_gap_inventory or gpu_integrity")
    ttl_seconds = _int(
        body.get("ttl_seconds", REPAIR_TTL_DEFAULT_SECONDS),
        "ttl_seconds",
        minimum=REPAIR_TTL_MIN_SECONDS,
    )
    if ttl_seconds > REPAIR_TTL_MAX_SECONDS:
        raise ValueError(
            "ttl_seconds must not exceed "
            f"{REPAIR_TTL_MAX_SECONDS} because it cannot outlast venue local archive retention"
        )
    ranges = body.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("ranges must be a non-empty list")
    if len(ranges) > REPAIR_MAX_INDEXES_PER_REQUEST:
        raise ValueError(f"ranges must not exceed {REPAIR_MAX_INDEXES_PER_REQUEST} segment indexes")
    normalized = []
    seen = set()
    for item in ranges:
        if not isinstance(item, dict):
            raise ValueError("each repair range must be a JSON object")
        _reject_extra_keys(item, {"segment_index", "kind"}, "repair range")
        segment_index = _int(item.get("segment_index"), "segment_index", minimum=0)
        kind = item.get("kind")
        if kind not in REPAIR_KINDS:
            raise ValueError("kind must be missing or corrupt")
        if segment_index in seen:
            raise ValueError(f"segment_index {segment_index} is listed more than once")
        seen.add(segment_index)
        normalized.append({"segment_index": segment_index, "kind": kind})
    normalized.sort(key=lambda item: item["segment_index"])
    return {
        "schema_version": SCHEMA_VERSION,
        "reason": reason,
        "ttl_seconds": ttl_seconds,
        "ranges": normalized,
    }


def repair_ranges_view(request, *, reveal_tokens=False):
    """Return the granted indexes of one repair request, ordered and safe to log."""
    view = []
    for grant in sorted(
        (request.get("grants") or {}).values(),
        key=lambda item: int(item["segment_index"]),
    ):
        entry = {
            "segment_index": int(grant["segment_index"]),
            "kind": grant["kind"],
            "state": grant["state"],
            "expected": dict(grant.get("expected") or {}),
            "grant_expires_at": grant["grant_expires_at"],
        }
        if grant.get("supersede_sha256"):
            entry["supersede_sha256"] = grant["supersede_sha256"]
        if grant.get("generation") is not None:
            entry["generation"] = int(grant["generation"])
        if grant.get("resolved_at"):
            entry["resolved_at"] = grant["resolved_at"]
        if reveal_tokens and grant["state"] == "open":
            entry["grant_token"] = grant.get("grant_token")
        else:
            entry["grant_token_issued"] = bool(grant.get("token_sha256"))
        view.append(entry)
    return view


def repair_window_response(session_id, request, *, reused=False):
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_session_id": session_id,
        "request_id": request["request_id"],
        "status": request["status"],
        "reason": request["reason"],
        "ttl_seconds": int(request["ttl_seconds"]),
        "granted_at": request["granted_at"],
        "window_expires_at": request["expires_at"],
        "reused": bool(reused),
        "ranges": repair_ranges_view(request, reveal_tokens=True),
    }


def repair_inventory_response(session_id, requests, *, window_open, window_expires_at):
    views = []
    for request in sorted(
        requests.values(), key=lambda item: (item.get("granted_at") or "", item["request_id"])
    ):
        summary = {}
        for grant in (request.get("grants") or {}).values():
            summary[grant["state"]] = summary.get(grant["state"], 0) + 1
        views.append({
            "request_id": request["request_id"],
            "status": request["status"],
            "reason": request["reason"],
            "granted_at": request["granted_at"],
            "window_expires_at": request["expires_at"],
            "summary": dict(sorted(summary.items())),
            "ranges": repair_ranges_view(request),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_session_id": session_id,
        "window_open": bool(window_open),
        "window_expires_at": window_expires_at,
        "requests": views,
    }


def repair_abandon_response(session_id, request, released_indexes):
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_session_id": session_id,
        "request_id": request["request_id"],
        "status": request["status"],
        "released_segment_indexes": sorted(int(index) for index in released_indexes),
        "window_expires_at": request["expires_at"],
    }


def create_session_response(session, status="accepted"):
    session_id = session["analysis_session_id"]
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_session_id": session_id,
        "status": status,
        "accepted_at": session["created_at"],
        "status_url": f"/api/v1/stream-sessions/{session_id}",
        "segments_url": f"/api/v1/stream-sessions/{session_id}/segments/{{segment_index}}",
        "complete_url": f"/api/v1/stream-sessions/{session_id}/complete",
    }


def segment_receipt_response(session, segment_index, receipt):
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_session_id": session["analysis_session_id"],
        "segment_index": int(segment_index),
        "session_status": session["status"],
        "receipt": receipt,
        "status_url": f"/api/v1/stream-sessions/{session['analysis_session_id']}",
    }


def session_status_response(
    session,
    progress,
    track_candidates,
    artifacts,
    current_stage,
    error=None,
    processing_errors=None,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_session_id": session["analysis_session_id"],
        "status": session["status"],
        "current_stage": current_stage,
        "updated_at": session["updated_at"],
        "sealed": bool(session.get("sealed", False)),
        "progress": progress,
        "track_candidates": track_candidates,
        "error": error,
        # Segment failures are non-terminal evidence gaps.  Keep them distinct
        # from ``error`` so clients do not confuse a partial match with a
        # completely failed session.
        "processing_errors": list(processing_errors or ()),
        "artifacts": artifacts,
        # A repair rewrites evidence inside an already-consumed time range, and the
        # event log is ordered by evidence time with a positional cursor.  That cursor
        # cannot surface the inserted rows, so the log carries a monotonic revision:
        # when it changes, consumers must re-pull from the beginning and de-duplicate
        # by ``event_id`` (which the writer already does).
        "events_revision": int(session.get("events_revision", 1)),
    }


def events_response(session, events, next_cursor):
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_session_id": session["analysis_session_id"],
        "events": events,
        "next_cursor": next_cursor,
        "events_revision": int(session.get("events_revision", 1)),
    }


def complete_session_response(
    session,
    status,
    sealed_at,
    missing_segment_indexes,
    failed_segment_indexes=None,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_session_id": session["analysis_session_id"],
        "status": status,
        "sealed_at": sealed_at,
        "missing_segment_indexes": missing_segment_indexes,
        "failed_segment_indexes": list(failed_segment_indexes or ()),
        "status_url": f"/api/v1/stream-sessions/{session['analysis_session_id']}",
    }


def error_response(code, message, retryable, details=None, request_id=None, analysis_session_id=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id or new_request_id(),
        "analysis_session_id": analysis_session_id,
        "error": {
            "code": code,
            "message": message,
            "retryable": bool(retryable),
            "details": details or {},
        },
    }
