"""Stream movement & candidate shot-route derivation for a completed anonymous session.

Shared across GPU-host CPU (``good_badminton_visual``) and business-host CPU
(``business_gateway``) so both compute routes execute the exact same adapter
and pure-calculation contracts without touching raw video or user identities.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .metrics.movement import generate_movement_metrics
from .metrics.stream_candidate_routes import (
    SOURCE as CANDIDATE_ROUTE_SOURCE,
    derive_stream_candidate_shot_routes,
)
from .result_contract import ResultContractError, finite, validate_page


_TERMINAL_WITH_EVIDENCE = {"finalized", "partial"}
_CANDIDATE_SHUTTLE_DETECTORS = {"tracknet_v3", "yolo"}


def derive_stream_movement_metrics(
    output_dir: Path | str,
    *,
    client,
    terminal_status: Mapping[str, Any],
    create_request: Mapping[str, Any],
    on_event_page=None,
) -> dict[str, Any]:
    """Fetch all event pages and derive auditable visual movement & route metrics."""

    status = str(terminal_status.get("status") or "")
    if status not in _TERMINAL_WITH_EVIDENCE:
        raise ValueError(
            "stream movement derivation requires finalized or partial session evidence"
        )

    output_dir = Path(output_dir)
    evidence_dir = output_dir / "evidence"
    derived_dir = output_dir / "derived"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    expected = getattr(getattr(client, "ledger", None), "analysis_session_id", None) or terminal_status.get(
        "analysis_session_id"
    )
    if not expected or terminal_status.get("analysis_session_id") != expected:
        raise ResultContractError("result_session_mismatch")
    started = time.monotonic()
    events = _read_all_events(client, on_event_page=on_event_page, expected_session_id=expected)
    timings = {"event_download_and_persistence_ms": (time.monotonic() - started) * 1000}
    materialize_started = time.monotonic()
    raw_events_path = evidence_dir / "stream_events.jsonl"
    _write_jsonl(raw_events_path, events)

    configuration = dict(create_request.get("configuration") or {})
    anchor_reference = None
    if configuration.get("roster_identity_policy") == "anchored_roster.v1":
        anchor_reference = _validate_anchored_roster(events, configuration.get("expected_player_count"))
    sample_hz = int(configuration.get("analysis_sample_hz") or 10)
    records = _materialize_detection_records(events, sample_hz=sample_hz)
    detections_path = evidence_dir / "stream_person_detections.jsonl"
    _write_jsonl(detections_path, records)

    duration_sec = float(
        ((terminal_status.get("progress") or {}).get("processed_source_time_sec"))
        or max((float(event.get("source_time_sec") or 0.0) for event in events), default=0.0)
    )
    metadata_path = evidence_dir / "stream_metadata.json"
    _write_json(
        metadata_path,
        {
            "schema_version": "business-stream-evidence.v1",
            "analysis_session_id": terminal_status.get("analysis_session_id"),
            "video": {"fps": sample_hz, "duration_sec": round(duration_sec, 6)},
            "analysis_mode": "person_only",
            "configuration": {
                key: configuration.get(key)
                for key in (
                    "analysis_sample_hz",
                    "pose_imgsz",
                    "shuttle_detector",
                    "tracker_backend",
                    "lock_match_roster",
                    "roster_stable_frames",
                    "expected_player_count",
                )
            },
            "source": "GPU stream-session.v1 events",
        },
    )
    spatial_summary_path = evidence_dir / "stream_spatial_match_summary.json"
    _write_json(spatial_summary_path, _spatial_summary(records))

    if configuration.get("roster_identity_policy") == "half_court_slots.v1":
        observations = [event for event in events if event.get("event_type") == "person_observation"]
        if not observations:
            raise ResultContractError("insufficient_player_evidence")

    use_candidate_routes = configuration.get("shuttle_detector") in _CANDIDATE_SHUTTLE_DETECTORS
    candidate_routes = None
    candidate_routes_path = None
    if use_candidate_routes:
        video_meta = dict(create_request.get("video") or terminal_status.get("video") or {})
        court_corners = create_request.get("court_corners")
        if court_corners is None:
            court_corners = terminal_status.get("court_corners")
        candidate_routes = derive_stream_candidate_shot_routes(
            events,
            court_corners=court_corners,
            video_width=video_meta.get("width"),
            video_height=video_meta.get("height"),
            video_fps=video_meta.get("fps"),
            match_mode=str(configuration.get("match_mode") or "doubles"),
            roster_identity_policy=configuration.get("roster_identity_policy"),
            duration_sec=duration_sec,
        )
        candidate_routes_path = derived_dir / "candidate_receiving_routes_v1.json"
        _write_json(candidate_routes_path, candidate_routes)

    timings["evidence_materialization_ms"] = (time.monotonic() - materialize_started) * 1000
    metrics_started = time.monotonic()
    metrics = generate_movement_metrics(
        output_dir=output_dir,
        detections_path=detections_path,
        spatial_summary_path=spatial_summary_path,
        metadata_path=metadata_path,
        candidate_shot_routes=candidate_routes,
        shot_route_mode=CANDIDATE_ROUTE_SOURCE if use_candidate_routes else None,
    )
    if anchor_reference is not None:
        metrics["identity_scope"] = "individual_tracks.v1"
        metrics["roster_identity_policy"] = "anchored_roster.v1"
        metrics["identity_anchor"] = anchor_reference
        for player in metrics["players"]:
            player["identity_scope"] = "individual_tracks.v1"
        _write_json(Path(metrics["metrics_path"]), metrics)
    if configuration.get("roster_identity_policy") == "half_court_slots.v1":
        observations = [event for event in events if event.get("event_type") == "person_observation"]
        if not observations:
            raise ResultContractError("insufficient_player_evidence")
        if any(
            (event.get("data", {}).get("track") or {}).get("identity_scope") != "half_court_slots.v1"
            for event in observations
        ):
            raise ResultContractError("invalid_identity_scope")
        metrics["identity_scope"] = "half_court_slots.v1"
        for player in metrics["players"]:
            player["identity_scope"] = "half_court_slots.v1"
        _write_json(Path(metrics["metrics_path"]), metrics)
    timings["statistics_generation_ms"] = (time.monotonic() - metrics_started) * 1000
    manifest = {
        "schema_version": "business-stream-movement-derivation.v1",
        "status": "succeeded",
        "analysis_session_id": terminal_status.get("analysis_session_id"),
        "source_session_status": status,
        "raw_events_path": str(raw_events_path),
        "materialized_detections_path": str(detections_path),
        "metadata_path": str(metadata_path),
        "spatial_summary_path": str(spatial_summary_path),
        **({"candidate_receiving_routes_path": str(candidate_routes_path)} if candidate_routes_path else {}),
        "event_count": len(events),
        "timings_ms": timings,
        "person_observation_count": sum(
            1 for event in events if event.get("event_type") == "person_observation"
        ),
        "metrics": metrics,
        "policy": (
            "Only detected, high-confidence court positions are admitted by the "
            "existing movement metrics reader; predicted/missing observations remain exclusions."
        ),
    }
    manifest_path = derived_dir / "stream_movement_derivation_manifest_v1.json"
    _write_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "movement_metrics_path": metrics.get("metrics_path"),
    }


def _validate_anchored_roster(events, expected_count):
    """Require the actual shared frame proof, never infer it from configuration."""
    import re

    observations = [e for e in events if e.get("event_type") == "person_observation"]
    if not observations:
        raise ResultContractError("insufficient_player_evidence")
    reference = (observations[0].get("data", {}).get("track") or {}).get("identity_anchor")
    if (
        not isinstance(reference, dict)
        or expected_count not in {2, 4}
        or reference.get("policy") != "anchored_roster.v1"
        or not isinstance(reference.get("track_ids"), list)
        or len(reference["track_ids"]) != expected_count
        or any(not isinstance(tid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", tid) for tid in reference["track_ids"])
        or len(set(reference["track_ids"])) != expected_count
        or type(reference.get("source_frame_index")) is not int
        or reference["source_frame_index"] < 0
        or not finite(reference.get("source_time_sec"))
        or reference["source_time_sec"] < 0
    ):
        raise ResultContractError("invalid_identity_scope")
    ids = set(reference["track_ids"])
    photos = {}
    for event in observations:
        data = event.get("data") or {}
        track = data.get("track") or {}
        if (
            track.get("identity_scope") != "individual_tracks.v1"
            or track.get("roster_identity_policy") != "anchored_roster.v1"
            or track.get("identity_anchor") != reference
            or track.get("track_id") not in ids
            or (event.get("evidence_state") == "detected" and track.get("identity_status") != "anchored")
        ):
            raise ResultContractError("invalid_identity_scope")
        photo = data.get("candidate_photo")
        if photo:
            if (
                not isinstance(photo, dict)
                or photo.get("track_id") != track.get("track_id")
                or photo.get("frozen") is not True
                or photo.get("selection_policy") != "same_frame_roster_v1"
                or photo.get("source_frame_index") != reference.get("source_frame_index")
                or not finite(photo.get("source_time_sec"))
                or abs(photo["source_time_sec"] - reference["source_time_sec"]) > 1e-6
                or abs(float(event["source_time_sec"]) - reference["source_time_sec"]) > 1e-6
                or not isinstance(photo.get("photo_set_id"), str)
                or not photo["photo_set_id"]
            ):
                raise ResultContractError("invalid_identity_scope")
            photos[track["track_id"]] = photo
    if (
        set(photos) != ids
        or len({p.get("photo_set_id") for p in photos.values()}) != 1
        or not all(p.get("photo_set_id") for p in photos.values())
    ):
        raise ResultContractError("invalid_identity_scope")
    return reference


def _read_all_events(client, *, on_event_page=None, expected_session_id=None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor = None
    seen_cursors = set()
    seen_events = {}
    while True:
        page = client.read_events(cursor=cursor, limit=1000)
        validate_page(
            page,
            expected_session_id or getattr(getattr(client, "ledger", None), "analysis_session_id", None),
        )
        page_events = page.get("events") or []
        for event in page_events:
            previous = seen_events.get(event["event_id"])
            if previous is not None and previous != event:
                raise ResultContractError("conflicting_event_identity")
            seen_events[event["event_id"]] = event
        events.extend(item for item in page_events if isinstance(item, dict))
        if page_events and on_event_page:
            on_event_page(page)
        next_cursor = page.get("next_cursor")
        if not next_cursor:
            return events
        if next_cursor in seen_cursors:
            raise ResultContractError("invalid_event_pagination")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _materialize_detection_records(events: list[dict[str, Any]], *, sample_hz: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        if event.get("event_type") != "person_observation":
            continue
        track = (event.get("data") or {}).get("track")
        if not isinstance(track, dict) or not track.get("track_id"):
            continue
        source_time = round(float(event.get("source_time_sec") or 0.0), 6)
        bucket = track.get("measurement_bucket")
        try:
            frame = int(bucket)
        except (TypeError, ValueError):
            frame = int(round(source_time * sample_hz))
        grouped[(frame, source_time)][str(track["track_id"])] = dict(track)

    records = []
    for (frame, source_time), tracks in sorted(grouped.items()):
        records.append(
            {
                "schema_version": "stream-materialized-detections.v1",
                "frame": frame,
                "time_sec": source_time,
                "spatial": {
                    "match": {"mode": "person_only"},
                    "tracks": [tracks[track_id] for track_id in sorted(tracks)],
                },
            }
        )
    return records


def _spatial_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_track: dict[str, dict[str, Any]] = {}
    for record in records:
        for track in (record.get("spatial") or {}).get("tracks") or []:
            track_id = str(track.get("track_id") or "")
            if not track_id:
                continue
            entry = per_track.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "detected_frames": 0,
                    "predicted_frames": 0,
                    "missing_frames": 0,
                    "distance_m": 0.0,
                    "zone_frames": Counter(),
                },
            )
            status = str(track.get("status") or "missing")
            key = f"{status}_frames"
            if key in entry:
                entry[key] += 1
            if status == "detected" and track.get("zone_id"):
                entry["zone_frames"][str(track["zone_id"])] += 1
    return {
        "schema_version": "business-stream-spatial-summary.v1",
        "match": {"mode": "person_only"},
        "score_policy": "unknown",
        "player_style_inputs": [
            {
                **entry,
                "zone_frames": dict(sorted(entry["zone_frames"].items())),
            }
            for _, entry in sorted(per_track.items())
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
