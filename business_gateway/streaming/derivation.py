"""Business-side movement derivation for a completed anonymous stream session.

The GPU service persists immutable stream events and checkpoints.  This module
is deliberately on the business side: it fetches those public anonymous events
after a terminal session, materialises a compact compatibility view, and runs
the existing movement-metrics code.  It never asks the GPU to create a legacy
video-analysis directory and never sends user identities back to the GPU.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from business_gateway.metrics.movement import generate_movement_metrics


_TERMINAL_WITH_EVIDENCE = {"finalized", "partial"}


def derive_stream_movement_metrics(
    output_dir: Path | str,
    *,
    client,
    terminal_status: Mapping[str, Any],
    create_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch all event pages and derive auditable visual movement metrics.

    The materialised JSONL only adapts the event contract to the existing
    neutral metrics reader.  It preserves original event IDs in a companion
    file and writes no predicted/missing point as a fresh movement sample.
    """

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

    events = _read_all_events(client)
    raw_events_path = evidence_dir / "stream_events.jsonl"
    _write_jsonl(raw_events_path, events)

    configuration = dict(create_request.get("configuration") or {})
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
    _write_json(
        spatial_summary_path,
        _spatial_summary(records),
    )

    metrics = generate_movement_metrics(
        output_dir=output_dir,
        detections_path=detections_path,
        spatial_summary_path=spatial_summary_path,
        metadata_path=metadata_path,
    )
    manifest = {
        "schema_version": "business-stream-movement-derivation.v1",
        "status": "succeeded",
        "analysis_session_id": terminal_status.get("analysis_session_id"),
        "source_session_status": status,
        "raw_events_path": str(raw_events_path),
        "materialized_detections_path": str(detections_path),
        "metadata_path": str(metadata_path),
        "spatial_summary_path": str(spatial_summary_path),
        "event_count": len(events),
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


def _read_all_events(client) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor = None
    seen_cursors = set()
    while True:
        page = client.read_events(cursor=cursor, limit=500)
        page_events = page.get("events") or []
        events.extend(item for item in page_events if isinstance(item, dict))
        next_cursor = page.get("next_cursor")
        if not next_cursor:
            return events
        if next_cursor in seen_cursors:
            raise RuntimeError("stream event pagination cursor repeated")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _materialize_detection_records(events: list[dict[str, Any]], *, sample_hz: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in events:
        if event.get("event_type") != "person_observation":
            continue
        track = ((event.get("data") or {}).get("track"))
        if not isinstance(track, dict) or not track.get("track_id"):
            continue
        source_time = round(float(event.get("source_time_sec") or 0.0), 6)
        bucket = ((track.get("measurement_bucket")))
        try:
            frame = int(bucket)
        except (TypeError, ValueError):
            frame = int(round(source_time * sample_hz))
        # One event per track/bucket is expected.  Keeping the later event ID
        # makes retry/replay duplicates deterministic without inventing data.
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
        for track in ((record.get("spatial") or {}).get("tracks") or []):
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
