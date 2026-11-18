"""Stream-adapted candidate shot-route diversity and opponent receiving heatmaps.

Consumes sealed anonymous stream events (``shuttle_observation`` and
``person_observation``) together with explicit court calibration and real
decoded video metadata.  Never fabricates default court corners, 1920x1080
dimensions, fake frame rates, or wrist-only fallback pairs.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import math
from typing import Any, Mapping, Sequence

from .court_contact_gate import person_court_gate, scoped_rows, valid_quad
from .heatmap_grid import receiving_heatmap
from .offline_shot_reconstruction import (
    _pixel_scale,
    build_shot_events,
    reconstruct_shuttle_track,
    refresh_event_evidence,
)
from .rally_segmentation import segment_rallies

SOURCE = "shot_candidates_receiving_region.v1"
SCHEMA_VERSION = "shot-route-profile.v1"
ALGORITHM_VERSION = "stream-candidate-routes.v1"
HALF_SLOT_SOURCES = frozenset(
    {
        "half_court_slot_opening",
        "half_court_slot_continuity",
        "half_court_slot_replacement",
    }
)
PERSON_TRACK_SOURCES = frozenset({"bytetrack", "court_association", "roster_bootstrap", "anchored_roster"})
MIN_BODY_CONFIDENCE = 0.5
MIN_LOCATION_CONFIDENCE = 0.5
MIN_IDENTITY_CONFIDENCE = 0.7
MAX_PERSON_MATCH_GAP_SEC = 0.150001
LIMITATION_COPY = (
    "Candidate shot routes from shuttle trajectory reconstruction and in-court "
    "nearest-player attribution; not confirmed contacts, true landing points, "
    "or competitive ratings."
)
REASONS = {
    "outside_active_rally": "局间／待重新开球，暂停计入击球与球路",
    "included": "已计入候选球路",
    "unknown_hitter": "本拍击球人待确认",
    "unknown_receiver": "下一拍击球人待确认",
    "same_side": "相邻候选在同侧，击球归属或重复候选待核验",
    "missing_actor_position": "本拍缺少可信场地位置",
    "missing_receiver_position": "下一拍缺少可信接球位置",
    "rally_end": "回合末拍，无下一次接球",
    "unknown_rally": "缺少回合归属",
    "epoch_discontinuity": "断流或修补边界前后不跨段配对",
    "occupancy_discontinuity": "席位占用段变化，跨换人边界不配对",
    "excluded": "已人工排除",
    "invalid_time": "时间无效或重复",
    "court_outside": "击球人物在标定四角场外，已拦截",
    "court_unverified": "缺少可信的场内人物证据，未触发本场击球",
    "receiver_court_unverified": "下一拍缺少可信场内人物证据，待核验",
}


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all(_finite(item) for item in value[:2])
    )


def _court_xy_valid(value: Any) -> bool:
    return _point(value) and 0.0 <= float(value[0]) <= 6.1 and 0.0 <= float(value[1]) <= 13.4


def _identity_confidence(track: Mapping[str, Any]) -> float:
    association = track.get("association") or {}
    source = association.get("source")
    if "identity_confidence" in association and _finite(association["identity_confidence"]):
        return float(association["identity_confidence"])
    if source in HALF_SLOT_SOURCES:
        return 0.95
    return {"bytetrack": 0.95, "court_association": 0.85, "roster_bootstrap": 0.95, "anchored_roster": 0.95}.get(
        source, 0.0
    )


def _sanitize_track(track: Mapping[str, Any], *, roster_identity_policy: str | None = None) -> dict[str, Any] | None:
    """Return a clean read-only track copy; strip stale hands on unmeasured rows."""
    if not isinstance(track, Mapping) or not track.get("track_id"):
        return None
    copy_track = dict(track)
    status = copy_track.get("status")
    loc = dict(copy_track.get("location_evidence") or {})
    pose = dict(copy_track.get("pose") or {})
    if status != "detected" or loc.get("is_current_measurement") is False:
        loc["hands_image"] = None
        loc["is_current_measurement"] = False
        pose["is_current_measurement"] = False
    if roster_identity_policy == "half_court_slots.v1":
        # Half-court slots reuse slot track_ids after misses or replacements;
        # pairing must still cut whenever occupancy_segment changes.
        scope = copy_track.get("identity_scope")
        end = copy_track.get("court_end")
        seg = copy_track.get("occupancy_segment")
        source = (copy_track.get("association") or {}).get("source")
        if (
            scope != "half_court_slots.v1"
            or copy_track.get("slot_court_end") != end
            or end not in {"end_a", "end_b"}
            or type(seg) is not int
            or seg < 1
            or source not in HALF_SLOT_SOURCES
        ):
            copy_track["_slot_invalid"] = True
    copy_track["location_evidence"] = loc
    copy_track["pose"] = pose
    return copy_track


def _validate_context(
    court_corners: Any,
    video_width: Any,
    video_height: Any,
    video_fps: Any,
) -> str | None:
    if not valid_quad(court_corners):
        return "missing_or_invalid_four_corners"
    if (
        not _finite(video_width)
        or not _finite(video_height)
        or isinstance(video_width, bool)
        or isinstance(video_height, bool)
        or float(video_width) <= 0
        or float(video_height) <= 0
    ):
        return "missing_video_dimensions"
    if not _finite(video_fps) or isinstance(video_fps, bool) or float(video_fps) <= 0:
        return "missing_video_fps"
    return None


def _extract_shuttle_observation(event: Mapping[str, Any], *, video_fps: float) -> dict[str, Any]:
    data = event.get("data") or {}
    measurement = data.get("measurement") or {}
    source_time = float(event.get("source_time_sec") or 0.0)
    raw_frame = data.get("source_frame_index")
    frame = (
        int(raw_frame)
        if type(raw_frame) is int and raw_frame >= 0
        else int(round(source_time * video_fps))
    )
    image = measurement.get("image")
    is_detected = (
        event.get("evidence_state") == "detected"
        and measurement.get("status") == "detected"
        and measurement.get("visible") is True
        and measurement.get("accepted") is True
        and _point(image)
    )
    conf = measurement.get("confidence") if _finite(measurement.get("confidence")) else event.get("confidence")
    return {
        "frame": frame,
        "time_sec": source_time,
        "continuity_epoch": int(
            data.get("continuity_epoch")
            if data.get("continuity_epoch") is not None
            else (event.get("continuity_epoch") or 0)
        ),
        "shuttlecock": {
            "status": "detected" if is_detected else "missing",
            "accepted": is_detected,
            "image": [float(image[0]), float(image[1])] if is_detected else None,
            "confidence": float(conf) if is_detected and _finite(conf) else 0.0,
        },
    }


def _deduplicate_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: dict[str, Mapping[str, Any]] = {}
    ordered: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        eid = event.get("event_id")
        if isinstance(eid, str) and eid:
            if eid in seen:
                continue
            seen[eid] = event
        ordered.append(event)
    ordered.sort(
        key=lambda ev: (
            float(ev.get("source_time_sec") or 0.0),
            int(ev.get("segment_index") or 0),
            str(ev.get("event_id") or ""),
        )
    )
    return ordered


def _event_epoch(event: Mapping[str, Any]) -> int:
    epoch = (event.get("data") or {}).get("continuity_epoch")
    return int(epoch if epoch is not None else (event.get("continuity_epoch") or 0))


def _event_position(event: Mapping[str, Any]) -> tuple[float, int]:
    return float(event.get("source_time_sec") or 0.0), int(event.get("segment_index") or 0)


def _event_source_frame(event: Mapping[str, Any]) -> int | None:
    data = event.get("data") or {}
    if event.get("event_type") == "person_observation":
        data = data.get("track") or {}
    frame = data.get("source_frame_index")
    return frame if type(frame) is int and frame >= 0 else None


def _partition_by_continuity(
    events: Sequence[Mapping[str, Any]],
    *,
    video_fps: float,
    match_mode: str,
    roster_identity_policy: str | None,
) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], dict[tuple[int, int], list[dict[str, Any]]], list[str], int, bool]:
    ordered = _deduplicate_events(events)
    boundaries: dict[int, set[tuple[float, int, int]]] = defaultdict(set)
    tied_positions: dict[int, set[tuple[float, int]]] = defaultdict(set)
    unordered_positions: dict[int, set[tuple[float, int]]] = defaultdict(set)
    rejected_segments = {
        (_event_epoch(event), _event_position(event)[1])
        for event in ordered
        if event.get("event_type") == "session_status"
        and (event.get("data") or {}).get("candidate_route_geometry") in (
            "calibration_frame_size_mismatch", "calibration_frame_size_unverified",
        )
    }
    resolution_change = None
    for event in ordered:
        data = event.get("data") or {}
        if event.get("event_type") != "session_status":
            continue
        position = _event_position(event)
        epoch = _event_epoch(event)
        geometry_rejected = (epoch, position[1]) in rejected_segments
        if data.get("ball_continuity") != "unknown" and not geometry_rejected:
            continue
        frame = _event_source_frame(event)
        boundary = (*position, frame if frame is not None else -1)
        boundaries[epoch].add(boundary)
        tied_positions[epoch].add(position)
        if frame is None:
            unordered_positions[epoch].add(position)
        if not geometry_rejected and data.get("reason") == "source_resolution_changed" and (
            resolution_change is None or boundary < resolution_change
        ):
            resolution_change = boundary

    cuts_by_epoch = {epoch: sorted(positions) for epoch, positions in boundaries.items()}
    shuttle_by_scope: dict[tuple[int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    person_by_scope: dict[tuple[int, int], dict[float, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    roster: set[str] = set()
    detected_shuttle_count = 0

    for event in ordered:
        etype = event.get("event_type")
        if etype not in {"shuttle_observation", "person_observation"}:
            continue
        position = _event_position(event)
        epoch = _event_epoch(event)
        if (epoch, position[1]) in rejected_segments:
            continue
        frame = _event_source_frame(event)
        if position in unordered_positions.get(epoch, ()) or (
            frame is None and position in tied_positions.get(epoch, ())
        ):
            continue
        order = (*position, frame if frame is not None else -1)
        if resolution_change is not None and (
            order >= resolution_change or (frame is None and position == resolution_change[:2])
        ):
            continue
        scope = (epoch, bisect_right(cuts_by_epoch.get(epoch, ()), order))
        data = event.get("data") or {}
        if etype == "shuttle_observation":
            row = _extract_shuttle_observation(event, video_fps=video_fps)
            if row["shuttlecock"]["status"] == "detected":
                detected_shuttle_count += 1
            existing = shuttle_by_scope[scope].get(row["frame"])
            if existing is None or (
                existing["shuttlecock"]["status"] != "detected"
                and row["shuttlecock"]["status"] == "detected"
            ):
                shuttle_by_scope[scope][row["frame"]] = row
        elif etype == "person_observation":
            raw_track = data.get("track")
            clean_track = _sanitize_track(raw_track, roster_identity_policy=roster_identity_policy)
            if clean_track is None:
                continue
            tid = str(clean_track["track_id"])
            roster.add(tid)
            source_time = round(float(event.get("source_time_sec") or 0.0), 6)
            person_by_scope[scope][source_time][tid] = clean_track

    shuttle_rows = {
        scope: sorted(frames_map.values(), key=lambda r: (r["frame"], r["time_sec"]))
        for scope, frames_map in shuttle_by_scope.items()
    }

    person_rows = {}
    for scope, time_map in person_by_scope.items():
        rows = []
        for source_time, tracks_by_id in sorted(time_map.items()):
            tracks = [tracks_by_id[tid] for tid in sorted(tracks_by_id)]
            frame_candidates = [
                t.get("source_frame_index")
                for t in tracks
                if type(t.get("source_frame_index")) is int and t["source_frame_index"] >= 0
            ]
            frame = frame_candidates[0] if frame_candidates else int(round(source_time * video_fps))
            rows.append(
                {
                    "frame": frame,
                    "time_sec": source_time,
                    "continuity_epoch": scope[0],
                    "spatial": {"match": {"mode": match_mode}, "tracks": tracks},
                }
            )
        person_rows[scope] = rows

    return shuttle_rows, person_rows, sorted(roster), detected_shuttle_count, bool(cuts_by_epoch)


def _align_shuttle_and_person_rows(
    shuttle_rows: list[dict[str, Any]],
    person_rows: list[dict[str, Any]],
    *,
    match_mode: str,
) -> list[dict[str, Any]]:
    """Attach nearest real measured person tracks (<= 150ms) to each shuttle frame.

    Does not duplicate rows back into ``person_rows`` so body-motion speeds in
    ``segment_rallies`` stay at true person measurement cadence.
    """
    if not person_rows:
        return [
            {**row, "spatial": {"match": {"mode": match_mode}, "tracks": []}}
            for row in shuttle_rows
        ]
    by_frame = {int(row["frame"]): row for row in person_rows}
    person_times = [float(row["time_sec"]) for row in person_rows]
    aligned = []
    for row in shuttle_rows:
        exact = by_frame.get(int(row["frame"]))
        if exact is not None:
            aligned.append({**row, "spatial": exact["spatial"]})
            continue
        t = float(row["time_sec"])
        idx = bisect_left(person_times, t)
        best = None
        best_dt = MAX_PERSON_MATCH_GAP_SEC
        for cand_idx in (idx - 1, idx):
            if 0 <= cand_idx < len(person_rows):
                dt = abs(person_times[cand_idx] - t)
                if dt <= best_dt:
                    best_dt = dt
                    best = person_rows[cand_idx]
        aligned.append(
            {
                **row,
                "spatial": best["spatial"] if best is not None else {"match": {"mode": match_mode}, "tracks": []},
            }
        )
    return aligned


def _stream_position(
    person_rows: list[dict[str, Any]],
    person_times: list[float],
    track_id: str,
    when: float,
    *,
    require_slot_valid: bool = False,
) -> dict[str, Any] | None:
    if not _finite(when) or not person_rows:
        return None
    lo = bisect_left(person_times, when - MAX_PERSON_MATCH_GAP_SEC)
    hi = bisect_right(person_times, when + MAX_PERSON_MATCH_GAP_SEC)
    nearby = sorted(person_rows[lo:hi], key=lambda r: abs(float(r["time_sec"]) - when))
    for row in nearby:
        for t in (row.get("spatial") or {}).get("tracks") or []:
            if t.get("track_id") != track_id or t.get("status") != "detected":
                continue
            if require_slot_valid and t.get("_slot_invalid"):
                continue
            xy = t.get("court_xy_m")
            if not _court_xy_valid(xy):
                continue
            source = (t.get("association") or {}).get("source")
            if source not in HALF_SLOT_SOURCES and source not in PERSON_TRACK_SOURCES:
                continue
            loc = t.get("location_evidence") or {}
            if loc.get("is_current_measurement") is False:
                continue
            values = (t.get("confidence"), loc.get("confidence"), _identity_confidence(t))
            if not all(_finite(v) for v in values):
                continue
            if (
                float(values[0]) < MIN_BODY_CONFIDENCE
                or float(values[1]) < MIN_LOCATION_CONFIDENCE
                or float(values[2]) < MIN_IDENTITY_CONFIDENCE
            ):
                continue
            end = t.get("court_end")
            if end not in {"end_a", "end_b"}:
                end = "end_a" if float(xy[1]) < 6.7 else "end_b"
            seg = t.get("occupancy_segment")
            return {
                "xy": [float(xy[0]), float(xy[1])],
                "end": end,
                "frame": row.get("frame"),
                "time_sec": float(row["time_sec"]),
                "occupancy_segment": seg if type(seg) is int and seg >= 1 else None,
                "identity_scope": t.get("identity_scope"),
                "association_source": source,
            }
    return None


def _build_occupancy_timeline(
    person_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[list[float], list[tuple[int | None, str | None]]]]:
    entries: dict[str, list[tuple[float, int | None, str | None]]] = defaultdict(list)
    for row in person_rows:
        t_sec = float(row["time_sec"])
        for t in (row.get("spatial") or {}).get("tracks") or []:
            tid = t.get("track_id")
            if not isinstance(tid, str) or not tid or t.get("status") != "detected":
                continue
            seg = t.get("occupancy_segment")
            seg_val = seg if type(seg) is int and seg >= 1 else None
            source = (t.get("association") or {}).get("source")
            entries[tid].append((t_sec, seg_val, source if isinstance(source, str) else None))
    timeline: dict[str, tuple[list[float], list[tuple[int | None, str | None]]]] = {}
    for tid, items in entries.items():
        items.sort(key=lambda item: item[0])
        timeline[tid] = (
            [item[0] for item in items],
            [(item[1], item[2]) for item in items],
        )
    return timeline


def _slot_occupancy_continuous(
    occ_timeline: Mapping[str, tuple[list[float], list[tuple[int | None, str | None]]]],
    *,
    actor_id: str,
    receiver_id: str,
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    hit_time_1: float,
    hit_time_2: float,
    require_slot_valid: bool,
) -> bool:
    uses_slots = (
        require_slot_valid
        or start.get("identity_scope") == "half_court_slots.v1"
        or end.get("identity_scope") == "half_court_slots.v1"
    )
    if not uses_slots:
        return True
    start_seg = start.get("occupancy_segment")
    end_seg = end.get("occupancy_segment")
    if type(start_seg) is not int or start_seg < 1 or type(end_seg) is not int or end_seg < 1:
        return False
    start_t = float(start["time_sec"])
    end_t = float(end["time_sec"])
    pair_lo = min(hit_time_1, start_t)
    pair_hi = max(hit_time_2, end_t)

    actor_times, actor_vals = occ_timeline.get(actor_id, ([], []))
    lo_a = bisect_left(actor_times, pair_lo - 1e-6)
    hi_a = bisect_right(actor_times, pair_hi + 1e-6)
    for obs_t, (seg, src) in zip(actor_times[lo_a:hi_a], actor_vals[lo_a:hi_a]):
        if seg != start_seg:
            return False
        if obs_t > start_t + 1e-6 and src == "half_court_slot_replacement":
            return False

    recv_times, recv_vals = occ_timeline.get(receiver_id, ([], []))
    lo_r = bisect_left(recv_times, pair_lo - 1e-6)
    hi_r = bisect_right(recv_times, pair_hi + 1e-6)
    prior_idx = bisect_right(recv_times, pair_lo + 1e-6) - 1
    if prior_idx >= 0 and recv_vals[prior_idx][0] != end_seg:
        return False
    for obs_t, (seg, src) in zip(recv_times[lo_r:hi_r], recv_vals[lo_r:hi_r]):
        if seg != end_seg:
            return False
        if obs_t > pair_lo + 1e-6 and src == "half_court_slot_replacement":
            return False
    return True


def _candidate_court_gate_indexed(
    when: Any,
    person_rows: list[dict[str, Any]],
    person_times: list[float],
    court_corners: Any,
    actor: str | None,
    shuttle_xy: Any,
    *,
    pixel_scale: float = 1.0,
) -> dict[str, Any]:
    if not valid_quad(court_corners):
        return {"status": "unknown", "reason": "missing_or_invalid_four_corners"}
    if not _finite(when) or not person_rows:
        return {"status": "unknown", "reason": "no_current_person_measurement"}
    lo = bisect_left(person_times, float(when) - MAX_PERSON_MATCH_GAP_SEC)
    hi = bisect_right(person_times, float(when) + MAX_PERSON_MATCH_GAP_SEC)
    nearby = person_rows[lo:hi]
    if not nearby:
        return {"status": "unknown", "reason": "no_current_person_measurement"}
    row = min(nearby, key=lambda r: abs(float(r["time_sec"]) - float(when)))
    tracks = (row.get("spatial") or {}).get("tracks") or []
    if actor:
        track = next((t for t in tracks if t.get("track_id") == actor), None)
        return {**person_court_gate(track, court_corners), "measurement_frame": row.get("frame")}
    supports = []
    if _point(shuttle_xy):
        for t in tracks:
            if t.get("status") != "detected":
                continue
            for hand in ((t.get("location_evidence") or {}).get("hands_image") or {}).values():
                if _point(hand):
                    dist = math.dist(hand[:2], shuttle_xy[:2])
                    if dist <= 150.0 * pixel_scale:
                        supports.append((dist, person_court_gate(t, court_corners)))
    if not supports:
        return {"status": "unknown", "reason": "no_in_court_person_action_support"}
    return min(supports, key=lambda pair: pair[0])[1]


def _assign_rallies_for_epoch(
    events: list[dict[str, Any]],
    rallies: list[dict[str, Any]],
    *,
    epoch: int,
    rally_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    normalized_rallies = []
    intervals = []
    for idx, rally in enumerate(rallies, start=1):
        gid = f"epoch_{epoch}_rally_{rally_offset + idx:04d}"
        start = float(rally["start_time_sec"])
        end = float(rally["end_time_sec"])
        entry = {**rally, "rally_id": gid, "continuity_epoch": epoch}
        normalized_rallies.append(entry)
        intervals.append((start, end, gid))

    assignments: dict[str, dict[str, Any]] = {}
    for ev in events:
        t = float(ev["hit_time_sec"])
        matched_gid = None
        for start, end, gid in intervals:
            if start - 1e-6 <= t <= end + 1e-6:
                matched_gid = gid
                break
        if matched_gid is not None:
            assignments[ev["shot_id"]] = {"rally_phase": "in_rally", "rally_id": matched_gid}
        else:
            assignments[ev["shot_id"]] = {"rally_phase": "awaiting_restart", "rally_id": None}
    return normalized_rallies, assignments


def _empty_profile(
    track_id: str,
    *,
    hit_candidate_count: int = 0,
    missing_reason: str | None = None,
    excluded: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": track_id,
        "source": SOURCE,
        "algorithm_version": ALGORITHM_VERSION,
        "metric": "opponent_receiving_region_diversity",
        "status": "insufficient_candidate_pairs",
        "sample_count": 0,
        "hit_candidate_count": int(hit_candidate_count),
        "minimum_samples": 1,
        "sample_quality": "small_candidate_sample",
        "value": None,
        "normalized_value": None,
        "unit": "normalized_entropy",
        "competitive_rating": False,
        "distribution": {},
        "receiving_heatmap": receiving_heatmap([]),
        "confirmed_sample_count": 0,
        "missing_reason": missing_reason,
        "excluded": dict(excluded or {}),
        "limitation": LIMITATION_COPY,
    }


def derive_stream_candidate_shot_routes(
    events: Sequence[Mapping[str, Any]],
    *,
    court_corners: Any,
    video_width: Any,
    video_height: Any,
    video_fps: Any,
    track_ids: Sequence[str] | None = None,
    match_mode: str = "doubles",
    roster_identity_policy: str | None = None,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    """Derive per-track candidate shot-route profiles from stream events."""
    roster_from_events = sorted(
        {
            str(((ev.get("data") or {}).get("track") or {}).get("track_id"))
            for ev in events
            if isinstance(ev, Mapping)
            and ev.get("event_type") == "person_observation"
            and ((ev.get("data") or {}).get("track") or {}).get("track_id")
        }
    )
    roster = sorted(set(track_ids or ()) | set(roster_from_events))
    context_error = _validate_context(court_corners, video_width, video_height, video_fps)
    if context_error is not None:
        return {
            "schema_version": "stream-candidate-receiving-routes.v1",
            "source": SOURCE,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "insufficient_context",
            "missing_reason": context_error,
            "track_ids": roster,
            "players": {
                tid: _empty_profile(tid, missing_reason=context_error) for tid in roster
            },
            "candidate_count": 0,
            "included_count": 0,
            "rally_count": 0,
            "reason_counts": {},
            "records": [],
            "rallies": [],
        }

    fps = float(video_fps)
    width = float(video_width)
    height = float(video_height)
    px_scale = _pixel_scale(width, height)
    shuttle_by_scope, person_by_scope, discovered_roster, detected_shuttle_count, has_explicit_boundary = _partition_by_continuity(
        events,
        video_fps=fps,
        match_mode=match_mode,
        roster_identity_policy=roster_identity_policy,
    )
    roster = sorted(set(roster) | set(discovered_roster))
    if detected_shuttle_count == 0:
        return {
            "schema_version": "stream-candidate-receiving-routes.v1",
            "source": SOURCE,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "insufficient_shuttle_evidence",
            "missing_reason": "insufficient_shuttle_evidence",
            "track_ids": roster,
            "players": {
                tid: _empty_profile(tid, missing_reason="insufficient_shuttle_evidence") for tid in roster
            },
            "candidate_count": 0,
            "included_count": 0,
            "rally_count": 0,
            "reason_counts": {},
            "records": [],
            "rallies": [],
        }

    require_slot_valid = roster_identity_policy == "half_court_slots.v1"
    all_candidates: list[dict[str, Any]] = []
    all_rallies: list[dict[str, Any]] = []
    all_assignments: dict[str, dict[str, Any]] = {}
    person_index_by_scope = {}
    occ_timeline_by_scope = {}
    shuttle_xy_by_scope_frame = {}
    shot_counter = 0

    for scope in sorted(set(shuttle_by_scope) | set(person_by_scope)):
        epoch = scope[0]
        s_rows = shuttle_by_scope.get(scope, [])
        p_rows = person_by_scope.get(scope, [])
        p_times = [float(r["time_sec"]) for r in p_rows]
        person_index_by_scope[scope] = (p_rows, p_times)
        occ_timeline_by_scope[scope] = _build_occupancy_timeline(p_rows)
        if not s_rows:
            continue
        track_rows = reconstruct_shuttle_track(
            s_rows,
            fps=fps,
            width=width,
            height=height,
            px_scale=px_scale,
        )
        for tr in track_rows:
            shuttle_xy_by_scope_frame[(scope, int(tr["frame"]))] = tr.get("image_xy")
        aligned_rows = _align_shuttle_and_person_rows(s_rows, p_rows, match_mode=match_mode)
        epoch_end_sec = max(
            [float(r["time_sec"]) for r in s_rows]
            + [float(r["time_sec"]) for r in p_rows]
            + ([float(duration_sec)] if _finite(duration_sec) and len(shuttle_by_scope) == 1 and not has_explicit_boundary else [0.0])
        )
        rally_stage = segment_rallies(
            track_rows,
            p_rows,
            court_corners,
            video_end_sec=epoch_end_sec,
            pixel_scale=px_scale,
        )
        # Never call infer_missing_shuttle_events: if trajectory cannot bridge a gap,
        # keep it broken until the ball reappears.
        hit_rows = scoped_rows(aligned_rows, court_corners)
        raw_events = build_shot_events(
            hit_rows,
            track_rows,
            fps=fps,
            px_scale=px_scale,
            court_polygon=court_corners,
        )
        raw_events = refresh_event_evidence(raw_events, track_rows)
        epoch_events = []
        for ev in raw_events:
            shot_counter += 1
            sid = f"epoch_{epoch}_shot_{shot_counter:04d}"
            ev_copy = {**ev, "shot_id": sid, "continuity_epoch": epoch, "_continuity_scope": scope}
            epoch_events.append(ev_copy)
        normalized_rallies, epoch_assignments = _assign_rallies_for_epoch(
            epoch_events,
            rally_stage.get("rallies") or [],
            epoch=epoch,
            rally_offset=len(all_rallies),
        )
        all_rallies.extend(normalized_rallies)
        all_assignments.update(epoch_assignments)
        all_candidates.extend(epoch_events)

    all_candidates.sort(
        key=lambda c: (
            float(c.get("hit_time_sec") if _finite(c.get("hit_time_sec")) else math.inf),
            int(c.get("continuity_epoch", 0)),
            str(c["shot_id"]),
        )
    )
    roster_set = set(roster)
    gates: dict[str, dict[str, Any]] = {}
    actors: dict[str, str | None] = {}
    for c in all_candidates:
        sid = c["shot_id"]
        scope = c["_continuity_scope"]
        p_rows, p_times = person_index_by_scope.get(scope, ([], []))
        raw_actor = (c.get("hitter") or {}).get("track_id")
        actor = str(raw_actor) if isinstance(raw_actor, str) and raw_actor in roster_set else None
        actors[sid] = actor
        shuttle_xy = shuttle_xy_by_scope_frame.get((scope, int(c.get("frame", -1))))
        gates[sid] = _candidate_court_gate_indexed(
            c.get("hit_time_sec"),
            p_rows,
            p_times,
            court_corners,
            actor,
            shuttle_xy,
            pixel_scale=px_scale,
        )

    hit_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    next_by_id = {a["shot_id"]: b for a, b in zip(all_candidates, all_candidates[1:])}

    for c in all_candidates:
        sid = c["shot_id"]
        epoch = int(c["continuity_epoch"])
        actor = actors[sid]
        when = c.get("hit_time_sec")
        gate = gates[sid]
        assign = all_assignments.get(sid, {"rally_phase": "awaiting_restart", "rally_id": None})
        if actor and gate["status"] == "inside" and assign.get("rally_phase") == "in_rally":
            hit_counts[actor] += 1

        rec: dict[str, Any] = {
            "shot_id": sid,
            "continuity_epoch": epoch,
            "time_sec": when if _finite(when) else None,
            "actor_track_id": actor,
            "rally_id": assign.get("rally_id"),
            "rally_phase": assign.get("rally_phase", "awaiting_restart"),
            "candidate_source": c.get("candidate_source"),
            "status": "pending",
            "court_gate": gate,
        }
        following = next_by_id.get(sid)
        receiver = actors[following["shot_id"]] if following else None
        reason: str
        if rec["rally_phase"] != "in_rally":
            reason, rec["status"] = "outside_active_rally", "waiting"
        elif gate["status"] == "outside":
            reason, rec["status"] = "court_outside", "filtered"
        elif gate["status"] != "inside":
            reason = "court_unverified"
        elif not _finite(when) or float(when) < 0:
            reason = "invalid_time"
        elif not actor:
            reason = "unknown_hitter"
        elif not rec["rally_id"]:
            reason = "unknown_rally"
        elif not following or all_assignments.get(following["shot_id"], {}).get("rally_id") != rec["rally_id"]:
            reason, rec["status"] = "rally_end", "terminal"
        elif following["_continuity_scope"] != c["_continuity_scope"]:
            reason = "epoch_discontinuity"
        elif not _finite(following.get("hit_time_sec")) or float(following["hit_time_sec"]) <= float(when):
            reason = "invalid_time"
        elif gates[following["shot_id"]]["status"] != "inside":
            reason = "receiver_court_unverified"
        elif not receiver:
            reason = "unknown_receiver"
        else:
            scope = c["_continuity_scope"]
            p_rows, p_times = person_index_by_scope.get(scope, ([], []))
            start = _stream_position(
                p_rows, p_times, actor, float(when), require_slot_valid=require_slot_valid
            )
            end = _stream_position(
                p_rows,
                p_times,
                receiver,
                float(following["hit_time_sec"]),
                require_slot_valid=require_slot_valid,
            )
            rec.update(
                receiver_shot_id=following["shot_id"],
                receiver_track_id=receiver,
                receiver_time_sec=following["hit_time_sec"],
            )
            if start:
                rec["actor_occupancy_segment"] = start.get("occupancy_segment")
            if end:
                rec["receiver_occupancy_segment"] = end.get("occupancy_segment")
            if not start:
                reason = "missing_actor_position"
            elif not end:
                reason = "missing_receiver_position"
            elif start["end"] == end["end"] or actor == receiver:
                reason = "same_side"
            elif not _slot_occupancy_continuous(
                occ_timeline_by_scope.get(scope, {}),
                actor_id=actor,
                receiver_id=receiver,
                start=start,
                end=end,
                hit_time_1=float(when),
                hit_time_2=float(following["hit_time_sec"]),
                require_slot_valid=require_slot_valid,
            ):
                reason = "occupancy_discontinuity"
            else:
                x, y = end["xy"]
                depth = abs(y - 6.7)
                rec.update(
                    status="included",
                    receiver_xy_m=end["xy"],
                    receiver_measurement_frame=end["frame"],
                    receiver_measurement_time_sec=end["time_sec"],
                    receiver_zone=min(2, int(depth / 6.7 * 3)) * 3 + min(2, int(x / 6.1 * 3)),
                    grid_row=min(6, int(depth / 6.7 * 7)),
                    grid_column=min(5, int(x / 6.1 * 6)),
                )
                reason = "included"
        rec["reason"] = reason
        rec["reason_label"] = REASONS.get(reason, reason)
        records.append(rec)

    profiles: dict[str, dict[str, Any]] = {}
    for tid in roster:
        pairs = [r for r in records if r["status"] == "included" and r["actor_track_id"] == tid]
        actor_excluded = Counter(
            r["reason"]
            for r in records
            if r["actor_track_id"] == tid and r["status"] != "included"
        )
        n = len(pairs)
        if n == 0:
            profiles[tid] = _empty_profile(
                tid,
                hit_candidate_count=hit_counts[tid],
                missing_reason="no_valid_candidate_pairs",
                excluded=dict(sorted(actor_excluded.items())),
            )
            continue
        zones = Counter(r["receiver_zone"] for r in pairs)
        entropy = -sum((count / n) * math.log(count / n) for count in zones.values()) / math.log(9)
        entropy = max(0.0, min(1.0, entropy))
        profiles[tid] = {
            "schema_version": SCHEMA_VERSION,
            "track_id": tid,
            "source": SOURCE,
            "algorithm_version": ALGORITHM_VERSION,
            "metric": "opponent_receiving_region_diversity",
            "status": "estimated",
            "sample_count": n,
            "hit_candidate_count": int(hit_counts[tid]),
            "minimum_samples": 1,
            "sample_quality": "small_candidate_sample" if n < 12 else "unvalidated_candidate_sample",
            "value": round(entropy, 4),
            "normalized_value": round(entropy * 100.0, 1),
            "unit": "normalized_entropy",
            "competitive_rating": False,
            "distribution": {str(k): v for k, v in sorted(zones.items())},
            "receiving_heatmap": receiving_heatmap(pairs),
            "confirmed_sample_count": 0,
            "missing_reason": None,
            "excluded": dict(sorted(actor_excluded.items())),
            "limitation": LIMITATION_COPY,
        }

    reasons = Counter(r["reason"] for r in records)
    return {
        "schema_version": "stream-candidate-receiving-routes.v1",
        "source": SOURCE,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "ready" if records or all_rallies else "no_candidates",
        "missing_reason": None,
        "track_ids": roster,
        "players": profiles,
        "candidate_count": len(all_candidates),
        "included_count": reasons["included"],
        "rally_count": len(all_rallies),
        "reason_counts": dict(sorted(reasons.items())),
        "records": records,
        "rallies": all_rallies,
    }
