"""Shot-type candidate generation and human-review storage.

This module deliberately separates three layers of information:

* ``detections.jsonl`` is the immutable model evidence produced by analysis.
* ``shot_candidates.json`` is a reproducible, low-confidence computer proposal.
* ``annotations.jsonl`` is an append-only human correction audit trail.

The current fixed-camera data only provides weak hit candidates.  Therefore a
proposal such as ``smash`` or ``drop`` is never treated as a score, error, or
player-skill fact until a reviewer confirms it.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2

from badminton_analysis.analysis.offline_shot_reconstruction import (
    build_rallies_from_manual_terminals,
    generate_offline_artifacts,
    load_jsonl as load_derived_jsonl,
    write_jsonl,
)
from webui.pipeline import _find_ffmpeg


SHOT_TYPES = {
    "unknown": "不确定",
    "clear": "高远球",
    "lift": "挑球",
    "drop": "吊球",
    "smash": "杀球",
    "drive": "平高/平抽球",
    "net": "网前球",
    "serve": "发球",
    "other": "其他",
}
REVIEW_DECISIONS = {
    "pending": "待复核",
    "confirmed": "确认自动建议",
    "corrected": "人工修正",
    "uncertain": "仍不确定",
    "excluded": "不是有效击球",
}
SESSION_FILENAME = "shot_candidates.json"
ANNOTATIONS_FILENAME = "annotations.jsonl"
RALLY_TERMINALS_FILENAME = "rally_terminals.jsonl"
REVIEWED_RALLIES_FILENAME = "reviewed_rallies_v2.jsonl"
CLIP_SECONDS_BEFORE = 1.0
CLIP_SECONDS_AFTER = 1.8
_RUN_TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(20\d{6}_\d{6}(?:_\d{6})?)(?!\d)")

RALLY_TERMINAL_OUTCOMES = {
    "out_of_bounds": "球出界",
    "landed_in_bounds": "球落地（界内）",
    "landed_out_of_bounds": "球落地（出界）",
    "landed_unknown": "球落地（界内外待定）",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def rally_terminal_path(analysis_dir):
    return Path(analysis_dir).expanduser().resolve() / "shot_review" / RALLY_TERMINALS_FILENAME


def reviewed_rallies_path(analysis_dir):
    return Path(analysis_dir).expanduser().resolve() / "shot_review" / REVIEWED_RALLIES_FILENAME


def load_manual_rally_terminals(analysis_dir):
    """Load append-only human terminal facts, newest duplicate kept only once."""
    path = rally_terminal_path(analysis_dir)
    if not path.is_file():
        return []
    latest_by_time = {}
    for item in load_derived_jsonl(path):
        try:
            time_sec = round(float(item.get("time_sec")), 3)
        except (TypeError, ValueError):
            continue
        if item.get("outcome") not in RALLY_TERMINAL_OUTCOMES:
            continue
        latest_by_time[time_sec] = item
    return [latest_by_time[key] for key in sorted(latest_by_time)]


def rebuild_reviewed_rallies(analysis_dir):
    """Build the display-only human-reviewed rally timeline beside raw output."""
    analysis_path = Path(analysis_dir).expanduser().resolve()
    terminals = load_manual_rally_terminals(analysis_path)
    output_path = reviewed_rallies_path(analysis_path)
    if not terminals:
        if output_path.exists():
            output_path.unlink()
        return [], output_path
    derived = generate_offline_artifacts(analysis_path / "detections.jsonl")
    events = load_derived_jsonl(derived["events_path"])
    reviewed = build_rallies_from_manual_terminals(events, terminals)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, reviewed)
    return reviewed, output_path


def add_manual_rally_terminal(analysis_dir, time_sec, outcome, reviewer="", note=""):
    """Append a reviewer-confirmed terminal boundary without touching raw data."""
    if outcome not in RALLY_TERMINAL_OUTCOMES:
        raise ValueError("未知的回合结束类型。")
    try:
        time_sec = round(float(time_sec), 3)
    except (TypeError, ValueError) as error:
        raise ValueError("回合结束时间必须是有效秒数。") from error
    if time_sec < 0:
        raise ValueError("回合结束时间不能小于 0 秒。")
    path = rally_terminal_path(analysis_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_manual_rally_terminals(analysis_dir)
    if any(abs(float(item["time_sec"]) - time_sec) <= 0.05 for item in existing):
        raise ValueError("该时间附近已经有人工确认的回合结束记录。")
    record = {
        "schema_version": "1.0",
        "terminal_id": f"terminal_{len(existing) + 1:04d}",
        "time_sec": time_sec,
        "outcome": outcome,
        "source": "human_review",
        "reviewer": (reviewer or "").strip(),
        "note": (note or "").strip(),
        "recorded_at": utc_now(),
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    reviewed, _ = rebuild_reviewed_rallies(analysis_dir)
    return record, reviewed


def reviewed_rally_table(session):
    """Rows displayed beside the player; machine and human boundaries stay distinct."""
    rows = []
    for item in session.get("reviewed_rallies") or []:
        terminal = item.get("terminal") or {}
        rows.append([
            item.get("rally_id"),
            round(float(item.get("start_time_sec", 0.0)), 2),
            round(float(item.get("end_time_sec", 0.0)), 2),
            int(item.get("shot_count", 0)),
            RALLY_TERMINAL_OUTCOMES.get(terminal.get("outcome"), terminal.get("outcome") or "未知"),
            "人工确认",
        ])
    return rows


def _attach_reviewed_rallies(session, analysis_path):
    """Attach a small reviewed view for playback without rewriting the session."""
    path = reviewed_rallies_path(analysis_path)
    if path.is_file():
        try:
            session["reviewed_rallies"] = load_derived_jsonl(path)
        except (OSError, ValueError):
            session.pop("reviewed_rallies", None)
    else:
        session.pop("reviewed_rallies", None)
    return session


def find_analysis_runs(base_dir="outputs"):
    """Return finished-analysis folders in stable analysis-time order.

    Directory mtime is not a valid "latest analysis" signal: opening a review
    session or creating derived files changes it.  New folders use a timestamp
    prefix; legacy folders are recognised by the timestamp that used to live at
    the end of their name, with mtime retained only as a fallback.
    """
    root = Path(base_dir)
    if not root.is_dir():
        return []
    runs = []
    for path in root.rglob("detections.jsonl"):
        parent = path.parent
        if "shot_review" in parent.parts:
            continue
        runs.append(parent)
    return [str(path) for path in sorted(set(runs), key=_analysis_run_sort_key, reverse=True)]


def analysis_run_label(path):
    """Return a timestamp-first dropdown label without renaming legacy data."""
    path = Path(path)
    timestamp = _analysis_run_timestamp(path) or datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
    name = path.name
    display_name = _RUN_TIMESTAMP_PATTERN.sub("", name).strip("_ -") or name
    if "remote_jobs" in path.parts:
        source = "远端 GPU"
    else:
        source = "本地"
    return f"{timestamp} · {source} · {display_name}"


def _analysis_run_sort_key(path):
    path = Path(path)
    timestamp = _analysis_run_timestamp(path)
    try:
        modified_at = path.stat().st_mtime
    except OSError:
        modified_at = 0.0
    # Timestamp-bearing names are analysis directories.  Unknown legacy/test
    # names sort below them, instead of a review-generated file moving to top.
    return (1 if timestamp else 0, timestamp or "", modified_at, str(path))


def _analysis_run_timestamp(path):
    # Prefer the leaf name, then allow the remote task folder immediately above
    # a future nested output directory.  It also supports the old timestamp
    # suffix without needing risky filesystem renames.
    for candidate in (Path(path), Path(path).parent):
        match = _RUN_TIMESTAMP_PATTERN.search(candidate.name)
        if match:
            return match.group(1)
    return None


def default_analysis_run(base_dir="outputs"):
    runs = find_analysis_runs(base_dir)
    return runs[0] if runs else ""


def create_or_load_review_session(analysis_dir, reference_video=None, regenerate=False):
    """Create computer proposals from one immutable analysis result directory.

    Existing review files are reused by default so pressing the UI button never
    erases a human correction.  ``regenerate`` is reserved for an explicit
    future rebuild workflow after the generator logic is versioned.
    """
    analysis_path = Path(analysis_dir).expanduser().resolve()
    detections_path = analysis_path / "detections.jsonl"
    if not detections_path.is_file():
        raise ValueError("分析目录中未找到 detections.jsonl。请选择已完成的视频分析输出目录。")

    review_dir = analysis_path / "shot_review"
    session_path = review_dir / SESSION_FILENAME
    previous_session = _load_json(session_path) if session_path.is_file() else None
    if previous_session is not None and not regenerate:
        return _attach_reviewed_rallies(previous_session, analysis_path)

    rows = _load_detections(detections_path)
    derived = generate_offline_artifacts(detections_path)
    derived_events = load_derived_jsonl(derived["events_path"])
    video_path = _resolve_reference_video(analysis_path, reference_video)
    session = {
        "schema_version": "2.0",
        "generator": {
            "name": "weak_evidence_shot_review",
            "version": "2.0",
            "generated_at": utc_now(),
            "policy": (
                "Shot type is an offline review proposal derived from immutable detector evidence, "
                "short bounded reconstruction and player spatial context. It is not a ground-truth "
                "score, error, or player ability metric until a human confirms it."
            ),
        },
        "source": {
            "analysis_dir": str(analysis_path),
            "detections_path": str(detections_path),
            "reference_video": str(video_path) if video_path else None,
            "detections_sha256": _sha256(detections_path),
            "derived_tracks_path": derived["tracks_path"],
            "derived_events_path": derived["events_path"],
            "derived_version": derived["version"],
        },
        "labels": SHOT_TYPES,
        "review_decisions": REVIEW_DECISIONS,
        "candidates": _build_candidates(rows, derived_events=derived_events),
    }
    if previous_session is not None:
        _preserve_manual_review_work(previous_session, session)
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_json(session_path, session)
    return _attach_reviewed_rallies(session, analysis_path)


def review_session_path(analysis_dir):
    return Path(analysis_dir).expanduser().resolve() / "shot_review" / SESSION_FILENAME


def review_summary(session):
    candidates = _active_candidates(session)
    decisions = {}
    for item in candidates:
        key = (item.get("review") or {}).get("decision", "pending")
        decisions[key] = decisions.get(key, 0) + 1
    return {
        "analysis_dir": (session.get("source") or {}).get("analysis_dir"),
        "candidate_count": len(candidates),
        "pending": decisions.get("pending", 0),
        "confirmed": decisions.get("confirmed", 0),
        "corrected": decisions.get("corrected", 0),
        "uncertain": decisions.get("uncertain", 0),
        "excluded": decisions.get("excluded", 0),
        "review_file": str(review_session_path((session.get("source") or {}).get("analysis_dir"))),
    }


def candidate_choices(session):
    choices = []
    for candidate in _active_candidates(session):
        proposed = SHOT_TYPES.get(candidate["proposal"]["label"], candidate["proposal"]["label"])
        review = candidate.get("review") or {}
        decision = REVIEW_DECISIONS.get(review.get("decision", "pending"), "待复核")
        choices.append((
            f"{candidate['shot_id']} · {candidate['hit_time_sec']:.2f}s · 建议：{proposed} · {decision}",
            candidate["shot_id"],
        ))
    return choices


def timeline_state(session, playback_sec):
    """Return the most recent and next active shot around a video time.

    The result intentionally follows the match timeline instead of assuming
    every automatic proposal is correct.  A reviewer can use ``current`` to
    edit the latest proposal or add a missing hit at the exact playback time.
    """
    try:
        playback_sec = max(0.0, float(playback_sec or 0.0))
    except (TypeError, ValueError):
        playback_sec = 0.0
    candidates = sorted(_active_candidates(session), key=lambda item: float(item["hit_time_sec"]))
    current = None
    next_candidate = None
    for candidate in candidates:
        if float(candidate["hit_time_sec"]) <= playback_sec:
            current = candidate
        else:
            next_candidate = candidate
            break
    return {
        "playback_sec": playback_sec,
        "current": current,
        "next": next_candidate,
        "candidate_count": len(candidates),
    }


def rally_playback_state(session, playback_sec):
    """Describe the candidate rally around a playback time without inventing one.

    ``rally_id`` and ``shot_index_in_rally`` originate in the offline derived
    artifacts.  A human-added touch has no assigned rally until a future,
    explicit re-segmentation step, so the UI must show that fact instead of
    silently placing it into a neighbouring rally.
    """
    try:
        playback_sec = max(0.0, float(playback_sec or 0.0))
    except (TypeError, ValueError):
        playback_sec = 0.0
    reviewed = session.get("reviewed_rallies") or []
    if reviewed:
        return _reviewed_rally_playback_state(reviewed, playback_sec)

    timeline = timeline_state(session, playback_sec)
    current = timeline["current"]
    if current is None:
        return {
            "status": "before_first_touch",
            "playback_sec": timeline["playback_sec"],
            "rally_id": None,
            "rally_number": None,
            "rally_count": 0,
            "shot_index": 0,
            "shot_count": 0,
        }

    evidence = current.get("evidence") or {}
    rally_id = evidence.get("rally_id")
    if not rally_id:
        return {
            "status": "unassigned_touch",
            "playback_sec": timeline["playback_sec"],
            "rally_id": None,
            "rally_number": None,
            "rally_count": 0,
            "shot_index": None,
            "shot_count": None,
        }

    candidates = sorted(_active_candidates(session), key=lambda item: float(item["hit_time_sec"]))
    rally_ids = []
    rally_candidates = []
    for candidate in candidates:
        candidate_rally_id = (candidate.get("evidence") or {}).get("rally_id")
        if candidate_rally_id and candidate_rally_id not in rally_ids:
            rally_ids.append(candidate_rally_id)
        if candidate_rally_id == rally_id:
            rally_candidates.append(candidate)

    shot_indexes = []
    for candidate in rally_candidates:
        try:
            shot_index = int((candidate.get("evidence") or {}).get("shot_index_in_rally"))
        except (TypeError, ValueError):
            continue
        if shot_index > 0:
            shot_indexes.append(shot_index)
    try:
        current_shot_index = int(evidence.get("shot_index_in_rally"))
    except (TypeError, ValueError):
        current_shot_index = 0
    if current_shot_index <= 0:
        current_shot_index = next(
            (index + 1 for index, item in enumerate(rally_candidates) if item["shot_id"] == current["shot_id"]),
            0,
        )

    return {
        "status": "assigned",
        "playback_sec": timeline["playback_sec"],
        "rally_id": rally_id,
        "rally_number": rally_ids.index(rally_id) + 1 if rally_id in rally_ids else None,
        "rally_count": len(rally_ids),
        "shot_index": current_shot_index,
        "shot_count": max(shot_indexes, default=len(rally_candidates)),
    }


def _reviewed_rally_playback_state(reviewed_rallies, playback_sec):
    """Use human terminal facts when available, without assigning extra shots."""
    for index, rally in enumerate(reviewed_rallies, start=1):
        start = float(rally.get("start_time_sec", 0.0))
        end = float(rally.get("end_time_sec", start))
        if start <= playback_sec <= end:
            event_times = [float(value) for value in rally.get("shot_times_sec") or []]
            completed = sum(time_sec <= playback_sec for time_sec in event_times)
            terminal = rally.get("terminal") or {}
            return {
                "status": "human_reviewed",
                "playback_sec": playback_sec,
                "rally_id": rally.get("rally_id"),
                "rally_number": index,
                "rally_count": len(reviewed_rallies),
                "shot_index": completed,
                "shot_count": int(rally.get("shot_count", 0)),
                "terminal_time_sec": end,
                "terminal_outcome": terminal.get("outcome"),
            }
    if playback_sec > float(reviewed_rallies[-1].get("end_time_sec", 0.0)):
        last = reviewed_rallies[-1]
        return {
            "status": "after_last_human_terminal",
            "playback_sec": playback_sec,
            "rally_id": last.get("rally_id"),
            "rally_number": len(reviewed_rallies),
            "rally_count": len(reviewed_rallies),
            "shot_index": int(last.get("shot_count", 0)),
            "shot_count": int(last.get("shot_count", 0)),
            "terminal_time_sec": float(last.get("end_time_sec", 0.0)),
            "terminal_outcome": (last.get("terminal") or {}).get("outcome"),
        }
    return {
        "status": "before_first_human_terminal",
        "playback_sec": playback_sec,
        "rally_id": None,
        "rally_number": 1,
        "rally_count": len(reviewed_rallies),
        "shot_index": 0,
        "shot_count": int(reviewed_rallies[0].get("shot_count", 0)),
        "terminal_time_sec": float(reviewed_rallies[0].get("end_time_sec", 0.0)),
        "terminal_outcome": (reviewed_rallies[0].get("terminal") or {}).get("outcome"),
    }


def candidate_table(session):
    """Compact table for review triage; detailed evidence stays with the clip."""
    table = []
    for candidate in _active_candidates(session):
        proposal = candidate["proposal"]
        table.append([
            candidate["shot_id"],
            round(candidate["hit_time_sec"], 2),
            SHOT_TYPES.get(proposal["label"], proposal["label"]),
            round(float(proposal.get("confidence", 0.0)), 3),
            candidate_source_display(candidate.get("candidate_source", "unknown")),
            review_result_display(candidate),
        ])
    return table


def candidate_at_table_row(session, row_index):
    """Return the active candidate represented by a visible table row."""
    try:
        row_index = int(row_index)
    except (TypeError, ValueError) as error:
        raise ValueError("未能识别所选触球记录。") from error
    candidates = _active_candidates(session)
    if row_index < 0 or row_index >= len(candidates):
        raise ValueError("所选触球记录已变化，请重新打开整场复核。")
    return candidates[row_index]


def candidate_source_display(source):
    return {
        "spatial_proximity": "Offline spatial proximity",
        "bidirectional_trajectory_turn": "Offline bidirectional trajectory turn",
        "spatial_hit_candidate": "人体/空间接近",
        "trajectory_turn": "轨迹方向变化",
        "trajectory_heading_change_near_hand": "轨迹急转且接近手部",
        "trajectory_gap_transition": "短暂断检衔接",
        "reconstructed_gap_hand_proximity": "短缺帧插值接近手部（推断）",
        "motion_inferred_missing_shuttle": "单打缺球动作约束候选",
        "hand_proximity_fallback": "手部接近（回退）",
        "manual_timeline": "人工补拍",
        "manual_merge": "人工合并",
    }.get(source, source)


def review_result_display(candidate):
    review = candidate.get("review") or {}
    decision = review.get("decision", "pending")
    label = review.get("label")
    if decision == "pending":
        return "待复核"
    if decision == "excluded":
        return "已排除"
    display = SHOT_TYPES.get(label, label or "不确定")
    return f"{REVIEW_DECISIONS.get(decision, decision)}：{display}"


def get_candidate(session, shot_id):
    for candidate in session.get("candidates", []):
        if candidate["shot_id"] == shot_id:
            return candidate
    raise ValueError("未找到所选击球候选。请先生成或打开复核队列。")


def candidate_view(session, shot_id):
    """Return form values and an optional playable context clip for a candidate."""
    candidate = get_candidate(session, shot_id)
    review = candidate.get("review") or {}
    clip = ensure_context_clip(session, candidate)
    details = {
        "shot_id": candidate["shot_id"],
        "hit_time_sec": candidate["hit_time_sec"],
        "candidate_source": candidate["candidate_source"],
        "hitter_track_id": candidate.get("hitter_track_id"),
        "proposal": candidate["proposal"],
        "evidence": candidate["evidence"],
        "review": review,
        "clip": clip,
    }
    return {
        "details": details,
        "label": review.get("label") or candidate["proposal"]["label"],
        "decision": review.get("decision", "pending"),
        "reviewer": review.get("reviewer", ""),
        "note": review.get("note", ""),
        "clip": clip,
    }


def save_human_review(session, shot_id, label, decision, reviewer="", note=""):
    """Persist a manual decision without changing raw detections or auto proposal."""
    if label not in SHOT_TYPES:
        raise ValueError("未知球种标签。")
    if decision not in REVIEW_DECISIONS:
        raise ValueError("未知复核状态。")
    candidate = get_candidate(session, shot_id)
    revision = {
        "at": utc_now(),
        "label": label,
        "decision": decision,
        "reviewer": (reviewer or "").strip(),
        "note": (note or "").strip(),
    }
    candidate["review"] = revision
    candidate.setdefault("review_history", []).append(revision)
    source_dir = Path(session["source"]["analysis_dir"])
    review_dir = source_dir / "shot_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_json(review_dir / SESSION_FILENAME, session)
    with (review_dir / ANNOTATIONS_FILENAME).open("a", encoding="utf-8") as output:
        output.write(json.dumps({"shot_id": shot_id, **revision}, ensure_ascii=False) + "\n")
    return session


def add_manual_candidate(session, hit_time_sec, reviewer=""):
    """Add a reviewer-selected time point without pretending it was detected."""
    hit_time_sec = float(hit_time_sec)
    if hit_time_sec < 0:
        raise ValueError("人工补拍时间不能小于 0 秒。")
    rows = _load_detections(Path(session["source"]["detections_path"]))
    row = min(rows, key=lambda item: abs(float(item.get("time_sec", 0.0)) - hit_time_sec))
    seed = {
        "row": row,
        "hit_time_sec": hit_time_sec,
        "hitter_track_id": None,
        "confidence": 1.0,
        "source": "manual_timeline",
        "reason": "reviewer_added_timepoint",
    }
    candidate = _candidate_from_seed(rows, seed, _next_manual_shot_id(session))
    candidate["manual_added_by"] = (reviewer or "").strip()
    session.setdefault("candidates", []).append(candidate)
    _sort_candidates(session)
    _persist_session(session)
    return session, candidate


def merge_review_candidates(session, shot_ids, reviewer=""):
    """Merge reviewer-selected overlapping candidates while retaining their history."""
    selected = [get_candidate(session, shot_id) for shot_id in (shot_ids or [])]
    selected = [candidate for candidate in selected if candidate.get("active", True)]
    if len(selected) < 2:
        raise ValueError("请至少选择两条仍有效的候选后再合并。")
    hit_time = sum(float(candidate["hit_time_sec"]) for candidate in selected) / len(selected)
    rows = _load_detections(Path(session["source"]["detections_path"]))
    row = min(rows, key=lambda item: abs(float(item.get("time_sec", 0.0)) - hit_time))
    seed = {
        "row": row,
        "hit_time_sec": hit_time,
        "hitter_track_id": None,
        "confidence": 1.0,
        "source": "manual_merge",
        "reason": "reviewer_merged_candidates",
    }
    merged = _candidate_from_seed(rows, seed, _next_manual_shot_id(session))
    merged["merged_from"] = [candidate["shot_id"] for candidate in selected]
    merged["manual_added_by"] = (reviewer or "").strip()
    for candidate in selected:
        candidate["active"] = False
        candidate["superseded_by"] = merged["shot_id"]
    session.setdefault("candidates", []).append(merged)
    _sort_candidates(session)
    _persist_session(session)
    return session, merged


def split_review_candidate(session, shot_id, split_offset_sec=0.18, reviewer=""):
    """Replace one ambiguous candidate with two reviewer-created time points."""
    original = get_candidate(session, shot_id)
    if not original.get("active", True):
        raise ValueError("所选候选已经被合并或拆分，不能再次拆分。")
    split_offset_sec = float(split_offset_sec)
    if not 0.05 <= split_offset_sec <= 2.0:
        raise ValueError("拆分间隔应在 0.05 到 2.0 秒之间。")
    center = float(original["hit_time_sec"])
    first_session, first = add_manual_candidate(session, max(0.0, center - split_offset_sec), reviewer)
    second_session, second = add_manual_candidate(first_session, center + split_offset_sec, reviewer)
    original["active"] = False
    original["split_into"] = [first["shot_id"], second["shot_id"]]
    _persist_session(second_session)
    return second_session, first, second


def ensure_context_clip(session, candidate):
    """Extract an H.264 review clip that browsers can play reliably."""
    source = Path((session.get("source") or {}).get("reference_video") or "")
    if not source.is_file():
        return None
    review_dir = Path(session["source"]["analysis_dir"]) / "shot_review" / "clips"
    # Earlier clips used OpenCV's mp4v writer.  Some Chromium/Gradio builds
    # show them as 0:00 / NaN:NaN.  A separate name bypasses that stale cache.
    clip_path = review_dir / f"review_{candidate['shot_id']}.mp4"
    if clip_path.is_file() and clip_path.stat().st_size > 0:
        return str(clip_path)
    review_dir.mkdir(parents=True, exist_ok=True)
    start_time = max(0.0, float(candidate["hit_time_sec"]) - CLIP_SECONDS_BEFORE)
    clip_duration = CLIP_SECONDS_BEFORE + CLIP_SECONDS_AFTER
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-ss", f"{start_time:.3f}", "-i", str(source),
                    "-t", f"{clip_duration:.3f}", "-map", "0:v:0", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=90,
                check=True,
            )
            if clip_path.is_file() and clip_path.stat().st_size > 0:
                return str(clip_path)
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    # Last-resort fallback if ffmpeg is not installed.  The review UI will
    # still return a file, but H.264 is the supported browser path above.
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        while capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 <= start_time + clip_duration:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        writer.release()
        capture.release()
    return str(clip_path) if clip_path.is_file() and clip_path.stat().st_size > 0 else None


def _build_candidates(rows, derived_events=None):
    if derived_events is not None:
        return [_candidate_from_derived_event(event) for event in derived_events]
    event_frames = []
    for row in rows:
        events = ((row.get("spatial") or {}).get("hit_events") or [])
        for event in events:
            if event.get("status") == "candidate":
                event_frames.append({
                    "row": row,
                    "hit_time_sec": float(row.get("time_sec", 0.0)),
                    "hitter_track_id": event.get("hitter_track_id"),
                    "confidence": float(event.get("confidence") or 0.0),
                    "source": "spatial_hit_candidate",
                    "reason": event.get("reason"),
                })

    # Hit records emitted by spatial tracking are useful but sparse.  Add two
    # deliberately high-recall trajectory signals: a local direction reversal
    # and a short accepted-detection gap.  Both remain review candidates, not
    # detected facts.  This lets a reviewer recover from missed hands/poses.
    event_frames.extend(_trajectory_candidates(rows))
    if not event_frames:
        event_frames = _hand_proximity_candidates(rows)
    grouped = _deduplicate_candidates(event_frames)
    return [_candidate_from_seed(rows, seed, f"shot_{index:04d}") for index, seed in enumerate(grouped, start=1)]


def _candidate_from_derived_event(event):
    """Adapt immutable offline artifacts to the existing review UI contract."""
    proposal = dict(event.get("proposal") or {})
    label = proposal.get("label", "unknown")
    if label not in SHOT_TYPES:
        label = "unknown"
    proposal["label"] = label
    proposal["label_display"] = SHOT_TYPES[label]
    proposal.setdefault("status", "needs_human_review")
    proposal.setdefault("confidence", 0.0)
    trajectory = dict(event.get("trajectory") or {})
    evidence = dict(trajectory)
    evidence.update({
        "trajectory": trajectory,
        "hitter": event.get("hitter") or {},
        "receiver": event.get("receiver") or {},
        "event_origin": event.get("event_origin"),
        "rally_id": event.get("rally_id"),
        "shot_index_in_rally": event.get("shot_index_in_rally"),
        "image_plane_only": True,
        "decision_contract": "machine candidate only; not eligible for statistics until confirmed human review",
    })
    return {
        "shot_id": event.get("event_id"),
        "hit_frame": int(event.get("frame", 0)),
        "hit_time_sec": float(event.get("hit_time_sec", 0.0)),
        "hitter_track_id": (event.get("hitter") or {}).get("track_id"),
        "candidate_source": event.get("candidate_source", "unknown"),
        "candidate_confidence": round(float(event.get("time_confidence") or 0.0), 4),
        "candidate_reason": event.get("candidate_reason"),
        "proposal": proposal,
        "evidence": evidence,
        "active": True,
        "review": {"decision": "pending", "label": None, "reviewer": "", "note": "", "at": None},
        "review_history": [],
    }


def _candidate_from_seed(rows, seed, shot_id):
    row = seed["row"]
    hit_time = float(seed.get("hit_time_sec", row.get("time_sec", 0.0)))
    evidence = _trajectory_evidence(rows, hit_time, seed)
    return {
        "shot_id": shot_id,
        "hit_frame": int(row.get("frame", 0)),
        "hit_time_sec": hit_time,
        "hitter_track_id": seed.get("hitter_track_id"),
        "candidate_source": seed["source"],
        "candidate_confidence": round(float(seed["confidence"]), 4),
        "candidate_reason": seed.get("reason"),
        "proposal": _propose_shot_type(evidence, seed["confidence"]),
        "evidence": evidence,
        "active": True,
        "review": {"decision": "pending", "label": None, "reviewer": "", "note": "", "at": None},
        "review_history": [],
    }


def _trajectory_candidates(rows):
    observations = []
    for row in rows:
        shuttle = row.get("shuttlecock") or {}
        if shuttle.get("accepted") and shuttle.get("image"):
            observations.append({
                "row": row,
                "time_sec": float(row.get("time_sec", 0.0)),
                "point": shuttle["image"],
                "confidence": float(shuttle.get("confidence") or 0.0),
            })
    seeds = []
    # Direction reversal: ball velocity changes sharply near a likely contact.
    for before, current, after in zip(observations, observations[1:], observations[2:]):
        dt_before = current["time_sec"] - before["time_sec"]
        dt_after = after["time_sec"] - current["time_sec"]
        if not (0.025 <= dt_before <= 0.30 and 0.025 <= dt_after <= 0.30):
            continue
        velocity_before = _velocity(before["point"], current["point"], dt_before)
        velocity_after = _velocity(current["point"], after["point"], dt_after)
        speed_before = math.hypot(*velocity_before)
        speed_after = math.hypot(*velocity_after)
        if min(speed_before, speed_after) < 120.0:
            continue
        cosine = _cosine(velocity_before, velocity_after)
        if cosine > -0.50:
            continue
        confidence = min(0.45, 0.12 + 0.33 * min(current["confidence"], 1.0))
        seeds.append({
            "row": current["row"],
            "hit_time_sec": current["time_sec"],
            "hitter_track_id": None,
            "confidence": confidence,
            "source": "trajectory_turn",
            "reason": f"2d_direction_reversal cosine={cosine:.2f}; human review required",
        })
    # A short observation gap can hide the contact itself.  It is intentionally
    # lower confidence than a direction turn, but improves recall for review.
    for before, after in zip(observations, observations[1:]):
        duration = after["time_sec"] - before["time_sec"]
        displacement = _distance(before["point"], after["point"])
        if not (0.20 <= duration <= 1.20 and displacement >= 35.0):
            continue
        hit_time = (before["time_sec"] + after["time_sec"]) / 2.0
        row = before["row"] if hit_time - before["time_sec"] <= after["time_sec"] - hit_time else after["row"]
        confidence = min(0.30, 0.05 + 0.25 * min(before["confidence"], after["confidence"]))
        seeds.append({
            "row": row,
            "hit_time_sec": hit_time,
            "hitter_track_id": None,
            "confidence": confidence,
            "source": "trajectory_gap_transition",
            "reason": f"accepted_shuttle_gap={duration:.2f}s displacement={displacement:.1f}px; human review required",
        })
    return seeds


def _deduplicate_candidates(seeds, min_gap_sec=0.40):
    grouped = []
    for seed in sorted(seeds, key=lambda item: float(item.get("hit_time_sec", item["row"].get("time_sec", 0.0)))):
        current_time = float(seed.get("hit_time_sec", seed["row"].get("time_sec", 0.0)))
        if grouped:
            previous = grouped[-1]
            previous_time = float(previous.get("hit_time_sec", previous["row"].get("time_sec", 0.0)))
            if current_time - previous_time <= min_gap_sec:
                if _seed_priority(seed) >= _seed_priority(previous):
                    grouped[-1] = seed
                continue
        grouped.append(seed)
    return grouped


def _hand_proximity_candidates(rows):
    seeds = []
    for row in rows:
        shuttle = row.get("shuttlecock") or {}
        point = shuttle.get("image")
        if not shuttle.get("accepted") or not point:
            continue
        closest = (None, None, float("inf"))
        for slot, player in (row.get("players") or {}).items():
            for hand in ((player.get("hands") or {}).values()):
                if not hand:
                    continue
                distance = _distance(point, hand)
                if distance < closest[2]:
                    closest = (slot, hand, distance)
        if closest[2] <= 110.0:
            confidence = max(0.0, min(0.35, (1.0 - closest[2] / 110.0) * float(shuttle.get("confidence") or 0.0)))
            seeds.append({
                "row": row,
                "hit_time_sec": float(row.get("time_sec", 0.0)),
                "hitter_track_id": None,
                "confidence": confidence,
                "source": "hand_proximity_fallback",
                "reason": "shuttle_near_visible_hand; review required",
            })
    return seeds


def _trajectory_evidence(rows, hit_time_sec, seed):
    observed = []
    for row in rows:
        time_sec = float(row.get("time_sec", 0.0))
        if time_sec < hit_time_sec - 0.16 or time_sec > hit_time_sec + 1.80:
            continue
        shuttle = row.get("shuttlecock") or {}
        if shuttle.get("accepted") and shuttle.get("image"):
            observed.append((int(row.get("frame", 0)), time_sec, shuttle["image"], float(shuttle.get("confidence") or 0.0)))
    after = [item for item in observed if item[1] >= hit_time_sec]
    if len(after) < 2:
        return {
            "observation_count": len(after),
            "outbound_speed_px_s": None,
            "flight_duration_sec": None,
            "image_plane_only": True,
            "limitations": ["Insufficient accepted shuttle detections after the candidate hit."],
        }
    first, last = after[0], after[-1]
    duration = max(0.0, last[1] - first[1])
    distance = _distance(first[2], last[2])
    short_speeds = []
    for before, current in zip(after, after[1:]):
        delta_time = current[1] - before[1]
        if delta_time > 0:
            short_speeds.append(_distance(before[2], current[2]) / delta_time)
    return {
        "observation_count": len(after),
        "outbound_speed_px_s": round(max(short_speeds) if short_speeds else 0.0, 2),
        "image_displacement_px": round(distance, 2),
        "flight_duration_sec": round(duration, 3),
        "first_image_xy": [round(float(value), 2) for value in first[2]],
        "last_image_xy": [round(float(value), 2) for value in last[2]],
        "ball_confidence_mean": round(sum(item[3] for item in after) / len(after), 4),
        "image_plane_only": True,
        "limitations": [
            "Trajectory uses accepted 2D shuttle detections only.",
            "The current hit candidate is weak spatial evidence and requires human review.",
        ],
    }


def _propose_shot_type(evidence, hit_confidence):
    """Return a conservative proposal; unknown is preferred to a false label."""
    speed = evidence.get("outbound_speed_px_s")
    duration = evidence.get("flight_duration_sec")
    distance = evidence.get("image_displacement_px")
    ball_quality = evidence.get("ball_confidence_mean", 0.0)
    if speed is None or duration is None or distance is None:
        return _proposal("unknown", 0.0, "有效羽毛球轨迹不足，不能自动判断球种。")

    # These are deliberately weak image-plane heuristics. They rank clips for
    # review; they must not be used for score/error/player ability statistics.
    confidence_cap = min(0.55, 0.15 + 0.4 * min(float(hit_confidence), float(ball_quality)))
    if duration >= 1.1 and distance >= 220:
        return _proposal("clear", confidence_cap, "飞行时间较长且二维位移较大，建议优先复核为高远球。")
    if speed >= 1100 and duration <= 0.8:
        return _proposal("smash", confidence_cap, "出球后二维瞬时速度较高、候选飞行窗口较短，建议复核为杀球。")
    if speed >= 650 and duration <= 1.0:
        return _proposal("drive", confidence_cap * 0.9, "二维速度较快且飞行窗口较短，建议复核为平抽/平高球。")
    if speed <= 550 and duration <= 1.0:
        return _proposal("drop", confidence_cap * 0.8, "二维速度较低且飞行窗口较短，建议优先复核为吊球或网前球。")
    return _proposal("unknown", confidence_cap * 0.5, "二维轨迹不足以区分高远、吊球与杀球，请人工确认。")


def _proposal(label, confidence, rationale):
    return {
        "label": label,
        "label_display": SHOT_TYPES[label],
        "confidence": round(float(confidence), 4),
        "status": "needs_human_review",
        "rationale": rationale,
    }


def _resolve_reference_video(analysis_path, supplied_path):
    if supplied_path:
        supplied = Path(supplied_path)
        if supplied.is_file():
            return supplied.resolve()
    preferred = [
        analysis_path / "web_detect_video.mp4",
        analysis_path / "web_annotated_video.mp4",
        analysis_path / "web_video.mp4",
        analysis_path / "web_skeleton_video.mp4",
    ]
    preferred.extend(sorted(analysis_path.glob("*.mp4")))
    for candidate in preferred:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_detections(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("detections.jsonl 为空，无法生成球路复核候选。")
    return rows


def _distance(left, right):
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _velocity(left, right, seconds):
    return (
        (float(right[0]) - float(left[0])) / seconds,
        (float(right[1]) - float(left[1])) / seconds,
    )


def _cosine(left, right):
    left_length = math.hypot(*left)
    right_length = math.hypot(*right)
    if left_length == 0 or right_length == 0:
        return 1.0
    return (left[0] * right[0] + left[1] * right[1]) / (left_length * right_length)


def _seed_priority(seed):
    source_priority = {
        "spatial_hit_candidate": 4,
        "trajectory_turn": 3,
        "trajectory_gap_transition": 2,
        "hand_proximity_fallback": 1,
    }
    return source_priority.get(seed.get("source"), 0), float(seed.get("confidence") or 0.0)


def _active_candidates(session):
    return [candidate for candidate in session.get("candidates", []) if candidate.get("active", True)]


def _next_manual_shot_id(session):
    existing = {candidate.get("shot_id") for candidate in session.get("candidates", [])}
    index = 1
    while f"manual_{index:04d}" in existing:
        index += 1
    return f"manual_{index:04d}"


def _sort_candidates(session):
    session["candidates"] = sorted(
        session.get("candidates", []),
        key=lambda candidate: (not candidate.get("active", True), float(candidate.get("hit_time_sec", 0.0)), candidate.get("shot_id", "")),
    )


def _persist_session(session):
    source_dir = Path(session["source"]["analysis_dir"])
    _write_json(source_dir / "shot_review" / SESSION_FILENAME, session)


def _preserve_manual_review_work(previous, rebuilt):
    """Carry human work across a machine-candidate regeneration.

    Automated candidates can change count/order after generator upgrades.  A
    nearby replacement inherits a non-pending review; manual candidates that
    no longer have a nearby machine seed are copied through unchanged.
    """
    candidates = rebuilt.get("candidates", [])
    for old in previous.get("candidates", []):
        review = old.get("review") or {}
        is_manual = str(old.get("candidate_source", "")).startswith("manual_")
        if not is_manual and review.get("decision", "pending") == "pending":
            continue
        old_time = float(old.get("hit_time_sec", 0.0))
        nearby = [
            candidate for candidate in candidates
            if abs(float(candidate.get("hit_time_sec", 0.0)) - old_time) <= 0.45
        ]
        if nearby:
            target = min(nearby, key=lambda candidate: abs(float(candidate["hit_time_sec"]) - old_time))
            target["review"] = review
            target["review_history"] = old.get("review_history", [])
            continue
        copied = dict(old)
        copied["active"] = old.get("active", True)
        copied["preserved_across_regeneration"] = True
        candidates.append(copied)
    _sort_candidates(rebuilt)


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


