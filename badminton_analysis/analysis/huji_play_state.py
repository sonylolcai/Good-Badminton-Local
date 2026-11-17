"""Huji-compatible coarse match-play evidence.

This module deliberately models only Huji's *scene* decision: whether a
sampled frame appears to be part of active play.  It does not turn a scene
model into shuttle, hit, landing, score, or player-identity evidence.

The optional generator accepts a separately supplied Ultralytics
classification checkpoint whose labels include ``play_ball``.  Huji's public
configuration references separate badminton singles/doubles ``best.pt`` files,
but does not make those trained weights a Good-Badminton dependency.  Keeping
the checkpoint explicit makes this feature deployable only after its model and
license have been reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path


HUJI_PLAY_STATE_FILENAME = "huji_play_state_v1.jsonl"
HUJI_PLAY_STATE_VERSION = "1.0"
PLAY_ACTION = "play_ball"


def generate_huji_play_state(video_path, model_path, output_path, sample_hz=6.0, image_size=640):
    """Run an explicit Huji-style action classifier and save immutable evidence.

    The sampling and rolling decision mirror Huji's published algorithm: 6
    samples per second, with a two-second window requiring at least six
    ``play_ball`` classifications.  This function is intentionally separate
    from the core detector so an unavailable optional model cannot stop the
    normal human/shuttle analysis.
    """
    import cv2
    from ultralytics import YOLO

    video_path = Path(video_path)
    model_path = Path(model_path)
    output_path = Path(output_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Huji play-state input video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Huji action-classification model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open Huji play-state input video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read FPS from Huji play-state input video: {video_path}")

    model = YOLO(str(model_path))
    sample_hz = max(1.0, float(sample_hz))
    sample_step = max(1, int(round(fps / sample_hz)))
    records = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_step:
            frame_index += 1
            continue
        prediction = model.predict(frame, imgsz=int(image_size), verbose=False)[0]
        names = prediction.names
        top_index = int(prediction.probs.top1)
        label = str(names[top_index])
        confidence = float(prediction.probs.top1conf)
        records.append(
            {
                "schema_version": HUJI_PLAY_STATE_VERSION,
                "time_sec": round(frame_index / fps, 6),
                "action_type": label,
                "confidence": round(confidence, 6),
                "source": "huji_compatible_ultralytics_action_classifier",
                "sample_hz": round(fps / sample_step, 6),
                "model_path": str(model_path),
            }
        )
        frame_index += 1
    cap.release()

    annotated = annotate_huji_play_segments(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in annotated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "path": str(output_path),
        "sample_count": len(annotated),
        "playing_sample_count": sum(item.get("is_playing") is True for item in annotated),
        "sample_hz": round(fps / sample_step, 6),
        "policy": "coarse scene evidence only; never score or terminal evidence by itself",
    }


def annotate_huji_play_segments(records, window_seconds=2.0, min_play_samples=6):
    """Attach Huji-equivalent rolling-window active-play flags to samples."""
    ordered = sorted((dict(item) for item in records), key=lambda item: float(item.get("time_sec", 0.0)))
    for index, record in enumerate(ordered):
        now = float(record.get("time_sec", 0.0))
        in_window = [
            item for item in ordered
            if now - float(window_seconds) < float(item.get("time_sec", 0.0)) <= now
        ]
        play_count = sum(item.get("action_type") == PLAY_ACTION for item in in_window)
        record["play_window_seconds"] = float(window_seconds)
        record["play_window_count"] = int(play_count)
        record["is_playing"] = bool(play_count >= int(min_play_samples))
        record["evidence_kind"] = "coarse_scene_classification"
    return ordered


def load_huji_play_state(path):
    """Load a generated play-state file without accepting malformed rows."""
    source = Path(path) if path else None
    if source is None or not source.is_file():
        return []
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict) and isinstance(item.get("time_sec"), (int, float)):
                rows.append(item)
        except json.JSONDecodeError:
            continue
    return annotate_huji_play_segments(rows)


def terminal_play_context(play_state_rows, time_sec, continuation_seconds=1.0):
    """Return conflict context only when play clearly continues past a terminal.

    A label at the exact candidate time is not enough: Huji's two-second window
    intentionally lags.  We therefore require active-play evidence at least
    one second later.  It remains an advisory conflict, not a proof that the
    shuttle did not land or go out.
    """
    if not play_state_rows:
        return None
    threshold = float(time_sec) + float(continuation_seconds)
    continuing = [
        item for item in play_state_rows
        if float(item.get("time_sec", -1.0)) >= threshold and item.get("is_playing") is True
    ]
    if not continuing:
        return None
    first = continuing[0]
    return {
        "source": str(first.get("source") or "huji_compatible_play_state"),
        "status": "active_play_continues_after_candidate",
        "first_continuing_time_sec": round(float(first["time_sec"]), 6),
        "continuation_seconds": float(continuation_seconds),
        "evidence_kind": "coarse_scene_classification",
    }
