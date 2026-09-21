"""Replay person-only rally boundaries for the approved stable-window sweep."""

from __future__ import annotations

import json
import time
from pathlib import Path

from badminton_analysis.analysis.fixed_camera_match import RallyStateMachine


WINDOWS_SECONDS = (0.5, 0.7, 1.0)


def evaluate_movement_rally_windows(detections_path, output_path, *, fps, match_mode="singles"):
    """Compare the three approved human-stability terminal windows.

    The replay consumes the same fresh/detected track records used online. It
    is not a ground-truth score evaluator; manual terminal references can be
    compared against this compact artifact later without rerunning models.
    """
    fps = max(1.0, float(fps or 0.0))
    expected_players = 2 if match_mode == "singles" else 4
    machines = {
        window: RallyStateMachine(
            fps=fps,
            min_active_frames=4,
            shuttle_enabled=False,
            expected_player_count=expected_players,
            settle_window_seconds=window,
        )
        for window in WINDOWS_SECONDS
    }
    source_frames = 0
    path = Path(detections_path)
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                spatial = record.get("spatial") or {}
                tracks = spatial.get("tracks")
                frame = record.get("frame")
                if not isinstance(frame, int) or not isinstance(tracks, list):
                    continue
                source_frames = max(source_frames, frame)
                for machine in machines.values():
                    machine.update(frame, tracks, None, [])
    results = {}
    for window, machine in machines.items():
        rallies = machine.finalize(source_frames or None)
        results[f"{window:.1f}"] = {
            "settle_window_seconds": window,
            "candidate_rally_count": len(rallies),
            "rallies": [
                {
                    **rally,
                    "start_time_sec": _frame_time(rally.get("start_frame"), fps),
                    "end_time_sec": _frame_time(rally.get("end_frame"), fps),
                }
                for rally in rallies
            ],
        }
    payload = {
        "schema_version": "1.0",
        "kind": "movement_only_rally_window_sweep",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "detections.jsonl spatial.tracks",
        "match_mode": match_mode,
        "fps": fps,
        "windows_seconds": list(WINDOWS_SECONDS),
        "policy": (
            "All expected players must be fresh detected and below the stable speed threshold. "
            "Gaps, predicted rows, missing rows, malformed coordinates and roster mismatch invalidate a window."
        ),
        "results": results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(destination), **payload}


def _frame_time(frame, fps):
    return round(float(frame) / fps, 3) if frame is not None else None
