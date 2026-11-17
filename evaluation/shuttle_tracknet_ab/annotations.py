"""Annotation contract for the shuttlecock A/B benchmark.

The benchmark deliberately keeps human labels separate from both model outputs.
It needs only a frame number, a review status, and either a visible shuttle
centre or an explicit ``not_visible`` label.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


VISIBILITY_STATES = {"visible", "not_visible", "ambiguous"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: every JSONL entry must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def validate_annotations(
    rows: list[dict],
    width: int,
    height: int,
    *,
    require_complete: bool = True,
) -> list[str]:
    """Return all validation errors without silently repairing human labels."""
    errors: list[str] = []
    seen_frames: set[int] = set()
    for row_number, row in enumerate(rows, 1):
        prefix = f"row {row_number}"
        frame_index = row.get("frame_index")
        if not isinstance(frame_index, int) or frame_index < 0:
            errors.append(f"{prefix}: frame_index must be a non-negative integer")
        elif frame_index in seen_frames:
            errors.append(f"{prefix}: duplicate frame_index {frame_index}")
        else:
            seen_frames.add(frame_index)

        time_sec = row.get("time_sec")
        if not isinstance(time_sec, (int, float)) or time_sec < 0:
            errors.append(f"{prefix}: time_sec must be a non-negative number")
        if require_complete and row.get("label_status") != "complete":
            errors.append(f"{prefix}: label_status is not 'complete'")

        shuttle = row.get("shuttle")
        if not isinstance(shuttle, dict):
            errors.append(f"{prefix}: shuttle must be an object")
            continue
        visibility = shuttle.get("visibility")
        if visibility not in VISIBILITY_STATES:
            errors.append(f"{prefix}: shuttle.visibility must be one of {sorted(VISIBILITY_STATES)}")
            continue
        point = shuttle.get("image_xy")
        if visibility == "visible":
            if not _valid_point(point, width, height):
                errors.append(f"{prefix}: a visible shuttle needs image_xy inside {width}x{height}")
        elif point is not None:
            errors.append(f"{prefix}: {visibility} shuttle must have image_xy=null")
    if not rows:
        errors.append("annotation file is empty")
    return errors


def _valid_point(point: object, width: int, height: int) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False
    if not all(isinstance(value, (int, float)) for value in point):
        return False
    x, y = point
    return 0 <= x < width and 0 <= y < height
