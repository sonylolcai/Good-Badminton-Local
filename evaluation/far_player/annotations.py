"""Annotation loading, validation and template generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


ALLOWED_VISIBILITY = {"visible", "partial", "not_visible"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_annotations(rows: list[dict], width: int, height: int, require_complete: bool = True) -> list[str]:
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
        if row.get("people_annotation_complete") not in (True, False):
            errors.append(f"{prefix}: people_annotation_complete must be true or false")
        for person_index, person in enumerate(row.get("people", [])):
            person_prefix = f"{prefix} person {person_index}"
            if not person.get("id"):
                errors.append(f"{person_prefix}: id is required")
            if person.get("role") not in {"far_player", "other_player", "non_player_person"}:
                errors.append(f"{person_prefix}: invalid role")
            if person.get("visibility", "visible") not in ALLOWED_VISIBILITY:
                errors.append(f"{person_prefix}: invalid visibility")
            _validate_box(person.get("bbox_xyxy"), width, height, person_prefix, errors)
        for region_index, region in enumerate(row.get("ignore_regions", [])):
            _validate_box(region, width, height, f"{prefix} ignore_region {region_index}", errors)
    if not rows:
        errors.append("annotation file is empty")
    return errors


def _validate_box(box, width: int, height: int, prefix: str, errors: list[str]) -> None:
    if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) for value in box):
        errors.append(f"{prefix}: bbox must be [x1, y1, x2, y2]")
        return
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        errors.append(f"{prefix}: bbox {box} is outside {width}x{height} or has invalid ordering")
