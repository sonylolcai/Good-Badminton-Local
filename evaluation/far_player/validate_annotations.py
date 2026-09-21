#!/usr/bin/env python
"""Validate a human-reviewed far-player annotation set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.far_player.annotations import load_jsonl, validate_annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    try:
        rows = load_jsonl(args.annotations)
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        video = metadata["video"]
        errors = validate_annotations(rows, int(video["width"]), int(video["height"]), require_complete=True)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    if errors:
        print(f"INVALID: {len(errors)} error(s)", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    far_count = sum(
        person.get("role") == "far_player" and person.get("visibility", "visible") in {"visible", "partial"}
        for row in rows
        for person in row.get("people", [])
    )
    if far_count == 0:
        print("INVALID: no visible far_player ground-truth boxes", file=sys.stderr)
        return 2
    print(f"VALID: {len(rows)} reviewed frames, {far_count} far-player boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
