"""Local fixed-video catalogue owned by the business gateway.

The catalogue deliberately keeps the operator's source path on the business
machine.  Public callers receive only an asset id, display metadata and the
analysis/calibration contract; they never receive a Windows path or a GPU
credential.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CATALOG_FILE = "fixed_video_catalog.local.json"


def catalog_path() -> Path:
    return Path(__file__).with_name(_CATALOG_FILE)


def load_fixed_video_catalog() -> list[dict[str, Any]]:
    """Load the uncommitted local catalogue and validate its small contract."""

    path = catalog_path()
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    videos = raw.get("videos")
    if raw.get("schema_version") != "fixed-video-catalog.v1" or not isinstance(videos, list):
        raise ValueError("fixed video catalogue must use fixed-video-catalog.v1 with a videos list")

    seen_ids: set[str] = set()
    for video in videos:
        if not isinstance(video, dict):
            raise ValueError("each fixed video entry must be an object")
        video_id = str(video.get("id", "")).strip()
        source_path = str(video.get("source_path", "")).strip()
        corners = video.get("court_corners")
        if not video_id or not source_path:
            raise ValueError("each fixed video requires id and source_path")
        if video_id in seen_ids:
            raise ValueError(f"duplicate fixed video id: {video_id}")
        if video.get("expected_player_count") not in {2, 4}:
            raise ValueError("expected_player_count must be 2 or 4")
        if video.get("match_format") not in {"singles", "doubles"}:
            raise ValueError("match_format must be singles or doubles")
        if not isinstance(corners, list) or len(corners) != 4 or any(
            not isinstance(point, list) or len(point) != 2 for point in corners
        ):
            raise ValueError("court_corners must contain four [x, y] points")
        seen_ids.add(video_id)
    return videos


def get_fixed_video(video_id: str) -> dict[str, Any]:
    """Return one business-only asset entry, including its local source path."""

    requested_id = str(video_id).strip()
    for video in load_fixed_video_catalog():
        if video["id"] == requested_id:
            return video
    raise KeyError(requested_id)


def list_public_fixed_videos() -> list[dict[str, Any]]:
    """Return safe catalogue fields for a client-facing selector."""

    return [
        {
            "id": video["id"],
            "label": video["label"],
            "match_format": video["match_format"],
            "expected_player_count": video["expected_player_count"],
            "analysis": video["analysis"],
            "calibration_id": video["calibration_id"],
        }
        for video in load_fixed_video_catalog()
    ]
