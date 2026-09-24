"""Business-owned video inventory, manual analysis, and physical deletion."""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from operator_api.services.remote_gpu import delete_remote_job, delete_remote_stream, submit_remote_job


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


class BusinessResourceService:
    def __init__(
        self,
        database,
        media_root: Path | None = None,
        *,
        submitter=submit_remote_job,
        remote_delete=delete_remote_job,
        remote_stream_delete=delete_remote_stream,
    ):
        self.database = database
        self.media_root = Path(
            media_root or os.environ.get("GOOD_BADMINTON_MEDIA_ROOT", "outputs/business-media")
        ).resolve()
        self.media_root.mkdir(parents=True, exist_ok=True)
        self.storage_roots = {
            self.media_root,
            Path(os.environ.get("GOOD_BADMINTON_EDGE_RECORDING_DIR", "outputs/edge_recordings")).resolve(),
        }
        self.submitter = submitter
        self.remote_delete = remote_delete
        self.remote_stream_delete = remote_stream_delete

    def target_path(self, asset_id: str, suffix: str) -> Path:
        if suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("unsupported video extension")
        target = (self.media_root / asset_id / f"source{suffix.lower()}").resolve()
        target.relative_to(self.media_root)
        target.parent.mkdir(parents=True, exist_ok=False)
        return target

    def register_uploaded_video(
        self,
        *,
        asset_id: str,
        path: Path,
        tenant_id: str,
        venue_id: str,
        player_id: str | None,
        match_id: str | None,
        media_type: str,
        original_filename: str,
        actor_admin_id: str,
    ) -> dict:
        path = self._safe_path(path)
        uploaded_at = datetime.now(timezone.utc)
        try:
            return self.database.register_media_asset(
                asset_id=asset_id,
                tenant_id=tenant_id,
                venue_id=venue_id,
                player_id=player_id,
                match_id=match_id,
                media_type=media_type,
                original_filename=original_filename,
                uploaded_at=uploaded_at,
                location_id=str(uuid.uuid4()),
                location_ref=str(path),
                sha256_digest=_sha256(path),
                size_bytes=path.stat().st_size,
                actor_admin_id=actor_admin_id,
            )
        except Exception:
            path.unlink(missing_ok=True)
            path.parent.rmdir()
            raise

    def list_assets(self, venue_id: str | None = None) -> list[dict]:
        return self.database.list_media_assets(venue_id)

    def delete_asset(self, asset_id: str, actor_admin_id: str, *, full: bool = False) -> dict:
        asset = self.database.get_media_asset(asset_id)
        results = []
        for location in asset["locations"]:
            if location["deletion_status"] == "deleted":
                results.append({**location, "status": "deleted"})
                continue
            self.database.mark_media_location(location["location_id"], "deleting")
            try:
                self._delete_location(location, full=full)
                self.database.mark_media_location(location["location_id"], "deleted")
                results.append({**location, "status": "deleted"})
            except Exception as exc:
                message = str(exc)[:2000]
                self.database.mark_media_location(location["location_id"], "failed", message)
                results.append({**location, "status": "failed", "error": message})
        failed = [item for item in results if item["status"] == "failed"]
        if failed:
            self.database.mark_media_asset_status(asset_id, "delete_failed")
        elif full:
            self.database.delete_media_asset(asset_id, actor_admin_id)
        else:
            self.database.mark_media_asset_status(asset_id, "resources_deleted")
        return {
            "id": asset_id,
            "venue_id": asset["venue_id"],
            "mode": "all" if full else "resources",
            "status": "partial" if failed else "deleted",
            "retryable": bool(failed),
            "record_deleted": bool(full and not failed),
            "locations": results,
        }

    def trigger_analysis(
        self,
        asset_id: str,
        actor_admin_id: str,
        template_path: Path,
        corners: list,
        options: dict | None = None,
    ) -> dict:
        asset = self.database.get_media_asset(asset_id)
        local = next(
            (
                item for item in asset["locations"]
                if item["storage_backend"] == "local_disk" and item["deletion_status"] != "deleted"
            ),
            None,
        )
        if local is None:
            raise ValueError("video resource has already been deleted")
        video = self._safe_path(Path(local["location_ref"]))
        normalized_corners = _corners(corners)
        job_id = self.database.create_manual_analysis_job(asset_id, actor_admin_id)
        try:
            receipt = self.submitter(
                video,
                Path(template_path),
                normalized_corners,
                {"sport_id": "badminton", **(options or {})},
                f"business-{job_id}",
            )
            remote_job_id = receipt["job_id"]
            self.database.bind_manual_analysis_job(job_id, asset_id, remote_job_id)
            return {
                "analysis_job_id": job_id,
                "remote_job_id": remote_job_id,
                "status": receipt.get("status", "queued"),
            }
        except Exception as exc:
            self.database.fail_manual_analysis_job(job_id, str(exc)[:2000])
            raise

    def _delete_location(self, location: dict, *, full: bool) -> None:
        backend = location["storage_backend"]
        if backend == "local_disk":
            path = Path(location["location_ref"]).resolve()
            if not any(_is_within(path, root) for root in self.storage_roots):
                raise ValueError("resource path is outside configured business storage")
            path.unlink(missing_ok=True)
            try:
                path.parent.rmdir()
            except OSError:
                pass
            return
        if backend == "gpu_http":
            kind, identifier = location["location_ref"].split(":", 1)
            if kind == "job":
                self.remote_delete(identifier, full=full)
            elif kind == "stream":
                self.remote_stream_delete(identifier, full=full)
            else:
                raise ValueError("unsupported GPU resource location")
            return
        raise ValueError(f"storage backend is not configured: {backend}")

    def _safe_path(self, path: Path) -> Path:
        resolved = Path(path).resolve()
        if not any(_is_within(resolved, root) for root in self.storage_roots):
            raise ValueError("resource path is outside configured business storage")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _corners(value: list) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("court_corners must contain four coordinate pairs")
    try:
        corners = [[float(point[0]), float(point[1])] for point in value]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("court_corners must contain four numeric coordinate pairs") from exc
    if any(len(point) != 2 for point in value):
        raise ValueError("court_corners must contain four numeric coordinate pairs")
    return corners
