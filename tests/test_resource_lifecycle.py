import tempfile
import unittest
from pathlib import Path

from operator_api.services.resource_lifecycle import BusinessResourceService


class FakeDatabase:
    def __init__(self, asset=None):
        self.asset = asset
        self.location_updates = []
        self.asset_status = []
        self.deleted_assets = []
        self.registered = None
        self.bound = None

    def register_media_asset(self, **values):
        self.registered = values
        return {"id": values["asset_id"], "upload_succeeded_at": values["uploaded_at"].isoformat()}

    def get_media_asset(self, _asset_id):
        return self.asset

    def mark_media_location(self, location_id, status, error=None):
        self.location_updates.append((location_id, status, error))

    def mark_media_asset_status(self, asset_id, status):
        self.asset_status.append((asset_id, status))

    def delete_media_asset(self, asset_id, actor_id):
        self.deleted_assets.append((asset_id, actor_id))

    def create_manual_analysis_job(self, _asset_id, _actor_id):
        return "analysis-job-1"

    def bind_manual_analysis_job(self, job_id, asset_id, remote_job_id):
        self.bound = (job_id, asset_id, remote_job_id)

    def fail_manual_analysis_job(self, _job_id, _message):
        pass


class BusinessResourceServiceTests(unittest.TestCase):
    def test_upload_success_time_is_recorded_after_file_is_durable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = FakeDatabase()
            service = BusinessResourceService(database, root)
            path = service.target_path("asset-1", ".mp4")
            path.write_bytes(b"video")

            result = service.register_uploaded_video(
                asset_id="asset-1", path=path, tenant_id="tenant-1", venue_id="venue-1",
                player_id="player-1", match_id=None, media_type="video/mp4",
                original_filename="match.mp4", actor_admin_id="admin-1",
            )

            self.assertTrue(path.is_file())
            self.assertEqual(result["id"], "asset-1")
            self.assertEqual(database.registered["size_bytes"], 5)
            self.assertIsNotNone(database.registered["uploaded_at"])

    def test_resource_delete_syncs_local_and_remote_but_keeps_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "asset" / "source.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            asset = _asset(video)
            database = FakeDatabase(asset)
            remote_calls = []
            service = BusinessResourceService(
                database, root, remote_delete=lambda job_id, full=False: remote_calls.append((job_id, full))
            )

            result = service.delete_asset("asset-1", "admin-1")

            self.assertEqual(result["status"], "deleted")
            self.assertFalse(video.exists())
            self.assertEqual(remote_calls, [("remote-1", False)])
            self.assertEqual(database.asset_status, [("asset-1", "resources_deleted")])
            self.assertEqual(database.deleted_assets, [])

    def test_full_delete_removes_record_only_after_every_location_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "asset" / "source.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            database = FakeDatabase(_asset(video))
            service = BusinessResourceService(database, root, remote_delete=lambda *_args, **_kwargs: None)

            result = service.delete_asset("asset-1", "admin-1", full=True)

            self.assertTrue(result["record_deleted"])
            self.assertEqual(database.deleted_assets, [("asset-1", "admin-1")])

    def test_manual_analysis_links_the_remote_job_to_the_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "asset" / "source.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            template = root / "court.png"
            template.write_bytes(b"image")
            database = FakeDatabase(_asset(video, remote=False))
            service = BusinessResourceService(
                database,
                root,
                submitter=lambda *_args: {"job_id": "remote-2", "status": "queued"},
            )

            result = service.trigger_analysis(
                "asset-1", "admin-1", template, [[1, 1], [2, 1], [2, 2], [1, 2]]
            )

            self.assertEqual(result["remote_job_id"], "remote-2")
            self.assertEqual(database.bound, ("analysis-job-1", "asset-1", "remote-2"))


def _asset(video, *, remote=True):
    locations = [{
        "location_id": "local-1", "storage_backend": "local_disk",
        "location_ref": str(video), "deletion_status": "active",
    }]
    if remote:
        locations.append({
            "location_id": "remote-1", "storage_backend": "gpu_http",
            "location_ref": "job:remote-1", "deletion_status": "active",
        })
    return {
        "id": "asset-1", "tenant_id": "tenant-1", "venue_id": "venue-1",
        "player_id": "player-1", "match_id": None, "media_type": "video/mp4",
        "original_filename": "match.mp4", "upload_succeeded_at": "2026-09-24T00:00:00Z",
        "status": "active", "locations": locations,
    }


if __name__ == "__main__":
    unittest.main()
