import unittest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from operator_api.services.retention import run_retention_once
from operator_api.services.resource_lifecycle import BusinessResourceService


class RetentionWorkerTests(unittest.TestCase):
    def test_disabled_or_not_due_policy_does_nothing(self):
        class Database:
            def claim_video_retention_run(self):
                return None

        result = run_retention_once(
            database=Database(),
            resource_service=None,
            evaluation_cleanup=lambda _days: self.fail("evaluation must not be called"),
        )

        self.assertEqual(result, {"status": "skipped"})

    def test_enabled_run_cleans_business_and_evaluation_then_marks_complete(self):
        class Database:
            completed = False

            def claim_video_retention_run(self):
                return {"retention_days": 7}

            def complete_video_retention_run(self):
                self.completed = True

        class Resources:
            def cleanup_expired_videos(self, days):
                return {"retention_days": days, "resources": ["asset-1"]}

        database = Database()
        result = run_retention_once(
            database=database,
            resource_service=Resources(),
            evaluation_cleanup=lambda days: {"retention_days": days, "runs": ["run-1"]},
        )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(database.completed)
        self.assertEqual(result["evaluation"]["retention_days"], 7)

    def test_cutoff_is_exactly_seven_days_from_upload_success_time(self):
        class Database:
            def __init__(self):
                self.cutoff = None

            def expired_media_asset_ids(self, cutoff):
                self.cutoff = cutoff
                return []

        database = Database()
        current = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temporary:
            service = BusinessResourceService(database, Path(temporary))
            service.cleanup_expired_videos(7, now=current)

        self.assertEqual(database.cutoff, datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
