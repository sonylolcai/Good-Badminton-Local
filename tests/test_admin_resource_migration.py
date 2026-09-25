import unittest
from pathlib import Path


class AdminResourceMigrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        migrations = Path(__file__).resolve().parents[1] / "business_gateway" / "migrations"
        cls.sql = "\n".join(path.read_text(encoding="utf-8").lower() for path in migrations.glob("*.sql"))

    def test_admin_auth_and_rbac_tables_exist(self):
        self.assertIn("business.admin_accounts", self.sql)
        self.assertIn("business.admin_role_assignments", self.sql)
        self.assertIn("business.admin_sessions", self.sql)
        self.assertIn("'gpu', 'admin'", self.sql)

    def test_media_inventory_and_retention_policy_tables_exist(self):
        self.assertIn("business.managed_media_resources", self.sql)
        self.assertIn("business.managed_media_resource_locations", self.sql)
        self.assertNotIn("create table if not exists business.media_assets", self.sql)
        self.assertIn("business.video_retention_policy", self.sql)

    def test_players_are_global_and_play_records_bind_venue_and_court(self):
        self.assertIn("business.player_play_records", self.sql)
        self.assertIn("mp.user_id as player_id", self.sql)
        self.assertIn("m.venue_id", self.sql)
        self.assertIn("m.court_id", self.sql)
        self.assertIn("must not represent player ownership", self.sql)


if __name__ == "__main__":
    unittest.main()
