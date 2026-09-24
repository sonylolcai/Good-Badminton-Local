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

    def test_media_inventory_and_retention_policy_tables_exist(self):
        self.assertIn("business.media_assets", self.sql)
        self.assertIn("business.media_asset_locations", self.sql)
        self.assertIn("business.video_retention_policy", self.sql)


if __name__ == "__main__":
    unittest.main()
