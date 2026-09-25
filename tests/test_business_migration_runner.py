import unittest
from pathlib import Path


class BusinessMigrationRunnerTests(unittest.TestCase):
    def test_records_description_for_existing_business_database(self):
        source = (Path(__file__).parents[1] / "deploy" / "business-server" / "apply_migrations.py").read_text(encoding="utf-8")

        self.assertIn("add column if not exists description", source)
        self.assertIn("(version, description)", source)


if __name__ == "__main__":
    unittest.main()
