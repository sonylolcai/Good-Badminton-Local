import copy
import json
import tempfile
import unittest
from pathlib import Path

from analysis_platform.contracts import contract_sha256
from analysis_platform.store import LocalManifestStore


FIXTURES = Path(__file__).parent / "fixtures"


def dataset_version():
    return {
        "schema_version": "analysis-platform.v1",
        "kind": "dataset_version",
        "dataset_version_id": "smoke-v1.0.0",
        "dataset_id": "smoke",
        "version": "1.0.0",
        "purpose": "local contract smoke",
        "split": "smoke",
        "created_at": "2026-09-14T00:00:00Z",
        "created_by": "agent_0",
        "cases": [
            {"case_id": "case_demo_mp4", "annotation_version_id": "annotation_demo_v1"}
        ],
        "license_and_consent_status": "internal_test_only",
    }


class LocalManifestStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = LocalManifestStore(self.temporary.name)
        self.contracts = json.loads(
            (FIXTURES / "analysis_contract_valid.json").read_text(encoding="utf-8")
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_draft_can_change_but_published_version_is_write_once(self):
        first = dataset_version()
        changed = copy.deepcopy(first)
        changed["purpose"] = "updated draft"

        self.store.save_draft("dataset_version", first)
        self.store.save_draft("dataset_version", changed)
        self.assertEqual(
            self.store.read_manifest("dataset_version", "smoke-v1.0.0", draft=True)["purpose"],
            "updated draft",
        )

        created = self.store.publish_manifest("dataset_version", first)
        repeated = self.store.publish_manifest("dataset_version", first)
        self.assertTrue(created["created"])
        self.assertFalse(repeated["created"])
        self.assertEqual(self.store.list_manifest_ids("dataset_version"), ["smoke-v1.0.0"])
        with self.assertRaisesRegex(ValueError, "immutable object"):
            self.store.publish_manifest("dataset_version", changed)

    def test_read_detects_a_tampered_published_manifest(self):
        stored = self.store.publish_manifest("dataset_version", dataset_version())
        path = Path(stored["path"])
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["payload"]["purpose"] = "tampered"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.store.read_manifest("dataset_version", "smoke-v1.0.0")

    def test_run_input_result_publication_and_baseline_form_one_hash_chain(self):
        task = self.contracts["analysis_task"]
        result = self.contracts["gpu_analysis_result"]
        published = self.contracts["published_result_ref"]

        self.store.create_run(task)
        self.store.write_run_result(result)
        self.store.publish_run_result(published)
        baseline = self.store.promote_baseline(
            "production-badminton",
            task["run_id"],
            actor="agent_0",
            reason="golden contract smoke passed",
        )

        self.assertTrue(Path(baseline["path"]).is_file())
        self.assertTrue((Path(self.temporary.name) / "audit.jsonl").is_file())
        self.assertEqual(published["source_result_sha256"], contract_sha256(result))

    def test_result_cannot_change_the_frozen_run_identity(self):
        self.store.create_run(self.contracts["analysis_task"])
        changed = copy.deepcopy(self.contracts["gpu_analysis_result"])
        changed["input_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "does not match"):
            self.store.write_run_result(changed)


if __name__ == "__main__":
    unittest.main()
