import copy
import json
import unittest
from pathlib import Path

from analysis_platform.contracts import (
    contract_sha256,
    validate_contract,
    verify_contract_sha256,
)


FIXTURES = Path(__file__).parent / "fixtures"


class AnalysisPlatformContractTests(unittest.TestCase):
    def test_valid_boundary_contracts(self):
        payloads = json.loads(
            (FIXTURES / "analysis_contract_valid.json").read_text(encoding="utf-8")
        )

        for name, payload in payloads.items():
            with self.subTest(name=name):
                self.assertEqual(validate_contract(payload), payload)
        self.assertEqual(
            payloads["published_result_ref"]["source_result_sha256"],
            contract_sha256(payloads["gpu_analysis_result"]),
        )

    def test_business_identity_is_rejected_from_analysis_parameters(self):
        payload = json.loads(
            (FIXTURES / "analysis_contract_invalid.json").read_text(encoding="utf-8")
        )

        with self.assertRaisesRegex(ValueError, "business identity"):
            validate_contract(payload)

    def test_contract_hash_detects_a_frozen_input_edit(self):
        task = json.loads(
            (FIXTURES / "analysis_contract_valid.json").read_text(encoding="utf-8")
        )["analysis_task"]
        frozen_sha256 = contract_sha256(task)
        verify_contract_sha256(task, frozen_sha256)

        changed = copy.deepcopy(task)
        changed["parameters"]["pose_imgsz"] = 1280
        with self.assertRaisesRegex(ValueError, "parameters_fingerprint does not match"):
            verify_contract_sha256(changed, frozen_sha256)

    def test_result_paths_cannot_escape_the_run_store(self):
        result = json.loads(
            (FIXTURES / "analysis_contract_valid.json").read_text(encoding="utf-8")
        )["gpu_analysis_result"]
        result["artifacts"][0]["path"] = "../metadata.json"

        with self.assertRaisesRegex(ValueError, "relative path"):
            validate_contract(result)


if __name__ == "__main__":
    unittest.main()
