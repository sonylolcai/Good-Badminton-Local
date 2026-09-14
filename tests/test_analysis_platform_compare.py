import copy
import unittest

from analysis_platform.compare import compare_reports


def metric(key, value, *, scope="run", scope_id="all", version="1.0.0", unit="ratio"):
    return {
        "metric_key": key,
        "metric_definition_version": version,
        "scope": scope,
        "scope_id": scope_id,
        "value": value,
        "unit": unit,
        "sample_count": 20,
        "eligible_sample_count": 20,
        "status": "valid",
    }


def report(run_id, metrics):
    return {
        "run_id": run_id,
        "dataset_version": "regression-v1.0.0",
        "dataset_manifest_sha256": "a" * 64,
        "hardware_fingerprint": "gpu-3090",
        "execution_mode": "batch-gpu",
        "metrics": metrics,
    }


class RunComparisonTests(unittest.TestCase):
    def test_recall_gain_does_not_hide_false_positive_regression(self):
        baseline = report(
            "run_base",
            [
                metric("person.recall", 0.8),
                metric("person.false_positive_count", 2, unit="count"),
            ],
        )
        candidate = report(
            "run_candidate",
            [
                metric("person.recall", 0.9),
                metric("person.false_positive_count", 5, unit="count"),
            ],
        )

        comparison = compare_reports(baseline, candidate)
        states = {row["metric_key"]: row["classification"] for row in comparison["metric_deltas"]}
        self.assertEqual(states["person.recall"], "improved")
        self.assertEqual(states["person.false_positive_count"], "regressed")

    def test_critical_slice_can_fail_while_overall_recall_improves(self):
        baseline = report(
            "run_base",
            [
                metric("shuttle.visible_recall", 0.7),
                metric(
                    "shuttle.visible_recall",
                    0.65,
                    scope="slice",
                    scope_id="difficulty:fast_shuttle",
                ),
            ],
        )
        candidate = report(
            "run_candidate",
            [
                metric("shuttle.visible_recall", 0.8),
                metric(
                    "shuttle.visible_recall",
                    0.55,
                    scope="slice",
                    scope_id="difficulty:fast_shuttle",
                ),
            ],
        )
        gate = {
            "profile_id": "release-v1",
            "checks": [
                {
                    "metric_key": "shuttle.visible_recall",
                    "scope": "slice",
                    "scope_id": "difficulty:fast_shuttle",
                    "minimum": 0.6,
                }
            ],
        }

        comparison = compare_reports(baseline, candidate, gate_profile=gate)
        self.assertEqual(comparison["gate_result"]["status"], "FAIL")
        overall = next(row for row in comparison["metric_deltas"] if row["metric_key"] == "shuttle.visible_recall")
        self.assertEqual(overall["classification"], "improved")

    def test_different_metric_versions_are_not_comparable(self):
        baseline = report("run_base", [metric("person.recall", 0.8)])
        candidate = copy.deepcopy(report("run_candidate", [metric("person.recall", 0.9)]))
        candidate["metrics"][0]["metric_definition_version"] = "2.0.0"

        comparison = compare_reports(baseline, candidate, gate_profile={"profile_id": "release-v1", "checks": []})
        self.assertEqual(comparison["comparability"]["status"], "not_comparable")
        self.assertEqual(comparison["gate_result"]["status"], "FAIL")
        self.assertTrue(
            any(issue["code"] == "metric_version_mismatch" for issue in comparison["comparability"]["issues"])
        )

    def test_matching_but_unregistered_metric_versions_are_not_comparable(self):
        baseline = report("run_base", [metric("person.recall", 0.8, version="2.0.0")])
        candidate = report("run_candidate", [metric("person.recall", 0.9, version="2.0.0")])

        comparison = compare_reports(baseline, candidate)

        self.assertEqual(comparison["comparability"]["status"], "not_comparable")
        self.assertTrue(
            any(issue["code"] == "unsupported_metric_version" for issue in comparison["comparability"]["issues"])
        )


if __name__ == "__main__":
    unittest.main()
