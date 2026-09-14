import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis_platform.app_service import (
    execute_local_run,
    execute_uploaded_stream_run,
    get_run_detail,
    execute_uploaded_run,
    clone_run_task,
    list_run_summaries,
    verify_published_manifest,
)
from analysis_platform.models import canonical_json_sha256


FIXTURES = Path(__file__).parent / "fixtures"


class AnalysisPlatformAppServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        (self.inputs / "video.mp4").write_bytes(b"fixed-video")
        (self.inputs / "court.png").write_bytes(b"fixed-court")
        contracts = json.loads((FIXTURES / "analysis_contract_valid.json").read_text(encoding="utf-8"))
        self.task = copy.deepcopy(contracts["analysis_task"])
        self.task["input_path"] = "video.mp4"
        self.task["input_sha256"] = hashlib.sha256(b"fixed-video").hexdigest()
        self.dataset = {
            "schema_version": "analysis-platform.v1",
            "kind": "dataset_version",
            "dataset_version_id": self.task["dataset_version"],
            "dataset_id": "smoke",
            "version": "1.0.0",
            "purpose": "local integration smoke",
            "split": "smoke",
            "created_at": "2026-09-14T00:00:00Z",
            "created_by": "agent_0",
            "cases": [{"case_id": self.task["case_id"], "annotation_version_id": "annotation_demo_v1"}],
            "license_and_consent_status": "internal_test_only",
        }
        self.config = {
            "template_path": "court.png",
            "corners": [[0, 0], [10, 0], [10, 20], [0, 20]],
            "template_sha256": hashlib.sha256(b"fixed-court").hexdigest(),
            "hardware_fingerprint": "cpu-test",
            "execution_mode": "batch-local",
        }
        self.task["parameters"]["court_corners"] = self.config["corners"]
        self.task["parameters"]["template_sha256"] = self.config["template_sha256"]
        self.task["parameters_fingerprint"] = canonical_json_sha256(self.task["parameters"])

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _runner(_video, _template, _corners, _options, **kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        metadata = output / "metadata.json"
        detections = output / "detections.jsonl"
        metadata.write_text('{"ok": true}\n', encoding="utf-8")
        detections.write_text('{"frame": 1}\n', encoding="utf-8")
        return {"metadata": str(metadata), "detections": str(detections), "warnings": []}

    @staticmethod
    def _remote_runner(_video, _template, _corners, _options, output_dir, **_kwargs):
        return AnalysisPlatformAppServiceTests._runner(
            _video, _template, _corners, _options, output_dir=output_dir
        )

    @staticmethod
    def _stream_runner(_video, _corners, _options, output_dir, **_kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "stream_events.jsonl").write_text('{"event": "ok"}\n', encoding="utf-8")
        yield {"phase": "session_accepted", "analysis_session_id": "session_test_001"}
        yield {"phase": "finalized", "analysis_session_id": "session_test_001"}

    @staticmethod
    def _evaluator(_name, _arguments, output):
        report = Path(output)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"metrics": true}\n', encoding="utf-8")
        return {
            "source_report": {
                "path": str(report),
                "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            },
            "metrics": [{
                "metric_key": "person.recall",
                "metric_definition_version": "1.0.0",
                "scope": "run",
                "scope_id": "all",
                "value": 0.9,
                "unit": "ratio",
                "sample_count": 10,
                "eligible_sample_count": 10,
                "status": "valid",
            }],
        }

    @mock.patch("analysis_platform.app_service.execute_evaluator", side_effect=_evaluator.__func__)
    @mock.patch("analysis_platform.app_service._run_analysis", side_effect=_runner.__func__)
    def test_one_local_run_freezes_input_and_outputs_static_report(self, _run, _evaluate):
        store = self.root / "store"
        baseline = {
            "run_id": "run_baseline",
            "dataset_version": self.task["dataset_version"],
            "dataset_manifest_sha256": canonical_json_sha256(self.dataset),
            "hardware_fingerprint": "cpu-test",
            "execution_mode": "batch-local",
            "metrics": [{
                "metric_key": "person.recall",
                "metric_definition_version": "1.0.0",
                "scope": "run",
                "scope_id": "all",
                "value": 0.8,
                "unit": "ratio",
                "sample_count": 10,
                "eligible_sample_count": 10,
                "status": "valid",
            }],
        }
        summary = execute_uploaded_run(
            store,
            self.dataset,
            self.task,
            self.inputs / "video.mp4",
            self.inputs / "court.png",
            self.config["corners"],
            "doubles",
            ["fixture.json"],
            baseline_report=baseline,
            gate_profile={
                "profile_id": "smoke-v1",
                "checks": [{"metric_key": "person.recall", "minimum": 0.85}],
            },
        )

        self.assertEqual(summary["status"], "succeeded")
        self.assertEqual(summary["metric_count"], 1)
        self.assertTrue(Path(summary["run_report"]).is_file())
        self.assertTrue(Path(summary["comparison_report"]).is_file())
        self.assertEqual(summary["gate_status"], "PASS")
        self.assertIn("--detections", _evaluate.call_args.args[1])
        self.assertTrue((store / "runs" / self.task["run_id"] / "result.json").is_file())
        verified = verify_published_manifest(store, "dataset_version", self.task["dataset_version"])
        self.assertTrue(verified["valid"])
        self.assertEqual(list_run_summaries(store)[0]["status"], "succeeded")
        detail_files = {Path(path).name for path in get_run_detail(store, self.task["run_id"])["files"]}
        self.assertTrue({"metadata.json", "detections.jsonl", "run-report.json", "comparison.json"}.issubset(detail_files))
        self.assertEqual(clone_run_task(store, self.task["run_id"], "run_cloned_20260914")["run_id"], "run_cloned_20260914")

    @mock.patch("analysis_platform.app_service.execute_evaluator", side_effect=_evaluator.__func__)
    @mock.patch("analysis_platform.app_service._run_remote_analysis", side_effect=_remote_runner.__func__)
    @mock.patch("analysis_platform.app_service._verify_remote_gpu", return_value={"sport_id": "badminton"})
    def test_uploaded_run_can_use_the_existing_remote_gpu_client(self, verify, remote, _evaluate):
        summary = execute_uploaded_run(
            self.root / "store",
            self.dataset,
            self.task,
            self.inputs / "video.mp4",
            self.inputs / "court.png",
            self.config["corners"],
            "doubles",
            ["--annotations", "truth.jsonl"],
            processing_target="remote_gpu",
            gpu_base_url="https://gpu.example.test",
        )

        self.assertEqual(summary["status"], "succeeded")
        verify.assert_called_once_with("badminton", "https://gpu.example.test")
        self.assertEqual(remote.call_args.kwargs["gpu_base_url"], "https://gpu.example.test")

    @mock.patch("analysis_platform.app_service._iter_remote_stream", side_effect=_stream_runner.__func__)
    def test_two_second_stream_is_recorded_as_an_immutable_run(self, _stream):
        summary = execute_uploaded_stream_run(
            self.root / "store",
            self.dataset,
            self.task,
            self.inputs / "video.mp4",
            self.config["corners"],
            gpu_base_url="https://gpu.example.test",
        )

        self.assertEqual(summary["execution_mode"], "two-second-direct-gpu")
        self.assertEqual(summary["analysis_session_id"], "session_test_001")

    @mock.patch("analysis_platform.app_service._run_analysis", side_effect=RuntimeError("model failed"))
    def test_failed_runner_keeps_failed_terminal_result(self, _run):
        store = self.root / "store"
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            execute_local_run(
                store,
                self.dataset,
                self.task,
                self.config,
                "doubles",
                [],
                input_root=self.inputs,
            )

        result = json.loads((store / "runs" / self.task["run_id"] / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["payload"]["status"], "failed")
        self.assertEqual(result["payload"]["error"]["code"], "analysis_failed")


if __name__ == "__main__":
    unittest.main()
