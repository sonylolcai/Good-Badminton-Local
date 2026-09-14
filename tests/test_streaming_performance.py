import tempfile
import unittest
from pathlib import Path

from badminton_analysis.streaming_validation import (
    DEFAULT_STREAMING_SLO,
    compare_stream_benchmark,
    new_benchmark,
    new_stage,
    run_reliability_scenarios,
    run_stream_replay,
    validate_trace,
)
from badminton_analysis.streaming_validation.baseline import build_gate_report
from tests.stream_test_utils import (
    counting_processor_factory,
    segment_metadata,
    write_video_segment_bytes,
)


CONFIG = {
    "video": "synthetic-replay",
    "resolution": "64x64",
    "sample_hz": 10,
    "pose_imgsz": 960,
    "shuttle_detector": "none",
    "generate_annotated_video": False,
    "gpu": "cpu-test-harness",
    "driver": "n/a",
    "match_mode": "singles",
    "switches": {"tracknet": False, "annotated_video": False},
}


class StreamingPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.data_dir = Path(self.temp_dir.name)
        self.segment_bytes = write_video_segment_bytes()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _segments(self, count):
        return [
            (index, self.segment_bytes, segment_metadata(index, self.segment_bytes, source_start=index * 1.0))
            for index in range(count)
        ]

    def test_replay_produces_a_valid_benchmark_trace(self):
        benchmark, _ = run_stream_replay(
            self.data_dir / "run", self._segments(2), counting_processor_factory(), CONFIG
        )
        self.assertEqual(benchmark["kind"], "good_badminton_stream_replay_benchmark")
        self.assertEqual(benchmark["segments"]["count"], 2)
        self.assertIsNotNone(benchmark["segments"]["p95_end_to_end_seconds"])
        self.assertFalse(benchmark["streaming_slo_proven"])
        self.assertTrue(benchmark["is_replay"])
        stage_names = {stage["name"] for stage in benchmark["stages"]}
        for required in ("decode", "pose_inference", "gpu_receive", "seal", "upload", "download"):
            self.assertIn(required, stage_names)
        self.assertEqual(validate_trace(benchmark), [])
        self.assertTrue((self.data_dir / "run" / "end_to_end_trace.json").is_file())

    def test_trace_validator_rejects_missing_required_stage(self):
        benchmark = new_benchmark(CONFIG)
        benchmark["stages"] = [new_stage("upload", "ok", elapsed_seconds=0.1)]
        errors = validate_trace(benchmark)
        self.assertTrue(any("missing required stage" in error for error in errors))

    def test_trace_validator_rejects_failed_stage_without_error(self):
        benchmark = new_benchmark(CONFIG)
        required_names = [
            "segment", "upload", "gpu_receive", "queue_wait", "decode", "sample",
            "pose_inference", "tracking", "event_write", "aggregate", "seal", "download",
        ]
        stages = [new_stage(name, "skipped") for name in required_names]
        stages = [stage for stage in stages if stage["name"] != "upload"]
        stages.append(new_stage("upload", "failed", elapsed_seconds=None))
        benchmark["stages"] = stages
        errors = validate_trace(benchmark)
        self.assertTrue(any("is failed but has no error" in error for error in errors))

    def test_gate_rejects_an_incomplete_trace_even_when_numbers_look_fast(self):
        benchmark = new_benchmark(CONFIG)
        benchmark["segments"]["p95_end_to_end_seconds"] = 0.01
        benchmark["completion"]["finalize_seconds"] = 0.01
        benchmark["completion"]["final_tail_seconds"] = 0.01
        report = build_gate_report(benchmark)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["checks"][0]["name"], "trace.valid")

    def test_gate_fails_when_segment_p95_exceeds_budget(self):
        benchmark, _ = run_stream_replay(
            self.data_dir / "run", self._segments(2), counting_processor_factory(), CONFIG
        )
        benchmark["segments"]["p95_end_to_end_seconds"] = 99.0
        checks = compare_stream_benchmark(benchmark, DEFAULT_STREAMING_SLO)
        failing = [check for check in checks if check["name"] == "segment.p95_seconds"]
        self.assertEqual(failing[0]["status"], "fail")

    def test_gate_passes_within_budget(self):
        benchmark, _ = run_stream_replay(
            self.data_dir / "run", self._segments(2), counting_processor_factory(), CONFIG
        )
        checks = compare_stream_benchmark(benchmark, DEFAULT_STREAMING_SLO)
        self.assertTrue(all(check["status"] in ("pass", "warn") for check in checks))

    def test_reliability_scenarios_all_pass(self):
        results = run_reliability_scenarios(
            self.data_dir / "reliability", counting_processor_factory(), self.segment_bytes
        )
        self.assertEqual(set(results), {
            "duplicate", "out_of_order", "missing_segment", "restart", "slow_upload", "backlog_at_seal",
        })
        for name, result in results.items():
            detail = result["detail"]
            self.assertEqual(result["status"], "ok", f"{name}: {detail}")


if __name__ == "__main__":
    unittest.main()
