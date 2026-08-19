import unittest

from evaluation.performance.performance_gate import FAIL, PASS, WARN, build_report


PROFILE = {
    "profile_id": "test-3090",
    "production_options": {
        "pose_imgsz": 960,
        "pose_sample_hz": 10.0,
        "shuttle_detector": "yolo",
        "max_llm_requests_per_match": 1,
    },
    "regression": {"max_stage_regression_percent": 15.0},
    "streaming_slo": {
        "max_segment_p95_seconds": 1.5,
        "max_backlog_seconds_at_video_end": 360.0,
        "max_finalize_seconds": 120.0,
        "max_llm_seconds": 75.0,
        "max_final_tail_seconds": 600.0,
    },
}


def make_trace(pose_sample_hz=10.0, human_seconds=8.0):
    return {
        "task": {
            "status": "succeeded",
            "started_at": "2026-08-18T00:00:00+00:00",
            "finished_at": "2026-08-18T00:00:10+00:00",
        },
        "options": {
            "pose_imgsz": 960,
            "pose_sample_hz": pose_sample_hz,
            "shuttle_detector": "yolo",
        },
        "progress": {"total_frames": 300},
        "timing": {
            "stages": [
                {"name": "human_frame_processing", "elapsed_seconds": human_seconds, "details": {"fps": 30}},
                {"name": "tracknet.inference", "elapsed_seconds": 1.0, "details": {}},
            ]
        },
    }


def make_stream_report(**overrides):
    report = {
        "kind": "good_badminton_stream_replay_benchmark",
        "segments": {"p95_end_to_end_seconds": 1.2},
        "queue": {"backlog_seconds_at_video_end": 120},
        "completion": {"finalize_seconds": 50, "final_tail_seconds": 180},
        "llm": {"request_count": 1, "elapsed_seconds": 12},
    }
    for section, values in overrides.items():
        report[section].update(values)
    return report


class PerformanceGateTests(unittest.TestCase):
    def test_batch_trace_with_locked_production_options_is_warning_not_stream_proof(self):
        report = build_report(make_trace(), PROFILE)

        self.assertEqual(report["status"], WARN)
        self.assertFalse(report["streaming_slo_proven"])
        self.assertEqual(report["observed_trace"]["source_duration_seconds"], 10.0)
        self.assertTrue(any(check["name"] == "streaming_slo.evidence" for check in report["checks"]))

    def test_production_parameter_drift_fails_gate(self):
        report = build_report(make_trace(pose_sample_hz=0.0), PROFILE)

        self.assertEqual(report["status"], FAIL)
        pose_check = next(check for check in report["checks"] if check["name"] == "production_option.pose_sample_hz")
        self.assertEqual(pose_check["status"], FAIL)

    def test_stream_replay_can_prove_slo(self):
        report = build_report(make_trace(), PROFILE, stream_replay=make_stream_report())

        self.assertEqual(report["status"], PASS)
        self.assertTrue(report["streaming_slo_proven"])

    def test_stage_regression_against_same_source_fails(self):
        report = build_report(make_trace(human_seconds=10.0), PROFILE, baseline=make_trace(human_seconds=8.0))

        self.assertEqual(report["status"], FAIL)
        regression = next(check for check in report["checks"] if check["name"] == "regression.stage.human_frame_processing")
        self.assertEqual(regression["status"], FAIL)

    def test_stream_report_rejects_multiple_llm_requests(self):
        report = build_report(make_trace(), PROFILE, stream_replay=make_stream_report(llm={"request_count": 2}))

        self.assertEqual(report["status"], FAIL)
        request_check = next(check for check in report["checks"] if check["name"] == "llm.request_count")
        self.assertEqual(request_check["status"], FAIL)


if __name__ == "__main__":
    unittest.main()
