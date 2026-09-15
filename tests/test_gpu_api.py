import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app


class GpuApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_key = os.environ.get("GOOD_BADMINTON_API_KEY")
        os.environ["GOOD_BADMINTON_API_KEY"] = "test-api-key"
        self.app = create_app(data_dir=Path(self.temp_dir.name), start_worker=False)
        self.client = TestClient(self.app)

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("GOOD_BADMINTON_API_KEY", None)
        else:
            os.environ["GOOD_BADMINTON_API_KEY"] = self.previous_key
        self.temp_dir.cleanup()

    def test_health_is_available_without_secret(self):
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_health_advertises_the_single_process_multi_sport_contract(self):
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["supported_sport_ids"], ["badminton", "tennis"])
        self.assertNotIn("sport_id", response.json())

    def test_job_submission_requires_api_key(self):
        response = self.client.post("/api/v1/jobs")

        self.assertEqual(response.status_code, 401)

    def test_invalid_upload_is_rejected_before_creating_job(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-0001"},
            files={
                "video": ("notes.txt", b"not a video", "text/plain"),
                "template": ("court.png", b"fake", "image/png"),
            },
            data={"court_corners": "[[1,1],[2,1],[2,2],[1,2]]"},
        )

        self.assertEqual(response.status_code, 422)

    def test_court_detection_rejects_non_video_uploads(self):
        response = self.client.post(
            "/api/v1/court/detect",
            headers={"X-API-Key": "test-api-key"},
            files={"video": ("notes.txt", b"not a video", "text/plain")},
        )

        self.assertEqual(response.status_code, 422)

    def test_court_detection_returns_detected_corners_for_a_video(self):
        self.app.state.court_detector = lambda _path: {
            "corners": [[10, 20], [30, 20], [30, 40], [10, 40]],
            "preview_bgr": None,
        }

        response = self.client.post(
            "/api/v1/court/detect",
            headers={"X-API-Key": "test-api-key"},
            files={"video": ("match.mp4", b"video-bytes", "video/mp4")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "corners": [[10, 20], [30, 20], [30, 40], [10, 40]],
                "preview_data_url": None,
            },
        )

    def test_valid_upload_creates_a_queued_job_with_stable_status_url(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-0001"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={"court_corners": "[[1,1],[2,1],[2,2],[1,2]]"},
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["execution"]["mode"], "remote_gpu")
        self.assertEqual(payload["timing"]["current_stage"], "queue_wait")
        self.assertEqual(payload["timing"]["stages"][0]["name"], "queue_wait")
        self.assertTrue(payload["receipt"]["accepted"])
        self.assertFalse(payload["receipt"]["reused"])
        job_id = payload["job_id"]

        status_response = self.client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-api-key"}
        )
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["job_id"], job_id)
        self.assertEqual(status_response.json()["timing"]["current_stage"], "queue_wait")

        repeated = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-0001"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={"court_corners": "[[1,1],[2,1],[2,2],[1,2]]"},
        )
        self.assertEqual(repeated.status_code, 202)
        self.assertEqual(repeated.json()["job_id"], job_id)
        self.assertTrue(repeated.json()["receipt"]["reused"])

        recovered = self.client.get(
            "/api/v1/jobs/by-idempotency/business-task-0001",
            headers={"X-API-Key": "test-api-key"},
        )
        self.assertEqual(recovered.status_code, 200)
        self.assertEqual(recovered.json()["job_id"], job_id)
        self.assertTrue(recovered.json()["receipt"]["reused"])

    def test_job_persists_the_request_sport_id(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-tennis"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "sport_id": "tennis",
            },
        )

        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertEqual(stored["input"]["sport_id"], "tennis")
        self.assertEqual(stored["options"]["sport_id"], "tennis")

    def test_job_rejects_an_unknown_sport_before_persisting_a_manifest(self):
        before = list(self.app.state.job_manager.jobs_dir.glob("*/manifest.json"))
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-unknown-sport"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "sport_id": "squash",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(list(self.app.state.job_manager.jobs_dir.glob("*/manifest.json")), before)

    def test_job_accepts_fixed_roster_options(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-roster"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": (
                    '{"match_mode":"doubles","lock_match_roster":true,'
                    '"roster_stable_frames":3}'
                ),
            },
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")

    def test_job_accepts_tracknet_as_explicit_primary_shuttle_source(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-tracknet"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": '{"shuttle_detector":"tracknet_v3"}',
            },
        )

        self.assertEqual(response.status_code, 202)
        job = response.json()
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["result"], None)
        stored = self.app.state.job_manager.get_job(job["job_id"])
        self.assertEqual(stored["options"]["shuttle_detector"], "tracknet_v3")

    def test_job_allows_shuttle_detection_to_be_explicitly_disabled(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-no-shuttle"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": '{"shuttle_detector":"none"}',
            },
        )

        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertEqual(stored["options"]["shuttle_detector"], "none")

    def test_person_only_job_accepts_one_of_the_approved_stability_windows(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-settle-window"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": '{"shuttle_detector":"none","movement_rally_settle_seconds":0.5}',
            },
        )
        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertEqual(stored["options"]["movement_rally_settle_seconds"], 0.5)

    def test_operator_can_cancel_a_queued_job_without_deleting_its_record(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-cancel"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={"court_corners": "[[1,1],[2,1],[2,2],[1,2]]"},
        )
        job_id = response.json()["job_id"]

        cancelled = self.client.delete(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-api-key"}
        )

        self.assertEqual(cancelled.status_code, 200)
        payload = cancelled.json()
        self.assertEqual(payload["status"], "cancelled")
        self.assertEqual(payload["error"]["type"], "TaskCancelled")
        self.assertIsNotNone(payload["finished_at"])
        self.assertEqual(payload["performance_trace"]["relative_path"], "performance_trace.json")
        self.assertTrue(any(item["event"] == "cancellation_requested" for item in payload["state_history"]))

        trace_response = self.client.get(
            f"/api/v1/jobs/{job_id}/performance-trace", headers={"X-API-Key": "test-api-key"}
        )
        self.assertEqual(trace_response.status_code, 200)
        trace = trace_response.json()
        self.assertEqual(trace["task"]["job_id"], job_id)
        self.assertEqual(trace["task"]["status"], "cancelled")
        self.assertEqual(trace["timing"]["current_stage"], "cancelled")

    def test_job_uses_960_and_10hz_pose_defaults(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-defaults"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={"court_corners": "[[1,1],[2,1],[2,2],[1,2]]"},
        )
        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertEqual(stored["options"]["pose_imgsz"], 960)
        self.assertEqual(stored["options"]["analysis_sample_hz"], 10.0)
        self.assertEqual(stored["options"]["pose_sample_hz"], 10.0)
        self.assertEqual(stored["options"]["tracker_backend"], "bytetrack")
        self.assertTrue(stored["options"]["enable_bytetrack"])
        self.assertEqual(stored["options"]["shuttle_detector"], "yolo")
        self.assertFalse(stored["options"]["generate_annotated_video"])
        self.assertFalse(stored["options"]["browser_video_reencode"])
        self.assertEqual(stored["tracking"]["phase"], "waiting_for_analysis")

    def test_job_can_explicitly_request_video_generation_and_browser_reencode(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-video-output"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": '{"generate_annotated_video":true,"browser_video_reencode":true}',
            },
        )
        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertTrue(stored["options"]["generate_annotated_video"])
        self.assertTrue(stored["options"]["browser_video_reencode"])

    def test_browser_reencode_is_omitted_when_video_generation_is_disabled(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-no-video-output"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": '{"browser_video_reencode":true}',
            },
        )
        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertFalse(stored["options"]["generate_annotated_video"])
        self.assertFalse(stored["options"]["browser_video_reencode"])

    def test_job_allows_an_explicit_full_frame_pose_evidence_run(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-full-pose"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": '{"pose_sample_hz":0}',
            },
        )

        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertEqual(stored["options"]["analysis_sample_hz"], 0.0)
        self.assertEqual(stored["options"]["pose_sample_hz"], 0.0)

    def test_shared_analysis_frequency_overrides_legacy_pose_frequency(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-cadence"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": '{"analysis_sample_hz":15,"pose_sample_hz":10}',
            },
        )

        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertEqual(stored["options"]["analysis_sample_hz"], 15.0)
        self.assertEqual(stored["options"]["pose_sample_hz"], 15.0)

    def test_job_keeps_an_opaque_match_reference_without_participant_identity(self):
        response = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "test-api-key", "X-Idempotency-Key": "business-task-context"},
            files={
                "video": ("match.mp4", b"video-bytes", "video/mp4"),
                "template": ("court.png", b"image-bytes", "image/png"),
            },
            data={
                "court_corners": "[[1,1],[2,1],[2,2],[1,2]]",
                "options_json": '{"match_session_ref":"match_check_batch_20260818"}',
            },
        )
        self.assertEqual(response.status_code, 202)
        stored = self.app.state.job_manager.get_job(response.json()["job_id"])
        self.assertEqual(stored["input"]["match_session_ref"], "match_check_batch_20260818")


if __name__ == "__main__":
    unittest.main()
