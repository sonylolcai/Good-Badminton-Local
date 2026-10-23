"""Real encoded media through ingest -> archive -> both local replay sources."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.media_archive import MediaInspector, StreamMediaArchive, input_snapshot, verify_segments
from api.media_storage import DirectoryObjectStore, sha256_file
from api.oss_replay import replay_archive
from api.stream_routes import register_stream_routes
from api.stream_runtime import StreamProcessorFactory
from api.stream_sessions import StreamSessionManager
from tests.stream_test_utils import CountingProcessor, counting_processor_factory, create_request, segment_metadata


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe are required")
class MediaArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = tempfile.TemporaryDirectory()
        root = Path(cls.fixture_dir.name)
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=2.3",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "10", "-keyint_min", "10", "-sc_threshold", "0",
            "-f", "segment", "-segment_time", "1", "-reset_timestamps", "1", str(root / "%03d.mp4"),
        ], check=True, capture_output=True)
        inspector = MediaInspector()
        cls.fixtures = [(path.read_bytes(), inspector.inspect(path)) for path in sorted(root.glob("*.mp4"))]

    @classmethod
    def tearDownClass(cls):
        cls.fixture_dir.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manager = StreamSessionManager(self.root / "gpu", processor_factory=counting_processor_factory(), start_worker=False)
        self.session_id = self.manager.create_session(create_request())[1]["analysis_session_id"]
        self.storage = DirectoryObjectStore(self.root / "objects")
        self.archive = StreamMediaArchive(self.manager, self.storage, retry_seconds=0)

    def ingest(self, indexes=(0, 1, 2), metadata_change=None):
        start, frame_start = 0.0, 0
        for index, (raw, evidence) in enumerate(self.fixtures):
            metadata = segment_metadata(index, raw, source_start=start, duration=evidence["duration_sec"])
            metadata.update(source_frame_start_index=frame_start, source_frame_count=evidence["frame_count"],
                            source_frame_time_offsets_sec=evidence["frame_offsets_sec"])
            if metadata_change:
                metadata_change(index, metadata)
            if index in indexes:
                self.manager.receive_segment(self.session_id, index, metadata, raw)
            start += evidence["duration_sec"]
            frame_start += evidence["frame_count"]

    def seal(self):
        self.manager.complete(self.session_id, {"schema_version": "stream-session.v1",
                                              "expected_last_segment_index": 2, "allow_partial": False})

    def prepare(self):
        self.ingest()
        self.seal()
        result = self.archive.process_session(self.session_id)
        self.assertEqual(result["status"], "ready", result)
        return result

    def test_sealed_backlog_archives_independently_then_master_and_fragments_match_gpu(self):
        result = self.prepare()
        self.assertEqual(self.manager.get_session(self.session_id)["status"], "draining")
        self.assertEqual(result["total_source_frames"], 23)
        self.assertFalse(result["analysis_quality_verified"])
        for source in ("fragments", "master"):
            run = replay_archive(
                self.storage, result["manifest_object_key"], manifest_sha256=result["manifest_sha256"],
                output_dir=self.root / source, source=source,
                processor_factory=lambda session: (CountingProcessor(), None),
            )
            self.assertEqual(run["source_frames"], 23)
            self.assertFalse(run["model_inference_tested"])
        def evidence(source):
            return [{k: v for k, v in json.loads(line).items() if k != "emitted_at"}
                    for line in (self.root / source / "events.jsonl").read_text().splitlines()]
        self.assertEqual(evidence("fragments"), evidence("master"))
        self.manager.drain()
        gpu_events = [e for e in self.manager._load_events(self.session_id) if e["event_type"] == "person_observation"]
        mac_events = [json.loads(line) for line in (self.root / "fragments/events.jsonl").read_text().splitlines()]
        mac_events = [e for e in mac_events if e["event_type"] == "person_observation"]
        fields = ("source_time_sec", "segment_index", "data")
        self.assertEqual([{k: e[k] for k in fields} for e in gpu_events], [{k: e[k] for k in fields} for e in mac_events])

    def test_reordered_delivery_is_archived_in_source_order(self):
        self.ingest((2,))
        self.ingest((0,))
        self.ingest((1,))
        self.seal()
        result = self.archive.process_session(self.session_id)
        self.assertEqual(result["status"], "ready", result)

    def test_ready_is_idempotent_and_restart_recovers_local_status(self):
        result = self.prepare()
        with patch.object(self.storage, "upload", side_effect=AssertionError("must not upload twice")):
            recovered = StreamMediaArchive(self.manager, self.storage)
            self.assertEqual(recovered.process_session(self.session_id), result)

    def test_upload_failure_never_commits_manifest_and_retries_after_restart(self):
        self.ingest()
        self.seal()
        upload = self.storage.upload
        def flaky(key, source):
            if key.endswith("master.mp4"):
                raise IOError("simulated network outage")
            upload(key, source)
        with patch.object(self.storage, "upload", side_effect=flaky):
            result = self.archive.process_session(self.session_id)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(list(self.storage.root.rglob("manifest.json")))
        master = self.archive._directory(self.session_id) / result["revision"] / "master.mp4"
        original_mtime = master.stat().st_mtime_ns
        recovered = StreamMediaArchive(self.manager, self.storage, retry_seconds=0)
        recovered.run_once()
        self.assertEqual(recovered.status(self.session_id)["status"], "ready")
        self.assertEqual(master.stat().st_mtime_ns, original_mtime, "do not invalidate SDK resume by rewriting the master")

    def test_low_disk_space_preserves_originals_and_does_not_publish(self):
        self.ingest()
        self.seal()
        with patch("api.media_archive.shutil.disk_usage") as disk:
            disk.return_value.free = 0
            result = self.archive.process_session(self.session_id)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(list(self.storage.root.rglob("manifest.json")))
        self.assertTrue(all(self.manager._segment_path(self.session_id, i).is_file() for i in range(3)))

    def test_broken_old_media_status_does_not_starve_other_cameras(self):
        self.prepare()
        (self.archive._directory(self.session_id) / "status.json").write_text("broken JSON")
        self.session_id = self.manager.create_session(create_request())[1]["analysis_session_id"]
        self.ingest()
        self.seal()
        with self.assertLogs("api.media_archive", level="ERROR"):
            self.archive.run_once()
        self.assertEqual(self.archive.status(self.session_id)["status"], "ready")

    def test_missing_tail_is_not_a_complete_archive(self):
        self.ingest((0, 1))
        self.manager.complete(self.session_id, {"schema_version": "stream-session.v1",
                                              "expected_last_segment_index": 2, "allow_partial": True})
        result = self.archive.process_session(self.session_id)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("missing", result["error"])
        self.assertFalse(list(self.storage.root.rglob("manifest.json")))

    def test_immediate_completion_keeps_final_index_with_an_existing_engine(self):
        self.ingest()
        self.manager.drain()
        self.seal()
        session = self.manager.get_session(self.session_id)
        self.assertTrue(session["sealed"])
        self.assertEqual(session["expected_last_segment_index"], 2)
        self.assertEqual(self.archive.process_session(self.session_id)["status"], "ready")

    def test_lost_or_corrupted_durable_bytes_do_not_archive(self):
        self.ingest()
        self.seal()
        self.manager._segment_path(self.session_id, 1).write_bytes(b"corrupted")
        result = self.archive.process_session(self.session_id)
        self.assertEqual(result["status"], "invalid")

    def test_valid_hash_but_undecodable_video_does_not_archive(self):
        self.ingest((0, 2))
        raw = b"not a video"
        metadata = segment_metadata(1, raw, source_start=1, duration=1)
        metadata.update(source_frame_start_index=10, source_frame_count=10,
                        source_frame_time_offsets_sec=[i / 10 for i in range(10)])
        self.manager.receive_segment(self.session_id, 1, metadata, raw)
        self.seal()
        self.assertEqual(self.archive.process_session(self.session_id)["status"], "invalid")

    def test_continuity_requires_real_frame_identity_and_source_time(self):
        for change in (lambda e: e.update(source_frame_start_index=11),
                       lambda e: e.update(source_start_time_sec=1.2),
                       lambda e: e.update(source_frame_count=9),
                       lambda e: e.update(source_frame_time_offsets_sec=None)):
            with self.subTest(change=change):
                self.ingest()
                snapshot = input_snapshot(self.manager.get_session(self.session_id))
                change(snapshot["segments"][1])
                with self.assertRaises(ValueError):
                    verify_segments(snapshot, lambda i: self.manager._segment_path(self.session_id, i), self.archive.inspector)

    def test_codec_parameter_drift_is_rejected(self):
        self.ingest()
        self.seal()
        actual = self.archive.inspector.inspect
        calls = [0]
        def inspect(path):
            evidence = actual(path)
            calls[0] += 1
            if calls[0] == 2:
                evidence["stream_signature"][0]["width"] = 128
            return evidence
        with patch.object(self.archive.inspector, "inspect", side_effect=inspect):
            result = self.archive.process_session(self.session_id)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("parameters changed", result["error"])

    def test_mac_rejects_corrupt_download_and_marks_run_failed(self):
        result = self.prepare()
        manifest = self.archive.manifest(self.session_id)
        self.storage._path(manifest["segments"][1]["object_key"]).write_bytes(b"corruption")
        with self.assertRaises(ValueError):
            replay_archive(self.storage, result["manifest_object_key"], manifest_sha256=result["manifest_sha256"],
                           output_dir=self.root / "bad-run", processing_mode="decode_smoke")
        self.assertEqual(json.loads((self.root / "bad-run/run.json").read_text())["status"], "failed")

    def test_mac_rejects_wrong_manifest_hash_and_never_overwrites_run(self):
        result = self.prepare()
        output = self.root / "run"
        with self.assertRaises(ValueError):
            replay_archive(self.storage, result["manifest_object_key"], manifest_sha256="0" * 64,
                           output_dir=output, processing_mode="decode_smoke")
        with self.assertRaises(FileExistsError):
            replay_archive(self.storage, result["manifest_object_key"], manifest_sha256=result["manifest_sha256"],
                           output_dir=output, processing_mode="decode_smoke")

    def test_decode_smoke_can_archive_without_model_weights(self):
        factory = StreamProcessorFactory(self.root / "smoke", processing_mode="decode_smoke")
        request = create_request()
        request["configuration"]["shuttle_detector"] = "tracknet_v3"
        factory.validate_session_request(request)  # No model path lookup or inference.
        processors = factory({**request, "processing_mode": "decode_smoke"})
        self.assertEqual(type(processors[0]).__name__, "DecodeSmokeProcessor")
        with self.assertRaisesRegex(ValueError, "different processing mode"):
            factory({**request, "processing_mode": "analysis"})
        result = self.prepare()
        run = replay_archive(self.storage, result["manifest_object_key"], manifest_sha256=result["manifest_sha256"],
                             output_dir=self.root / "smoke-run", processing_mode="decode_smoke")
        self.assertEqual(run["source_frames"], 23)
        self.assertFalse(run["model_inference_tested"])

    def test_media_routes_require_management_auth_and_publish_ready_only(self):
        from fastapi import Header, HTTPException
        def authenticate(x_api_key: str = Header(default="")):
            if x_api_key != "test-key":
                raise HTTPException(401)
        app = FastAPI()
        register_stream_routes(app, stream_manager=self.manager, require_api_key=authenticate,
                               require_upload_access=authenticate, health_payload=lambda: {}, media_archive=self.archive)
        client = TestClient(app)
        route = f"/api/v1/stream-sessions/{self.session_id}/media"
        self.assertEqual(client.get(route).status_code, 401)
        headers = {"X-API-Key": "test-key"}
        self.assertEqual(client.post(route + "/prepare", headers=headers).status_code, 409)
        self.assertEqual(client.get(route + "/manifest", headers=headers).status_code, 409)
        self.prepare()
        self.assertEqual(client.get(route, headers=headers).json()["status"], "ready")
        self.assertEqual(client.get(route + "/manifest", headers=headers).json()["sport_id"], "badminton")

    def test_unarchived_evidence_is_protected_from_ttl_cleanup(self):
        self.prepare()
        self.manager.drain()
        with patch.dict(os.environ, {"GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED": "1"}):
            from datetime import datetime, timedelta, timezone
            future = datetime.now(timezone.utc) + timedelta(days=20)
            self.assertEqual(len(self.manager.cleanup_expired_terminal_sessions(now=future)), 1)
            (self.archive._directory(self.session_id) / "status.json").unlink()
            self.assertEqual(self.manager.cleanup_expired_terminal_sessions(now=future), [])

    def test_restart_with_archiving_disabled_preserves_uncommitted_media(self):
        self.prepare()
        self.manager.drain()
        state_path = self.archive._directory(self.session_id) / "status.json"
        state_path.write_text(json.dumps({"status": "failed"}))
        with patch.dict(os.environ, {"GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED": "0"}):
            recovered = StreamSessionManager(self.root / "gpu", processor_factory=counting_processor_factory(), start_worker=False)
            from datetime import datetime, timedelta, timezone
            future = datetime.now(timezone.utc) + timedelta(days=20)
            self.assertEqual(recovered.cleanup_expired_terminal_sessions(now=future, dry_run=False), [])
        self.assertTrue(recovered._segment_path(self.session_id, 0).is_file())


if __name__ == "__main__":
    unittest.main()
