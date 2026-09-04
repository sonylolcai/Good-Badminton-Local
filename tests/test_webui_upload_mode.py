"""Regression coverage for the WebUI's selected GPU upload transport."""

import json
import tempfile
import unittest
from pathlib import Path

from webui.app import (
    _find_remote_stream_session_workdir,
    _full_video_upload_update,
    _two_second_segment_upload_update,
)


class WebUiUploadModeTests(unittest.TestCase):
    def test_full_video_keeps_all_existing_outputs_and_marks_transport(self):
        full_update = tuple(range(15))

        mapped = _full_video_upload_update(full_update)

        self.assertEqual(mapped[:15], full_update)
        self.assertEqual(mapped[15]["mode"], "full_video_direct_gpu")

    def test_two_second_segments_expose_direct_gpu_transport_state(self):
        stream_status = {
            "phase": "segment_accepted",
            "analysis_session_id": "ssn_test",
            "remote_base_url": "http://gpu.example",
        }

        mapped = _two_second_segment_upload_update(stream_status)

        self.assertEqual(len(mapped), 19)
        self.assertIsNone(mapped[8])
        self.assertIsNone(mapped[13])
        self.assertEqual(mapped[14]["upload_mode"], "two_second_segments")
        self.assertEqual(mapped[14]["segment_seconds"], 2.0)
        self.assertEqual(mapped[14]["remote_base_url"], "http://gpu.example")
        self.assertEqual(mapped[15], mapped[14])
        self.assertIsNone(mapped[16])
        self.assertEqual(mapped[17], [])
        self.assertEqual(mapped[18]["schema_version"], "webui-player-result.v1")

    def test_find_remote_stream_session_workdir_requires_matching_local_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_dir = root / "20260904_092548_712038"
            session_dir.mkdir()
            (session_dir / "delivery-ledger.json").write_text(
                json.dumps({"analysis_session_id": "ssn_abc123"}), encoding="utf-8"
            )
            self.assertEqual(
                _find_remote_stream_session_workdir("ssn_abc123", root), session_dir,
            )
            self.assertIsNone(_find_remote_stream_session_workdir("ssn_missing", root))
            self.assertIsNone(_find_remote_stream_session_workdir("../unsafe", root))


if __name__ == "__main__":
    unittest.main()
