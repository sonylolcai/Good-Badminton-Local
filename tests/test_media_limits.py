"""Small real spools verify both transport channels and assembly reservations."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from api.media_limits import MediaLimits, ensure_capacity
from api.stream_errors import StreamSessionError
from api.stream_sessions import StreamSessionManager
from tests.stream_test_utils import counting_processor_factory, create_request, segment_metadata


class MediaLimitsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.env=patch.dict('os.environ',{
            'GOOD_BADMINTON_STREAM_MIN_FREE_BYTES':'1',
            'GOOD_BADMINTON_STREAM_MAX_CACHE_BYTES':str(1024**3),
            'GOOD_BADMINTON_STREAM_MAX_SESSION_BYTES':'100',
            'GOOD_BADMINTON_STREAM_MAX_SOURCE_SECONDS':'3',
            'GOOD_BADMINTON_STREAM_MAX_SEGMENTS':'8',
            'GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED':'0',
        });self.env.start();self.addCleanup(self.env.stop)
        self.manager=StreamSessionManager(Path(self.temp.name),start_worker=False,processor_factory=counting_processor_factory())
        self.session_id=self.manager.create_session(create_request())[1]['analysis_session_id']

    def send(self,index,data=b'1234',start=None):
        meta=segment_metadata(index,data,source_start=2*index if start is None else start,duration=2)
        return self.manager.receive_segment(self.session_id,index,meta,data)

    def test_half_hour_defaults_and_invalid_settings_cannot_disable_guards(self):
        with patch.dict('os.environ',{},clear=True):
            limits=MediaLimits.from_environment()
            self.assertEqual(limits.max_source_seconds,2700)
            self.assertEqual(limits.max_session_bytes,8*1024**3)
            self.assertEqual(limits.min_free_bytes,10*1024**3)
        for invalid in ('0','-1','invalid'):
            with patch.dict('os.environ',{'GOOD_BADMINTON_STREAM_MAX_CACHE_BYTES':invalid}),self.assertRaises(ValueError):
                MediaLimits.from_environment()

    def test_duration_refuses_new_data_but_keeps_old_segments_and_duplicate_receipts(self):
        self.send(0)
        with self.assertRaises(StreamSessionError) as error:self.send(1)
        self.assertEqual(error.exception.code,'gpu_media_duration_limit')
        self.assertEqual(error.exception.status_code,413)
        self.assertFalse(self.manager._segment_path(self.session_id,1).exists())
        self.assertTrue(self.manager._segment_path(self.session_id,0).is_file())
        with patch('api.media_limits.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            self.assertEqual(self.send(0)[0],200)
        # Completion still works when further input is refused.
        self.manager.complete(self.session_id,{'schema_version':'stream-session.v1','expected_last_segment_index':0,'allow_partial':False})
        self.assertTrue(self.manager.get_session(self.session_id)['sealed'])

    def test_long_preview_offset_is_not_mistaken_for_long_recording(self):
        self.send(0,start=10000)
        self.assertEqual(self.manager.get_session(self.session_id)['segments']['0']['source_start_time_sec'],10000)

    def test_byte_limit_and_fragment_count_are_hard_limits(self):
        with patch.dict('os.environ',{'GOOD_BADMINTON_STREAM_MAX_SESSION_BYTES':'3'}):
            with self.assertRaises(StreamSessionError) as error:self.send(0)
        self.assertEqual(error.exception.code,'gpu_media_session_size_limit')
        self.send(0)
        with patch.dict('os.environ',{'GOOD_BADMINTON_STREAM_MAX_SEGMENTS':'1'}):
            with self.assertRaises(StreamSessionError) as error:self.send(1)
        self.assertEqual(error.exception.code,'gpu_media_segment_limit')

    def test_ordinary_service_key_upload_also_respects_free_space(self):
        with patch('api.media_limits.shutil.disk_usage',return_value=SimpleNamespace(free=4)):
            with self.assertRaises(StreamSessionError) as error:self.send(0)
        self.assertEqual(error.exception.status_code,507)
        self.assertTrue(error.exception.retryable)
        self.assertFalse(self.manager._segment_path(self.session_id,0).exists())

    def test_new_sessions_stop_at_cache_budget_but_idempotent_creation_still_works(self):
        key='safe-create-'+'a'*32
        original=self.manager.create_session(create_request(),idempotency_key=key)[1]
        with patch.dict('os.environ',{'GOOD_BADMINTON_STREAM_MAX_CACHE_BYTES':'1'}):
            with self.assertRaises(StreamSessionError) as error:self.manager.create_session(create_request())
            self.assertEqual(error.exception.code,'gpu_media_cache_limit')
            self.assertEqual(self.manager.create_session(create_request(),idempotency_key=key)[1],original)

    def test_reserves_capacity_for_all_pending_assembled_videos(self):
        self.send(0)
        self.manager.media_archive_enabled=True
        # 4 source bytes reserve rounded-up 10% mux overhead plus 1 MiB.
        with patch('api.media_limits.shutil.disk_usage',return_value=SimpleNamespace(free=1024**2+5)):
            with self.assertRaises(StreamSessionError) as error:ensure_capacity(self.manager,0)
        self.assertEqual(error.exception.code,'gpu_media_storage_full')
        self.assertEqual(error.exception.details['pending_assembly_bytes'],1024**2+5)

    def test_corrupt_file_restoration_cannot_bypass_disk_guard(self):
        self.send(0);self.manager._segment_path(self.session_id,0).unlink()
        with patch('api.media_limits.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaises(StreamSessionError):self.send(0)
        self.assertFalse(self.manager._segment_path(self.session_id,0).exists())


if __name__=='__main__':unittest.main()
