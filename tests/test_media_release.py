"""Real archived videos are released only after the cloud commit and GPU terminal state."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

from api.media_archive import StreamMediaArchive
from api.stream_sessions import StreamSessionManager
from tests import test_media_archive as fixtures
from tests.stream_test_utils import counting_processor_factory


class MediaReleaseTests(unittest.TestCase):
    setUpClass=classmethod(fixtures.MediaArchiveTests.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.MediaArchiveTests.tearDownClass.__func__)
    setUp=fixtures.MediaArchiveTests.setUp
    ingest=fixtures.MediaArchiveTests.ingest
    seal=fixtures.MediaArchiveTests.seal
    prepare=fixtures.MediaArchiveTests.prepare

    def terminal(self):
        state=self.prepare();self.manager.drain()
        self.assertEqual(self.manager.get_session(self.session_id)['status'],'finalized')
        return state

    def test_releases_duplicate_bytes_preserves_metadata_and_durable_receipt_after_restart(self):
        state=self.terminal()
        source=self.manager._segment_path(self.session_id,0)
        master=self.archive._directory(self.session_id)/state['revision']/'master.mp4'
        self.assertTrue(source.is_file());self.assertTrue(master.is_file())
        self.assertTrue(self.archive.release_verified_media(self.session_id,now=state['ready_at']+301))
        self.assertFalse(source.exists());self.assertFalse(master.exists())
        self.assertTrue(self.manager._events_path(self.session_id).is_file())
        self.assertTrue(self.manager._checkpoint_path(self.session_id).is_file())
        self.assertEqual(self.archive.manifest(self.session_id)['status'],'ready')
        receipt=self.manager.get_segment_receipt(self.session_id,0)
        self.assertTrue(receipt['gpu_accepted'])
        self.assertEqual(receipt['durable_location']['storage'],'object_storage')
        recovered=StreamSessionManager(self.root/'gpu',processor_factory=counting_processor_factory(),start_worker=False)
        self.assertTrue(recovered.get_segment_receipt(self.session_id,0)['gpu_accepted'])
        self.assertFalse(StreamMediaArchive(recovered,self.storage).release_verified_media(self.session_id,now=state['ready_at']+302))

    def test_ready_archive_does_not_release_while_inference_is_draining_or_worker_is_active(self):
        state=self.prepare()
        self.assertFalse(self.archive.release_verified_media(self.session_id,now=state['ready_at']+301))
        self.manager.drain()
        self.manager._active_segment={'session_id':self.session_id}
        self.assertFalse(self.archive.release_verified_media(self.session_id,now=state['ready_at']+301))
        self.assertTrue(self.manager._segment_path(self.session_id,0).is_file())

    def test_network_failure_or_missing_remote_manifest_keeps_every_local_video(self):
        state=self.terminal()
        with patch.object(self.storage,'head',side_effect=OSError('temporary cloud failure')):
            self.assertFalse(self.archive.release_verified_media(self.session_id,now=state['ready_at']+301))
        self.assertTrue(all(self.manager._segment_path(self.session_id,i).is_file() for i in range(3)))
        remote=self.storage._path(state['manifest_object_key']);remote.unlink()
        self.assertFalse(self.archive.release_verified_media(self.session_id,now=state['ready_at']+301))
        self.assertTrue(self.manager._segment_path(self.session_id,0).is_file())

    def test_grace_period_disable_switch_and_expired_cloud_lifetime_protect_inputs(self):
        state=self.terminal()
        self.assertFalse(self.archive.release_verified_media(self.session_id,now=state['ready_at']+10))
        with patch.dict('os.environ',{'GOOD_BADMINTON_MEDIA_LOCAL_CLEANUP_ENABLED':'0'}):
            self.assertFalse(self.archive.release_verified_media(self.session_id,now=state['ready_at']+301))
        self.assertFalse(self.archive.release_verified_media(self.session_id,now=state['ready_at']+8*86400))
        self.assertTrue(self.manager._segment_path(self.session_id,0).is_file())

    def test_interrupted_release_recovers_without_deleting_events_or_losing_receipts(self):
        state=self.terminal()
        original=Path.unlink
        blocked=self.manager._segment_path(self.session_id,1)
        def interrupted(path,*args,**kwargs):
            if path==blocked:raise OSError('simulated interrupted cleanup')
            return original(path,*args,**kwargs)
        with patch.object(Path,'unlink',interrupted),self.assertRaises(OSError):
            self.archive.release_verified_media(self.session_id,now=state['ready_at']+301)
        self.assertFalse(self.manager._segment_path(self.session_id,0).exists())
        self.assertTrue(self.manager.get_segment_receipt(self.session_id,0)['gpu_accepted'])
        recovered=StreamMediaArchive(self.manager,self.storage)
        self.assertTrue(recovered.release_verified_media(self.session_id,now=state['ready_at']+302))
        self.assertTrue(self.manager._events_path(self.session_id).is_file())
        for index in range(3):self.assertTrue(self.manager.get_segment_receipt(self.session_id,index)['gpu_accepted'])

    def test_release_never_follows_a_symlink_or_claims_durability_after_expiration(self):
        state=self.terminal()
        source=self.manager._segment_path(self.session_id,0)
        source.unlink();outside=self.root/'unique-evidence.bin';outside.write_bytes(b'unique')
        source.symlink_to(outside)
        with self.assertRaises(ValueError):self.archive.release_verified_media(self.session_id,now=state['ready_at']+301)
        self.assertEqual(outside.read_bytes(),b'unique')
        source.unlink();source.write_bytes(self.fixtures[0][0])
        self.assertTrue(self.archive.release_verified_media(self.session_id,now=state['ready_at']+302))
        session=self.manager.get_session(self.session_id)
        session['local_media_release']['expires_at']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        self.manager._save(session)
        self.assertFalse(self.manager.get_segment_receipt(self.session_id,0)['gpu_accepted'])


if __name__=='__main__':unittest.main()
