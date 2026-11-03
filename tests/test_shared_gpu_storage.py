"""One GPU serves two business environments without changing replay scope."""
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from api.dogecloud_storage import DogeCloudObjectStore
from api.media_storage import EnvironmentObjectStore, oss_from_environment


class SharedGpuStorageTests(unittest.TestCase):
    def store(self, root):
        store = DogeCloudObjectStore(access_key='test-only', secret_key='test-only',
                                     bucket='test-only', allowed_root=root)
        for method in ('upload', 'download', 'head'):
            setattr(store, method, Mock())
        return store

    def test_routes_all_operations_without_crossing_child_scope(self):
        test, production = self.store('testVideo'), self.store('productionVideo')
        router = EnvironmentObjectStore({'testVideo': test, 'productionVideo': production})
        for store, root in ((test, 'testVideo'), (production, 'productionVideo')):
            key = root + '/venue/master.mp4'
            router.upload(key, Path('/tmp/evidence.mp4'))
            router.download(key, Path('/tmp/download.mp4'))
            router.head(key)
            store.upload.assert_called_once_with(key, Path('/tmp/evidence.mp4'))
            store.download.assert_called_once_with(key, Path('/tmp/download.mp4'))
            store.head.assert_called_once_with(key)
        self.assertTrue(router.requires_media_identity)
        for key in ('other/video.mp4', 'testVideo/../productionVideo/x',
                    'testVideo/%2e%2e/x', 'productionVideo//x'):
            with self.subTest(key=key), self.assertRaises(ValueError):router.head(key)
        self.assertEqual(test.head.call_count, 1)
        self.assertEqual(production.head.call_count, 1)
        with self.assertRaises(ValueError):
            EnvironmentObjectStore({'testVideo': production})

    def test_only_archive_factory_can_enable_both_environments(self):
        config = {'GOOD_BADMINTON_APP_ENV': 'production',
                  'GOOD_BADMINTON_MEDIA_STORAGE_PROVIDER': 'dogecloud',
                  'DOGECLOUD_MEDIA_ENVIRONMENT': 'production',
                  'GOOD_BADMINTON_MEDIA_ARCHIVE_ENVIRONMENTS': 'test,production',
                  'DOGECLOUD_ACCESS_KEY': 'test-only', 'DOGECLOUD_SECRET_KEY': 'test-only',
                  'DOGECLOUD_BUCKET': 'test-only'}
        with patch.dict(os.environ, config, clear=True), \
                patch('api.dogecloud_storage.load_dogecloud_environment'):
            archive = oss_from_environment('/tmp/checkpoints', allow_environment_routing=True)
            self.assertEqual(set(archive.stores), {'testVideo', 'productionVideo'})
            replay = oss_from_environment('/tmp/checkpoints')
            self.assertEqual(replay.allowed_root, 'productionVideo')
            with self.assertRaises(ValueError):replay._key('testVideo/venue/video.mp4')
            for invalid in ('', 'test,', 'test,test', 'test,other', 'test, production'):
                with self.subTest(invalid=invalid), patch.dict(os.environ, {
                        'GOOD_BADMINTON_MEDIA_ARCHIVE_ENVIRONMENTS': invalid}):
                    with self.assertRaises(ValueError):
                        oss_from_environment('/tmp/checkpoints', allow_environment_routing=True)


class SharedArchiveIdentityTests(unittest.TestCase):
    def test_router_still_requires_business_issued_identity(self):
        from tests.test_media_identity_archive import BusinessIssuedArchiveTests
        case = BusinessIssuedArchiveTests('test_dogecloud_never_archives_unregistered_legacy_sessions')
        case.setUpClass()
        try:
            case.setUp()
            try:
                case.archive.storage = EnvironmentObjectStore({
                    'testVideo': DogeCloudObjectStore(access_key='test-only', secret_key='test-only',
                                                     bucket='test-only', allowed_root='testVideo')})
                case.ingest();case.seal()
                result = case.archive.process_session(case.session_id)
                self.assertEqual(result['status'], 'invalid')
                self.assertIn('business-issued', result['error'])
            finally:
                case.tearDown()
                case.doCleanups()
        finally:case.tearDownClass()
