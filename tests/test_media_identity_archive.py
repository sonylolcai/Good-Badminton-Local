from copy import deepcopy
from uuid import uuid4
import unittest
from unittest.mock import patch

from api.media_identity import UUID_FIELDS, media_prefix
from api.stream_errors import StreamSessionError
from api.oss_replay import validate_manifest
from tests import test_media_archive as archive_tests
from tests.stream_test_utils import create_request


class BusinessIssuedArchiveTests(unittest.TestCase):
    setUpClass = classmethod(archive_tests.MediaArchiveTests.setUpClass.__func__)
    tearDownClass = classmethod(archive_tests.MediaArchiveTests.tearDownClass.__func__)
    setUp = archive_tests.MediaArchiveTests.setUp
    ingest = archive_tests.MediaArchiveTests.ingest
    seal = archive_tests.MediaArchiveTests.seal
    prepare = archive_tests.MediaArchiveTests.prepare
    def business_session(self):
        identity={key:str(uuid4()) for key in UUID_FIELDS}
        identity.update(schema_version='media-identity.v1',environment='test',venue_slug='haoyu',capture_date='2026-10-23',daily_sequence=1)
        request=create_request()
        request.update(camera_id=identity['camera_id'],calibration_id=identity['calibration_id'],
                       client_reference=identity['edge_session_id'],media_identity=identity)
        self.session_id=self.manager.create_session(request,idempotency_key='business-session-'+'a'*32)[1]['analysis_session_id']
        return identity,request

    def test_business_path_identity_and_replay_validation_are_frozen_and_deterministic(self):
        identity,request=self.business_session()
        result=self.prepare()
        self.assertEqual(result['manifest_object_key'],media_prefix(identity)+'/manifest.json')
        manifest=self.archive.manifest(self.session_id)
        self.assertEqual(manifest['media_identity'],identity)
        self.assertEqual(validate_manifest(manifest,result['manifest_object_key']),manifest)
        status,reused=self.manager.create_session(request,idempotency_key='business-session-'+'a'*32)
        self.assertEqual(status,200);self.assertEqual(reused['analysis_session_id'],self.session_id)
        changed=deepcopy(request);changed['media_identity']['daily_sequence']=2
        with self.assertRaises(StreamSessionError):self.manager.create_session(changed,idempotency_key='business-session-'+'a'*32)

    def test_mismatched_camera_identity_is_rejected_before_creating_a_session(self):
        identity,request=self.business_session()
        request['camera_id']=str(uuid4())
        with self.assertRaises(StreamSessionError):self.manager.create_session(request)

    def test_dogecloud_never_archives_unregistered_legacy_sessions(self):
        self.ingest();self.seal()
        self.storage.allowed_root='testVideo'
        result=self.archive.process_session(self.session_id)
        self.assertEqual(result['status'],'invalid')
        self.assertIn('business-issued',result['error'])


if __name__=='__main__':unittest.main()
