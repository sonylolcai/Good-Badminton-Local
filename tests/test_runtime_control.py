import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.app import create_app
from tests.stream_test_utils import create_request, counting_processor_factory

class RuntimeControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        env=patch.dict(os.environ,{'GOOD_BADMINTON_API_KEY':'runtime-test-key',
            'GOOD_BADMINTON_UPLOAD_TOKEN_TTL_SECONDS':'2700','GOOD_BADMINTON_STREAM_MIN_FREE_BYTES':'1',
            'GOOD_BADMINTON_MEDIA_ARCHIVE_ENABLED':'0','GPU_RESULT_CALLBACK_URL':''})
        env.start();self.addCleanup(env.stop)
        self.app=create_app(data_dir=Path(self.temp.name),start_worker=False,stream_processor_factory=counting_processor_factory())
        self.client=TestClient(self.app);self.headers={'X-API-Key':'runtime-test-key'}

    def test_management_needs_service_key_and_unknown_metrics_are_not_zero(self):
        self.assertEqual(self.client.get('/api/v1/runtime/status').status_code,401)
        with patch('api.runtime_control.subprocess.run',side_effect=FileNotFoundError):
            result=self.client.get('/api/v1/runtime/status',headers=self.headers).json()
        self.assertIsNone(result['gpu']);self.assertFalse(result['busy'])
        self.assertEqual(result['upload_token_ttl_seconds'],2700)

    def test_drain_blocks_new_work_and_reports_existing_session(self):
        result=self.client.post('/api/v1/runtime/drain',json={'enabled':True},headers=self.headers)
        self.assertFalse(result.json()['busy'])
        self.assertEqual(self.client.post('/api/v1/stream-sessions',json=create_request(),headers=self.headers).status_code,503)
        self.client.post('/api/v1/runtime/drain',json={'enabled':False},headers=self.headers)
        self.assertEqual(self.client.post('/api/v1/stream-sessions',json=create_request(),headers=self.headers).status_code,202)
        result=self.client.post('/api/v1/runtime/drain',json={'enabled':True},headers=self.headers).json()
        self.assertTrue(result['busy']);self.assertEqual(result['active_stream_sessions'],1)

    def test_forty_five_minute_grant_is_session_scoped_and_revocable(self):
        first=self.client.post('/api/v1/stream-sessions',json=create_request(),headers=self.headers).json()['analysis_session_id']
        request=create_request();request['client_reference']='another-session'
        second=self.client.post('/api/v1/stream-sessions',json=request,headers=self.headers).json()['analysis_session_id']
        url=f'/api/v1/stream-sessions/{first}/upload-authorization'
        auth=self.client.post(url,headers=self.headers).json()
        self.assertEqual(auth['ttl_seconds'],2700)
        self.assertTrue(self.app.state.stream_manager.verify_upload_token(first,auth['upload_token']))
        self.assertFalse(self.app.state.stream_manager.verify_upload_token(second,auth['upload_token']))
        self.assertEqual(self.client.post('/api/v1/runtime/drain',json={'enabled':True},headers={'X-Upload-Token':auth['upload_token']}).status_code,401)
        renewed=self.client.post(url,headers=self.headers).json()
        self.assertFalse(self.app.state.stream_manager.verify_upload_token(first,auth['upload_token']))
        self.assertTrue(self.app.state.stream_manager.verify_upload_token(first,renewed['upload_token']))
        self.client.delete(url,headers=self.headers)
        self.assertFalse(self.app.state.stream_manager.verify_upload_token(first,renewed['upload_token']))

if __name__=='__main__':unittest.main()
