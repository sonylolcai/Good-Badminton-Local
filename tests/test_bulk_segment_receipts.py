"""Bulk proof reads share a manifest, preserve integrity, and require management auth."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from api.stream_sessions import StreamSessionManager
from api.stream_routes import register_stream_routes
from api.stream_errors import StreamSessionError
from tests.stream_test_utils import create_request, segment_metadata, counting_processor_factory


class BulkReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=StreamSessionManager(Path(self.tmp.name),processor_factory=counting_processor_factory(),start_worker=False)
        self.sid=self.manager.create_session(create_request())[1]['analysis_session_id']
        for i in (1,0):
            raw=('saved-original-'+str(i)).encode()
            self.manager.receive_segment(self.sid,i,segment_metadata(i,raw,source_start=i),raw)

    def test_one_manifest_read_preserves_requested_order_and_missing_indexes(self):
        original=self.manager._load
        with patch.object(self.manager,'_load',wraps=original) as load:
            response=self.manager.get_segment_receipts(self.sid,[1,0,2])
        self.assertEqual(load.call_count,1)
        self.assertEqual(response['analysis_session_id'],self.sid)
        self.assertEqual([r['segment_index'] for r in response['receipts']],[1,0])
        self.assertEqual(response['missing_indexes'],[2])
        self.assertTrue(all(r['gpu_accepted'] for r in response['receipts']))
        self.assertEqual(response['receipts'][0],self.manager.get_segment_receipt(self.sid,1))

    def test_corrupted_or_absent_bytes_never_claim_durable_acceptance(self):
        self.manager._segment_path(self.sid,0).write_bytes(b'corrupt')
        response=self.manager.get_segment_receipts(self.sid,[0,1])
        self.assertFalse(response['receipts'][0]['gpu_accepted'])
        self.assertTrue(response['receipts'][1]['gpu_accepted'])

    def test_limits_duplicates_boolean_and_foreign_session_rejected(self):
        for indexes in ([],list(range(16)),[0,0],[True],[-1],[1000001],['0']):
            with self.subTest(indexes=indexes),self.assertRaises(StreamSessionError):
                self.manager.get_segment_receipts(self.sid,indexes)
        with self.assertRaises(StreamSessionError):self.manager.get_segment_receipts('ssn_missing',[0])

    def test_http_endpoint_requires_management_auth_and_valid_bounded_query(self):
        def auth(x_api_key: str=Header(default='')):
            if x_api_key!='test-management':raise HTTPException(401)
        app=FastAPI();register_stream_routes(app,stream_manager=self.manager,require_api_key=auth,
            require_upload_access=auth,health_payload=lambda:{})
        client=TestClient(app);path=f'/api/v1/stream-sessions/{self.sid}/segment-receipts'
        self.assertEqual(client.get(path+'?indexes=0,1',headers={'X-Upload-Token':'scoped-upload-token'}).status_code,401)
        headers={'X-API-Key':'test-management'}
        response=client.get(path+'?indexes=1,0,2',headers=headers)
        self.assertEqual(response.status_code,200,response.text);self.assertEqual(response.json()['missing_indexes'],[2])
        for query in ('','-1','0,0','0,,1',','.join(map(str,range(16))),'0'*129,'１'):
            with self.subTest(query=query):
                self.assertEqual(client.get(path,params={'indexes':query},headers=headers).status_code,422)
        self.assertEqual(client.get('/api/v1/health').json()['segment_receipts_version'],'segment-receipts.v1')
