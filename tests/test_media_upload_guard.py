import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace

from fastapi.testclient import TestClient
from api.gpu_stream_app import create_gpu_stream_app
from api.media_upload_guard import MediaUploadGuard
from tests.stream_test_utils import counting_processor_factory,create_request


class MediaUploadGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.env=patch.dict('os.environ',{
            'GOOD_BADMINTON_API_KEY':'test-key','GOOD_BADMINTON_STREAM_MIN_FREE_BYTES':'1',
            'GOOD_BADMINTON_STREAM_MAX_REQUEST_BYTES':'1024',
        });self.env.start();self.addCleanup(self.env.stop)
        self.app=create_gpu_stream_app(data_dir=Path(self.temp.name),start_worker=False,
            stream_processor_factory=counting_processor_factory())
        self.client=TestClient(self.app)
        self.session=self.client.post('/api/v1/stream-sessions',json=create_request(),headers={'X-API-Key':'test-key'}).json()['analysis_session_id']

    def test_known_oversized_request_is_denied_before_multipart_parser_or_file_creation(self):
        path=f'/api/v1/stream-sessions/{self.session}/segments/0'
        with patch('starlette.formparsers.MultiPartParser.parse',side_effect=AssertionError('must not spool')):
            response=self.client.post(path,content=b'x'*2048,headers={'X-API-Key':'test-key','Content-Type':'multipart/form-data; boundary=x'})
        self.assertEqual(response.status_code,413,response.text)
        self.assertEqual(response.json()['error']['code'],'gpu_media_request_limit')
        self.assertEqual(self.app.state.stream_manager.get_session(self.session)['segments'],{})

    def test_chunked_or_false_content_length_still_counts_actual_received_bytes(self):
        async def run(headers):
            writes=[]
            async def app(scope,receive,send):
                while True:
                    part=await receive();writes.append(len(part.get('body',b'')))
                    if not part.get('more_body'):break
            manager=self.app.state.stream_manager
            middleware=MediaUploadGuard(app,manager=manager)
            parts=iter([{'type':'http.request','body':b'x'*600,'more_body':True},
                        {'type':'http.request','body':b'x'*600,'more_body':False}])
            async def receive():return next(parts)
            messages=[]
            async def send(message):messages.append(message)
            await middleware({'type':'http','method':'POST','path':f'/api/v1/stream-sessions/{self.session}/segments/0','headers':headers},receive,send)
            self.assertEqual(writes,[600])
            self.assertEqual(messages[0]['status'],413)
            self.assertEqual(middleware._inflight,0)
        asyncio.run(run([]));asyncio.run(run([(b'content-length',b'1')]))

    def test_streaming_overflow_closes_the_actual_multipart_temporary_file(self):
        buffers=[]
        factory=tempfile.SpooledTemporaryFile
        def record(*args,**kwargs):
            handle=factory(*args,**kwargs);buffers.append(handle);return handle
        async def run():
            prefix=(b'--test-boundary\r\nContent-Disposition: form-data; name="segment"; filename="segment.mp4"'
                    b'\r\nContent-Type: video/mp4\r\n\r\n')
            parts=iter([{'type':'http.request','body':prefix+b'x'*600,'more_body':True},
                        {'type':'http.request','body':b'x'*600+b'\r\n--test-boundary--\r\n','more_body':False}])
            async def receive():return next(parts)
            messages=[]
            async def send(message):messages.append(message)
            scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
                'scheme':'http','path':f'/api/v1/stream-sessions/{self.session}/segments/0','root_path':'',
                'query_string':b'','server':('testserver',80),'client':('127.0.0.1',1234),
                'headers':[(b'content-type',b'multipart/form-data; boundary=test-boundary'),(b'x-api-key',b'test-key')]}
            with patch('starlette.formparsers.SpooledTemporaryFile',side_effect=record):
                await self.app(scope,receive,send)
            self.assertEqual(messages[0]['status'],413)
        asyncio.run(run())
        self.assertTrue(buffers)
        self.assertTrue(all(handle.closed for handle in buffers))
        self.assertEqual(self.app.state.stream_manager.get_session(self.session)['segments'],{})

    def test_exhausted_upload_slots_or_temp_disk_never_start_parsing(self):
        async def run(kind):
            calls=[]
            async def app(scope,receive,send):calls.append(True)
            guard=MediaUploadGuard(app,manager=self.app.state.stream_manager)
            if kind=='slots':guard._inflight=4
            async def receive():raise AssertionError('request body must not be read')
            messages=[]
            async def send(message):messages.append(message)
            # Capacity is already covered by separate tests. Isolate the system-temp check here.
            with patch('api.media_upload_guard.ensure_capacity'),patch('api.media_upload_guard.shutil.disk_usage',return_value=SimpleNamespace(free=0 if kind=='temp' else 10**12)):
                await guard({'type':'http','method':'POST','path':f'/api/v1/stream-sessions/{self.session}/segments/0/repair','headers':[]},receive,send)
            self.assertEqual(calls,[])
            self.assertEqual(messages[0]['status'],429 if kind=='slots' else 507)
        asyncio.run(run('slots'));asyncio.run(run('temp'))


if __name__=='__main__':unittest.main()
