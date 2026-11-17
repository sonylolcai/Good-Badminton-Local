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
            await middleware({'type':'http','method':'POST','path':f'/api/v1/stream-sessions/{self.session}/segments/0','headers':[(b'x-api-key',b'test-key'),*headers]},receive,send)
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
                await guard({'type':'http','method':'POST','path':f'/api/v1/stream-sessions/{self.session}/segments/0/repair','headers':[(b'x-api-key',b'test-key')]},receive,send)
            self.assertEqual(calls,[])
            self.assertEqual(messages[0]['status'],429 if kind=='slots' else 507)
        asyncio.run(run('slots'));asyncio.run(run('temp'))

    def test_unauthenticated_segment_and_repair_requests_fail_before_slot_or_body_read(self):
        from datetime import datetime,timezone
        import hashlib
        from api.capture_leases import iso

        async def run():
            manager=self.app.state.stream_manager
            now_ts=datetime.now(timezone.utc).timestamp()
            tok1='fixture-upload-token-1'
            manager.register_capture_lease(self.session,{
                'schema_version':'capture-lease.v1','lease_id':'lease_1','match_id':'match_1',
                'edge_session_id':'edge_1','device_id':'device_1','camera_id':'camera_1','sport_id':'badminton',
                'started_at':iso(now_ts),'expires_at':iso(now_ts+1800),
                'token_sha256':hashlib.sha256(tok1.encode()).hexdigest(),
            })
            other=self.client.post('/api/v1/stream-sessions',json=create_request(),headers={'X-API-Key':'test-key'}).json()['analysis_session_id']
            tok2='fixture-upload-token-2'
            manager.register_capture_lease(other,{
                'schema_version':'capture-lease.v1','lease_id':'lease_2','match_id':'match_2',
                'edge_session_id':'edge_2','device_id':'device_2','camera_id':'camera_2','sport_id':'badminton',
                'started_at':iso(now_ts),'expires_at':iso(now_ts+1800),
                'token_sha256':hashlib.sha256(tok2.encode()).hexdigest(),
            })

            app_calls=[]
            async def dummy_app(scope,receive,send):
                app_calls.append(scope['path'])
                await send({'type':'http.response.start','status':202,'headers':[]})
                await send({'type':'http.response.body','body':b'{}'})

            guard=MediaUploadGuard(dummy_app,manager=manager)
            async def forbidden_receive():
                raise AssertionError('unauthenticated upload must not read body')

            bad_cases=[
                (f'/api/v1/stream-sessions/{self.session}/segments/0',[]),
                (f'/api/v1/stream-sessions/{self.session}/segments/0/repair',[]),
                (f'/api/v1/stream-sessions/{self.session}/segments/0',[(b'x-api-key',b'wrong-key')]),
                (f'/api/v1/stream-sessions/{self.session}/segments/0',[(b'x-upload-token',tok2.encode())]),
                ('/api/v1/stream-sessions/../segments/0',[(b'x-upload-token',tok1.encode())]),
                ('/api/v1/stream-sessions/invalid.id/segments/0',[(b'x-upload-token',tok1.encode())]),
                (f'/api/v1/stream-sessions/{self.session}/segments/+0',[]),
                (f'/api/v1/stream-sessions/{self.session}/segments/-1',[]),
                (f'/api/v1/stream-sessions/{self.session}/segments/1_0/repair',[]),
                (f'/api/v1/stream-sessions/{self.session}/segments/0',[(b'x-api-key',b'\xff')]),
                (f'/api/v1/stream-sessions/{self.session}/segments/0',[(b'x-upload-token',b'\xff')]),
                (f'/api/v1/stream-sessions/{self.session}/segments/0',[(b'x-upload-token',b'wrong'),(b'x-upload-token',tok1.encode())]),
            ]
            with patch.dict('os.environ',{'GOOD_BADMINTON_STREAM_MAX_UPLOAD_REQUESTS':'1'}):
                for target_path,hdrs in bad_cases*2:
                    msgs=[]
                    async def send(msg):msgs.append(msg)
                    await guard({'type':'http','method':'POST','path':target_path,'headers':hdrs},forbidden_receive,send)
                    self.assertEqual(msgs[0]['status'],401)
                    self.assertEqual(guard._inflight,0)
                    self.assertEqual(guard._reserved,0)
                self.assertEqual(app_calls,[])

                # Valid upload token still succeeds immediately without encountering 429
                ok_msgs=[]
                async def ok_receive():return {'type':'http.request','body':b'','more_body':False}
                async def ok_send(msg):ok_msgs.append(msg)
                await guard({'type':'http','method':'POST','path':f'/api/v1/stream-sessions/{self.session}/segments/0','headers':[(b'x-upload-token',tok1.encode())]},ok_receive,ok_send)
                self.assertEqual(ok_msgs[0]['status'],202)
                self.assertEqual(guard._inflight,0)
                self.assertEqual(guard._reserved,0)
        asyncio.run(run())

    def test_slow_authenticated_upload_times_out_closes_spool_and_releases_slot(self):
        from datetime import datetime,timezone
        import hashlib
        from api.capture_leases import iso
        buffers=[]
        factory=tempfile.SpooledTemporaryFile
        def record(*args,**kwargs):
            handle=factory(*args,**kwargs);buffers.append(handle);return handle

        async def run():
            manager=self.app.state.stream_manager
            token='fixture-slow-token'
            now_ts=datetime.now(timezone.utc).timestamp()
            manager.register_capture_lease(self.session,{
                'schema_version':'capture-lease.v1','lease_id':'lease_1','match_id':'match_1',
                'edge_session_id':'edge_1','device_id':'device_1','camera_id':'camera_1','sport_id':'badminton',
                'started_at':iso(now_ts),'expires_at':iso(now_ts+1800),
                'token_sha256':hashlib.sha256(token.encode()).hexdigest(),
            })
            prefix=(b'--slow-boundary\r\nContent-Disposition: form-data; name="segment"; filename="segment.mp4"'
                    b'\r\nContent-Type: video/mp4\r\n\r\n')
            step=0
            async def slow_receive():
                nonlocal step
                step+=1
                if step==1:
                    return {'type':'http.request','body':prefix+b'x'*32,'more_body':True}
                await asyncio.sleep(5)
                return {'type':'http.request','body':b'\r\n--slow-boundary--\r\n','more_body':False}

            guard=None
            # Locate MediaUploadGuard wrapper on the FastAPI middleware stack
            cur=self.app.middleware_stack or self.app.build_middleware_stack()
            while cur is not None:
                if isinstance(cur,MediaUploadGuard):
                    guard=cur
                    break
                cur=getattr(cur,'app',None)
            self.assertIsNotNone(guard)

            msgs=[]
            async def send(msg):msgs.append(msg)
            scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
                'scheme':'http','path':f'/api/v1/stream-sessions/{self.session}/segments/0','root_path':'',
                'query_string':b'','server':('testserver',80),'client':('127.0.0.1',1234),
                'headers':[(b'content-type',b'multipart/form-data; boundary=slow-boundary'),(b'x-upload-token',token.encode())]}
            with patch.dict('os.environ',{'GOOD_BADMINTON_STREAM_UPLOAD_BODY_TIMEOUT_SECONDS':'0.05'}),patch('starlette.formparsers.SpooledTemporaryFile',side_effect=record):
                await self.app(scope,slow_receive,send)
            self.assertEqual(msgs[0]['status'],408)
            self.assertEqual(guard._inflight,0)
            self.assertEqual(guard._reserved,0)
        asyncio.run(run())
        self.assertTrue(buffers)
        self.assertTrue(all(handle.closed for handle in buffers))

    def test_cancelled_upload_and_closed_lease_token_fail_closed_without_leaking_slots(self):
        from datetime import datetime,timezone
        import hashlib,json
        from api.capture_leases import iso
        from tests.stream_test_utils import segment_metadata,write_video_segment_bytes

        async def run():
            manager=self.app.state.stream_manager
            token='fixture-lease-token'
            now_ts=datetime.now(timezone.utc).timestamp()
            manager.register_capture_lease(self.session,{
                'schema_version':'capture-lease.v1','lease_id':'lease_1','match_id':'match_1',
                'edge_session_id':'edge_1','device_id':'device_1','camera_id':'camera_1','sport_id':'badminton',
                'started_at':iso(now_ts),'expires_at':iso(now_ts+1800),
                'token_sha256':hashlib.sha256(token.encode()).hexdigest(),
            })

            started=asyncio.Event()
            async def stalled_app(scope,receive,send):
                started.set()
                await receive()

            guard=MediaUploadGuard(stalled_app,manager=manager)
            async def hanging_receive():
                await asyncio.sleep(5)
                return {'type':'http.request','body':b'','more_body':False}

            msgs=[]
            async def send(msg):msgs.append(msg)
            scope={'type':'http','method':'POST','path':f'/api/v1/stream-sessions/{self.session}/segments/0','headers':[(b'x-upload-token',token.encode())]}
            task=asyncio.create_task(guard(scope,hanging_receive,send))
            await started.wait()
            await asyncio.sleep(0.01)
            self.assertEqual(guard._inflight,1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(guard._inflight,0)
            self.assertEqual(guard._reserved,0)

            # Mid-flight lease close race after pre-guard auth still fails closed cleanly in handler
            raw=write_video_segment_bytes()[:64]
            meta=json.dumps(segment_metadata(0,raw))
            boundary='race-boundary'
            body=(f'--{boundary}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n{meta}\r\n'
                  f'--{boundary}\r\nContent-Disposition: form-data; name="segment"; filename="seg.mp4"\r\nContent-Type: video/mp4\r\n\r\n').encode()+raw+f'\r\n--{boundary}--\r\n'.encode()
            part1,part2=body[:len(body)//2],body[len(body)//2:]
            chunk_idx=0
            async def racing_receive():
                nonlocal chunk_idx
                chunk_idx+=1
                if chunk_idx==1:
                    return {'type':'http.request','body':part1,'more_body':True}
                manager.interrupt_capture(self.session,{'lease_id':'lease_1','reason':'terminal_restarted'})
                return {'type':'http.request','body':part2,'more_body':False}

            race_msgs=[]
            async def race_send(msg):race_msgs.append(msg)
            race_scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
                'scheme':'http','path':f'/api/v1/stream-sessions/{self.session}/segments/0','root_path':'',
                'query_string':b'','server':('testserver',80),'client':('127.0.0.1',1234),
                'headers':[(b'content-type',f'multipart/form-data; boundary={boundary}'.encode()),(b'x-upload-token',token.encode())]}
            await self.app(race_scope,racing_receive,race_send)
            self.assertEqual(race_msgs[0]['status'],401)
            self.assertEqual(manager.get_session(self.session)['segments'],{})

            # Subsequent attempt with closed lease token is rejected 401 before slot acquisition
            closed_msgs=[]
            async def closed_send(msg):closed_msgs.append(msg)
            await guard(scope,hanging_receive,closed_send)
            self.assertEqual(closed_msgs[0]['status'],401)
            self.assertEqual(guard._inflight,0)
        asyncio.run(run())

    def test_upload_body_timeout_environment_validation(self):
        from api.media_limits import MediaLimits
        for bad in ('0','-1','nan','inf','-inf','abc'):
            with patch.dict('os.environ',{'GOOD_BADMINTON_STREAM_UPLOAD_BODY_TIMEOUT_SECONDS':bad}):
                with self.assertRaises(ValueError):
                    MediaLimits.from_environment()
        with patch.dict('os.environ',{'GOOD_BADMINTON_STREAM_UPLOAD_BODY_TIMEOUT_SECONDS':'2.5'}):
            self.assertEqual(MediaLimits.from_environment().upload_body_timeout_seconds,2.5)


if __name__=='__main__':unittest.main()
