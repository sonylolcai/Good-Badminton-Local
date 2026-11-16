import asyncio,json,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from api.gpu_stage_timing import stage,span,summary

class StageTimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=SimpleNamespace(sessions_dir=Path(self.tmp.name));self.path=Path(self.tmp.name)/'session'
        self.path.mkdir();(self.path/'manifest.json').write_text('{}')
    def test_nested_stages_and_original_return_and_exception(self):
        @stage('outer',manager=self.manager)
        def run(session_id):
            with span(self.manager,session_id,'inner'):return 17
        self.assertEqual(run('session'),17)
        with self.assertRaisesRegex(ValueError,'original'):
            with span(self.manager,'session','failed'):raise ValueError('original')
        result=summary(self.path)
        self.assertEqual(result['stages']['outer']['count'],1)
        self.assertEqual(result['stages']['inner']['returned_count'],1)
        self.assertEqual(result['stages']['failed']['error_count'],1)
        self.assertEqual(len({r['clock_id'] for r in result['anchors']}),1)
        self.assertGreaterEqual(result['stages']['outer']['total_ms'],result['stages']['inner']['total_ms'])
    def test_timing_write_failure_does_not_change_work_or_its_exception(self):
        @stage('work',manager=self.manager)
        def run(session_id):return 19
        with patch('api.gpu_stage_timing.os.open',side_effect=OSError('disk full')):
            self.assertEqual(run('session'),19)
            with self.assertRaisesRegex(RuntimeError,'work error'):
                with span(self.manager,'session','work'):raise RuntimeError('work error')
        self.assertEqual(summary(self.path)['anchor_count'],0)
    def test_concurrent_append_truncation_and_corruption_are_visible(self):
        def run():
            for _ in range(10):
                with span(self.manager,'session','worker'):pass
        threads=[threading.Thread(target=run) for _ in range(4)]
        for t in threads:t.start()
        for t in threads:t.join()
        with (self.path/'gpu_stage_timing.jsonl').open('a') as f:f.write('broken\n')
        result=summary(self.path,limit=6)
        self.assertEqual(result['stages']['worker']['count'],40)
        self.assertEqual(result['anchor_count'],80);self.assertEqual(result['anchors_omitted'],74)
        self.assertEqual(result['invalid_lines'],1)
    def test_async_signature_and_bound_session_dictionary(self):
        @stage('async',manager=self.manager)
        async def run(session_id:str,segment_index:int):return segment_index
        self.assertEqual(asyncio.run(run('session',7)),7)
        class Owner:
            manager=self.manager
            @stage('archive')
            def work(self,snapshot):return True
        self.assertTrue(Owner().work({'analysis_session_id':'session'}))
        self.assertEqual(summary(self.path)['stages']['archive']['count'],1)
    def test_unsafe_identifier_cannot_write_outside_session(self):
        with span(self.manager,'../escape','unsafe'):pass
        self.assertFalse((Path(self.tmp.name).parent/'escape/gpu_stage_timing.jsonl').exists())

    def test_ingress_middleware_measures_body_read_without_storing_body(self):
        from api.gpu_stage_timing import UploadTimingMiddleware
        async def app(scope,receive,send):
            await receive();await send({'type':'http.response.start','status':202});await send({'type':'http.response.body','body':b''})
        async def receive():return {'type':'http.request','body':b'secret-video-content'}
        async def send(_):pass
        middleware=UploadTimingMiddleware(app,self.manager)
        asyncio.run(middleware({'type':'http','method':'POST','path':'/api/v1/stream-sessions/session/segments/0'},receive,send))
        result=summary(self.path);self.assertEqual(result['stages']['http_upload_request']['count'],1)
        self.assertNotIn('secret-video-content',json.dumps(result))
        asyncio.run(middleware({'type':'http','method':'POST','path':'/api/v1/stream-sessions/foreign/segments/0'},receive,send))
        self.assertFalse((Path(self.tmp.name)/'foreign').exists())
    def test_queue_and_seal_intervals_require_one_process_clock(self):
        from api.gpu_stage_timing import append
        rows=[('a','segment_accepted','point',1,0),('a','segment_processing_total','start',5,0),('a','input_sealed','point',6,None),('b','algorithm_finalize','end',9,None)]
        for clock,name,edge,t,index in rows:
            append(self.path,{'clock_id':clock,'stage':name,'edge':edge,'monotonic_ns':t*1000000,'segment_index':index,'duration_ms':1,'outcome':'returned'})
        result=summary(self.path);self.assertEqual(result['queue_wait']['mean_ms'],4)
        self.assertIsNone(result['input_seal_to_finalize_return_ms'])
    def test_real_registered_upload_route_and_trace_include_timing_without_running_inference(self):
        import os
        from fastapi.testclient import TestClient
        from api.app import create_app
        from api.stream_sessions import StreamSessionManager
        from tests.stream_test_utils import create_request,segment_metadata,counting_processor_factory
        manager=StreamSessionManager(Path(self.tmp.name)/'api',processor_factory=counting_processor_factory(),start_worker=False)
        _,created=manager.create_session(create_request(),'timing-contract-local-test')
        sid=created['analysis_session_id'];raw=b'fake-video-no-decoder-runs'
        with patch.dict(os.environ,{'GOOD_BADMINTON_API_KEY':'timing-test-key'}):
            client=TestClient(create_app(data_dir=Path(self.tmp.name)/'app',stream_manager=manager,start_worker=False))
            response=client.post(f'/api/v1/stream-sessions/{sid}/segments/0',headers={'X-API-Key':'timing-test-key'},files={'segment':('segment.mp4',raw,'video/mp4')},data={'metadata':json.dumps(segment_metadata(0,raw))})
            self.assertEqual(response.status_code,202,response.text)
            result=client.get(f'/api/v1/stream-sessions/{sid}/trace',headers={'X-API-Key':'timing-test-key'}).json()
        timing=result['gpu_stage_timing']
        self.assertEqual(timing['stages']['http_upload_request']['count'],1)
        self.assertEqual(timing['stages']['gpu_receive_validate_persist']['count'],1)
        self.assertNotIn('decode_sample_models_tracking',timing['stages'])
