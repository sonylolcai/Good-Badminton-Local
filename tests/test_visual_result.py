import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import Future

from api.visual_result import (compute_visual_result, input_identity, result_path,
    VisualResultService, LocalEvidenceClient, register_visual_result_routes, encode)
from good_badminton_visual.derivation import derive_stream_movement_metrics

def fixtures(root):
    from api.visual_result import implementation
    sid = 'ssn_visual_result_test'
    session = {'analysis_session_id': sid, 'status': 'finalized', 'sealed': True,
        'sealed_at': '2026-11-14T00:00:00Z', 'events_revision': 1, 'continuity_epoch': 0,
        'visual_result': {'schema_version': 'stream-visual-result.v1', 'implementation_sha256': implementation()},
        'expected_last_segment_index': 0, 'configuration': {'analysis_sample_hz': 10},
        'segments': {'0': {'processed': True, 'duration_sec': 2}}}
    events = []
    for i, point in enumerate(([1., 2.], [1.2, 2.], [1.4, 2.])):
        events.append({'schema_version': 'stream-session.v1', 'analysis_session_id': sid,
            'event_id': f'evt_{i}', 'event_type': 'person_observation', 'segment_index': 0,
            'source_time_sec': i*.1, 'data': {'track': {'track_id': 'track_001', 'status': 'detected',
                'confidence': .95, 'court_xy_m': point, 'zone_id': 'mid_center',
                'measurement_bucket': i, 'location_evidence': {'confidence': .9},
                'association': {'source': 'bytetrack', 'identity_confidence': .95}}}})
    root.mkdir(parents=True, exist_ok=True)
    (root/'manifest.json').write_text(json.dumps(session))
    (root/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    progress = {'received_segments': 1, 'processed_segments': 1, 'failed_segments': 0,
        'received_duration_sec': 2, 'processed_source_time_sec': 2, 'missing_segment_indexes': [],
        'failed_segment_indexes': [], 'backlog_duration_sec': 0}
    terminal = {'analysis_session_id': sid, 'status': 'finalized', 'progress': progress}
    manager = Mock(); manager._session_dir.return_value = root
    manager._load.side_effect = lambda _: json.loads((root/'manifest.json').read_text())
    manager._list_sessions.return_value = [session]
    manager._compute_progress.return_value = progress
    manager.worker_running = False
    return session, events, terminal, manager

def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()
                if k not in ('generated_at', 'metrics_path') and not k.endswith('_path')}
    if isinstance(value, list): return [normalized(v) for v in value]
    return value

class VisualResultTests(unittest.TestCase):
    def test_failure_artifact_disk_error_keeps_task_scheduled_until_storage_recovers(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,terminal,manager=fixtures(root)
            executor=Mock();resubmitted=Future();executor.submit.return_value=resubmitted
            service=VisualResultService(manager,active=True,executor=executor)
            original=Future();original.set_exception(OSError('aggregation disk error'))
            identity=input_identity(root,s);service.futures[s['analysis_session_id']]=(original,identity)
            with patch('api.visual_result.atomic_json',side_effect=OSError('failure record disk error')):
                with self.assertRaises(OSError):service.tick()
            self.assertEqual(service.pending_count,1)
            self.assertEqual(service.read(s)[0],202)
            service.tick();executor.submit.assert_called_once()
            self.assertEqual(compute_visual_result(str(root),identity,terminal),{'ready':True})
            resubmitted.set_result({'ready':True});service.tick()
            self.assertEqual(service.read(s)[1]['state'],'ready')
            self.assertEqual(service.outstanding_count([s]),0)

    def test_accepted_summary_recovers_when_admissions_are_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,_,manager=fixtures(root)
            executor=Mock();executor.submit.return_value=Future()
            service=VisualResultService(manager,active=False,executor=executor)
            with patch('api.visual_result.threading.Thread'):
                service.start()
                self.assertEqual(service.pending_count,1)
                service.tick()
            executor.submit.assert_called_once()

    def test_accepted_summary_remains_readable_when_admissions_are_disabled(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,terminal,manager=fixtures(root)
            compute_visual_result(str(root),input_identity(root,s),terminal)
            app=FastAPI()
            with patch.dict('os.environ',{'GOOD_BADMINTON_VISUAL_RESULT_ENABLED':'0'}):
                register_visual_result_routes(app,manager,lambda:None)
            response=TestClient(app).get('/api/v1/stream-sessions/'+s['analysis_session_id']+'/visual-result')
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['state'],'ready')
            manager._load_events.assert_not_called()

    def test_empty_half_court_evidence_fails_before_generating_metrics(self):
        from good_badminton_visual.result_contract import ResultContractError
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,terminal,_=fixtures(root)
            with patch('good_badminton_visual.derivation.generate_movement_metrics') as generate:
                with self.assertRaisesRegex(ResultContractError,'insufficient_player_evidence'):
                    derive_stream_movement_metrics(root/'empty',
                        client=LocalEvidenceClient([],s['analysis_session_id'],1),
                        terminal_status=terminal,
                        create_request={'configuration':{'roster_identity_policy':'half_court_slots.v1'}})
                generate.assert_not_called()

    def test_disabled_capability_still_counts_durable_unfinished_summaries_as_busy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,terminal,manager=fixtures(root)
            service=VisualResultService(manager,active=False)
            self.assertEqual(service.outstanding_count([s]),1)
            compute_visual_result(str(root),input_identity(root,s),terminal)
            self.assertEqual(service.outstanding_count([s]),0)

    def test_business_route_never_schedules_summary_even_when_capability_is_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,_,manager=fixtures(root);s.pop('visual_result')
            (root/'manifest.json').write_text(json.dumps(s))
            service=VisualResultService(manager,active=True,executor=Mock())
            service.notify(s['analysis_session_id']);service.tick();service.executor.submit.assert_not_called()
            self.assertEqual(service.read(s)[1]['error_code'],'visual_result_not_requested')

    def test_create_projection_is_persisted_and_idempotency_includes_route(self):
        from api.stream_sessions import StreamSessionManager
        from api.stream_errors import StreamSessionError
        from api.visual_result import implementation
        from tests.stream_test_utils import create_request
        with tempfile.TemporaryDirectory() as temp:
            manager=StreamSessionManager(temp,start_worker=False)
            request=create_request();request['visual_result']={'schema_version':'stream-visual-result.v1',
                'implementation_sha256':implementation()}
            with patch.dict('os.environ',{'GOOD_BADMINTON_VISUAL_RESULT_ENABLED':'1'}):
                _,created=manager.create_session(request,'routing-test-idempotent-001')
            sid=created['analysis_session_id'];self.assertEqual(manager._load(sid)['visual_result'],request['visual_result'])
            # A retry of an existing accepted request remains idempotent when
            # admissions for NEW summary requests have subsequently been disabled.
            with patch.dict('os.environ',{'GOOD_BADMINTON_VISUAL_RESULT_ENABLED':'0'}):
                self.assertEqual(manager.create_session(request,'routing-test-idempotent-001')[0],200)
                with self.assertRaises(StreamSessionError):manager.create_session(request,'routing-test-other-session')
            legacy=create_request()
            with self.assertRaises(StreamSessionError):manager.create_session(legacy,'routing-test-idempotent-001')

    def test_unsupported_tracking_policy_rejected_before_any_session_or_inference(self):
        from api.stream_sessions import StreamSessionManager
        from api.stream_errors import StreamSessionError
        from api.visual_result import implementation
        from tests.stream_test_utils import create_request
        with tempfile.TemporaryDirectory() as temp:
            manager=StreamSessionManager(temp,start_worker=False)
            anchored=create_request()
            anchored['visual_result']={'schema_version':'stream-visual-result.v1','implementation_sha256':implementation()}
            anchored['configuration']['roster_identity_policy']='anchored_roster.v1'
            anchored['configuration']['expected_player_count']=4
            unsupported=create_request()
            unsupported['visual_result']={'schema_version':'stream-visual-result.v1','implementation_sha256':implementation()}
            unsupported['configuration']['sport_id']='tennis'
            with patch.dict('os.environ',{'GOOD_BADMINTON_VISUAL_RESULT_ENABLED':'1'}):
                self.assertEqual(manager.create_session(anchored,'anchored-allowed-001')[0],202)
                with self.assertRaises(StreamSessionError):manager.create_session(unsupported,'unsupported-sport-001')
            self.assertEqual(len(manager._list_sessions()),1)

    def test_terminal_save_only_notifies_requested_summary_sessions(self):
        from api.stream_sessions import StreamSessionManager
        with tempfile.TemporaryDirectory() as temp:
            manager=StreamSessionManager(temp,start_worker=False)
            root=manager._session_dir('ssn_visual_result_test');s,_,_,_=fixtures(root)
            service=Mock();service.enabled=True;manager.visual_result_service=service
            legacy={k:v for k,v in s.items() if k!='visual_result'}
            manager._save(legacy);service.notify.assert_not_called()
            manager._save(s);service.notify.assert_called_once_with(s['analysis_session_id'])
            self.assertTrue(manager._load(s['analysis_session_id'])['visual_result_required'])

    def test_local_cpu_metrics_equal_existing_paged_derivation_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'session';s,events,terminal,manager=fixtures(root)
            expected=derive_stream_movement_metrics(Path(temp)/'legacy',
                client=LocalEvidenceClient(events,s['analysis_session_id'],1),
                terminal_status=terminal,create_request={'configuration':s['configuration']})
            identity=input_identity(root,s)
            with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
                self.assertEqual(compute_visual_result(str(root),identity,terminal),{'ready':True})
            code,result=VisualResultService(manager,active=True).read(s)
            self.assertEqual(code,200)
            self.assertEqual(normalized(result['metrics']),normalized(expected['metrics']))
            self.assertEqual(result['source']['event_count'],3)
            self.assertEqual(result['source']['events_sha256'],hashlib.sha256((root/'events.jsonl').read_bytes()).hexdigest())
            self.assertNotIn('metrics_path',result['metrics'])
            self.assertEqual(result['completeness']['status'],'partial')

    def test_revision_change_invalidates_ready_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,terminal,manager=fixtures(root)
            identity=input_identity(root,s);compute_visual_result(str(root),identity,terminal)
            newer={**s,'events_revision':2};(root/'manifest.json').write_text(json.dumps(newer))
            code,value=VisualResultService(manager,active=True).read(newer)
            self.assertEqual((code,value['state']),(202,'pending'))
            self.assertEqual(compute_visual_result(str(root),identity,terminal),{'stale':True})

    def test_evidence_change_during_aggregation_cannot_publish_old_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,terminal,_=fixtures(root);identity=input_identity(root,s)
            def changed(*args,**kwargs):
                result=derive_stream_movement_metrics(*args,**kwargs)
                with (root/'events.jsonl').open('a') as f:f.write('\n')
                return result
            with patch('good_badminton_visual.derivation.derive_stream_movement_metrics',side_effect=changed):
                self.assertEqual(compute_visual_result(str(root),identity,terminal),{'stale':True})
            self.assertFalse(result_path(root,identity).exists())

    def test_pending_read_does_not_load_raw_events_or_schedule_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,_,manager=fixtures(root)
            service=VisualResultService(manager,active=True,executor=Mock())
            for _ in range(3):self.assertEqual(service.read(s)[0],202)
            manager._load_events.assert_not_called();service.executor.submit.assert_not_called()

    def test_checksum_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,terminal,manager=fixtures(root);identity=input_identity(root,s)
            compute_visual_result(str(root),identity,terminal);p=result_path(root,identity)
            value=json.loads(p.read_text());value['metrics']['track_count']=99;p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'checksum'):
                VisualResultService(manager,active=True).read(s)

    def test_scheduler_deduplicates_and_tracks_cpu_work_for_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,_,manager=fixtures(root);executor=Mock();future=Future()
            executor.submit.return_value=future;service=VisualResultService(manager,active=True,executor=executor)
            service.notify(s['analysis_session_id']);service.notify(s['analysis_session_id'])
            self.assertEqual(service.pending_count,1);service.tick()
            self.assertEqual(service.pending_count,1);service.tick();executor.submit.assert_called_once()
            future.set_result({'ready':True});service.tick();self.assertEqual(service.pending_count,0)

    def test_missing_finalized_evidence_fails_instead_of_pending_forever(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,_,manager=fixtures(root)
            (root/'events.jsonl').unlink()
            code,value=VisualResultService(manager,active=True).read(s)
            self.assertEqual((code,value['state']),(200,'failed'))

    def test_cpu_failure_persists_and_restart_does_not_rerun_failed_evidence(self):
        from good_badminton_visual.result_contract import ResultContractError
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,_,manager=fixtures(root);executor=Mock();future=Future()
            executor.submit.return_value=future;service=VisualResultService(manager,active=True,executor=executor)
            service.notify(s['analysis_session_id']);service.tick();future.set_exception(ResultContractError('insufficient_player_evidence'))
            service.tick();code,value=service.read(s)
            self.assertEqual((code,value['state']),(200,'failed'))
            self.assertEqual(value['detail_code'],'insufficient_player_evidence')
            restored=VisualResultService(manager,active=True,executor=Mock())
            restored.notify(s['analysis_session_id']);restored.tick();restored.executor.submit.assert_not_called()

    def test_visual_result_capability_defaults_enabled_when_unset(self):
        from api.visual_result import enabled
        with patch.dict('os.environ',{},clear=False):
            import os
            previous = os.environ.pop('GOOD_BADMINTON_VISUAL_RESULT_ENABLED', None)
            try:
                self.assertTrue(enabled())
                with patch.dict('os.environ',{'GOOD_BADMINTON_VISUAL_RESULT_ENABLED':'0'}):
                    self.assertFalse(enabled())
            finally:
                if previous is not None:
                    os.environ['GOOD_BADMINTON_VISUAL_RESULT_ENABLED'] = previous

    def test_submit_failure_requeues_for_restart_safe_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,_,manager=fixtures(root);executor=Mock()
            executor.submit.side_effect=RuntimeError('pool unavailable')
            service=VisualResultService(manager,active=True,executor=executor);service.notify(s['analysis_session_id'])
            with self.assertRaises(RuntimeError):service.tick()
            self.assertEqual(service.pending_count,1);self.assertIsNone(service.executor)

    def test_corrupted_artifact_is_regenerated_by_scheduled_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,terminal,manager=fixtures(root);identity=input_identity(root,s)
            compute_visual_result(str(root),identity,terminal);result_path(root,identity).write_text('{')
            executor=Mock();executor.submit.return_value=Future()
            service=VisualResultService(manager,active=True,executor=executor)
            service.notify(s['analysis_session_id']);service.tick();executor.submit.assert_called_once()

    def test_route_requires_auth_and_does_not_accept_path_escape(self):
        from fastapi import FastAPI,Header,HTTPException
        from fastapi.testclient import TestClient
        def auth(x_api_key: str = Header(default='')):
            if x_api_key!='test':raise HTTPException(401,'unauthorized')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);s,_,_,manager=fixtures(root);app=FastAPI()
            with patch.dict('os.environ',{'GOOD_BADMINTON_VISUAL_RESULT_ENABLED':'1'}):
                register_visual_result_routes(app,manager,auth)
            client=TestClient(app);path='/api/v1/stream-sessions/'+s['analysis_session_id']+'/visual-result'
            self.assertEqual(client.get(path).status_code,401)
            self.assertEqual(client.get(path,headers={'X-API-Key':'test'}).status_code,202)
            self.assertEqual(client.get('/api/v1/stream-sessions/invalid!id/visual-result',headers={'X-API-Key':'test'}).status_code,404)
            manager._load_events.assert_not_called()

if __name__=='__main__':unittest.main()


class PreviousImplementationTests(unittest.TestCase):
    def test_unfinished_admitted_session_keeps_previous_result_identity(self):
        from api.visual_result import PREVIOUS_IMPLEMENTATION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, _, terminal, manager = fixtures(root)
            session['visual_result']['implementation_sha256'] = PREVIOUS_IMPLEMENTATION
            (root/'manifest.json').write_text(json.dumps(session))
            identity = input_identity(root, session)
            self.assertEqual(identity['implementation_sha256'], PREVIOUS_IMPLEMENTATION)
            self.assertEqual(compute_visual_result(root, identity, terminal), {'ready': True})
            self.assertEqual(VisualResultService(manager).read(session)[1]['state'], 'ready')
