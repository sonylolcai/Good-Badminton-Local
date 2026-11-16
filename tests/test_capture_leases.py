"""Fixed capture deadlines and signed lifecycle delivery without inference."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from api.capture_leases import iso
from api.capture_notifications import CaptureNotifications, lifecycle_url
from api.stream_errors import StreamSessionError
from api.stream_routes import register_stream_routes
from api.stream_sessions import StreamSessionManager
from tests.stream_test_utils import create_request, segment_metadata, counting_processor_factory


class CaptureLeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.clock=[datetime.now(timezone.utc).timestamp()]
        self.manager=StreamSessionManager(Path(self.tmp.name),processor_factory=counting_processor_factory(),
            start_worker=False,capture_clock=lambda:self.clock[0])
        self.sid=self.manager.create_session(create_request())[1]['analysis_session_id']
        self.token='fixture-business-issued-token'
        self.body={'schema_version':'capture-lease.v1','lease_id':'lease_1','match_id':'match_1',
            'edge_session_id':'edge_1','device_id':'device_1','camera_id':'camera_1','sport_id':'badminton',
            'started_at':iso(self.clock[0]),'expires_at':iso(self.clock[0]+1800),
            'token_sha256':hashlib.sha256(self.token.encode()).hexdigest()}

    def register(self):return self.manager.register_capture_lease(self.sid,self.body)
    def signal(self,sequence=1,token=None,lease='lease_1'):
        return self.manager.capture_signal(self.sid,{'lease_id':lease,'sequence':sequence},self.token if token is None else token)

    def test_activation_delay_does_not_consume_signal_window_and_register_retry_does_not_reset_it(self):
        for delay in (0,119,121):
            with self.subTest(activation_delay=delay):
                fixture=CaptureLeaseTests();fixture.setUp()
                try:
                    deadline=fixture.body['expires_at'];fixture.clock[0]+=delay
                    activation=fixture.clock[0];fixture.register()
                    lease=fixture.manager.get_session(fixture.sid)['capture_lease']
                    self.assertEqual(lease['last_signal_at'],iso(activation))
                    self.assertEqual(lease['started_at'],fixture.body['started_at'])
                    self.assertEqual(lease['expires_at'],deadline)
                    fixture.clock[0]+=119;fixture.manager.check_capture_leases()
                    self.assertEqual(fixture.manager.get_session(fixture.sid)['capture_lease']['state'],'active')
                    fixture.register()
                    self.assertEqual(fixture.manager.get_session(fixture.sid)['capture_lease']['last_signal_at'],iso(activation))
                    fixture.clock[0]+=1;fixture.manager.check_capture_leases()
                    ended=fixture.manager.get_session(fixture.sid)
                    self.assertEqual(ended['capture_end_reason'],'input_signal_timeout')
                    self.assertEqual(ended['capture_lease']['expires_at'],deadline)
                finally:fixture.doCleanups()

    def test_delayed_activation_cannot_extend_fixed_business_deadline(self):
        deadline=self.body['expires_at'];self.clock[0]+=1790;self.register()
        self.clock[0]+=9;self.manager.check_capture_leases()
        self.assertTrue(self.manager.verify_upload_token(self.sid,self.token))
        self.clock[0]+=1;self.manager.check_capture_leases()
        ended=self.manager.get_session(self.sid)
        self.assertEqual(ended['capture_end_reason'],'token_expired')
        self.assertEqual(ended['capture_lease']['expires_at'],deadline)
        self.assertFalse(self.manager.verify_upload_token(self.sid,self.token))

    def test_fixed_deadline_never_extends_on_register_retry_signal_or_legacy_authorization(self):
        self.register();deadline=self.body['expires_at'];self.clock[0]+=100;self.signal()
        self.register()
        self.assertEqual(self.manager.get_session(self.sid)['capture_lease']['expires_at'],deadline)
        with patch.dict('os.environ',{'GOOD_BADMINTON_UPLOAD_TOKEN_TTL_SECONDS':'2700','GOOD_BADMINTON_STREAM_MIN_FREE_BYTES':'1'}):
            with self.assertRaises(StreamSessionError):self.manager.issue_upload_token(self.sid)
        replaced=dict(self.body,expires_at=iso(self.clock[0]+1800),started_at=iso(self.clock[0]))
        with self.assertRaises(StreamSessionError):self.manager.register_capture_lease(self.sid,replaced)

    def test_120_second_timeout_from_activation_revokes_and_persists_event(self):
        self.register();self.clock[0]+=119;self.manager.check_capture_leases()
        self.assertTrue(self.manager.verify_upload_token(self.sid,self.token))
        self.clock[0]+=1;self.manager.check_capture_leases()
        session=self.manager.get_session(self.sid)
        self.assertEqual(session['status'],'cancelled');self.assertEqual(session['capture_end_reason'],'input_signal_timeout')
        self.assertFalse(self.manager.verify_upload_token(self.sid,self.token))
        self.assertEqual(session['capture_lifecycle_event']['reason'],'input_signal_timeout')
        self.assertNotIn('media_status',session['capture_lifecycle_event'])
        event=session['capture_lifecycle_event'];self.manager.check_capture_leases()
        self.assertEqual(self.manager.get_session(self.sid)['capture_lifecycle_event'],event)

    def test_only_fresh_authenticated_signals_or_new_segments_refresh_liveness(self):
        self.register();self.clock[0]+=100;self.signal()
        saved=self.manager.get_session(self.sid)['capture_lease']['last_signal_at']
        for seq,token,lease in [(1,None,'lease_1'),(2,'wrong','lease_1'),(2,None,'foreign')]:
            with self.subTest(seq=seq,token=token,lease=lease),self.assertRaises(StreamSessionError):self.signal(seq,token,lease)
        self.assertEqual(self.manager.get_session(self.sid)['capture_lease']['last_signal_at'],saved)
        self.clock[0]+=119;raw=b'accepted-source';meta=segment_metadata(0,raw,source_start=0)
        self.manager.receive_segment(self.sid,0,meta,raw)
        self.clock[0]+=119;self.manager.receive_segment(self.sid,0,meta,raw)  # duplicate must not refresh
        self.clock[0]+=1;self.manager.check_capture_leases()
        self.assertEqual(self.manager.get_session(self.sid)['capture_end_reason'],'input_signal_timeout')
        self.assertEqual(self.manager._segment_path(self.sid,0).read_bytes(),raw)

    def test_expiry_closes_input_at_fixed_time_even_with_live_signals_and_preserves_partial_bytes(self):
        self.register();raw=b'accepted-source';self.manager.receive_segment(self.sid,0,segment_metadata(0,raw,source_start=0),raw)
        for sequence in range(1,18):
            self.clock[0]+=100;self.signal(sequence)
        self.clock[0]+=99;self.signal(18);self.clock[0]+=1
        self.manager.check_capture_leases();session=self.manager.get_session(self.sid)
        self.assertEqual(session['status'],'draining');self.assertTrue(session['capture_incomplete'])
        self.assertEqual(session['capture_end_reason'],'token_expired');self.assertEqual(session['expected_last_segment_index'],0)
        self.assertFalse(self.manager.verify_upload_token(self.sid,self.token))
        with self.assertRaises(StreamSessionError):self.manager.receive_segment(self.sid,1,segment_metadata(1,b'late',source_start=1),b'late')
        self.assertEqual(self.manager._segment_path(self.sid,0).read_bytes(),raw)
        self.manager._finalize_complete(session)
        self.assertEqual(self.manager.get_session(self.sid)['status'],'partial')

    def test_restart_restores_timeout_state_and_deadline_without_new_window(self):
        self.register();self.clock[0]+=80;self.signal();self.clock[0]+=120
        restarted=StreamSessionManager(Path(self.tmp.name),processor_factory=counting_processor_factory(),
            start_worker=False,capture_clock=lambda:self.clock[0])
        restarted.check_capture_leases();session=restarted.get_session(self.sid)
        self.assertEqual(session['capture_end_reason'],'input_signal_timeout')
        self.assertEqual(session['capture_lease']['expires_at'],self.body['expires_at'])

    def test_manually_revoked_token_cannot_refresh_capture_signal_or_accept_input(self):
        self.register();self.clock[0]+=20;self.manager.revoke_upload_token(self.sid)
        before=self.manager.get_session(self.sid)['capture_lease']['last_signal_at']
        with self.assertRaises(StreamSessionError):self.signal()
        self.assertFalse(self.manager.verify_upload_token(self.sid,self.token))
        raw=b'late';metadata=segment_metadata(0,raw,source_start=0)
        with self.assertRaises(StreamSessionError):self.manager.receive_segment(self.sid,0,metadata,raw)
        self.assertEqual(self.manager.get_session(self.sid)['capture_lease']['last_signal_at'],before)

    def test_normal_sealed_processing_is_not_interrupted_by_missing_capture_signals(self):
        self.register();session=self.manager.get_session(self.sid);session.update(sealed=True,status='draining')
        self.manager._save(session);self.clock[0]+=1900;self.manager.check_capture_leases()
        session=self.manager.get_session(self.sid)
        self.assertEqual(session['status'],'draining');self.assertNotIn('capture_lifecycle_event',session)

    def test_interruption_is_bound_idempotent_and_preserves_first_reason(self):
        self.register()
        with self.assertRaises(StreamSessionError):self.manager.interrupt_capture(self.sid,{'lease_id':'other','reason':'terminal_restarted'})
        result=self.manager.interrupt_capture(self.sid,{'lease_id':'lease_1','reason':'terminal_restarted'})
        self.assertEqual(result['state'],'ended')
        event=self.manager.get_session(self.sid)['capture_lifecycle_event']
        self.clock[0]+=2000;self.manager.interrupt_capture(self.sid,{'lease_id':'lease_1','reason':'token_expired'})
        self.assertEqual(self.manager.get_session(self.sid)['capture_lifecycle_event'],event)

    def test_registration_input_binding_validation_and_cannot_replace_existing_input(self):
        for change in ({'expires_at':iso(self.clock[0]+2700)},{'started_at':iso(self.clock[0]+100)},
                       {'token_sha256':'bad'},{'lease_id':'a'*129},{'extra':True},{'sport_id':'tennis'}):
            with self.subTest(change=change),self.assertRaises(StreamSessionError):self.manager.register_capture_lease(self.sid,self.body|change)
        raw=b'input';self.manager.receive_segment(self.sid,0,segment_metadata(0,raw,source_start=0),raw)
        with self.assertRaises(StreamSessionError):self.register()

    def test_event_delivery_is_durable_and_independent_of_archive_with_lost_response_and_restart(self):
        self.register();self.manager.interrupt_capture(self.sid,{'lease_id':'lease_1','reason':'terminal_restarted'})
        calls=[]
        def fail(value):calls.append(deepcopy(value));raise OSError('lost response')
        sender=CaptureNotifications(self.manager,url='https://business.test/api/v1/edge/gpu-lifecycle-events',
            secret='s'*32,send=fail,clock=lambda:self.clock[0]);sender.run_once()
        self.assertEqual(sender.pending_count(),1)
        state=json.loads(sender._path(self.sid).read_text());self.assertEqual(state['attempts'],1)
        self.clock[0]+=3
        restarted=CaptureNotifications(self.manager,url='https://business.test/api/v1/edge/gpu-lifecycle-events',
            secret='s'*32,send=lambda value:calls.append(deepcopy(value)),clock=lambda:self.clock[0]);restarted.run_once()
        self.assertEqual(calls[0],calls[1]);self.assertEqual(restarted.pending_count(),0)
        self.assertEqual(lifecycle_url('https://business.test/api/v1/edge/gpu-results'),'https://business.test/api/v1/edge/gpu-lifecycle-events')

    def test_ended_management_status_exposes_durable_proof_without_token_or_hash(self):
        self.register();self.manager.interrupt_capture(self.sid,{'lease_id':'lease_1','reason':'terminal_restarted'})
        status=self.manager.get_status(self.sid)[1]
        proof=status['capture_lease']
        self.assertEqual(set(proof),{'lease_id','state','end_reason','expires_at','ended_at'})
        self.assertEqual(proof['state'],'ended');self.assertEqual(proof['end_reason'],'terminal_restarted')
        self.assertEqual(proof['expires_at'],self.body['expires_at'])
        self.assertNotIn(self.token,json.dumps(status));self.assertNotIn(self.body['token_sha256'],json.dumps(status))

    def test_http_management_binding_signals_and_callback_readiness(self):
        def auth(x_api_key:str=Header(default='')):
            if x_api_key!='fixture-management':raise HTTPException(401)
        app=FastAPI();register_stream_routes(app,stream_manager=self.manager,require_api_key=auth,
            require_upload_access=auth,health_payload=lambda:{})
        client=TestClient(app);base=f'/api/v1/stream-sessions/{self.sid}';headers={'X-API-Key':'fixture-management'}
        self.assertEqual(client.post(base+'/capture-lease',json=self.body).status_code,401)
        self.assertEqual(client.post(base+'/capture-lease',json=self.body,headers=headers).status_code,503)
        app.state.capture_notifications=object()
        self.assertEqual(client.post(base+'/capture-lease',json=self.body,headers=headers).status_code,200)
        body={'lease_id':'lease_1','sequence':1}
        self.assertEqual(client.post(base+'/capture-signals',json=body,headers=headers).status_code,401)
        self.assertEqual(client.post(base+'/capture-signals',json=body,headers={'X-Upload-Token':self.token}).status_code,200)
        self.assertEqual(client.post(base+'/capture-interrupt',json={'lease_id':'lease_1','reason':'terminal_restarted'}).status_code,401)
        self.assertEqual(client.post(base+'/capture-interrupt',json={'lease_id':'lease_1','reason':'terminal_restarted'},headers=headers).status_code,200)
