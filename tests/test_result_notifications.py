import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
from api.result_notifications import ResultNotifications

class ResultNotificationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.session={'analysis_session_id':'ssn_notice','status':'finalized','configuration':{'sport_id':'badminton'},
                      'media_identity':{'edge_session_id':'business-session'}}
        self.manager=Mock();self.manager._list_sessions.return_value=[self.session]
        self.manager._session_dir.return_value=self.root
        self.archive=Mock();self.archive.status.return_value={'status':'ready','revision':'a'*64,
            'manifest_object_key':'testVideo/folder/manifest.json','manifest_sha256':'b'*64,'total_source_frames':23}
        self.send=Mock();self.now=1000

    def outbox(self):
        return ResultNotifications(self.manager,self.archive,url='https://business.example.test/api/v1/gpu/result-notifications',
            secret='test-result-notification-secret-32bytes',send=self.send,clock=lambda:self.now)

    def test_blocked_delivery_recovers_after_configuration_repair_without_restart(self):
        from urllib.error import HTTPError
        self.send.side_effect = HTTPError('https://business.test', 401, 'unauthorized', {}, None)
        box = self.outbox(); box.run_once(); box.run_once()
        self.assertEqual(self.send.call_count, 1)
        self.assertEqual(box.pending_count(), 1)
        self.send.side_effect = None; self.now += 901
        self.outbox().run_once()
        self.assertEqual(self.send.call_count, 2)
        self.assertEqual(box.pending_count(), 0)
        self.assertEqual(box.blocked_count(), 0)

    def test_acknowledged_result_is_not_resent_after_restart(self):
        box=self.outbox();self.assertEqual(box.pending_count(),1)
        box.run_once();self.assertEqual(box.pending_count(),0)
        self.outbox().run_once();self.send.assert_called_once()
        self.assertEqual(self.send.call_args.args[0]['media_identity'],self.session['media_identity'])

    def test_network_failure_retries_from_durable_outbox_without_losing_result(self):
        self.send.side_effect=OSError('network unavailable')
        box=self.outbox();box.run_once();self.assertEqual(box.pending_count(),1)
        self.assertTrue((self.root/'result-notification.json').is_file())
        self.send.side_effect=None;self.now+=30;self.outbox().run_once()
        self.assertEqual(self.outbox().pending_count(),0)

    def test_analysis_and_upload_both_finish_before_success_notice(self):
        self.archive.status.return_value={'status':'uploading'}
        self.outbox().run_once();self.send.assert_not_called()
        self.archive.status.return_value={'status':'ready'};self.session['status']='draining'
        self.outbox().run_once();self.send.assert_not_called()

    def test_failed_archive_then_ready_emits_distinct_events(self):
        self.archive.status.return_value={'status':'failed','error':'NetworkError'}
        self.outbox().run_once();first=self.send.call_args.args[0]['event_id']
        self.archive.status.return_value={'status':'ready','manifest_sha256':'a'*64}
        self.outbox().run_once();second=self.send.call_args.args[0]['event_id']
        self.assertNotEqual(first,second);self.assertEqual(self.send.call_count,2)

    def test_retrying_archive_is_not_a_terminal_failure(self):
        self.archive.status.return_value = {'status':'retrying'}
        self.outbox().run_once(); self.send.assert_not_called()
        self.archive.status.return_value = {'status':'ready'}
        self.outbox().run_once(); self.send.assert_called_once()

    def test_unsealed_failed_analysis_still_notifies_business(self):
        self.session.update(status='failed', sealed=False)
        self.archive.status.return_value = {'status':'awaiting_seal'}
        self.outbox().run_once()
        self.assertEqual(self.send.call_args.args[0]['media_status'], 'failed')

    def test_permanent_callback_failure_is_bounded_and_can_be_retried(self):
        from urllib.error import HTTPError
        self.send.side_effect = HTTPError('https://callback.test', 401, 'denied', {}, None)
        box = self.outbox(); box.run_once(); self.now += 100; box.run_once()
        self.assertEqual(self.send.call_count, 1)
        self.assertEqual(box.blocked_count(), 1)
        self.assertEqual(box.pending_count(), 1)  # unacknowledged evidence still blocks shutdown
        self.send.side_effect = None; box.retry_blocked(); box.run_once()
        self.assertEqual(box.pending_count(), 0)

    def test_callback_cannot_use_http_or_credential_urls(self):
        for url in ('http://business.example.test','https://secret@business.example.test','https://business.example.test/?token=secret'):
            with self.assertRaises(ValueError):ResultNotifications(self.manager,self.archive,url=url,secret='a'*32)

    def test_explicit_cutover_preserves_history_without_replaying_legacy_test_sessions(self):
        self.session['created_at']='2026-10-23T10:00:00Z'
        box=ResultNotifications(self.manager,self.archive,url='https://business.example.test/callback',secret='x'*32,
            send=self.send,not_before='2026-10-24T10:00:00+08:00')
        box.run_once();self.assertEqual(box.pending_count(),0);self.send.assert_not_called()
        self.session['created_at']='2026-10-24T02:00:00Z'
        box.run_once();self.send.assert_called_once();self.assertEqual(box.pending_count(),0)

    def test_cutover_never_hides_an_unknown_creation_time(self):
        box=ResultNotifications(self.manager,self.archive,url='https://business.example.test/callback',secret='x'*32,
            send=self.send,not_before='2026-10-24T02:00:00Z')
        self.assertEqual(box.pending_count(),1)
        box.run_once();self.send.assert_not_called()

if __name__=='__main__':unittest.main()
