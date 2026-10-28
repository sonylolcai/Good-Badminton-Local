"""Verify terminal evidence commit and notification retry without cloud access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from uuid import uuid4

from api.analysis_archive import upload_analysis_archive
from api.media_identity import UUID_FIELDS, media_prefix
from api.media_storage import DirectoryObjectStore
from api.result_notifications import ResultNotifications
from good_badminton_contracts.gpu_notifications import encode


class AnalysisArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        identity = {key: str(uuid4()) for key in UUID_FIELDS}
        identity.update(schema_version='media-identity.v1', environment='test', venue_slug='court',
                        capture_date='2026-10-24', daily_sequence=1)
        self.session = {'analysis_session_id': 'ssn_result', 'status': 'finalized',
                        'configuration': {'sport_id': 'badminton'}, 'media_identity': identity}
        self.manager = Mock()
        self.manager._session_dir.return_value = self.root
        self.manager._list_sessions.return_value = [self.session]
        self.manager._events_path.return_value = self.root / 'events.jsonl'
        self.manager.trace_path.return_value = self.root / 'trace.json'
        (self.root / 'events.jsonl').write_text('{"evidence":"unknown"}\n')
        (self.root / 'trace.json').write_text(json.dumps({'terminal_status': 'finalized', 'finished_at': '2026-10-24T00:00:00Z'}))
        self.storage = DirectoryObjectStore(self.root / 'oss')

    def test_terminal_evidence_commits_immutable_manifest_last(self):
        descriptor = upload_analysis_archive(self.manager, self.storage, self.session)
        manifest = descriptor['manifest']
        self.assertEqual(manifest['media_identity'], self.session['media_identity'])
        self.assertTrue(descriptor['manifest_object_key'].startswith(media_prefix(self.session['media_identity']) + '/analysis/'))
        head = self.storage.head(descriptor['manifest_object_key'])
        self.assertEqual(head['sha256'], hashlib.sha256(encode(manifest)).hexdigest())
        self.assertEqual(head['size_bytes'], descriptor['content_length_bytes'])
        self.assertEqual({entry['role'] for entry in manifest['objects']}, {'events', 'trace'})
        self.assertEqual(upload_analysis_archive(self.manager, self.storage, self.session), descriptor)

    def test_incomplete_trace_cannot_publish_final_manifest(self):
        (self.root / 'trace.json').write_text('{"terminal_status":null}')
        with self.assertRaises(IOError):
            upload_analysis_archive(self.manager, self.storage, self.session)
        self.assertEqual(list((self.root / 'oss').rglob('manifest.json')), [])

    def test_storage_failure_blocks_notice_then_retry_commits_before_delivery(self):
        archive = Mock()
        archive.storage = self.storage
        archive.status.return_value = {'status': 'ready', 'manifest_sha256': 'a' * 64}
        sender = Mock()
        now = [1000]
        box = ResultNotifications(self.manager, archive, url='https://business.example.test/callback',
                                  secret='x' * 32, send=sender, clock=lambda: now[0], archive_results=True)
        original = self.storage.upload
        self.storage.upload = Mock(side_effect=IOError('offline'))
        box.run_once()
        sender.assert_not_called()
        self.assertEqual(box.pending_count(), 1)
        self.storage.upload = original
        now[0] += 30
        sender.side_effect = IOError('business offline')
        box.run_once()
        descriptor = sender.call_args.args[0]['analysis_archive']
        self.assertEqual(self.storage.head(descriptor['manifest_object_key'])['sha256'], descriptor['manifest_sha256'])
        self.assertEqual(box.pending_count(), 1)
        now[0] += 30
        sender.side_effect = None
        box.run_once()
        self.assertEqual(sender.call_args.args[0]['analysis_archive'], descriptor)
        self.assertEqual(box.pending_count(), 0)
        box.run_once()
        self.assertEqual(sender.call_count, 2)
