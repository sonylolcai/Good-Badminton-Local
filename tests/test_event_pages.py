import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from api.event_pages import EventPageReader


class EventPageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'events.jsonl'
        self.reader = EventPageReader()
        self.write([('late', 2), ('early-b', 1), ('early-a', 1)])

    def write(self, entries, path=None):
        (path or self.path).write_text(''.join(json.dumps({'event_id': name, 'source_time_sec': t}) + '\n'
                                             for name, t in entries))

    def test_pages_keep_contract_order_and_unknown_cursor_returns_empty(self):
        first, cursor = self.reader.page(self.path, revision=1, limit=2)
        second, last = self.reader.page(self.path, revision=1, cursor=cursor)
        self.assertEqual([e['event_id'] for e in first + second], ['early-a', 'early-b', 'late'])
        self.assertEqual(self.reader.page(self.path, revision=1, cursor=last), ([], None))
        self.assertEqual(self.reader.page(self.path, revision=1, cursor='unknown'), ([], None))

    def test_next_page_decodes_only_the_requested_rows(self):
        self.reader.page(self.path, revision=1, limit=1)
        with patch('api.event_pages.json.loads', wraps=json.loads) as decode:
            page, _ = self.reader.page(self.path, revision=1, cursor='early-a', limit=1)
        self.assertEqual(page[0]['event_id'], 'early-b'); self.assertEqual(decode.call_count, 1)

    def test_append_rebuilds_index_and_does_not_lose_new_events(self):
        self.reader.page(self.path, revision=1)
        with self.path.open('a') as f:f.write(json.dumps({'event_id': 'last', 'source_time_sec': 3}) + '\n')
        page, _ = self.reader.page(self.path, revision=1)
        self.assertEqual([e['event_id'] for e in page], ['early-a', 'early-b', 'late', 'last'])

    def test_repair_replacement_and_revision_invalidate_cached_offsets(self):
        self.reader.page(self.path, revision=1)
        replacement=self.path.with_suffix('.new'); self.write([('repair', .5)], replacement)
        replacement.replace(self.path)
        page, _ = self.reader.page(self.path, revision=2)
        self.assertEqual(page, [{'event_id':'repair','source_time_sec':.5}])

    def test_missing_malformed_and_cross_session_files_do_not_reuse_old_events(self):
        self.reader.page(self.path, revision=1)
        other=self.path.with_name('other.jsonl')
        self.assertEqual(self.reader.page(other, revision=1), ([], None))
        other.write_text('malformed\n\n' + json.dumps({'event_id':'other','source_time_sec':0}) + '\n')
        self.assertEqual(self.reader.page(other, revision=1)[0][0]['event_id'], 'other')
        self.assertEqual(self.reader.page(self.path, revision=1)[0][0]['event_id'], 'early-a')

    def test_response_mutation_cannot_change_later_pages(self):
        page, _ = self.reader.page(self.path, revision=1)
        page[0]['event_id'] = 'caller-modified'
        self.assertEqual(self.reader.page(self.path, revision=1)[0][0]['event_id'], 'early-a')
