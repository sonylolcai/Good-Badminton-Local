"""Receiving-route contracts for the independently versioned GPU CPU bundle."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.visual_result import (
    compute_visual_result, implementation, input_identity, result_path,
    validate_projection_request,
)
from good_badminton_visual.metrics.receiving_route_estimate import (
    estimate_receiving_routes, receiving_route_estimate_profile,
)
from tests.test_visual_result import fixtures


def observations(slot=True, source='half_court_slot_continuity', same_side=False, hz=10):
    """Two known wrist accelerations at .3s and 1.1s; not video ground truth."""
    rows = []
    for i in range(6 * hz):
        tracks = []
        for tid, trigger, end, xy in [
            ('track_001', round(.3 * hz), 'end_a', [1, 3]),
            ('track_002', round(1.1 * hz), 'end_a' if same_side else 'end_b', [5, 4 if same_side else 10]),
        ]:
            offset = 0 if i < trigger else 1 if i == trigger else 3
            points = [[120, 120] for _ in range(17)]
            for index, point in {5: [100, 100], 6: [140, 100], 11: [105, 150],
                                 12: [135, 150], 9: [100, 120],
                                 10: [120 + 50 * offset, 120]}.items():
                points[index] = point
            track = {
                'track_id': tid, 'status': 'detected', 'court_end': end,
                'court_xy_m': xy, 'confidence': .9,
                'location_evidence': {'confidence': .9},
                'association': {'source': source if slot else 'bytetrack', 'identity_confidence': .95},
                'pose': {'format': 'coco17_image_v1', 'is_current_measurement': True,
                         'keypoints_image': points, 'keypoint_scores': [.9] * 17},
            }
            if slot:
                track.update(identity_scope='half_court_slots.v1', occupancy_segment=1,
                             slot_court_end=end)
            tracks.append(track)
        rows.append({'frame': i, 'time_sec': i / hz, 'spatial': {'tracks': tracks}})
    return rows


class VisualReceivingRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'detections.jsonl'

    def estimate(self, rows):
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return estimate_receiving_routes(self.path)

    def test_each_slot_source_produces_an_estimate_with_occupant_provenance(self):
        for source in ('half_court_slot_opening', 'half_court_slot_continuity',
                       'half_court_slot_replacement'):
            with self.subTest(source=source):
                result = self.estimate(observations(source=source))
                self.assertEqual(result['pose_peak_count'], 2)
                self.assertEqual(len(result['events']), 1)
                event = result['events'][0]
                self.assertEqual((event['actor_track_id'], event['receiver_track_id']),
                                 ('track_001', 'track_002'))
                self.assertEqual(event['receiver_zone'], 5)
                self.assertEqual(event['actor_identity_scope'], 'half_court_slots.v1')
                self.assertEqual(event['receiver_identity_scope'], 'half_court_slots.v1')
                self.assertEqual(event['actor_occupancy_segment'], 1)
                self.assertEqual(event['receiver_occupancy_segment'], 1)
                profile = receiving_route_estimate_profile('track_001', result)
                self.assertEqual(profile['status'], 'estimated')
                self.assertEqual(profile['sample_count'], 1)
                self.assertEqual(profile['normalized_value'], 0)
                self.assertFalse(profile['competitive_rating'])

    def test_inconsistent_slot_identity_is_excluded(self):
        for field, value in [('identity_scope', None), ('identity_scope', 'individual_tracks.v1'),
                             ('occupancy_segment', None), ('occupancy_segment', 0),
                             ('occupancy_segment', True), ('occupancy_segment', 1.5),
                             ('slot_court_end', 'end_b')]:
            with self.subTest(field=field, value=value):
                rows = observations()
                for row in rows:
                    row['spatial']['tracks'][0][field] = value
                self.assertEqual(self.estimate(rows)['events'], [])

    def test_legacy_source_cannot_bypass_half_slot_scope_checks(self):
        rows = observations()
        for row in rows:
            row['spatial']['tracks'][0]['association']['source'] = 'bytetrack'
        self.assertEqual(self.estimate(rows)['events'], [])

    def test_occupant_boundary_cannot_create_a_wrist_peak(self):
        rows = observations()
        for i, row in enumerate(rows):
            if i >= 3:
                track = row['spatial']['tracks'][0]
                track['occupancy_segment'] = 2
                track['association']['source'] = 'half_court_slot_replacement'
        result = self.estimate(rows)
        self.assertEqual(result['pose_peak_count'], 1)
        self.assertEqual(result['events'], [])

    def test_either_slot_changing_occupant_between_peaks_prevents_pairing(self):
        for side in (0, 1):
            with self.subTest(side=side):
                rows = observations()
                for row in rows[7:]:
                    track = row['spatial']['tracks'][side]
                    track['occupancy_segment'] = 2
                    track['association']['source'] = 'half_court_slot_replacement'
                result = self.estimate(rows)
                self.assertEqual(result['pose_peak_count'], 2)
                self.assertEqual(result['events'], [])

    def test_unknown_tracking_or_unobserved_gap_cannot_prove_pair_continuity(self):
        for kind in ('missing', 'predicted', 'low_confidence', 'gap', 'conflict'):
            with self.subTest(kind=kind):
                rows = observations()
                track = rows[7]['spatial']['tracks'][0]
                if kind in ('missing', 'predicted'):
                    track['status'] = kind
                elif kind == 'low_confidence':
                    track['confidence'] = .49
                elif kind == 'gap':
                    rows[7]['spatial']['tracks'].pop(0)
                else:
                    duplicate = copy.deepcopy(rows[7])
                    duplicate['spatial']['tracks'][0]['occupancy_segment'] = 2
                    rows.append(duplicate)
                self.assertEqual(self.estimate(rows)['events'], [])

    def test_pose_absence_alone_is_not_an_occupant_change(self):
        rows = observations()
        rows[7]['spatial']['tracks'][0].pop('pose')
        self.assertEqual(len(self.estimate(rows)['events']), 1)
        # A replacement outside the pair interval does not invalidate the pair.
        rows = observations()
        for row in rows[15:]:
            row['spatial']['tracks'][0]['occupancy_segment'] = 2
        self.assertEqual(len(self.estimate(rows)['events']), 1)

    def test_high_frequency_absent_slot_empty_frame_and_missing_spatial_are_unknown(self):
        for hz in (20, 40, 100):
            self.assertEqual(len(self.estimate(observations(hz=hz))['events']), 1)
            for kind in ('actor', 'receiver', 'empty', 'spatial'):
                with self.subTest(hz=hz, kind=kind):
                    rows = observations(hz=hz)
                    missing = rows[round(.65 * hz)]
                    if kind in ('actor', 'receiver'):
                        missing['spatial']['tracks'].pop(0 if kind == 'actor' else 1)
                    elif kind == 'empty':
                        missing['spatial']['tracks'] = []
                    else:
                        missing.pop('spatial')
                    self.assertEqual(self.estimate(rows)['events'], [])
            rows = observations(hz=hz)
            rows[round(.65 * hz)]['spatial']['tracks'][0].pop('pose')
            self.assertEqual(len(self.estimate(rows)['events']), 1)

    def test_high_frequency_missing_slot_cannot_create_a_wrist_peak(self):
        for hz in (20, 40, 100):
            with self.subTest(hz=hz):
                rows = observations(hz=hz)
                rows[round(.3 * hz) - 1]['spatial']['tracks'].pop(0)
                self.assertEqual(self.estimate(rows)['events'], [])
        # At 100 Hz, .25s precedes the real pulse's .29-.31s window.
        # A resolved absence outside both the pulse and pair must not erase it.
        rows = observations(hz=100)
        rows[25]['spatial']['tracks'].pop(0)
        self.assertEqual(len(self.estimate(rows)['events']), 1)

    def test_predicted_stale_or_low_confidence_pose_never_fills_routes(self):
        for kind in ('predicted', 'stale', 'body', 'location', 'identity', 'wrists'):
            with self.subTest(kind=kind):
                rows = observations()
                for row in rows:
                    track = row['spatial']['tracks'][0]
                    if kind == 'predicted':
                        track['status'] = 'predicted'
                    elif kind == 'stale':
                        track['pose']['is_current_measurement'] = False
                    elif kind == 'body':
                        track['confidence'] = .49
                    elif kind == 'location':
                        track['location_evidence']['confidence'] = .49
                    elif kind == 'identity':
                        track['association']['identity_confidence'] = .69
                    else:
                        for wrist in (9, 10):
                            track['pose']['keypoint_scores'][wrist] = .49
                self.assertEqual(self.estimate(rows)['events'], [])

    def test_legacy_pair_counts_times_and_distribution_are_preserved(self):
        result = self.estimate(observations(slot=False))
        self.assertEqual(result['pose_peak_count'], 2)
        event, = result['events']
        self.assertEqual(event['source_time_sec'], .3)
        self.assertEqual(event['receiver_time_sec'], 1.1)
        profile = receiving_route_estimate_profile('track_001', result)
        self.assertEqual(profile['distribution'], {'5': 1})
        self.assertEqual(profile['sample_count'], 1)
        self.assertEqual(profile['normalized_value'], 0)
        self.assertEqual(profile['source'], 'pose_wrist_pair_estimate.v1')

    def test_duplicates_do_not_increase_counts_and_same_side_is_not_paired(self):
        rows = observations()
        self.assertEqual(len(self.estimate(rows + copy.deepcopy(rows))['events']), 1)
        result = self.estimate(observations(same_side=True))
        self.assertEqual(result['events'], [])
        profile = receiving_route_estimate_profile('track_001', result)
        self.assertEqual(profile['status'], 'not_collected')
        self.assertIsNone(profile['normalized_value'])

    def test_slot_estimate_reaches_compact_report_without_model_or_network_io(self):
        session_root = self.root / 'session'
        session, _, terminal, _ = fixtures(session_root)
        session['configuration']['roster_identity_policy'] = 'half_court_slots.v1'
        events = []
        for row in observations():
            for track in row['spatial']['tracks']:
                events.append({
                    'schema_version': 'stream-session.v1',
                    'analysis_session_id': session['analysis_session_id'],
                    'event_id': f"{track['track_id']}_{row['frame']}",
                    'event_type': 'person_observation', 'segment_index': 0,
                    'source_time_sec': row['time_sec'], 'data': {'track': track},
                })
        terminal['progress']['processed_source_time_sec'] = 6
        (session_root / 'manifest.json').write_text(json.dumps(session))
        (session_root / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        original = (session_root / 'events.jsonl').read_bytes()
        identity = input_identity(session_root, session)
        with patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            self.assertEqual(compute_visual_result(str(session_root), identity, terminal), {'ready': True})
        result = json.loads(result_path(session_root, identity).read_text())
        actor, receiver = result['metrics']['players']
        self.assertEqual(actor['shot_route_profile']['status'], 'estimated')
        self.assertEqual(actor['shot_route_profile']['sample_count'], 1)
        self.assertEqual(actor['shot_route_profile']['normalized_value'], 0)
        self.assertEqual(actor['identity_scope'], 'half_court_slots.v1')
        self.assertEqual(receiver['shot_route_profile']['sample_count'], 0)
        self.assertIsNone(receiver['shot_route_profile']['normalized_value'])
        self.assertEqual(result['completeness']['status'], 'partial')
        self.assertEqual(set(result['completeness']['missing_stages']),
                         {'rally_segmentation', 'hit_moments', 'hitter_attribution'})
        self.assertEqual((session_root / 'events.jsonl').read_bytes(), original)
        self.assertEqual(result['source']['events_sha256'], hashlib.sha256(original).hexdigest())

    def test_new_implementation_rejects_requests_pinned_to_old_bundle(self):
        baseline = '57b2fb9752214a940a089976e8a0dbf620223e5a8c0bfddc3e00112c6c4d2740'
        self.assertNotEqual(implementation(), baseline)
        with patch.dict('os.environ', {'GOOD_BADMINTON_VISUAL_RESULT_ENABLED': '1'}):
            with self.assertRaisesRegex(ValueError, 'visual_result_implementation_mismatch'):
                validate_projection_request({'visual_result': {'implementation_sha256': baseline}})


if __name__ == '__main__':
    unittest.main()
