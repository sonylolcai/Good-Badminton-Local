import copy
import unittest
from badminton_analysis.analysis.rally_activity import derive_rally_activity, find_settle_terminals

QUAD = [[0, 0], [500, 0], [500, 500], [0, 500]]

def point(t, x, y=100, status='detected'):
    return {'time_sec': t, 'image_xy': [x, y], 'confidence': .5, 'status': status}

def event(i, t, actor='a', y=2, **extra):
    return {'event_id': f's{i}', 'hit_time_sec': t, 'actor_track_id': actor,
            'actor_xy_m': [2, y], **extra}

def flight(start):
    return [point(start + i * .1, 50 + i * 30) for i in range(6)]

class RallyActivityTests(unittest.TestCase):
    def test_end_requires_rearming_not_the_next_swing_or_same_side_contact(self):
        events = [event(1,1),event(2,1.5,'b',10),event(3,3),event(4,3.5,'c',3),
                  event(5,5),event(6,5.5,'b',10)]
        before = copy.deepcopy(events)
        result = derive_rally_activity(events, flight(1)+flight(3)+flight(5),
                    [{'time_sec':2,'source':'human_review'}], video_end_sec=8)
        self.assertEqual([(1,2),(5,8)],[(r['start_time_sec'],r['end_time_sec']) for r in result['rallies']])
        self.assertEqual('awaiting_restart',result['assignments']['s3']['rally_phase'])
        self.assertEqual('awaiting_restart',result['assignments']['s4']['rally_phase'])
        self.assertNotEqual(result['assignments']['s2']['rally_id'],result['assignments']['s5']['rally_id'])
        self.assertEqual(before,events)
        self.assertFalse(result['score_inference'])

    def test_opponents_without_observed_flight_cannot_rearm_and_unknowns_are_not_skipped(self):
        events=[event(1,1),event(2,1.5,'b',10),event(3,4),event(4,4.2,None),event(5,4.5,'b',10)]
        for tracks in [flight(1), flight(1)+flight(4)]:
            r=derive_rally_activity(events,tracks,[{'time_sec':2,'source':'human_review'}],video_end_sec=7)
            self.assertEqual(3,r['waiting_restart_count'])
        events.pop(3)
        tracks=flight(1)+[dict(p,status='reconstructed') for p in flight(4)]
        self.assertEqual(2,derive_rally_activity(events,tracks,[{'time_sec':2,'source':'human_review'}],video_end_sec=7)['waiting_restart_count'])

    def test_gap_direction_reversal_alone_does_not_end_an_active_rally(self):
        events=[event(1,1),event(2,1.5,'b',10),event(3,4)]
        r=derive_rally_activity(events,flight(1),[{'time_sec':2,'source':'direction_reversal_after_gap','confidence':.85}],video_end_sec=6)
        self.assertEqual(1,len(r['rallies']))
        self.assertEqual('in_rally',r['assignments']['s3']['rally_phase'])
        self.assertEqual([],r['terminals'])

    def test_descent_compact_cluster_is_a_candidate_not_score_and_near_hand_rejects_it(self):
        tracks=[point(4.25,200,200),point(4.35,200,250),point(4.45,200,310),
                point(4.85,200,400),point(4.95,200,398),point(5.05,200,410)]
        # A short outlier interruption is allowed; unsupported long gaps are not.
        tracks.insert(3,point(4.55,200,370))
        r=find_settle_terminals(tracks,[],QUAD)
        self.assertEqual(1,len(r));self.assertEqual(4.85,r[0]['time_sec'])
        self.assertFalse(r[0]['eligible_for_scoring'])
        people=[{'time_sec':4.85,'spatial':{'tracks':[{'status':'detected','location_evidence':{'hands_image':{'right':[200,405]}}}]}}]
        self.assertEqual([],find_settle_terminals(tracks,people,QUAD))
        self.assertEqual([],find_settle_terminals([dict(p,status='reconstructed') for p in tracks],[],QUAD))
        self.assertEqual([],find_settle_terminals(tracks,[],[[600,0],[800,0],[800,500],[600,500]]))

    def test_human_end_interval_is_separate_from_automatic_detection(self):
        terminals=[{'time_sec':4.85,'source':'descent_then_compact_settle','confidence':.72},
                   {'time_sec':5,'time_interval_sec':[4.55,5],'source':'user_reported_interval'}]
        events=[event(1,1),event(2,1.5,'b',10),event(3,6.86)]
        r=derive_rally_activity(events,flight(1),terminals,video_end_sec=9)
        self.assertEqual([5],[t['time_sec'] for t in r['terminals']])
        self.assertEqual([4.55,5],r['rallies'][0]['terminal']['time_interval_sec'])
        self.assertEqual('awaiting_restart',r['assignments']['s3']['rally_phase'])
        auto=derive_rally_activity(events,flight(1),terminals[:1],video_end_sec=9)
        self.assertEqual(4.85,auto['rallies'][0]['end_time_sec'])

    def test_explicit_excluded_contact_does_not_start_a_rally(self):
        events=[event(1,1,excluded=True),event(2,1.5,'b',10)]
        r=derive_rally_activity(events,flight(1),[],video_end_sec=3)
        self.assertEqual([],r['rallies'])
        self.assertEqual('human_excluded',r['assignments']['s1']['rally_phase'])

    def test_calibrated_offline_output_applies_gate_and_keeps_all_original_event_ids(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from badminton_analysis.analysis import offline_shot_reconstruction as offline
        events=[event(1,1),event(2,1.5,'b',10),event(3,3),event(4,3.5,'c',3),event(5,5),event(6,5.5,'b',10)]
        tracks=flight(1)+flight(3)+flight(5)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'detections.jsonl').write_text(json.dumps({'frame':1,'time_sec':1})+'\n')
            (root/'metadata.json').write_text(json.dumps({'court':{'corners':QUAD},'video':{'width':1920,'height':1080}}))
            with patch.object(offline,'reconstruct_shuttle_track',return_value=tracks), \
                 patch.object(offline,'build_shot_events',return_value=events), \
                 patch.object(offline,'infer_missing_shuttle_events',side_effect=lambda rows,events: events), \
                 patch.object(offline,'refresh_event_evidence',side_effect=lambda events,tracks: events), \
                 patch.object(offline,'build_rallies',return_value=[]), \
                 patch.object(offline,'detect_rally_terminal_candidates',return_value=[{'time_sec':2,'source':'descent_then_compact_settle','confidence':.72}]):
                output=offline.generate_offline_artifacts(root/'detections.jsonl',fps=30)
            effective=offline.load_jsonl(output['events_path'])
            self.assertEqual([f's{i}' for i in range(1,7)],[e['event_id'] for e in effective])
            self.assertEqual('awaiting_restart',effective[2]['rally_phase'])
            self.assertIsNone(effective[2]['rally_id'])
            self.assertEqual(2,output['rally_count'])
            self.assertTrue((root/'derived/rally_activity_v1.json').is_file())




if __name__ == "__main__":unittest.main()
