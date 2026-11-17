import copy
import unittest
from badminton_analysis.analysis.rally_segmentation import segment_rallies
QUAD=[[0,0],[500,0],[500,500],[0,500]]
def ball(t,x,status='detected'):
    return {'time_sec':round(t,6),'image_xy':[x,100],'status':status,'confidence':.5}
def person_rows(end=4, inside=True):
    return [{'time_sec':i/10,'spatial':{'tracks':[{'track_id':'temporal_body_key','status':'detected',
             'image_xy':[100,200] if inside else [700,200], 'court_xy_m':[2,3],
             'location_evidence':{'confidence':.9,'is_current_measurement':True}}]}} for i in range(int(end*10)+1)]
def flight(start,n=10):return [ball(start+i*.1,40+i*25) for i in range(n)]
class StageOneSegmentationTests(unittest.TestCase):
    def run_stage(self,balls,rows=None,**kw):
        return segment_rallies(balls,person_rows() if rows is None else rows,QUAD,video_end_sec=4,**kw)
    def test_no_hit_events_or_hitter_required_and_input_not_mutated(self):
        rows=person_rows();before=copy.deepcopy(rows)
        r=self.run_stage(flight(.2),rows)
        self.assertTrue(r['uncertain_intervals']);self.assertFalse(r['hitter_analysis_performed']);self.assertFalse(r['contact_analysis_performed'])
        self.assertEqual(before,rows)
        for row in rows:row['spatial']['hit_events']=[{'hitter_track_id':'WRONG'}];row['hitter_review']='WRONG'
        self.assertEqual(r,self.run_stage(flight(.2),rows))
    def test_predicted_ball_and_outside_people_cannot_open_a_rally(self):
        self.assertEqual([],self.run_stage([dict(b,status='reconstructed') for b in flight(.2)])['rallies'])
        self.assertEqual([],self.run_stage(flight(.2),person_rows(inside=False))['rallies'])
        self.assertEqual([],self.run_stage(flight(.2),person_rows(inside=False))['uncertain_intervals'])
    def test_missing_ball_has_explicit_full_video_coverage(self):
        r=self.run_stage([])
        self.assertEqual([],r['rallies']);self.assertEqual([{'start_time_sec':0.,'end_time_sec':4,'status':'between_rallies_or_unresolved'}],r['non_rally_or_unresolved_intervals'])
    def test_manual_end_and_restart_cannot_reuse_pre_end_window(self):
        r=self.run_stage(flight(.2),manual_terminals=[{'time_sec':1.05,'source':'human_review'}])
        self.assertEqual(1,len(r['uncertain_intervals']));self.assertEqual(1.05,r['uncertain_intervals'][0]['end_time_sec'])
        self.assertEqual('human_review',r['uncertain_intervals'][0]['end_reason'])
    def test_later_flight_can_rearm_without_opponent_hitter(self):
        r=self.run_stage(flight(.2)+flight(2.5))
        self.assertEqual(2,len(r['uncertain_intervals']))
        self.assertEqual('needs_review',r['uncertain_intervals'][1]['status'])
    def test_timeline_is_nonoverlapping_complete_and_zero_not_missing(self):
        r=self.run_stage(flight(.2)+flight(2.5))
        intervals=sorted(r['rallies']+r['uncertain_intervals']+r['non_rally_or_unresolved_intervals'],key=lambda i:i['start_time_sec'])
        self.assertEqual(0,intervals[0]['start_time_sec'])
        self.assertEqual(4,intervals[-1]['end_time_sec'])
        self.assertAlmostEqual(4,sum(i['end_time_sec']-i['start_time_sec'] for i in intervals),5)
        for a,b in zip(intervals,intervals[1:]):self.assertAlmostEqual(a['end_time_sec'],b['start_time_sec'],5)
    def test_invalid_calibration_and_duration_fail(self):
        with self.assertRaises(ValueError):segment_rallies([],[],[],video_end_sec=4)
        with self.assertRaises(ValueError):segment_rallies([],[],QUAD,video_end_sec=float('nan'))
if __name__=='__main__':unittest.main()
