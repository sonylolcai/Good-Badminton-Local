import unittest
from unittest.mock import patch
from badminton_analysis.analysis.rally_segmentation import segment_rallies,posture_evidence
QUAD=[[0,0],[500,0],[500,500],[0,500]]
def flight(start,n=8):
    return [{'time_sec':round(start+i*.1,6),'image_xy':[40+i*28,100], 'status':'detected','confidence':.5} for i in range(n)]
def bodies(duration=4,reactions=(2.,2.),ready=True,pickup=False):
    return [{'time_sec':i/10,'arm_motion_by_half':list(reactions),'measured_in_court_people':4,
             'ready_by_half':[ready,ready],'pickup_by_half':[pickup,pickup],'posture_known_by_half':[True,True]}
            for i in range(int(duration*10)+1)]
class PrecisionGateTests(unittest.TestCase):
    def run_with_bodies(self,balls,motion,**kwargs):
        duration=kwargs.pop("video_end_sec",4)
        with patch('badminton_analysis.analysis.rally_segmentation.body_activity',return_value=motion):
            return segment_rallies(balls,[],QUAD,video_end_sec=duration,**kwargs)
    def test_presence_and_single_sided_motion_do_not_promote_toss_to_rally(self):
        for reactions in [(0.,0.),(2.,0.),(0.,2.)]:
            r=self.run_with_bodies(flight(.2),bodies(reactions=reactions))
            self.assertEqual([],r['rallies'])
            self.assertTrue(r['uncertain_intervals'])
            self.assertEqual('unverified_activity',r['uncertain_intervals'][0]['classification'])
    def test_short_serve_remains_recoverable_in_uncertain_queue(self):
        r=self.run_with_bodies(flight(.2,4),bodies(reactions=(2.,0.)))
        self.assertEqual([],r['rallies']);self.assertTrue(r['uncertain_intervals'])
        self.assertIn('may_be_between_rally_action_or_single_shot_serve',r['uncertain_intervals'][0]['review_warnings'])
    def test_bilateral_response_supports_rally_without_hitter_identity(self):
        r=self.run_with_bodies(flight(.2),bodies())
        self.assertEqual(1,len(r['rallies']));self.assertEqual([],r['uncertain_intervals'])
        self.assertFalse(r['hitter_analysis_performed'])
    def test_jitter_path_length_does_not_replace_actual_spatial_span(self):
        balls=flight(.2)
        for i,b in enumerate(balls):b['image_xy']=[100+(25 if i%2 else 0),100]
        r=self.run_with_bodies(balls,bodies());self.assertEqual([],r['rallies']);self.assertEqual([],r['uncertain_intervals'])
    def test_disconnected_links_and_old_window_do_not_open_a_flight(self):
        balls=flight(.2,2)+flight(1.2,2)
        r=self.run_with_bodies(balls,bodies());self.assertEqual([],r['rallies']);self.assertEqual([],r['uncertain_intervals'])
    def test_walking_or_pickup_cannot_bridge_an_unobserved_gap(self):
        balls=flight(.2)+flight(3.5)
        for motion in [bodies(duration=5,reactions=(2.,0.)),bodies(duration=5,ready=False),bodies(duration=5,pickup=True)]:
            # Motion is absent in the middle, then a one-sided burst occurs;
            # old v1 any-arm criterion would allow merging both activity groups.
            for row in motion:
                if 1.0<=row['time_sec']<=3.4:row['arm_motion_by_half']=[2.,0.]
            r=self.run_with_bodies(balls,motion,video_end_sec=5)
            all_intervals=r['rallies']+r['uncertain_intervals']
            self.assertEqual(2,len(all_intervals))
            self.assertFalse(any(x.get('bridged_unknown_ball_gaps') for x in all_intervals))
    def test_pickup_and_missing_pose_are_separate(self):
        kp=[[100.,20.] for _ in range(17)]
        kp[5]=[80,90];kp[6]=[120,90];kp[11]=[90,100];kp[12]=[110,100];kp[0]=[100,110]
        r=posture_evidence(kp,[],200);self.assertTrue(r['pickup']);self.assertFalse(r['ready'])
        self.assertIsNone(posture_evidence([],[],200)['pickup'])
    def test_current_unassigned_half_motion_does_not_require_roster_or_hitter_identity(self):
        from badminton_analysis.analysis.rally_segmentation import body_activity
        rows=[]
        for i in range(40):
            players={}
            for side,y in [('upper',3),('lower',10)]:
                players[side]={'image':[150,150],'court':[2,y],'hands':{'right':[100+(70 if i%2 else 0),60]},
                              'position_evidence':{'status':'detected','confidence':.9,'person_confidence':.9,'bbox_xyxy':[70,20,230,220]}}
            rows.append({'time_sec':i/10,'players':players,'spatial':{'tracks':[]}})
        motion=body_activity(rows,QUAD)
        self.assertTrue(all(v>1.3 for v in motion[2]['arm_motion_by_half']))
        r=segment_rallies(flight(.2),rows,QUAD,video_end_sec=4)
        self.assertEqual(1,len(r['rallies']));self.assertFalse(r['hitter_analysis_performed'])
        self.assertIn('current_unassigned_person_motion_used_without_hitter_identity',r['rallies'][0]['review_warnings'])
        for row in rows:
            for player in row['players'].values():player['position_evidence']['status']='predicted'
        r=segment_rallies(flight(.2),rows,QUAD,video_end_sec=4)
        self.assertEqual([],r['rallies']);self.assertEqual([],r['uncertain_intervals'])

if __name__=='__main__':unittest.main()
