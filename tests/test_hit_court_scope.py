import unittest
from badminton_analysis.analysis.fixed_camera_match import FixedCameraMatchPipeline
from badminton_analysis.analysis.court_contact_gate import inside_quad
from badminton_analysis.analysis.offline_shot_reconstruction import build_shot_events

QUAD = [[0, 0], [600, 0], [600, 1200], [0, 1200]]


def person(identity, foot, hand):
    return {'track_id': identity, 'status': 'detected', 'confidence': .9,
            'image_xy': foot, 'court_xy_m': [2,2],
            'location_evidence': {'confidence': .9, 'is_current_measurement': True,
                                  'bbox_xyxy': [80,30,200,330], 'hands_image': {'right': hand}}}


class HitCourtScopeTests(unittest.TestCase):
    def test_outside_person_cannot_trigger_with_hand_and_ball_inside_image_quad(self):
        pipeline=FixedCameraMatchPipeline(QUAD,fps=10)
        ball={'status':'approximate','confidence':.9,'image_xy':[150,160]}
        outside=person('outside',[-10,330],[150,160])
        self.assertEqual([],pipeline._detect_hit_events([outside],ball,10))
        inside=person('inside',[150,330],[180,160])
        self.assertEqual([],pipeline._detect_hit_events([outside,inside],ball,11))
        self.assertEqual('inside',pipeline._detect_hit_events([inside],ball,12)[0]['hitter_track_id'])

    def test_boundary_person_may_hit_with_racket_outside(self):
        pipeline=FixedCameraMatchPipeline(QUAD,fps=10)
        t=person('a',[0,330],[-15,160])
        ball={'status':'approximate','confidence':.9,'image_xy':[-15,160]}
        self.assertEqual(1,len(pipeline._detect_hit_events([t],ball,10)))

    def test_predicted_or_missing_foot_is_not_a_current_in_court_person(self):
        pipeline=FixedCameraMatchPipeline(QUAD,fps=10)
        ball={'status':'approximate','confidence':.9,'image_xy':[150,160]}
        for value in [None,[-1,330]]:
            self.assertEqual([],pipeline._detect_hit_events([person('a',value,[150,160])],ball,10))

    def test_perspective_quad_does_not_expand_to_bounding_rectangle(self):
        quad=[[100,0],[200,0],[300,300],[0,300]]
        self.assertFalse(inside_quad([10,10],quad))
        self.assertTrue(inside_quad([150,100],quad))

    def test_offline_generation_cannot_reintroduce_outside_contacts(self):
        raw={'frame':10,'time_sec':1,'spatial':{'tracks':[person('a',[-10,330],[150,160])],
             'hit_events':[{'status':'candidate','hitter_track_id':'a','confidence':.4}]}}
        shuttle=[{'frame':10,'time_sec':1,'status':'detected','image_xy':[150,160]}]
        self.assertEqual([],build_shot_events([raw],shuttle,court_polygon=QUAD))


if __name__=='__main__':unittest.main()
