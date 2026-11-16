"""Known generated identities exercise crossings, occlusion and gallery safety.

The oracle is the renderer's actor ID, independent of tracker keys, court side
or the association algorithm. These tests do not estimate real-video accuracy.
"""
from copy import deepcopy
import unittest

import cv2
import numpy as np

from badminton_analysis.tracking.anchored_roster import prepare_anchor_observations, appearance_distance
from tests import test_anchored_roster as fixtures
from tests.test_anchored_roster import context, scene


def render(actors, *, twins=False):
    frame=np.zeros((650,650,3),np.uint8);observations=[];oracle={}
    colours=[(35,35,210),(35,210,35),(210,35,35),(35,210,210)]
    for actor,x,y in actors:
        px,py=50+80*x,70+40*y;box=[round(px-35),round(py-140),round(px+35),round(py)]
        x1,y1,x2,y2=box
        frame[y1:y2,x1:x2]=colours[0 if twins and actor<2 else actor]
        cv2.rectangle(frame,(x1+2,y1+2),(x2-2,y2-2),(255,255,255),2)
        points=[[0.,0.] for _ in range(17)];scores=[0.]*17
        for joint,rx,ry in ((5,.3,.2),(6,.7,.2),(11,.35,.5),(12,.65,.5),(15,.35,.92),(16,.65,.92)):
            points[joint]=[x1+(x2-x1)*rx,y1+(y2-y1)*ry];scores[joint]=.95
        observations.append(dict(court_xy=[x,y],image_xy=[px,py],confidence=.95,location_confidence=.95,
            bbox_xyxy=box,keypoints_image=points,keypoint_scores=scores,
            # Deliberately misleading keys cannot supply the identity oracle.
            association_key=f'key-{3-actor}'))
        oracle[tuple(box)]=actor
    return frame,observations,oracle


class IdentityRecoveryTests(unittest.TestCase):
    def tracker(self,count=4):
        return fixtures.AnchoredRosterTests().tracker(count)

    def update(self,tracker,bucket,actors,*,twins=False):
        image,rows,oracle=render(actors,twins=twins)
        snapshot=tracker.update(bucket,prepare_anchor_observations(image,rows,context(bucket)))
        for track in snapshot['tracks']:
            if track['status']=='detected':
                actor=oracle[tuple(track['location_evidence']['bbox_xyxy'])]
                self.assertEqual(track['track_id'],f'track_{actor+1:03d}',f'wrong owner at bucket {bucket}')
        self.assertEqual(len({tuple(t['location_evidence']['bbox_xyxy']) for t in snapshot['tracks'] if t['status']=='detected'}),
                         sum(t['status']=='detected' for t in snapshot['tracks']))
        return snapshot

    def open(self,tracker,*,twins=False):
        for bucket in range(11):
            snapshot=self.update(tracker,bucket,[(0,1,3),(1,5,3),(2,1,11),(3,5,11)],twins=twins)
        self.assertEqual(len(snapshot['tracks']),4)

    def test_rendered_crossing_has_no_wrong_owner_and_recovers_after_separation(self):
        tracker=self.tracker();self.open(tracker)
        unknown=0
        for bucket in range(11,52):
            progress=(bucket-11)/40
            snapshot=self.update(tracker,bucket,[(0,1+4*progress,3),(1,5-4*progress,3),(2,1,11),(3,5,11)])
            unknown+=sum(t['identity_status']=='unknown' for t in snapshot['tracks'][:2])
        self.assertGreater(unknown,0,'overlap must not fabricate two measurements')
        self.assertTrue(all(t['identity_status']=='anchored' for t in snapshot['tracks']))

    def test_two_people_return_with_swapped_positions_after_long_occlusion(self):
        tracker=self.tracker();self.open(tracker)
        for bucket in range(11,41):
            snapshot=self.update(tracker,bucket,[(2,1,11),(3,5,11)])
            self.assertTrue(all(t['identity_status']=='unknown' for t in snapshot['tracks'][:2]))
        distance={tid:m['distance_m'] for tid,m in tracker._tracker.track_metrics.items()}
        for bucket in (41,42):
            snapshot=self.update(tracker,bucket,[(0,5,3),(1,1,3),(2,1,11),(3,5,11)])
            self.assertTrue(all(t['identity_status']=='unknown' for t in snapshot['tracks'][:2]))
        snapshot=self.update(tracker,43,[(0,5,3),(1,1,3),(2,1,11),(3,5,11)])
        self.assertTrue(all(t['identity_status']=='anchored' for t in snapshot['tracks']))
        for track in snapshot['tracks'][:2]:
            self.assertIsNone(track['motion']['current_speed_mps'])
            self.assertEqual(tracker._tracker.track_metrics[track['track_id']]['distance_m'],distance[track['track_id']])

    def test_indistinguishable_twins_after_occlusion_stay_unknown_instead_of_filling_nearest_slot(self):
        tracker=self.tracker();self.open(tracker,twins=True)
        for bucket in range(11,41):self.update(tracker,bucket,[(2,1,11),(3,5,11)],twins=True)
        for bucket in range(41,61):
            snapshot=self.update(tracker,bucket,[(0,5,3),(1,1,3),(2,1,11),(3,5,11)],twins=True)
            self.assertTrue(all(t['identity_status']=='unknown' for t in snapshot['tracks'][:2]))
        self.assertEqual([len(tracker._tracker.appearance_gallery[t]) for t in ('track_001','track_002')],[1,1])

    def test_reference_gallery_records_only_confirmed_continuous_views_and_survives_restart(self):
        frame,rows=scene();tracker=self.tracker();fixtures.AnchoredRosterTests().open(tracker,frame,rows)
        original=deepcopy(tracker._tracker.appearance_anchors)
        # Four disjoint colour signatures give a known owner for each actor.
        for i,tid in enumerate(sorted(tracker._tracker.tracks)):
            signature=[0.]*192;signature[i]=1.
            tracker._tracker.appearance_anchors[tid]=signature
            tracker._tracker.appearance_gallery[tid]=[signature.copy()]
        fixed=deepcopy(tracker._tracker.appearance_anchors)
        for bucket in range(11,23):
            prepared=prepare_anchor_observations(frame,rows,context(bucket))
            for i,row in enumerate(prepared):row['anchor_appearance']=fixed[f'track_{i+1:03d}'].copy()
            fraction=[.1,.3,.5,.6][(bucket-11)//3]
            prepared[0]['anchor_appearance'][0]=1-fraction;prepared[0]['anchor_appearance'][10]=fraction
            snapshot=tracker.update(bucket,prepared)
            self.assertEqual(snapshot['tracks'][0]['identity_status'],'anchored')
        self.assertEqual(tracker._tracker.appearance_anchors,fixed)
        self.assertGreater(len(tracker._tracker.appearance_gallery['track_001']),1)
        state=tracker.snapshot_state();restored=self.tracker();restored.restore_state(state)
        self.assertEqual(restored._tracker.appearance_gallery,tracker._tracker.appearance_gallery)
        for bucket in (100,101,102):
            prepared=prepare_anchor_observations(frame,rows,context(bucket))
            for i,row in enumerate(prepared):row['anchor_appearance']=fixed[f'track_{i+1:03d}'].copy()
            prepared[0]['anchor_appearance'][0]=.4;prepared[0]['anchor_appearance'][10]=.6
            snapshot=restored.update(bucket,prepared)
        self.assertGreater(appearance_distance(fixed['track_001'],prepared[0]['anchor_appearance']),.35)
        self.assertEqual(snapshot['tracks'][0]['identity_status'],'anchored')
        self.assertEqual(snapshot['tracks'][0]['association']['source'],'anchored_roster_reference_recovery')
        gallery=deepcopy(restored._tracker.appearance_gallery)
        # Another actor's actual clothing cannot be added as a new view of A.
        for bucket in range(103,110):
            bad=deepcopy(prepared);bad[0]['anchor_appearance']=fixed['track_002'].copy()
            snapshot=restored.update(bucket,bad)
            self.assertEqual(snapshot['tracks'][0]['identity_status'],'unknown')
        self.assertEqual(restored._tracker.appearance_gallery,gallery)
        legacy=deepcopy(state)
        for key in ('appearance_gallery','gallery_evidence','gallery_version','pending_gallery'):
            legacy['court_tracker']['anchored_roster'].pop(key,None)
        migrated=self.tracker();migrated.restore_state(legacy)
        self.assertEqual(migrated._tracker.appearance_gallery,{tid:[v] for tid,v in fixed.items()})
        bad_version=deepcopy(state);bad_version['court_tracker']['anchored_roster']['gallery_version']='unknown-version'
        with self.assertRaisesRegex(ValueError,'gallery checkpoint'):self.tracker().restore_state(bad_version)
        state['court_tracker']['anchored_roster']['appearance_gallery']['track_001'][0]=fixed['track_002']
        with self.assertRaisesRegex(ValueError,'gallery checkpoint'):self.tracker().restore_state(state)

    def test_interrupted_or_ambiguous_new_views_never_enter_reference_gallery(self):
        frame,rows=scene();tracker=self.tracker();fixtures.AnchoredRosterTests().open(tracker,frame,rows)
        for i,tid in enumerate(sorted(tracker._tracker.tracks)):
            v=[0.]*192;v[i]=1.
            tracker._tracker.appearance_anchors[tid]=v
            tracker._tracker.appearance_gallery[tid]=[v.copy()]
        fixed=deepcopy(tracker._tracker.appearance_anchors)
        for bucket in range(11,25):
            prepared=prepare_anchor_observations(frame,rows,context(bucket))
            for i,row in enumerate(prepared):row['anchor_appearance']=fixed[f'track_{i+1:03d}'].copy()
            prepared[0]['anchor_appearance'][0]=.9;prepared[0]['anchor_appearance'][10]=.1
            if bucket%3==1:
                prepared=prepared[1:]
            tracker.update(bucket,prepared)
        self.assertEqual(len(tracker._tracker.appearance_gallery['track_001']),1)
        self.assertEqual(tracker._tracker.appearance_anchors,fixed)

    def test_position_continuity_cannot_override_a_clear_other_person_appearance(self):
        frame,rows=scene();tracker=self.tracker();fixtures.AnchoredRosterTests().open(tracker,frame,rows)
        original=deepcopy(tracker._tracker.appearance_anchors)
        prepared=prepare_anchor_observations(frame,rows,context(11))
        prepared[0]['anchor_appearance']=original['track_002'].copy()
        result=tracker.update(11,prepared)
        self.assertEqual(result['tracks'][0]['identity_status'],'unknown')
        self.assertEqual(tracker._tracker.appearance_anchors,original)

    def test_gap_does_not_teleport_identity_across_net_but_measured_crossing_is_allowed(self):
        tracker=self.tracker();self.open(tracker)
        for bucket in range(11,41):
            self.update(tracker,bucket,[(1,5,3),(2,1,11),(3,5,11)])
        for bucket in range(41,46):
            result=self.update(tracker,bucket,[(0,3,11),(1,5,3),(2,1,11),(3,5,11)])
            self.assertEqual(result['tracks'][0]['identity_status'],'unknown')
        # Court end itself is not identity. An actually measured transition
        # through the net line must keep the original ID.
        tracker=self.tracker();self.open(tracker)
        for bucket in range(11,92):
            progress=(bucket-11)/80
            result=self.update(tracker,bucket,[(0,1+2*progress,3+8*progress),(1,5,3),(2,1,11),(3,5,11)])
        self.assertEqual(result['tracks'][0]['identity_status'],'anchored')
        self.assertEqual(result['tracks'][0]['court_end'],'end_b')


if __name__=='__main__':unittest.main()
