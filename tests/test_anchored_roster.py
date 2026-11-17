"""Labelled synthetic frames test identity invariants, not video accuracy."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from api.candidate_photos import CandidatePhotoCollector
from api.stream_runtime import CompositeMeasurementProcessor
from badminton_analysis.streaming.models import FrameContext
from badminton_analysis.tracking.anchored_roster import ANCHORED_ROSTER, prepare_anchor_observations
from badminton_analysis.tracking.person_only import PersonOnlyTracker, PersonOnlyFrameProcessor


CORNERS = [(0, 0), (610, 0), (610, 1340), (0, 1340)]
BOXES = [[20, 20, 100, 160], [140, 20, 220, 160], [280, 180, 360, 320], [420, 180, 500, 320]]
COLOURS = [(40, 40, 200), (40, 200, 40), (200, 40, 40), (40, 200, 200)]


def context(bucket):
    return FrameContext(analysis_session_id='same-frame-test', segment_index=bucket // 20,
                        source_frame_index=bucket * 3, source_time_sec=bucket / 10,
                        is_measurement_frame=True, measurement_bucket=bucket)


def scene(count=4):
    image = np.zeros((360, 640, 3), np.uint8)
    rows = []
    for i in (range(4) if count == 4 else (0, 2)):
        x1, y1, x2, y2 = BOXES[i]
        image[y1:y2, x1:x2] = COLOURS[i]
        cv2.rectangle(image, (x1+3, y1+3), (x2-3, y2-3), (255,255,255), 2)
        cv2.line(image, (x1+5,y1+5), (x2-5,y2-5), (0,0,0), 2)
        points=[[0.,0.] for _ in range(17)];scores=[0.]*17
        for joint,rx,ry in ((5,.3,.20),(6,.7,.20),(11,.35,.50),(12,.65,.50),(15,.35,.92),(16,.65,.92)):
            points[joint]=[x1+(x2-x1)*rx,y1+(y2-y1)*ry];scores[joint]=.95
        rows.append(dict(court_xy=[[1,3],[5,3],[1,11],[5,11]][i], image_xy=[(x1+x2)/2,y2],
                         confidence=.95, location_confidence=.95, bbox_xyxy=BOXES[i], association_key=f'visual-{i}',
                         keypoints_image=points,keypoint_scores=scores))
    return image, rows


class AnchoredRosterTests(unittest.TestCase):
    def tracker(self, count=4):
        return PersonOnlyTracker(CORNERS, fps=10, lock_match_roster=True,
            expected_roster_count=count, match_mode='doubles' if count==4 else 'singles',
            roster_discovery_seconds=1, roster_identity_policy=ANCHORED_ROSTER)

    def open(self, tracker, frame, rows):
        processor = PersonOnlyFrameProcessor(tracker, lambda f,c: rows)
        for bucket in range(11):
            events = list(processor.process_frame(frame, context(bucket)))
        self.assertEqual(len([e for e in events if e.event_type=='person_observation']), len(rows))
        return processor, events

    def test_singles_and_doubles_capture_one_shared_frame_and_never_replace_individually(self):
        for count in (2,4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                frame, rows = scene(count); tracker = self.tracker(count)
                collector = CandidatePhotoCollector(directory, anchored_roster_count=count)
                processor = CompositeMeasurementProcessor(PersonOnlyFrameProcessor(tracker, lambda f,c: rows), candidate_photos=collector)
                for bucket in range(10):
                    events = list(processor.process_frame(frame, context(bucket)))
                    self.assertFalse(any(e.data.get('candidate_photo') for e in events))
                    self.assertFalse((Path(directory)/'anchor_set').exists())
                events = list(processor.process_frame(frame, context(10)))
                photos = [e.data['candidate_photo'] for e in events]
                self.assertEqual(len(photos),count)
                self.assertEqual({p['source_frame_index'] for p in photos},{30})
                self.assertEqual({p['source_time_sec'] for p in photos},{1.})
                self.assertEqual(len({p['photo_set_id'] for p in photos}),1)
                self.assertEqual(len({p['sha256'] for p in photos}),count)
                before = {p.name:p.read_bytes() for p in (Path(directory)/'anchor_set').iterdir()}
                later=list(processor.process_frame(255-frame,context(11)))
                self.assertFalse(any(e.data.get('candidate_photo') for e in later))
                self.assertEqual(before,{p.name:p.read_bytes() for p in (Path(directory)/'anchor_set').iterdir()})

    def test_crossing_order_keeps_original_people_despite_changed_or_swapped_keys(self):
        frame, rows=scene(); tracker=self.tracker(); self.open(tracker,frame,rows)
        for bucket in range(11,52):
            moved=deepcopy(rows); progress=(bucket-11)/40
            moved[0]['court_xy']=[1+4*progress,3]
            moved[1]['court_xy']=[5-4*progress,3.6]
            moved[0]['association_key'],moved[1]['association_key']='visual-1','visual-0'
            snapshot=tracker.update(bucket,prepare_anchor_observations(frame,list(reversed(moved)),context(bucket)))
        a,b=snapshot['tracks'][:2]
        self.assertEqual(a['court_xy_m'],[5.,3.]);self.assertEqual(b['court_xy_m'],[1.,3.6])
        self.assertTrue(all(t['identity_status']=='anchored' for t in snapshot['tracks']))

    def test_baseline_rounding_does_not_discard_a_pose_admitted_by_observation_contract(self):
        frame,rows=scene();rows[0]['court_xy']=[1,.14]
        tracker=self.tracker();self.open(tracker,frame,rows)
        for bucket,y in ((11,-.04),(12,-.10),(13,.05)):
            moved=deepcopy(rows);moved[0]['court_xy']=[1,y]
            snapshot=tracker.update(bucket,prepare_anchor_observations(frame,moved,context(bucket)))
            self.assertTrue(tracker.contains_athlete([1,y]))
            self.assertEqual(snapshot['tracks'][0]['status'],'detected')
            self.assertEqual(snapshot['tracks'][0]['identity_status'],'anchored')
        outside=deepcopy(rows);outside[0]['court_xy']=[1,-.36]
        snapshot=tracker.update(14,prepare_anchor_observations(frame,outside,context(14)))
        self.assertFalse(tracker.contains_athlete([1,-.36]))
        self.assertEqual(snapshot['tracks'][0]['identity_status'],'unknown')

    def test_one_clear_frame_after_presence_window_is_enough_for_complete_photo_set(self):
        frame,rows=scene();tracker=self.tracker()
        processor=PersonOnlyFrameProcessor(tracker,lambda f,c:rows)
        # Bodies are present throughout; earlier images cannot supply clear
        # clothing references. The single clear frame must still be usable.
        for bucket in range(10):
            self.assertFalse(any(e.event_type=='person_observation' for e in processor.process_frame(np.zeros_like(frame),context(bucket))))
        selected=list(processor.process_frame(frame,context(10)))
        self.assertEqual(len(selected),4)
        with tempfile.TemporaryDirectory() as directory:
            photos=CandidatePhotoCollector(directory,anchored_roster_count=4).attach(frame,context(10),selected)
            self.assertEqual({e.data['candidate_photo']['source_frame_index'] for e in photos},{30})

    def test_missing_foreign_or_similarly_dressed_pose_does_not_inherit_empty_identity(self):
        frame, rows=scene(); tracker=self.tracker(); self.open(tracker,frame,rows)
        initial=deepcopy(tracker._tracker.appearance_anchors)
        tracker.update(11,prepare_anchor_observations(frame,rows[1:],context(11)))
        changed=frame.copy();x1,y1,x2,y2=BOXES[0];changed[y1:y2,x1:x2]=(170,40,170)
        cv2.line(changed,(x1,y1),(x2,y2),(255,255,255),2)
        snapshot=tracker.update(12,prepare_anchor_observations(changed,rows,context(12)))
        self.assertEqual(snapshot['tracks'][0]['identity_status'],'unknown')
        self.assertNotEqual(snapshot['tracks'][0]['status'],'detected')
        ambiguous=prepare_anchor_observations(frame,rows,context(13))
        ambiguous[0]['anchor_appearance']=ambiguous[1]['anchor_appearance']
        for row in ambiguous[:2]:
            row['court_xy']=[3,3]
            row['bbox_xyxy']=[20,20,220,160]
        snapshot=tracker.update(13,ambiguous)
        self.assertTrue(all(t['identity_status']=='unknown' for t in snapshot['tracks'][:2]))
        self.assertEqual(tracker._tracker.appearance_anchors,initial)

    def test_duplicate_boxes_blur_clipping_and_incomplete_group_cannot_lock(self):
        frame, rows=scene()
        cases=[(frame,rows[:3]),(np.zeros_like(frame),rows)]
        duplicate=deepcopy(rows);duplicate[1]['bbox_xyxy']=duplicate[0]['bbox_xyxy']
        clipped=deepcopy(rows);clipped[0]['bbox_xyxy']=[-10,20,100,160]
        legs_only=deepcopy(rows)
        for i in (5,6,11,12):legs_only[0]['keypoint_scores'][i]=.01
        cases.extend([(frame,duplicate),(frame,clipped),(frame,legs_only)])
        for image,observations in cases:
            with self.subTest(kind=len(cases)):
                tracker=self.tracker();processor=PersonOnlyFrameProcessor(tracker,lambda f,c: observations)
                for bucket in range(25):
                    events=list(processor.process_frame(image,context(bucket)))
                self.assertFalse(any(e.event_type=='person_observation' for e in events))

    def test_short_gap_recovery_excludes_gap_distance_but_long_gap_stays_unknown(self):
        frame, rows=scene();tracker=self.tracker();self.open(tracker,frame,rows)
        tracker.update(11,prepare_anchor_observations(frame,rows[1:],context(11)))
        before=tracker._tracker.track_metrics['track_001']['distance_m']
        returned=deepcopy(rows);returned[0]['court_xy']=[1.3,3]
        s=tracker.update(12,prepare_anchor_observations(frame,returned,context(12)))
        self.assertEqual(s['tracks'][0]['identity_status'],'anchored')
        self.assertEqual(tracker._tracker.track_metrics['track_001']['distance_m'],before)
        self.assertIsNone(s['tracks'][0]['motion']['current_speed_mps'])
        s=tracker.update(100,prepare_anchor_observations(frame,rows,context(100)))
        self.assertTrue(all(t['identity_status']=='unknown' for t in s['tracks']))
        # One returning pose never fills a slot after a long absence. Only
        # repeated unique original-reference evidence permits a new segment.
        s=tracker.update(101,prepare_anchor_observations(frame,rows,context(101)))
        self.assertTrue(all(t['identity_status']=='unknown' for t in s['tracks']))
        state=tracker.snapshot_state();restored=self.tracker();restored.restore_state(state)
        before={tid:m['distance_m'] for tid,m in restored._tracker.track_metrics.items()}
        s=restored.update(102,prepare_anchor_observations(frame,rows,context(102)))
        self.assertTrue(all(t['identity_status']=='anchored' for t in s['tracks']))
        self.assertTrue(all(t['association']['source']=='anchored_roster_reference_recovery' for t in s['tracks']))
        self.assertEqual(before,{tid:m['distance_m'] for tid,m in restored._tracker.track_metrics.items()})

    def test_checkpoint_preserves_anchors_and_frozen_set_and_rejects_corruption(self):
        frame,rows=scene();tracker=self.tracker();person,events=self.open(tracker,frame,rows)
        with tempfile.TemporaryDirectory() as directory:
            collector=CandidatePhotoCollector(directory,anchored_roster_count=4)
            captured=collector.attach(frame,context(10),events)
            tracker_state=json.loads(json.dumps(tracker.snapshot_state()))
            restored=self.tracker();restored.restore_state(tracker_state)
            again=restored.update(11,prepare_anchor_observations(frame,rows,context(11)))
            self.assertTrue(all(t['identity_status']=='anchored' for t in again['tracks']))
            photo_state=collector.snapshot_state();other=CandidatePhotoCollector(directory,anchored_roster_count=4)
            other.restore_state(photo_state)
            self.assertFalse(any(e.data.get('candidate_photo') for e in other.attach(frame,context(11),list(person.process_frame(frame,context(11))))))
            # Rename committed the group but process died before checkpoint:
            # replaying its exact frame emits the same metadata and pixels.
            replay=CandidatePhotoCollector(directory,anchored_roster_count=4)
            self.assertEqual([e.data['candidate_photo'] for e in replay.attach(frame,context(10),events)], [e.data['candidate_photo'] for e in captured])
            (Path(directory)/'anchor_set'/'track_001.jpg').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'mismatch'):other.restore_state(photo_state)
            tracker_state['court_tracker']['anchored_roster']['appearance_anchors'].pop('track_001')
            with self.assertRaisesRegex(ValueError,'Incomplete'):self.tracker().restore_state(tracker_state)

    def test_atomic_photo_write_failure_publishes_no_partial_group(self):
        frame,rows=scene();_,events=self.open(self.tracker(),frame,rows)
        with tempfile.TemporaryDirectory() as directory:
            collector=CandidatePhotoCollector(directory,anchored_roster_count=4)
            with patch('api.candidate_photos.os.replace',side_effect=OSError('injected full disk')):
                with self.assertRaises(OSError):collector.attach(frame,context(10),events)
            self.assertFalse((Path(directory)/'anchor_set').exists())
            self.assertIsNone(collector.snapshot_state()['frozen_set'])
            self.assertEqual(len(collector.attach(frame,context(10),events)),4)

    def test_photo_collector_rejects_duplicate_body_box_even_with_distinct_track_ids(self):
        frame,rows=scene();_,events=self.open(self.tracker(),frame,rows)
        events[1].data['track']['location_evidence']['bbox_xyxy']=events[0].data['track']['location_evidence']['bbox_xyxy']
        with tempfile.TemporaryDirectory() as directory:
            result=CandidatePhotoCollector(directory,anchored_roster_count=4).attach(frame,context(10),events)
            self.assertFalse(any(e.data.get('candidate_photo') for e in result))
            self.assertFalse((Path(directory)/'anchor_set').exists())

    def test_session_keeps_photo_set_proof_after_finalize_and_serves_only_committed_group(self):
        from api.stream_sessions import StreamSessionManager
        from tests.stream_test_utils import create_request
        with tempfile.TemporaryDirectory() as directory:
            manager=StreamSessionManager(directory,start_worker=False)
            request=create_request();request['configuration'].update(roster_identity_policy=ANCHORED_ROSTER,expected_player_count=4)
            _,created=manager.create_session(request,'anchored-status-proof')
            sid=created['analysis_session_id'];session=manager._load(sid)
            photo_dir=manager._session_dir(sid)/'candidate_photos'
            photo_dir.mkdir(parents=True,exist_ok=True)
            (photo_dir/'track_001.jpg').write_bytes(b'uncommitted-loose-photo')
            self.assertIsNone(manager.candidate_photo_path(sid,'track_001'))
            frame,rows=scene();_,events=self.open(self.tracker(),frame,rows)
            collector=CandidatePhotoCollector(photo_dir,anchored_roster_count=4)
            photos=collector.attach(frame,context(10),events)
            for event in photos:
                manager._append_event(session,event.event_type,1.,0,.9,'detected',event.data)
            manager._append_event(session,'session_status',2.,0,1.,'derived',{'track_candidates':[{'track_id':f'track_{i+1:03d}','state':'closed'} for i in range(4)]})
            manager._save(session)
            _,status=manager.get_status(sid)
            for candidate in status['track_candidates']:
                self.assertTrue(candidate['candidate_photo']['frozen'])
                self.assertEqual(candidate['candidate_photo']['source_frame_index'],30)
                self.assertEqual(candidate['candidate_photo']['photo_set_id'],'anchor_30_10')
                self.assertEqual(manager.candidate_photo_path(sid,candidate['track_id']).parent.name,'anchor_set')

    def test_configuration_is_explicit_and_incompatible_sport_or_roster_is_rejected(self):
        from api.stream_models import validate_configuration
        from api.mode_sync import VisionModeSynchronizer
        from api.vision_profiles import TENNIS_PROFILE
        config=dict(analysis_sample_hz=10,pose_imgsz=960,shuttle_detector='tracknet_v3',generate_annotated_video=False,
                    expected_player_count=4,roster_identity_policy=ANCHORED_ROSTER)
        self.assertEqual(validate_configuration(config)['roster_identity_policy'],ANCHORED_ROSTER)
        with self.assertRaises(ValueError):validate_configuration(dict(config,lock_match_roster=False))
        # The fixed tennis process cannot acquire a badminton roster policy.
        with self.assertRaises(ValueError):VisionModeSynchronizer(TENNIS_PROFILE).synchronize(config)

    def test_far_crop_for_anchored_people_includes_body_above_calibrated_floor(self):
        from api.stream_runtime import _derived_far_court_roi
        corners=[[711.5,350],[1179.5,361],[1568,946],[313.5,919.5]]
        legacy=_derived_far_court_roi(corners,(1080,1920,3))
        anchored=_derived_far_court_roi(corners,(1080,1920,3),include_upper_body=True)
        self.assertEqual(legacy[1],323)
        self.assertLessEqual(anchored[1],200)
        self.assertEqual(anchored[0],legacy[0]);self.assertEqual(anchored[2],legacy[2])
        self.assertEqual(corners[0],[711.5,350])


if __name__=='__main__':
    unittest.main()
