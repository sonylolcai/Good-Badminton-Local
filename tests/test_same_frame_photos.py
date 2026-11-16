from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.same_frame_photos import SameFramePhotoCollector
from badminton_analysis.streaming.models import FrameContext, ProcessorEvent
from good_badminton_contracts.photo_roster import validate_photo_roster

SID = 'ssn_same_frame'


def context(t=1, segment=0):
    return FrameContext(analysis_session_id=SID, segment_index=segment, source_frame_index=int(t*30),
                        source_time_sec=t, is_measurement_frame=True, measurement_bucket=int(t*10))


def scene(count=4):
    frame = np.zeros((400, 720, 3), dtype=np.uint8)
    boxes = [[20,20,110,175], [200,20,290,175], [380,200,470,355], [560,200,650,355]]
    events = []
    for i, box in enumerate(boxes[:count]):
        x1,y1,x2,y2=box;frame[y1:y2,x1:x2] = [(i+1)*40, 220-i*35, i*70]
        points, scores = [[0.,0.] for _ in range(17)], [0.]*17
        for j,ry in [(5,.2),(6,.2),(11,.5),(12,.5),(15,.9),(16,.9)]:
            points[j] = [(x1+x2)/2, y1+(y2-y1)*ry];scores[j]=.95
        events.append(ProcessorEvent(event_type='person_observation', evidence_state='detected', confidence=.95,
            data={'track':{'track_id':f'track_{i+1:03}', 'status':'detected','confidence':.95,
                'location_evidence':{'bbox_xyxy':box},'pose':{'keypoints_image':points,'keypoint_scores':scores}}}))
    return frame,events


class SameFramePhotoTests(unittest.TestCase):
    def test_singles_and_doubles_are_same_frame_distinct_and_frozen(self):
        for count in (2,4):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as root:
                frame,events=scene(count);c=SameFramePhotoCollector(root,count=count,session_id=SID)
                output=c.attach(frame,context(),events);proofs={e.data['track']['track_id']:e.data['candidate_photo'] for e in output}
                receipt=validate_photo_roster(proofs,proofs,SID)
                self.assertEqual(receipt['source_time_sec'],1.)
                self.assertEqual(receipt['source_frame_index'],30)
                self.assertEqual(len({p['sha256'] for p in proofs.values()}),count)
                path=Path(root)/'same_frame_set';before={p.name:p.read_bytes() for p in path.iterdir()}
                self.assertFalse(any('candidate_photo' in e.data for e in c.attach(255-frame,context(2),events)))
                self.assertEqual(before,{p.name:p.read_bytes() for p in path.iterdir()})

    def test_missing_person_or_predicted_frame_waits_without_partial_group(self):
        frame,events=scene()
        for selected in (events[:3], [ProcessorEvent(event_type='person_observation',evidence_state='predicted',confidence=.5,data=e.data) for e in events]):
            with tempfile.TemporaryDirectory() as root:
                c=SameFramePhotoCollector(root,count=4,session_id=SID);c.attach(frame,context(),selected)
                self.assertFalse((Path(root)/'same_frame_set').exists())

    def test_duplicate_ids_overlapping_body_and_incomplete_pose_wait(self):
        frame,events=scene()
        changes=[lambda e:e[1].data['track'].update(track_id='track_001'),
                 lambda e:e[1].data['track'].update(location_evidence=e[0].data['track']['location_evidence']),
                 lambda e:e[0].data['track']['pose'].update(keypoint_scores=[0.]*17)]
        for mutate in changes:
            with tempfile.TemporaryDirectory() as root:
                bad=deepcopy(events);mutate(bad);c=SameFramePhotoCollector(root,count=4,session_id=SID)
                c.attach(frame,context(),bad);self.assertFalse((Path(root)/'same_frame_set').exists())

    def test_write_failure_exposes_no_group_and_retry_commits_all(self):
        frame,events=scene()
        with tempfile.TemporaryDirectory() as root:
            c=SameFramePhotoCollector(root,count=4,session_id=SID)
            with patch('api.same_frame_photos.os.replace',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):c.attach(frame,context(),events)
            self.assertEqual(list(Path(root).iterdir()),[])
            self.assertEqual(len(c.attach(frame,context(2),events)),4)
            self.assertTrue((Path(root)/'same_frame_set'/'manifest.json').is_file())

    def test_checkpoint_restores_original_group_and_rejects_corruption(self):
        frame,events=scene()
        with tempfile.TemporaryDirectory() as root:
            c=SameFramePhotoCollector(root,count=4,session_id=SID);c.attach(frame,context(),events)
            state=c.snapshot_state();restored=SameFramePhotoCollector(root,count=4,session_id=SID)
            restored.restore_state(state);self.assertEqual(restored.snapshot_state(),state)
            (Path(root)/'same_frame_set'/'track_001.jpg').write_bytes(b'wrong image')
            with self.assertRaisesRegex(ValueError,'file_mismatch'):
                SameFramePhotoCollector(root,count=4,session_id=SID).restore_state(state)

    def test_rename_before_checkpoint_replays_only_committed_frame(self):
        frame,events=scene()
        with tempfile.TemporaryDirectory() as root:
            original=SameFramePhotoCollector(root,count=4,session_id=SID)
            empty=original.snapshot_state();original.attach(frame,context(2),events)
            recovered=SameFramePhotoCollector(root,count=4,session_id=SID);recovered.restore_state(empty)
            earlier=recovered.attach(frame,context(1),events)
            self.assertFalse(any('candidate_photo' in e.data for e in earlier))
            replay=recovered.attach(frame,context(2),events)
            self.assertEqual({e.data['candidate_photo']['source_time_sec'] for e in replay},{2.})
            for later in (3, 5, 7):
                fresh=SameFramePhotoCollector(root,count=4,session_id=SID)
                result=fresh.attach(frame,context(later),events)
                self.assertFalse(any('candidate_photo' in e.data for e in result))
                self.assertEqual(fresh.snapshot_state()['frozen'],original.snapshot_state()['frozen'])

    def test_manifest_contract_rejects_mixed_frame_duplicate_image_and_foreign_session(self):
        frame,events=scene()
        with tempfile.TemporaryDirectory() as root:
            SameFramePhotoCollector(root,count=4,session_id=SID).attach(frame,context(),events)
            photos=json.loads((Path(root)/'same_frame_set'/'manifest.json').read_text())['photos']
            for mutate in (lambda p:p['track_001'].update(source_time_sec=2),
                           lambda p:p['track_002'].update(sha256=p['track_001']['sha256']),
                           lambda p:p['track_001'].update(analysis_session_id='foreign')):
                bad=deepcopy(photos);mutate(bad)
                with self.assertRaises(ValueError):validate_photo_roster(bad,photos,SID)

    def test_protected_photo_lookup_requires_committed_set_and_checks_hash(self):
        from api.stream_sessions import StreamSessionManager
        frame,events=scene()
        with tempfile.TemporaryDirectory() as root:
            manager=Mock();manager._session_dir.return_value=Path(root)
            manager._load.return_value={'configuration':{'candidate_photo_policy':'same_frame_roster.v1','expected_player_count':4}}
            self.assertIsNone(StreamSessionManager.candidate_photo_path(manager,SID,'track_001'))
            SameFramePhotoCollector(Path(root)/'candidate_photos',count=4,session_id=SID).attach(frame,context(),events)
            self.assertTrue(StreamSessionManager.candidate_photo_path(manager,SID,'track_001').is_file())
            self.assertIsNone(StreamSessionManager.candidate_photo_path(manager,SID,'../secret'))
            (Path(root)/'candidate_photos'/'same_frame_set'/'track_001.jpg').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'file_mismatch'):
                StreamSessionManager.candidate_photo_path(manager,SID,'track_001')

    def test_photo_roster_endpoint_is_authenticated_and_does_not_scan_events(self):
        from api.stream_routes import register_stream_routes
        frame,events=scene()
        with tempfile.TemporaryDirectory() as root:
            SameFramePhotoCollector(Path(root)/'candidate_photos',count=4,session_id=SID).attach(frame,context(),events)
            manager=Mock();manager._session_dir.return_value=Path(root);manager.worker_running=False
            manager.get_session.return_value={'configuration':{'candidate_photo_policy':'same_frame_roster.v1','expected_player_count':4}}
            app=FastAPI();register_stream_routes(app,stream_manager=manager,require_api_key=lambda:None,
                                                require_upload_access=lambda:None,health_payload=lambda:{})
            response=TestClient(app).get(f'/api/v1/stream-sessions/{SID}/photo-roster')
            self.assertEqual(response.status_code,200);self.assertEqual(len(response.json()['photos']),4)
            manager._load_events.assert_not_called()
            def denied():raise HTTPException(401,'unauthorized')
            app=FastAPI();register_stream_routes(app,stream_manager=manager,require_api_key=denied,
                                                require_upload_access=lambda:None,health_payload=lambda:{})
            self.assertEqual(TestClient(app).get(f'/api/v1/stream-sessions/{SID}/photo-roster').status_code,401)

    def test_legacy_configuration_is_unchanged_and_policy_requires_locked_roster(self):
        from api.stream_models import validate_configuration
        config={'analysis_sample_hz':10,'pose_imgsz':960,'shuttle_detector':'none','generate_annotated_video':False}
        self.assertNotIn('candidate_photo_policy',validate_configuration(config))
        for count in (2,4):
            good={**config,'candidate_photo_policy':'same_frame_roster.v1','expected_player_count':count,'lock_match_roster':True}
            self.assertEqual(validate_configuration(good)['candidate_photo_policy'],'same_frame_roster.v1')
            with self.assertRaises(ValueError):validate_configuration({**good,'lock_match_roster':False})

    def test_photo_endpoint_distinguishes_terminal_missing_from_pending_and_gap(self):
        from api.stream_routes import register_stream_routes
        with tempfile.TemporaryDirectory() as root:
            manager=Mock();manager._session_dir.return_value=Path(root);manager.worker_running=False
            session={'configuration':{'candidate_photo_policy':'same_frame_roster.v1','expected_player_count':4},'status':'processing'}
            manager.get_session.return_value=session
            app=FastAPI();register_stream_routes(app,stream_manager=manager,require_api_key=lambda:None,
                                                require_upload_access=lambda:None,health_payload=lambda:{})
            client=TestClient(app);url=f'/api/v1/stream-sessions/{SID}/photo-roster'
            self.assertEqual(client.get(url).status_code,409)
            session['status']='finalized'
            reply=client.get(url);self.assertEqual(reply.status_code,422)
            self.assertEqual(reply.json()['detail']['code'],'incomplete_same_frame_photos')
            frame,events=scene();SameFramePhotoCollector(Path(root)/'candidate_photos',count=4,session_id=SID).attach(frame,context(),events)
            session['continuity_epoch']=1
            reply=client.get(url);self.assertEqual(reply.status_code,422)
            self.assertEqual(reply.json()['detail']['code'],'same_frame_photo_continuity_unverified')
