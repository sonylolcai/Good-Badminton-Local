"""Freeze one complete photo roster independently of the tracking algorithm."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .candidate_photos import CandidatePhotoCollector
from badminton_analysis.streaming.models import ProcessorEvent
from good_badminton_contracts.photo_roster import validate_photo_roster, POLICY, SELECTION


class SameFramePhotoCollector:
    def __init__(self, directory, *, count, session_id):
        if type(count) is not int or count not in (2, 4):
            raise ValueError('A photo roster requires two or four people')
        self.directory = Path(directory)
        self.count = count
        self.session_id = session_id
        self._frozen = None

    def _read(self):
        path = self.directory / 'same_frame_set' / 'manifest.json'
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
        if value.get('policy') != POLICY or len(value.get('photos', {})) != self.count:
            raise ValueError('invalid_same_frame_photo_manifest')
        validate_photo_roster(value['photos'], value['photos'], self.session_id)
        for tid, proof in value['photos'].items():
            photo = path.parent / (tid + '.jpg')
            if not photo.is_file() or hashlib.sha256(photo.read_bytes()).hexdigest() != proof['sha256']:
                raise ValueError('same_frame_photo_file_mismatch')
        return value

    def attach(self, frame, context, events):
        events = list(events)
        if self._frozen is not None:
            return events
        # A rename may have committed the photos before the event/checkpoint
        # transaction. Replay publishes the original proofs only at that frame.
        committed = self._read()
        if committed is not None:
            first = next(iter(committed['photos'].values()))
            target = (first['source_segment_index'], first['source_time_sec'], first['source_frame_index'])
            current = (context.segment_index, round(float(context.source_time_sec), 6), context.source_frame_index)
            if current < target:
                return events
            if current > target:
                # A replacement engine may begin after a failed segment,
                # without the old collector checkpoint. Keep the committed
                # photos; do not make the remaining video fail as well. The
                # photo API rejects changed continuity epochs before claiming.
                self._frozen = committed
                return events
            photos = committed['photos']
        else:
            selected = [e for e in events if e.event_type == 'person_observation']
            if len(selected) != self.count or any(e.evidence_state != 'detected' for e in selected):
                return events
            ids = [(e.data.get('track') or {}).get('track_id') for e in selected]
            if len(set(ids)) != self.count:
                return events
            for event in selected:
                track = event.data.get('track') or {}
                box = (track.get('location_evidence') or {}).get('bbox_xyxy')
                if (not isinstance(box, (list, tuple)) or len(box) != 4
                        or not all(type(v) in (int, float) and np.isfinite(v) for v in box)):
                    return events
                pose = track.get('pose') or {}
                points, scores = pose.get('keypoints_image') or [], pose.get('keypoint_scores') or []
                def visible(j):
                    return (j < len(points) and j < len(scores) and isinstance(points[j], (list, tuple))
                            and len(points[j]) >= 2 and type(scores[j]) in (int, float) and scores[j] >= .25
                            and all(type(v) in (int, float) and np.isfinite(v) for v in points[j][:2])
                            and box[0] <= points[j][0] <= box[2] and box[1] <= points[j][1] <= box[3])
                if not all(any(visible(j) for j in pair) for pair in ((5, 6), (11, 12), (15, 16))):
                    return events
            self.directory.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix='.same-frame-', dir=self.directory))
            try:
                cropper = CandidatePhotoCollector(stage)
                photos = {}
                height, width = np.asarray(frame).shape[:2]
                for event in selected:
                    proof = cropper._capture(frame, context, event)
                    if proof is None:
                        return events
                    track = event.data['track']
                    x1, y1, x2, y2 = (float(v) for v in track['location_evidence']['bbox_xyxy'])
                    px, py = max(4, int(round((x2-x1)*.12))), max(4, int(round((y2-y1)*.08)))
                    crop = [max(0, int(x1)-px), max(0, int(y1)-py), min(width, int(x2)+px), min(height, int(y2)+py)]
                    tid = proof['track_id']
                    proof.update(analysis_session_id=self.session_id, selection_policy=SELECTION, frozen=True,
                                 source_segment_index=int(context.segment_index), source_frame_index=int(context.source_frame_index),
                                 crop_xyxy=crop, sha256=hashlib.sha256((stage/(tid+'.jpg')).read_bytes()).hexdigest())
                    photos[tid] = proof
                key = [self.session_id, context.segment_index, context.source_frame_index,
                       round(float(context.source_time_sec), 6), sorted(photos)]
                set_id = 'frame_' + hashlib.sha256(json.dumps(key).encode()).hexdigest()
                for proof in photos.values():
                    proof['photo_set_id'] = set_id
                try:
                    validate_photo_roster(photos, ids, self.session_id)
                except ValueError:
                    return events
                committed = {'policy': POLICY, 'photos': photos}
                (stage/'manifest.json').write_text(json.dumps(committed, sort_keys=True))
                os.replace(stage, self.directory/'same_frame_set')
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        emitted = set()
        output = []
        for event in events:
            tid = (event.data.get('track') or {}).get('track_id')
            if event.event_type == 'person_observation' and event.evidence_state == 'detected' and tid in photos:
                data = deepcopy(dict(event.data)); data['candidate_photo'] = deepcopy(photos[tid])
                event = ProcessorEvent(event_type=event.event_type, evidence_state=event.evidence_state,
                                       confidence=event.confidence, data=data, source_time_sec=event.source_time_sec,
                                       segment_index=event.segment_index)
                emitted.add(tid)
            output.append(event)
        if emitted != set(photos):
            raise ValueError('same_frame_photo_event_roster_mismatch')
        self._frozen = committed
        return output

    def snapshot_state(self):
        return {'state_version': POLICY, 'count': self.count, 'session_id': self.session_id,
                'frozen': deepcopy(self._frozen)}

    def restore_state(self, state):
        if (state.get('state_version') != POLICY or state.get('count') != self.count
                or state.get('session_id') != self.session_id):
            raise ValueError('same_frame_photo_checkpoint_mismatch')
        frozen = state.get('frozen')
        if frozen is not None and self._read() != frozen:
            raise ValueError('same_frame_photo_checkpoint_mismatch')
        self._frozen = deepcopy(frozen)


def result_photo_roster(root, session_id, expected_ids):
    path = Path(root)/'candidate_photos'/'same_frame_set'/'manifest.json'
    if not path.is_file():
        raise ValueError('incomplete_same_frame_photos')
    manifest = json.loads(path.read_text())
    if manifest.get('policy') != POLICY:
        raise ValueError('invalid_same_frame_photo_manifest')
    photos = manifest['photos']
    receipt = validate_photo_roster(photos, expected_ids, session_id)
    for tid, proof in photos.items():
        if hashlib.sha256((path.parent/(tid+'.jpg')).read_bytes()).hexdigest() != proof['sha256']:
            raise ValueError('same_frame_photo_file_mismatch')
    return photos, receipt
