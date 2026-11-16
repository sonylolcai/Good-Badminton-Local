"""Same-source-frame photo proof. This does not certify tracking identity."""
import math
import re

POLICY = 'same_frame_roster.v1'
SELECTION = 'same_frame_roster_v1'


def validate_photo_roster(photos, expected_ids, session_id):
    ids = set(expected_ids)
    if (len(ids) not in (2, 4) or not isinstance(photos, dict) or set(photos) != ids
            or any(not isinstance(tid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', tid) for tid in ids)):
        raise ValueError('incomplete_same_frame_photos')
    signatures, hashes, boxes = set(), set(), []
    for tid, p in photos.items():
        if (not isinstance(p, dict) or p.get('track_id') != tid or p.get('frozen') is not True
                or p.get('selection_policy') != SELECTION or p.get('analysis_session_id') != session_id
                or not isinstance(p.get('photo_set_id'), str) or not p['photo_set_id']):
            raise ValueError('invalid_same_frame_photo_identity')
        stamp = p.get('source_time_sec')
        if (type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0
                or any(type(p.get(k)) is not int or p[k] < 0
                       for k in ('source_frame_index', 'source_segment_index'))):
            raise ValueError('invalid_same_frame_photo_source')
        digest = p.get('sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid_same_frame_photo_checksum')
        box = p.get('crop_xyxy')
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(v) is not int or v < 0 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError('invalid_same_frame_photo_box')
        if any(min(box[2], b[2]) > max(box[0], b[0]) and min(box[3], b[3]) > max(box[1], b[1]) for b in boxes):
            raise ValueError('overlapping_same_frame_photos')
        boxes.append(box)
        hashes.add(digest)
        signatures.add((p['photo_set_id'], p['source_segment_index'], p['source_frame_index'], stamp))
    if len(signatures) != 1:
        raise ValueError('mixed_frame_photos')
    if len(hashes) != len(ids):
        raise ValueError('duplicate_same_frame_photos')
    signature = next(iter(signatures))
    return {'policy': POLICY, 'photo_set_id': signature[0], 'source_segment_index': signature[1],
            'source_frame_index': signature[2], 'source_time_sec': signature[3], 'track_ids': sorted(ids)}
