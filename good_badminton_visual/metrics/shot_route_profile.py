"""One descriptive radar axis using only newly supplied paired-contact routes.

Receiving regions describe where an opponent contacts a shot, not its landing.
Legacy TrackNet candidates, movement proxies and LLM grades are not inputs.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from .heatmap_grid import receiving_heatmap

VERSION = 'shot-route-profile.v1'
EVENT_VERSION = 'personal-shot-route-events.v1'
SOURCE = 'paired_contacts_receiving_region'
MIN_SAMPLES = 12


def shot_route_profile(track_id, path, *, duration_sec=None):
    path = Path(path)
    profile = {'schema_version': VERSION, 'track_id': track_id, 'source': SOURCE,
               'metric': 'opponent_receiving_region_diversity',
               'status': 'not_collected', 'value': None, 'normalized_value': None,
               'sample_count': 0, 'minimum_samples': MIN_SAMPLES, 'unit': 'normalized_entropy',
               'competitive_rating': False, 'distribution': {}, 'event_ids': [],
               'source_sha256': None, 'excluded': {},
               'formula': '-sum(p*ln(p))/ln(9); display = 100*value'}
    profile['receiving_heatmap'] = receiving_heatmap([])
    if not path.is_file():
        return profile
    raw = path.read_bytes()
    profile['source_sha256'] = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        profile['status'] = 'invalid_source'
        return profile
    if (not isinstance(document, dict) or document.get('schema_version') != EVENT_VERSION
            or document.get('source') != SOURCE or not isinstance(document.get('events'), list)):
        profile['status'] = 'unsupported_source'
        return profile
    counts, excluded, ids, seen = Counter(), Counter(), [], set()
    accepted = []
    for event in document['events']:
        if not isinstance(event, dict) or event.get('actor_track_id') != track_id:
            continue
        event_id = event.get('event_id')
        if not isinstance(event_id, str) or not event_id.strip():
            excluded['missing_event_id'] += 1
            continue
        if event_id in seen:
            excluded['duplicate_event'] += 1
            continue
        seen.add(event_id)
        if (not isinstance(event.get('receiver_track_id'), str)
                or not event['receiver_track_id'] or event['receiver_track_id'] == track_id
                or any(event.get(key) is not True for key in
                       ('actor_identity_verified', 'receiver_identity_verified', 'opponent_verified', 'calibration_verified'))
                or event.get('contact_status') != 'confirmed' or event.get('receiver_contact_status') != 'confirmed'
                or event.get('target_kind') != 'receiver_contact_position'):
            excluded['unverified_pair'] += 1
            continue
        start, end = event.get('source_time_sec'), event.get('receiver_time_sec')
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end
                or duration_sec is not None and duration_sec > 0 and end > duration_sec):
            excluded['invalid_source_time'] += 1
            continue
        zone = event.get('receiver_zone')
        if type(zone) is not int or not 0 <= zone < 9:
            excluded['invalid_receiving_region'] += 1
            continue
        counts[zone] += 1
        ids.append(event_id)
        accepted.append(event)
    samples = sum(counts.values())
    ready = samples >= MIN_SAMPLES
    value = -sum((count / samples) * math.log(count / samples) for count in counts.values()) / math.log(9) if ready else None
    profile.update(status='observed' if ready else 'insufficient_verified_samples',
                   receiving_heatmap=receiving_heatmap(accepted),
                   sample_count=samples, distribution={str(k): v for k, v in sorted(counts.items())},
                   event_ids=ids, excluded=dict(excluded),
                   value=round(max(0.0, min(1.0, value)), 4) if value is not None else None,
                   normalized_value=round(max(0.0, min(100.0, value * 100)), 1) if value is not None else None)
    return profile
