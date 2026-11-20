"""Stage-one rally proposals from ball flight and in-court body motion only.

No hit events, stroke labels, hitter identity, or review-candidate files are read.
Heuristic intervals are review proposals, never score or physical landing truth.
"""
import hashlib
import json
import math
from pathlib import Path
from bisect import bisect_left, bisect_right
from .court_contact_gate import valid_quad, person_court_gate
from .rally_activity import find_settle_terminals

VERSION = 'rally-segmentation.stage1.v2.2'
POLICY = {'sample_step_sec': .1, 'flight_window_sec': .5, 'flight_min_points': 3,
          'flight_min_path_px': 55, 'flight_min_speed_px_s': 100,
          'flight_max_speed_px_s': 2500, 'flight_max_step_gap_sec': .22,
          'quiet_end_sec': .8, 'body_supported_gap_sec': 1.8,
          'slow_cluster_sec': .45, 'slow_cluster_radius_px': 18,
          'arm_reaction_body_heights_sec': 1.3, 'minimum_interval_sec': .35,
          'restart_guard_sec': .25, 'bridge_unconfirmed_gap_sec': 3., 'video_entry_tolerance_sec': 2.0,
          'bilateral_min_reaction_samples': 2, 'gap_ready_coverage_min': .3,
          'gap_pickup_coverage_max': .2, 'gap_context_sec': .6}


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def point(v):
    return isinstance(v, (list, tuple)) and len(v) >= 2 and all(finite(x) for x in v[:2])



def posture_evidence(kp, scores, height):
    """Coarse readiness/pickup evidence, not a stroke or contact classifier."""
    needed=(0,5,6,11,12,13,14,15,16)
    if len(kp)<17 or any(not point(kp[i]) for i in needed):
        return {'ready': None, 'pickup': None}
    if scores and (len(scores)<17 or any(not finite(scores[i]) or scores[i]<.4 for i in needed)):
        return {'ready': None, 'pickup': None}
    shoulder=sum(kp[i][1] for i in (5,6))/2
    hip=sum(kp[i][1] for i in (11,12))/2
    torso=(hip-shoulder)/height
    pickup=torso<.12 or kp[0][1]>hip-.08*height
    def angle(a,b,c):
        u=[a[i]-b[i] for i in (0,1)];v=[c[i]-b[i] for i in (0,1)]
        length=math.hypot(*u)*math.hypot(*v)
        return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(u,v))/length)))) if length>1e-8 else 180
    knees=[angle(kp[a],kp[b],kp[c]) for a,b,c in [(11,13,15),(12,14,16)]]
    width=abs(kp[15][0]-kp[16][0])/height
    ready=not pickup and width>=.20 and max(knees)<170
    return {'ready':ready,'pickup':pickup}


def body_activity(rows, polygon):
    """Anonymous motion support; temporal track keys are not hitter labels."""
    previous, result = {}, []
    for row in sorted(rows, key=lambda r: r['time_sec']):
        t = row['time_sec']
        sides = [0., 0.]
        measured_people = 0
        measured_halves=[False,False];fallback=[False,False]
        ready=[False,False];pickup=[False,False];posture_known=[False,False]
        for person in (row.get('spatial') or {}).get('tracks') or []:
            if person_court_gate(person, polygon)['status'] != 'inside':
                continue
            xy = person.get('court_xy_m')
            if not point(xy) or not 0 <= xy[0] <= 6.1 or not 0 <= xy[1] <= 13.4:
                continue
            measured_people += 1
            side = 0 if xy[1] < 6.7 else 1
            measured_halves[side]=True
            pose = person.get('pose') or {}
            loc = person.get('location_evidence') or {}
            kp, bbox = pose.get('keypoints_image') or [], loc.get('bbox_xyxy')
            if pose.get('is_current_measurement') is False or len(kp) < 7 or not point(kp[5]) or not point(kp[6]):
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(finite(v) for v in bbox) or bbox[3] - bbox[1] < 20:
                continue
            shoulder = [(kp[5][i] + kp[6][i]) / 2 for i in (0, 1)]
            posture=posture_evidence(kp,pose.get('keypoint_scores') or [],bbox[3]-bbox[1])
            if posture['ready'] is not None:
                posture_known[side]=True
                ready[side]=ready[side] or posture['ready']
                pickup[side]=pickup[side] or posture['pickup']
            hands = {k: [v[i] - shoulder[i] for i in (0, 1)]
                     for k, v in (loc.get('hands_image') or {}).items() if point(v)}
            key = person.get('track_id')
            prior = previous.get(key) if key else None
            if prior and 0 < t - prior[0] <= .16:
                speeds = [math.dist(v, prior[1][k]) / (t - prior[0]) / (bbox[3] - bbox[1])
                          for k, v in hands.items() if k in prior[1]]
                if posture['pickup'] is not True:
                    sides[side] = max([sides[side], *speeds])
            if key:
                previous[key] = (t, hands)
        # A locked roster is not a prerequisite for rally segmentation.
        # Legacy half observations contain current measured hands/feet even
        # when a visible person was not reassociated with a durable track.
        # Use them only for halves without a measured roster member; never
        # assign an athlete/hitter identity and never use predicted positions.
        for player in (row.get('players') or {}).values():
            if not isinstance(player,dict):
                continue
            evidence=player.get('position_evidence') or {}
            xy=player.get('court');image=player.get('image');bbox=evidence.get('bbox_xyxy')
            if evidence.get('status')!='detected' or not point(xy) or not point(image):
                continue
            if not 0<=xy[0]<=6.1 or not 0<=xy[1]<=13.4:
                continue
            side=0 if xy[1]<6.7 else 1
            if measured_halves[side] or not isinstance(bbox,(list,tuple)) or len(bbox)!=4 or not all(finite(v) for v in bbox) or bbox[3]-bbox[1]<20:
                continue
            person={'status':'detected','image_xy':image,'location_evidence':{'confidence':evidence.get('confidence')}}
            if person_court_gate(person,polygon)['status']!='inside' or not finite(evidence.get('person_confidence')) or evidence['person_confidence']<.5:
                continue
            measured_people+=1
            center=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2];height=bbox[3]-bbox[1]
            hands={k:[v[i]-center[i] for i in (0,1)] for k,v in (player.get('hands') or {}).items() if point(v)}
            key=('measured_half',side);prior=previous.get(key)
            if prior and 0<t-prior[0]<=.16 and math.dist(xy,prior[2])/(t-prior[0])<=6:
                speeds=[math.dist(v,prior[1][k])/(t-prior[0])/height for k,v in hands.items() if k in prior[1]]
                sides[side]=max([sides[side],*speeds]);fallback[side]=bool(speeds)
            previous[key]=(t,hands,list(xy))
        result.append({'time_sec': t, 'arm_motion_by_half': sides, 'measured_in_court_people': measured_people,
                       'ready_by_half':ready,'pickup_by_half':pickup,'posture_known_by_half':posture_known,'unassigned_person_motion_by_half':fallback})
    return result


def segment_rallies(track_rows, person_rows, polygon, *, video_end_sec, pixel_scale=1, manual_terminals=None):
    if not valid_quad(polygon):
        raise ValueError('Stage-one rally segmentation requires valid four-corner calibration')
    if not finite(video_end_sec) or video_end_sec <= 0 or not finite(pixel_scale) or pixel_scale <= 0:
        raise ValueError('Invalid video duration or pixel scale')
    raw = sorted((r for r in track_rows if r.get('status') == 'detected' and finite(r.get('time_sec'))
                  and 0 <= r['time_sec'] <= video_end_sec and point(r.get('image_xy'))
                  and finite(r.get('confidence')) and r['confidence'] >= .35), key=lambda r: r['time_sec'])
    # Airborne ball points need room above the floor quad. Exclude only clear
    # lateral spill; never crop the flight to the ground polygon.
    left, right = min(p[0] for p in polygon), max(p[0] for p in polygon)
    margin = (right - left) * .2
    balls = [r for r in raw if left - margin <= r['image_xy'][0] <= right + margin]
    times = [r['time_sec'] for r in balls]
    people = body_activity([r for r in person_rows if finite(r.get('time_sec'))], polygon)
    person_times = [r['time_sec'] for r in people]
    auto_terminals = find_settle_terminals(balls, person_rows, polygon, pixel_scale)
    manual = [dict(t) for t in manual_terminals or [] if t.get('source') in ('human_review', 'user_reported_interval')
              and finite(t.get('time_sec')) and 0 <= t['time_sec'] <= video_end_sec]
    ends = sorted([t for t in auto_terminals if not any(abs(t['time_sec'] - h['time_sec']) <= .6 for h in manual)] + manual,
                  key=lambda r: r['time_sec'])
    frames = []
    for i, ball in enumerate(balls):
        t = ball['time_sec']
        window = balls[max(0, bisect_left(times, t - POLICY['flight_window_sec'])):i+1]
        # A flight needs a contiguous measured run ending at THIS point.
        # Summing disconnected links or old movement permits jitter/held-ball
        # tails to trigger a fresh start after an actual terminal.
        links=[]
        for a,b in zip(window,window[1:]):
            dt=b['time_sec']-a['time_sec']
            distance=math.dist(a['image_xy'],b['image_xy'])
            speed=distance/dt/pixel_scale if dt>0 else math.inf
            if 0<dt<=POLICY['flight_max_step_gap_sec'] and POLICY['flight_min_speed_px_s']<=speed<=POLICY['flight_max_speed_px_s']:
                links.append((a,b,distance))
            else:
                links=[]
        linked_points=[links[0][0]]+[v[1] for v in links] if links else []
        span=max((math.dist(a['image_xy'],b['image_xy']) for a in linked_points for b in linked_points),default=0)
        supported=len(linked_points)>=POLICY['flight_min_points'] and span>=POLICY['flight_min_path_px']*pixel_scale
        frames.append({'time_sec': t, 'flight': supported,
                       'flight_start_sec': links[0][0]['time_sec'] if links else t,
                       'measured_links': len(links)})
    # Low-motion runs end at their first stable point. This is explicitly not
    # labelled as a verified landing; held balls and apex ambiguity remain visible.
    slow = []
    for i, ball in enumerate(balls):
        t = ball['time_sec'];cluster = []
        for b in balls[i:bisect_right(times, t + POLICY['slow_cluster_sec'] + .15)]:
            if math.dist(ball['image_xy'], b['image_xy']) > POLICY['slow_cluster_radius_px'] * pixel_scale:
                break
            if cluster and b['time_sec'] - cluster[-1]['time_sec'] > .22:
                break
            cluster.append(b)
        if len(cluster) >= 4 and cluster[-1]['time_sec'] - t >= POLICY['slow_cluster_sec']:
            if not slow or t - slow[-1]['time_sec'] > .8:
                slow.append({'time_sec': t, 'source': 'sustained_low_ball_motion', 'status': 'candidate',
                             'evidence': {'end_sec': cluster[-1]['time_sec'], 'points': len(cluster)}})
    ends = sorted(ends + [t for t in slow if not any(abs(t['time_sec'] - e['time_sec']) < .6 for e in ends)], key=lambda r: r['time_sec'])
    rallies, active_start, last_flight, last_support, end_index = [], None, None, None, 0
    reset_after = None
    warnings = []
    def finish(end, source, evidence=None):
        nonlocal active_start, last_flight, last_support, warnings, reset_after
        reset_after = end
        if active_start is None:
            return
        if end > active_start and end - active_start >= POLICY['minimum_interval_sec']:
            support = people[bisect_left(person_times, active_start):bisect_right(person_times, end)]
            reactions = [sum(p['arm_motion_by_half'][side] >= POLICY['arm_reaction_body_heights_sec'] for p in support) for side in (0,1)]
            checks = list(warnings)
            if any(any(p.get('unassigned_person_motion_by_half',[])) for p in support):
                checks.append('current_unassigned_person_motion_used_without_hitter_identity')
            if end - active_start < 1.5:
                checks.append('short_flight_may_be_serve_or_between_rally_return')
            if not all(reactions):
                checks.append('bilateral_body_reaction_not_observed')
            if source == 'sustained_low_ball_motion':
                checks.append('slow_ball_may_be_held_or_apex_not_proven_landing')
            rallies.append({'rally_id': f'stage1_rally_{len(rallies)+1:04d}', 'start_time_sec': round(active_start,6),
                            'end_time_sec': round(end,6), 'duration_sec': round(end-active_start,6),
                            'status': 'needs_review', 'start_reason': 'measured_ball_flight_with_in_court_people',
                            'end_reason': source, 'terminal': evidence, 'review_warnings': checks,
                            'body_reaction_samples_by_half': reactions, 'hitter_analysis_performed': False,
                            'contact_analysis_performed': False})
        active_start, last_flight, last_support, warnings = None, None, None, []
    # Drive on real video time, not just observed points, so a missing-ball tail
    # and a video containing no ball samples still have explicit coverage.
    frame_times = [r['time_sec'] for r in frames]
    for step in range(int(math.ceil(video_end_sec / .1)) + 1):
        t = min(step * .1, video_end_sec)
        while end_index < len(ends) and ends[end_index]['time_sec'] <= t:
            terminal = ends[end_index];finish(terminal['time_sec'], terminal['source'], terminal);end_index += 1
        recent = frames[bisect_left(frame_times,t-.100001):bisect_right(frame_times,t)]
        body = people[bisect_left(person_times,t-.4):bisect_right(person_times,t)]
        measured = max((p['measured_in_court_people'] for p in body),default=0)
        strong_body = any(max(p['arm_motion_by_half']) >= POLICY['arm_reaction_body_heights_sec']
                          and not any(p['pickup_by_half']) for p in body)
        flight = [f for f in recent if f['flight'] and (reset_after is None or
                  (f['flight_start_sec'] > reset_after + .000001 and f['time_sec'] - reset_after >= POLICY['restart_guard_sec']))]
        if flight and measured:
            if active_start is None:
                boundary = ends[end_index-1]['time_sec'] if end_index else 0
                active_start = max(boundary, flight[0]['flight_start_sec'])
                if not rallies and end_index == 0 and active_start <= POLICY['video_entry_tolerance_sec']:
                    active_start = 0.
                    warnings.append('video_begins_during_or_near_rally')
            last_flight = flight[-1]['time_sec'];last_support = t
        elif active_start is not None:
            if strong_body and last_flight is not None and t - last_flight < POLICY['body_supported_gap_sec']:
                last_support = t
            if last_support is not None and t - last_support >= POLICY['quiet_end_sec']:
                finish(last_flight, 'ball_flight_evidence_ended_not_proven_landing')
        if t == video_end_sec:
            break
    if active_start is not None:
        finish(video_end_sec,'video_end_natural_terminal_unconfirmed')
    # A lost-ball gap alone is weak. Bridge it when in-court body motion
    # continues; retain that uncertainty for a reviewer instead of inventing a
    # physical terminal event. Explicit low-motion/manual terminals are not merged.
    merged = []
    for rally in rallies:
        prior = merged[-1] if merged else None
        gap_people = people[bisect_left(person_times, prior['end_time_sec']):bisect_right(person_times, rally['start_time_sec'])] if prior else []
        context=people[bisect_left(person_times, prior['end_time_sec']-POLICY['gap_context_sec']):bisect_right(person_times,rally['start_time_sec']+POLICY['gap_context_sec'])] if prior else []
        readiness=sum(all(p['ready_by_half']) for p in gap_people)/len(gap_people) if gap_people else 0
        pickups=sum(any(p['pickup_by_half']) for p in gap_people)/len(gap_people) if gap_people else 0
        reactions=[sum(p['arm_motion_by_half'][side]>=POLICY['arm_reaction_body_heights_sec'] for p in context) for side in (0,1)]
        if (prior and prior['end_reason'] == 'ball_flight_evidence_ended_not_proven_landing'
                and rally['start_time_sec'] - prior['end_time_sec'] <= POLICY['bridge_unconfirmed_gap_sec']
                and not any(prior['end_time_sec'] < e['time_sec'] <= rally['start_time_sec'] for e in ends)
                and readiness>=POLICY['gap_ready_coverage_min'] and pickups<=POLICY['gap_pickup_coverage_max']
                and all(n>=POLICY['bilateral_min_reaction_samples'] for n in reactions)):
            prior.setdefault('bridged_unknown_ball_gaps', []).append({'start_time_sec':prior['end_time_sec'], 'end_time_sec':rally['start_time_sec'],
                         'ready_coverage':round(readiness,3),'pickup_coverage':round(pickups,3),'context_reactions_by_half':reactions})
            prior.update(end_time_sec=rally['end_time_sec'], duration_sec=round(rally['end_time_sec']-prior['start_time_sec'],6),
                         end_reason=rally['end_reason'], terminal=rally['terminal'])
            prior['review_warnings'] = sorted(set(prior['review_warnings'] + rally['review_warnings'] + ['ball_gap_bridged_by_body_activity']))
            prior['body_reaction_samples_by_half'] = [a+b for a,b in zip(prior['body_reaction_samples_by_half'],rally['body_reaction_samples_by_half'])]
        else:
            merged.append(rally)
    rallies = merged
    for index, rally in enumerate(rallies,1):
        rally['rally_id'] = f'stage1_rally_{index:04d}'
        rally['review_priority'] = 'high' if rally['duration_sec'] < 1.5 or not all(rally['body_reaction_samples_by_half']) else 'normal'
    # Keep unsupported activity separate instead of advertising every toss as
    # a rally. Single-shot serves remain recoverable here; duration alone never
    # deletes them. No human exclusion labels are fed into automatic prediction.
    activities=rallies
    primary=[];uncertain=[]
    for r in activities:
        both=all(n>=POLICY['bilateral_min_reaction_samples'] for n in r['body_reaction_samples_by_half'])
        r['classification']='rally_candidate' if both else 'unverified_activity'
        r['classification_evidence']={'reaction_samples_by_half':r['body_reaction_samples_by_half'],
                                      'minimum_samples_each_half':POLICY['bilateral_min_reaction_samples']}
        if both:
            primary.append(r)
        else:
            r['review_priority']='high'
            r['review_warnings']=sorted(set(r['review_warnings']+['may_be_between_rally_action_or_single_shot_serve']))
            uncertain.append(r)
    # Explicit complement prevents silent loss of unclassified video time.
    gaps, cursor = [], 0.
    for rally in rallies:
        if rally['start_time_sec'] > cursor:
            gaps.append({'start_time_sec': cursor, 'end_time_sec': rally['start_time_sec'], 'status':'between_rallies_or_unresolved'})
        cursor = rally['end_time_sec']
    if cursor < video_end_sec:
        gaps.append({'start_time_sec': cursor, 'end_time_sec': video_end_sec, 'status':'between_rallies_or_unresolved'})
    return {'schema_version':VERSION,'stage':'rally_segmentation_only','status':'needs_review','video_duration_sec':video_end_sec,
            'policy':dict(POLICY),'inputs_used':['observed_ball_track','in_court_person_motion','four_corner_calibration'],
            'inputs_not_used':['shot_candidates','hit_events','hitter_review','stroke_labels'],
            'rallies':primary,'uncertain_intervals':uncertain,'activity_interval_count':len(activities),
            'non_rally_or_unresolved_intervals':gaps,'automatic_terminal_candidates':auto_terminals,
            'manual_terminal_constraints':manual,'contact_analysis_performed':False,'hitter_analysis_performed':False,
            'accuracy':'not_measured_no_complete_human_boundary_reference'}


def derivation_provenance(detections_path, *, fps, play_state_path=None):
    def hash_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    detection_file = Path(detections_path)
    inputs = {"detections": hash_file(detection_file)}
    metadata = detection_file.parent / "metadata.json"
    if metadata.is_file():
        source = json.loads(metadata.read_text(encoding="utf-8"))
        calibration = {"video": source.get("video"), "court": source.get("court")}
        encoded = json.dumps(calibration, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        inputs["video_and_calibration"] = hashlib.sha256(encoded).hexdigest()
    if play_state_path is not None and Path(play_state_path).is_file():
        inputs["play_state"] = hash_file(play_state_path)
    sources = ("court_contact_gate.py", "offline_shot_reconstruction.py", "rally_activity.py", "rally_segmentation.py")
    return {
        "input_sha256": inputs,
        "algorithm_sha256": {name: hash_file(Path(__file__).with_name(name)) for name in sources},
        "fps": fps,
    }


def generate_rally_stage_artifacts(detections_path):
    """Export only upstream observations and stage-one proposals, never shots."""
    from .offline_shot_reconstruction import reconstruct_shuttle_track, load_jsonl, write_jsonl
    path=Path(detections_path);directory=path.parent
    metadata=json.loads((directory/'metadata.json').read_text())
    video=metadata['video'];polygon=metadata['court']['corners']
    rows=load_jsonl(path)
    scale=video.get('width',1920)/1920
    tracks=reconstruct_shuttle_track(rows,fps=video.get('fps',30),width=video.get('width',1920),height=video.get('height',1080),px_scale=scale,court_polygon=polygon)
    context=segment_rallies(tracks,rows,polygon,video_end_sec=video['duration_sec'],pixel_scale=scale)
    context['provenance'] = derivation_provenance(path, fps=video.get('fps', 30))
    target=directory/'derived';target.mkdir(parents=True,exist_ok=True)
    track_path=target/'shuttle_tracks_v2.jsonl';rally_path=target/'rallies_stage1.json'
    write_jsonl(track_path,tracks)
    rally_path.write_text(json.dumps(context,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'version':VERSION,'status':'rally_stage_only','tracks_path':str(track_path),'rallies_path':str(rally_path),
            'frame_count':len(tracks),'rally_count':len(context['rallies']),
            'provenance':context['provenance'],
            'uncertain_interval_count':len(context.get('uncertain_intervals',[])),
            'event_count':0,'contact_analysis_performed':False,'hitter_analysis_performed':False}
