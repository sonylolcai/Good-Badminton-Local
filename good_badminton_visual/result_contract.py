"""Reject malformed or foreign evidence before any event/metric publication."""
import math
import re


class ResultContractError(ValueError):
    def __init__(self,code='invalid_result_structure'):
        self.code=code
        super().__init__(code)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate_page(page,expected):
    if not isinstance(page,dict) or not isinstance(page.get('events'),list):raise ResultContractError()
    if page.get('analysis_session_id')!=expected:raise ResultContractError('result_session_mismatch')
    if page.get('schema_version')!='stream-session.v1':raise ResultContractError()
    if page.get('next_cursor') is not None and not isinstance(page['next_cursor'],str):raise ResultContractError()
    for event in page['events']:
        if not isinstance(event,dict):raise ResultContractError()
        if event.get('analysis_session_id')!=expected:raise ResultContractError('result_session_mismatch')
        if event.get('schema_version')!='stream-session.v1' or not isinstance(event.get('event_id'),str) or not event['event_id']:
            raise ResultContractError()
        if not finite(event.get('source_time_sec')) or event['source_time_sec']<0 or type(event.get('segment_index')) is not int or event['segment_index']<0:
            raise ResultContractError()
        if not isinstance(event.get('data'),dict):raise ResultContractError()
        if event.get('event_type')=='person_observation':
            track=event['data'].get('track')
            if not isinstance(track,dict) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',str(track.get('track_id') or '')):
                raise ResultContractError()
            if track.get('status') not in ('detected','predicted','missing'):raise ResultContractError()
            point=track.get('court_xy_m')
            if point is not None and (not isinstance(point,(list,tuple)) or len(point)!=2 or not all(finite(x) for x in point)):
                raise ResultContractError()
            if track.get('status')=='detected' and point is None:raise ResultContractError()
    return page


def validate_metrics(metrics):
    if not isinstance(metrics,dict) or not isinstance(metrics.get('players'),list) or not metrics['players']:
        raise ResultContractError('empty_player_result')
    if len(metrics['players'])>4:raise ResultContractError('invalid_player_count')
    ids=[]
    def walk(v):
        if isinstance(v,float) and not math.isfinite(v):raise ResultContractError()
        if isinstance(v,dict):
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(metrics)
    for p in metrics['players']:
        if not isinstance(p,dict) or not isinstance(p.get('track_id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',p['track_id']):raise ResultContractError()
        movement=p.get('movement')
        coverage=p.get('measurement_coverage')
        if not isinstance(movement,dict) or not isinstance(coverage,dict):raise ResultContractError()
        for field in ('distance_m','moving_time_sec','mean_speed_mps','high_intensity_movement_time_sec','direction_change_count'):
            value=movement.get(field)
            if field not in movement or value is not None and (not finite(value) or value<0):raise ResultContractError()
        ratio=coverage.get('usable_measurement_ratio')
        if 'usable_measurement_ratio' not in coverage or ratio is not None and (not finite(ratio) or not 0<=ratio<=1):raise ResultContractError()
        if metrics.get('kind')=='visual_track_movement_metrics':
            if not isinstance(p.get('receiving_profile'),dict) or not isinstance(p.get('combat_radar_profile'),dict):
                raise ResultContractError()
        if 'receiving_profile' in p and p['receiving_profile'] is not None:
            rp=p['receiving_profile']
            if (not isinstance(rp,dict) or rp.get('schema_version')!='personal-receiving-profile.v1'
                    or rp.get('track_id')!=p['track_id'] or rp.get('status') not in ('estimated','not_collected','observed')):
                raise ResultContractError()
        if 'combat_radar_profile' in p and p['combat_radar_profile'] is not None:
            cr=p['combat_radar_profile']
            if (not isinstance(cr,dict) or cr.get('schema_version')!='combat-radar-profile.v1'
                    or cr.get('track_id')!=p['track_id'] or cr.get('status') not in ('observed','insufficient_measurements')
                    or not isinstance(cr.get('dimensions'),list) or len(cr['dimensions'])!=6):
                raise ResultContractError()
        if 'shot_route_profile' in p and isinstance(p['shot_route_profile'],dict) and p['shot_route_profile'].get('source')=='shot_candidates_receiving_region.v1':
            sr=p['shot_route_profile']
            n=sr.get('sample_count');hits=sr.get('hit_candidate_count');val=sr.get('value');norm=sr.get('normalized_value')
            if (sr.get('schema_version')!='shot-route-profile.v1' or sr.get('track_id')!=p['track_id']
                    or sr.get('metric')!='opponent_receiving_region_diversity'
                    or sr.get('status') not in ('estimated','insufficient_candidate_pairs')
                    or type(n) is not int or n<0 or type(hits) is not int or hits<0
                    or sr.get('competitive_rating') is not False or 'events' in sr or 'event_ids' in sr):
                raise ResultContractError()
            if sr.get('status')=='estimated':
                if n<1 or not finite(val) or not 0<=val<=1 or not finite(norm) or not 0<=norm<=100 or abs(norm-val*100)>.15:
                    raise ResultContractError()
            elif n!=0 or val is not None or norm is not None:
                raise ResultContractError()
            dist=sr.get('distribution') or {}
            if not isinstance(dist,dict) or any(k not in {str(i) for i in range(9)} or type(v) is not int or v<0 for k,v in dist.items()) or sum(dist.values())!=n:
                raise ResultContractError()
            hm=sr.get('receiving_heatmap')
            if (not isinstance(hm,dict) or hm.get('schema_version')!='court-heatmap-grid.v1'
                    or hm.get('coordinate_space')!='opponent_half' or hm.get('rows')!=7 or hm.get('columns')!=6
                    or not isinstance(hm.get('cells'),list) or len(hm['cells'])!=42
                    or hm.get('sample_count')!=n or hm.get('total_route_samples')!=n
                    or sum(int(c.get('count') or 0) for c in hm['cells'] if isinstance(c,dict))!=n):
                raise ResultContractError()
        ids.append(p['track_id'])
    if len(set(ids))!=len(ids):raise ResultContractError()
