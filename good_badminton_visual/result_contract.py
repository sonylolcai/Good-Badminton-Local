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
        ids.append(p['track_id'])
    if len(set(ids))!=len(ids):raise ResultContractError()
