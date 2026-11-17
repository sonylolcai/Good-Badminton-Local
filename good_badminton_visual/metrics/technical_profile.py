"""Evidence gates for future video calibration; never competitive level scores."""
from __future__ import annotations
from collections import Counter
import math

VERSION='technical-profile.v1'
MIN_SAMPLES=12
SHOT_TYPES=('clear','smash','drop','drive','lift','net')


def _feature(values, measure):
    ready=len(values)>=MIN_SAMPLES
    return {'status':'observed' if ready else 'insufficient_verified_samples',
            'sample_count':len(values),'value':measure(values) if ready else None,
            'competitive_score':None}


def technical_profile(events=(), *, identity_verified=False, court_calibration_verified=False, flight_speed_calibration_verified=False):
    """Events must bind actor, confirmed contact and observed shot endpoint.

    Estimated landings, unknown actors and proposed shot types remain excluded.
    Physical shuttle speed requires a flight/height model, not floor homography.
    """
    confirmed=[e for e in events if identity_verified is True and e.get('actor_verified') is True and e.get('contact_status')=='confirmed']
    landings=[e['landing_zone'] for e in confirmed if court_calibration_verified is True and e.get('landing_status')=='observed'
              and type(e.get('landing_zone')) is int and 0<=e['landing_zone']<9]
    types=[e['shot_type'] for e in confirmed if e.get('shot_type_status')=='confirmed' and e.get('shot_type') in SHOT_TYPES]
    speeds=[float(e['ball_speed_mps']) for e in confirmed if flight_speed_calibration_verified is True and e.get('ball_speed_status')=='validated'
            and type(e.get('ball_speed_mps')) in (int,float) and math.isfinite(e['ball_speed_mps']) and e['ball_speed_mps']>0]
    def entropy(values, classes):
        counts=Counter(values);n=len(values)
        return round(-sum(count/n*math.log(count/n) for count in counts.values())/math.log(classes),4)
    return {'schema_version':VERSION,'kind':'technical_observation','competitive_rating':False,
            'features':{
                'landing_distribution':_feature(landings,lambda x:{str(k):v for k,v in sorted(Counter(x).items())}),
                'landing_spread':_feature(landings,lambda x:entropy(x,9)),
                'noncentral_landing_ratio':_feature(landings,lambda x:round(sum(z!=4 for z in x)/len(x),4)),
                'ball_speed_mps':_feature(speeds,lambda x:round(sum(x)/len(x),3)),
                'shot_type_diversity':_feature(types,lambda x:entropy(x,len(SHOT_TYPES))),
                'shot_combination':{'status':'awaiting_verified_sequences','value':None,'competitive_score':None},
            },'policy':'Shot variety and spread describe choices, not automatically good technique. No gender/age skill caps.'}
