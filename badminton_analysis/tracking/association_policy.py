"""Shared, auditable person-association thresholds for batch and streaming."""


def association_policy(sport_id="badminton"):
    badminton = sport_id == "badminton"
    return {
        "version": "person_association.v2" if badminton else "person_association.v1",
        "bytetrack_match_cost_limit": 0.70 if badminton else 0.50,
        "recovery_body_min": 0.50 if badminton else 0.65,
        "recovery_location_min": 0.45 if badminton else 0.65,
        "anchor_body_min": 0.55 if badminton else 0.70,
        "anchor_location_min": 0.50 if badminton else 0.70,
        "anchor_identity_min": 0.75 if badminton else 0.80,
        "weak_recovery_max_distance_m": 1.50,
        "weak_recovery_slot_margin_m": 0.35,
        "weak_pair_cost_margin_m": 0.50,
        "direct_key_motion_guard": badminton,
        "direct_key_min_gate_m": 0.80,
        "direct_key_motion_tolerance_m": 0.35,
    }
