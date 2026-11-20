"""Badminton Biomechanical and Tactical Pose Feature Extractor.

Extracts deterministic biomechanical and tactical features from human pose keypoints
(YOLO-Pose 17 joints), player court positions, and shuttle trajectory/hit events.
This module performs NO LLM calls or model training; it provides grounded, auditable
quantitative metrics (angles, velocities, phase lags, stability variances) to feed
the LLM-based coaching evaluation system.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (max(0.0, min(100.0, float(pct))) / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    weight = pos - lo
    return float(s[lo] * (1.0 - weight) + s[hi] * weight)


def _argmax(values: Sequence[float]) -> int:
    if not values:
        return 0
    return max(range(len(values)), key=lambda i: values[i])

# COCO 17 Keypoints Index Definition
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

DEFAULT_MIN_KEYPOINT_CONF = 0.40
DEFAULT_SAMPLING_HZ = 10.0
DEEP_LUNGE_MAX_KNEE_ANGLE_DEG = 110.0
DEEP_LUNGE_MIN_PHYSIOLOGICAL_ANGLE_DEG = 60.0
DEEP_LUNGE_DEBOUNCE_GAP_SEC = 1.0


def compute_joint_angle(
    p_a: Union[List[float], Tuple[float, float], Sequence[float]],
    p_b: Union[List[float], Tuple[float, float], Sequence[float]],
    p_c: Union[List[float], Tuple[float, float], Sequence[float]],
) -> Optional[float]:
    """Compute 2D angle (in degrees, [0, 180]) at vertex p_b formed by segment BA and BC."""
    if p_a is None or p_b is None or p_c is None:
        return None
    try:
        ba_x = float(p_a[0]) - float(p_b[0])
        ba_y = float(p_a[1]) - float(p_b[1])
        bc_x = float(p_c[0]) - float(p_b[0])
        bc_y = float(p_c[1]) - float(p_b[1])

        dot = ba_x * bc_x + ba_y * bc_y
        mag_ba = math.hypot(ba_x, ba_y)
        mag_bc = math.hypot(bc_x, bc_y)

        if mag_ba < 1e-4 or mag_bc < 1e-4:
            return None

        cos_angle = dot / (mag_ba * mag_bc)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return round(math.degrees(math.acos(cos_angle)), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def compute_torso_tilt(
    left_shoulder: Union[List[float], Sequence[float]],
    right_shoulder: Union[List[float], Sequence[float]],
    left_hip: Union[List[float], Sequence[float]],
    right_hip: Union[List[float], Sequence[float]],
) -> Optional[float]:
    """Compute torso axial tilt angle in degrees relative to the vertical line.
    
    0 deg = perfectly upright. Higher values indicate lateral tilt or forward lean.
    """
    if any(p is None for p in (left_shoulder, right_shoulder, left_hip, right_hip)):
        return None
    try:
        sh_x = (float(left_shoulder[0]) + float(right_shoulder[0])) / 2.0
        sh_y = (float(left_shoulder[1]) + float(right_shoulder[1])) / 2.0
        hip_x = (float(left_hip[0]) + float(right_hip[0])) / 2.0
        hip_y = (float(left_hip[1]) + float(right_hip[1])) / 2.0

        dx = sh_x - hip_x
        dy = sh_y - hip_y  # In image coordinates, y increases downward, so dy < 0 when upright

        if math.hypot(dx, dy) < 1e-4:
            return None

        # Angle with vertical axis (dy):
        tilt = math.degrees(math.atan2(abs(dx), abs(dy)))
        return round(tilt, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def compute_knee_flexion(
    hip: Union[List[float], Sequence[float]],
    knee: Union[List[float], Sequence[float]],
    ankle: Union[List[float], Sequence[float]],
) -> Optional[float]:
    """Compute knee flexion angle in degrees (angle H-K-A).
    
    ~180 deg = straight standing leg.
    ~90-120 deg = deep lunge (弓箭步/跨步落地缓冲).
    """
    return compute_joint_angle(hip, knee, ankle)


def compute_elbow_extension(
    shoulder: Union[List[float], Sequence[float]],
    elbow: Union[List[float], Sequence[float]],
    wrist: Union[List[float], Sequence[float]],
) -> Optional[float]:
    """Compute elbow extension angle in degrees (angle S-E-W).
    
    160-180 deg = full overhead arm extension (展臂充分).
    < 135 deg = cramped / bent arm (大臂未完全展开).
    """
    return compute_joint_angle(shoulder, elbow, wrist)


def classify_stroke_hand_side(
    wrist_x: float,
    torso_cx: float,
    shoulder_width: float,
    dominant_hand: str = "right",
    is_near: bool = True,
) -> str:
    """Classify Forehand ('forehand') vs Backhand ('backhand') using biomechanical projection.
    
    For a right-handed player:
      - Near court (facing away from camera):
        Right hand on right side of torso center -> Forehand
        Right hand crossing torso to the left -> Backhand
      - Far court (facing camera):
        Right hand on left side of screen (player's right) -> Forehand
        Right hand crossing to right side of screen -> Backhand
    For a left-handed player, symmetric logic applies.
    """
    if shoulder_width <= 1e-3:
        return "unknown"

    margin = 0.10 * shoulder_width
    is_left_hand = (dominant_hand.lower() == "left")

    if is_near:
        if not is_left_hand:
            # Right hand, near court
            return "forehand" if wrist_x >= torso_cx - margin else "backhand"
        else:
            # Left hand, near court
            return "forehand" if wrist_x <= torso_cx + margin else "backhand"
    else:
        # Far court (facing camera, left/right inverted on screen)
        if not is_left_hand:
            # Right hand, far court
            return "forehand" if wrist_x <= torso_cx + margin else "backhand"
        else:
            # Left hand, far court
            return "forehand" if wrist_x >= torso_cx - margin else "backhand"


def classify_hitting_height(
    wrist_y: float,
    shoulder_y: float,
    torso_cy: float,
    shoulder_width: float,
) -> str:
    """Classify stroke contact height zone: 'overhead', 'waist', or 'underhand'."""
    if shoulder_width <= 1e-3:
        return "unknown"

    # In image coordinates, smaller Y means higher in physical space
    if wrist_y < shoulder_y - 0.15 * shoulder_width:
        return "overhead"
    elif wrist_y > torso_cy + 1.2 * shoulder_width:
        return "underhand"
    else:
        return "waist"


def calculate_entropy(values: List[Any], num_categories: int) -> float:
    """Calculate Shannon entropy normalized to [0.0, 1.0]."""
    if not values or num_categories <= 1:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy_val = -sum((c / total) * math.log(c / total) for c in counts.values())
    max_entropy = math.log(num_categories)
    return round(entropy_val / max_entropy, 4) if max_entropy > 0 else 0.0


def extract_player_biomechanical_metrics(
    player_frames: List[Dict[str, Any]],
    *,
    player_ref: str,
    dominant_hand: str = "right",
    is_near: Optional[bool] = None,
    hit_events: Optional[List[Dict[str, Any]]] = None,
    sample_hz: float = DEFAULT_SAMPLING_HZ,
) -> Dict[str, Any]:
    """Extract full biomechanical and kinetic profile for a specific player across frames."""
    if not player_frames:
        return {
            "player_ref": player_ref,
            "status": "insufficient_data",
            "frame_count": 0,
            "confidence_ratio": 0.0,
        }

    is_left_handed = (dominant_hand.lower() == "left")
    hitting_sh_idx = KP_LEFT_SHOULDER if is_left_handed else KP_RIGHT_SHOULDER
    hitting_el_idx = KP_LEFT_ELBOW if is_left_handed else KP_RIGHT_ELBOW
    hitting_wr_idx = KP_LEFT_WRIST if is_left_handed else KP_RIGHT_WRIST

    non_hitting_sh_idx = KP_RIGHT_SHOULDER if is_left_handed else KP_LEFT_SHOULDER
    non_hitting_el_idx = KP_RIGHT_ELBOW if is_left_handed else KP_LEFT_ELBOW
    non_hitting_wr_idx = KP_RIGHT_WRIST if is_left_handed else KP_LEFT_WRIST

    torso_tilts = []
    elbow_extensions = []
    knee_flexions = []
    height_counts = Counter()
    hand_counts = Counter()

    time_series_data = []
    valid_frames_count = 0

    # Auto-detect near/far court side if not provided
    if is_near is None:
        y_coords = []
        for f in player_frames:
            box = f.get("box") or []
            if len(box) >= 4:
                y_coords.append(box[1])
        if y_coords:
            avg_y = sum(y_coords) / len(y_coords)
            is_near = (avg_y > 1180.0)  # Standard baseline threshold in court camera
        else:
            is_near = True

    for item in player_frames:
        kp = item.get("keypoints")
        conf = item.get("keypoint_confidences")
        t_sec = float(item.get("source_pts_sec") or item.get("time_sec") or 0.0)

        if not kp or len(kp) < 17:
            continue

        # Check keypoint confidences if provided
        def get_kp(idx):
            if conf and len(conf) > idx and conf[idx] < DEFAULT_MIN_KEYPOINT_CONF:
                return None
            pt = kp[idx]
            if len(pt) >= 2 and (pt[0] > 0 or pt[1] > 0):
                return pt[:2]
            return None

        ls = get_kp(KP_LEFT_SHOULDER)
        rs = get_kp(KP_RIGHT_SHOULDER)
        lh = get_kp(KP_LEFT_HIP)
        rh = get_kp(KP_RIGHT_HIP)

        # 1. Torso stability
        tilt = compute_torso_tilt(ls, rs, lh, rh)
        if tilt is not None:
            torso_tilts.append(tilt)

        # 2. Elbow extension of hitting arm
        sh = get_kp(hitting_sh_idx)
        el = get_kp(hitting_el_idx)
        wr = get_kp(hitting_wr_idx)
        elbow_angle = compute_elbow_extension(sh, el, wr)
        if elbow_angle is not None:
            elbow_extensions.append(elbow_angle)

        # 3. Knee flexion (measure both legs, take active/lunge flexion)
        lk_angle = compute_knee_flexion(lh, get_kp(KP_LEFT_KNEE), get_kp(KP_LEFT_ANKLE))
        rk_angle = compute_knee_flexion(rh, get_kp(KP_RIGHT_KNEE), get_kp(KP_RIGHT_ANKLE))
        angles = [a for a in (lk_angle, rk_angle) if a is not None]
        if angles:
            knee_flexions.append((t_sec, min(angles)))  # More flexed leg represents lunge or squat

        # 4. Hitting height & Forehand/Backhand (use anatomical torso length to avoid sideways collapse)
        if ls and rs and wr:
            sh_cx = (ls[0] + rs[0]) / 2.0
            sh_cy = (ls[1] + rs[1]) / 2.0
            horiz_shoulder = abs(rs[0] - ls[0])
            if lh and rh:
                hip_cx = (lh[0] + rh[0]) / 2.0
                hip_cy = (lh[1] + rh[1]) / 2.0
                torso_len = math.hypot(sh_cx - hip_cx, sh_cy - hip_cy)
                torso_cy = (sh_cy + hip_cy) / 2.0
            else:
                torso_len = horiz_shoulder
                torso_cy = sh_cy
            anatomical_scale = max(torso_len, horiz_shoulder)
            shoulder_y = min(ls[1], rs[1])

            height = classify_hitting_height(wr[1], shoulder_y, torso_cy, anatomical_scale)
            if height != "unknown":
                height_counts[height] += 1

            side = classify_stroke_hand_side(wr[0], sh_cx, anatomical_scale, dominant_hand, is_near)
            if side != "unknown":
                hand_counts[side] += 1

        # Record temporal angles for kinetic chain analysis
        if ls and rs and lh and rh:
            hip_angle = math.atan2(rh[1] - lh[1], rh[0] - lh[0])
            sh_angle = math.atan2(rs[1] - ls[1], rs[0] - ls[0])
            wr_pt = wr if wr is not None else [0.0, 0.0]
            time_series_data.append({
                "t": t_sec,
                "hip_angle": hip_angle,
                "sh_angle": sh_angle,
                "wr_pt": wr_pt,
            })
            valid_frames_count += 1

    # Kinetic Chain Phase Lag Analysis
    kinetic_chain_lag_ms = None
    kinetic_coordination_rating = "unknown"
    if len(time_series_data) >= 8:
        # Compute angular velocity derivatives
        hip_w = []
        sh_w = []
        wr_v = []
        for i in range(1, len(time_series_data)):
            dt = max(1e-3, time_series_data[i]["t"] - time_series_data[i-1]["t"])
            hw = abs(time_series_data[i]["hip_angle"] - time_series_data[i-1]["hip_angle"]) / dt
            sw = abs(time_series_data[i]["sh_angle"] - time_series_data[i-1]["sh_angle"]) / dt
            p1 = time_series_data[i]["wr_pt"]
            p0 = time_series_data[i-1]["wr_pt"]
            wv = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt
            hip_w.append(hw)
            sh_w.append(sw)
            wr_v.append(wv)

        if hip_w and sh_w and wr_v:
            hip_peak_idx = _argmax(hip_w)
            sh_peak_idx = _argmax(sh_w)
            wr_peak_idx = _argmax(wr_v)

            # Proper sequence: Hip initiates -> Shoulder follows -> Wrist accelerates through
            hip_t = time_series_data[hip_peak_idx]["t"]
            sh_t = time_series_data[sh_peak_idx]["t"]
            wr_t = time_series_data[wr_peak_idx]["t"]

            lag_sh_hip = (sh_t - hip_t) * 1000.0  # ms
            lag_wr_sh = (wr_t - sh_t) * 1000.0   # ms

            # If lag is within biomechanical window (20ms - 150ms), sequence is coordinated
            if 15.0 <= lag_sh_hip <= 250.0 and -50.0 <= lag_wr_sh <= 200.0:
                kinetic_coordination_rating = "fluid_kinetic_whip"
                kinetic_chain_lag_ms = round(lag_sh_hip, 1)
            elif lag_sh_hip < 0:
                kinetic_coordination_rating = "upper_body_dominant"
                kinetic_chain_lag_ms = round(lag_sh_hip, 1)
            else:
                kinetic_coordination_rating = "acceptable_coordination"
                kinetic_chain_lag_ms = round(lag_sh_hip, 1)

    # Core stability stats
    mean_torso_tilt = round(_mean(torso_tilts), 2) if torso_tilts else 0.0
    torso_tilt_std = round(_std(torso_tilts), 2) if len(torso_tilts) > 1 else 0.0

    # Arm extension stats
    mean_elbow_ext = round(_mean(elbow_extensions), 2) if elbow_extensions else 0.0
    p90_elbow_ext = round(_percentile(elbow_extensions, 90), 2) if elbow_extensions else 0.0

    # Knee flexion stats (with debounced deep lunge event counting, gap >= 1.0s, physiological filter [60°, 110°])
    knee_angle_vals = [a for _, a in knee_flexions]
    valid_knee_vals = [a for a in knee_angle_vals if a >= DEEP_LUNGE_MIN_PHYSIOLOGICAL_ANGLE_DEG]
    deep_lunge_ratio = (
        round(sum(1 for a in valid_knee_vals if a <= DEEP_LUNGE_MAX_KNEE_ANGLE_DEG) / len(valid_knee_vals), 3)
        if valid_knee_vals else 0.0
    )
    mean_knee_flex = round(_mean(valid_knee_vals), 2) if valid_knee_vals else 0.0
    min_knee_flex = round(float(min(valid_knee_vals)), 1) if valid_knee_vals else None
    deep_lunge_count = 0
    last_lunge_t = -999.0
    for t_val, k_ang in knee_flexions:
        if DEEP_LUNGE_MIN_PHYSIOLOGICAL_ANGLE_DEG <= k_ang <= DEEP_LUNGE_MAX_KNEE_ANGLE_DEG:
            if t_val - last_lunge_t >= DEEP_LUNGE_DEBOUNCE_GAP_SEC:
                deep_lunge_count += 1
            last_lunge_t = t_val

    # Height and Hand ratios
    total_height = sum(height_counts.values()) or 1
    overhead_ratio = round(height_counts["overhead"] / total_height, 3)
    waist_ratio = round(height_counts["waist"] / total_height, 3)
    underhand_ratio = round(height_counts["underhand"] / total_height, 3)

    total_hand = sum(hand_counts.values()) or 1
    forehand_ratio = round(hand_counts["forehand"] / total_hand, 3)
    backhand_ratio = round(hand_counts["backhand"] / total_hand, 3)

    return {
        "player_ref": player_ref,
        "is_near_court": is_near,
        "dominant_hand": dominant_hand,
        "sample_count": len(player_frames),
        "valid_frames_count": valid_frames_count,
        "core_stability": {
            "mean_torso_tilt_deg": mean_torso_tilt,
            "torso_tilt_std_deg": torso_tilt_std,
            "stability_rating": "high_stability" if torso_tilt_std <= 8.5 else ("moderate_stability" if torso_tilt_std <= 14.0 else "unstable_core"),
        },
        "arm_mechanics": {
            "mean_elbow_extension_deg": mean_elbow_ext,
            "peak_overhead_extension_deg": p90_elbow_ext,
            "extension_adequacy": "fully_extended" if p90_elbow_ext >= 155.0 else ("adequate" if p90_elbow_ext >= 138.0 else "cramped_stroke"),
        },
        "lunge_mechanics": {
            "mean_knee_flexion_deg": mean_knee_flex,
            "min_knee_flexion_deg": min_knee_flex,
            "deep_lunge_count": deep_lunge_count,
            "deep_lunge_ratio": deep_lunge_ratio,
            "knee_cushion_rating": "resilient_cushion" if deep_lunge_ratio >= 0.25 else "shallow_lunge",
        },
        "stroke_posture_distribution": {
            "overhead_ratio": overhead_ratio,
            "waist_ratio": waist_ratio,
            "underhand_ratio": underhand_ratio,
            "forehand_ratio": forehand_ratio,
            "backhand_ratio": backhand_ratio,
        },
        "kinetic_chain": {
            "phase_lag_ms": kinetic_chain_lag_ms,
            "coordination_rating": kinetic_coordination_rating,
        },
    }


def extract_shot_tactical_metrics(
    routes: List[Dict[str, Any]],
    player_ref: str,
) -> Dict[str, Any]:
    """Extract shot selection, trajectory depth, and landing entropy for a player."""
    player_routes = []
    for r in routes:
        anchor = r.get("launch_anchor") or {}
        actor = anchor.get("actor") or r.get("actor")
        if actor == player_ref:
            player_routes.append(r)

    if not player_routes:
        return {
            "shot_count": 0,
            "deep_court_ratio": 0.0,
            "landing_entropy": 0.0,
            "shot_types": {},
            "speed_mps": {},
        }

    shot_types = Counter()
    depth_categories = []
    speeds = []

    for r in player_routes:
        # Stroke type
        m = r.get("metrics") or {}
        stype = m.get("stroke_type") or r.get("shot_type") or "rally_shot"
        shot_types[stype] += 1

        # Landing depth (based on end_image_xy or target coordinates)
        end_xy = r.get("end_image_xy") or m.get("end_image_xy")
        if end_xy and len(end_xy) >= 2:
            # In vertical court projection, distance from net represents depth
            # Simplified depth: Net (< 0.35), Mid (0.35 - 0.70), Deep (> 0.70)
            # Use y coordinate normalization or displacement
            depth_categories.append("deep" if end_xy[1] > 1400 or end_xy[1] < 1000 else "mid_or_net")

        # Speed samples
        speed_list = m.get("speed_samples") or []
        for s in speed_list:
            v_px = s.get("image_displacement_px_per_s")
            if v_px and v_px > 0:
                speeds.append(float(v_px))

    total_shots = len(player_routes)
    deep_count = sum(1 for d in depth_categories if d == "deep")
    deep_ratio = round(deep_count / total_shots, 3) if total_shots else 0.0

    landing_entropy = calculate_entropy(depth_categories, 2)

    return {
        "shot_count": total_shots,
        "deep_court_ratio": deep_ratio,
        "landing_entropy": landing_entropy,
        "shot_types": dict(shot_types),
        "mean_pixel_speed_pps": round(_mean(speeds), 1) if speeds else 0.0,
        "peak_pixel_speed_pps": round(_percentile(speeds, 95), 1) if speeds else 0.0,
    }


def extract_match_technical_features(
    pose_frames: List[Dict[str, Any]],
    routes: Optional[List[Dict[str, Any]]] = None,
    hit_events: Optional[List[Dict[str, Any]]] = None,
    player_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate all player biomechanical, footwork, and tactical features into one match manifest."""
    routes = routes or []
    hit_events = hit_events or []
    player_configs = player_configs or {}

    # Group pose frames by player ref or track_id
    frames_by_player: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in pose_frames:
        players = f.get("players") or f.get("near_players", []) + f.get("far_players", [])
        f_idx = f.get("frame_index") or f.get("frame")
        t_sec = f.get("source_pts_sec") or f.get("time_sec")
        for p in players:
            ref = str(p.get("player_ref") or p.get("track_id") or "")
            if not ref:
                continue
            item = dict(p)
            item["frame_index"] = f_idx
            item["source_pts_sec"] = t_sec
            frames_by_player[ref].append(item)

    players_profile = {}
    for ref, pframes in sorted(frames_by_player.items()):
        cfg = player_configs.get(ref) or {}
        hand = cfg.get("dominant_hand") or "right"
        is_near = cfg.get("is_near")

        biomech = extract_player_biomechanical_metrics(
            pframes,
            player_ref=ref,
            dominant_hand=hand,
            is_near=is_near,
            hit_events=hit_events,
        )
        shots = extract_shot_tactical_metrics(routes, ref)

        players_profile[ref] = {
            "biomechanics": biomech,
            "shot_tactics": shots,
        }

    return {
        "schema_version": "match-technical-features.v1",
        "total_analyzed_frames": len(pose_frames),
        "player_count": len(players_profile),
        "players": players_profile,
    }


TACTICAL_JOINT_SCHEMA_VERSION = "tactical-and-joint-profile.v1"
MIN_FATIGUE_VIDEO_DURATION_SEC = 600.0
MIN_FATIGUE_MOVING_TIME_SEC = 180.0


def _analyze_coco17_frame(kps: Any, scores: Any, min_conf: float = 0.35) -> Optional[Dict[str, Any]]:
    if not isinstance(kps, list) or not isinstance(scores, list) or len(kps) < 17 or len(scores) < 17:
        return None
    try:
        if any(not isinstance(p, (list, tuple)) or len(p) < 2
               or any(type(v) not in (int, float) or not math.isfinite(v) for v in p[:2])
               for p in kps[:17]):
            return None
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in scores[:17]):
            return None
        if min(float(scores[5]), float(scores[6]), float(scores[11]), float(scores[12])) < min_conf:
            return None
        ls, rs = kps[5], kps[6]
        lh, rh = kps[11], kps[12]
        sh_mid = ((float(ls[0]) + float(rs[0])) / 2.0, (float(ls[1]) + float(rs[1])) / 2.0)
        hip_mid = ((float(lh[0]) + float(rh[0])) / 2.0, (float(lh[1]) + float(rh[1])) / 2.0)
        torso_len = math.hypot(sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])
        if torso_len < 12.0:
            return None
        wrists = []
        for idx in (9, 10):
            if float(scores[idx]) >= min_conf and isinstance(kps[idx], (list, tuple)) and len(kps[idx]) >= 2:
                wrists.append((float(kps[idx][0]), float(kps[idx][1])))
        rel_y = None
        stroke_zone = None
        if wrists:
            highest_w = min(wrists, key=lambda pt: pt[1])
            rel_y = (sh_mid[1] - highest_w[1]) / torso_len
            if rel_y >= 0.15:
                stroke_zone = "overhead"
            elif rel_y >= -0.85:
                stroke_zone = "drive"
            else:
                stroke_zone = "underhand"
        knees = []
        for h_i, k_i, a_i in ((11, 13, 15), (12, 14, 16)):
            if min(float(scores[h_i]), float(scores[k_i]), float(scores[a_i])) >= min_conf:
                ang = compute_joint_angle(kps[h_i], kps[k_i], kps[a_i])
                if ang is not None:
                    knees.append(ang)
        min_knee = min(knees) if knees else None
        torso_tilt = compute_torso_tilt(ls, rs, lh, rh)
        return {
            "rel_y": rel_y,
            "stroke_zone": stroke_zone,
            "min_knee": min_knee,
            "torso_tilt": torso_tilt,
        }
    except (TypeError, ValueError, IndexError):
        return None


COMBAT_RADAR_SCHEMA_VERSION = "combat-radar-profile.v1"
MIN_CORE_STABILITY_TORSO_SAMPLES = 30
MIN_CORE_STABILITY_SPAN_SEC = 3.0
MIN_CORE_STABILITY_LUNGE_RECOVERIES = 3
MAX_POSE_CONTINUITY_GAP_SEC = 0.60
MAX_TRAJECTORY_JOIN_GAP_SEC = 0.50
HALF_SLOT_SOURCES = {
    "half_court_slot_opening",
    "half_court_slot_continuity",
    "half_court_slot_replacement",
}
TRUSTED_PERSON_SOURCES = {
    "bytetrack",
    "court_association",
    "roster_bootstrap",
    "anchored_roster_opening",
    "anchored_roster_continuity",
    "anchored_roster_appearance",
    "anchored_roster_reference_recovery",
}
CANDIDATE_HIT_SOURCES = {
    "pose_wrist_pair_estimate.v1",
    "shot_candidates_receiving_region.v1",
    "ShuttleEventCandidates",
    "shot_events_v2",
    "candidate-receiving-routes.v1",
}


def _finite_num(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _valid_court_xy(xy: Any) -> bool:
    return (
        isinstance(xy, (list, tuple))
        and len(xy) >= 2
        and _finite_num(xy[0])
        and _finite_num(xy[1])
        and 0.0 <= float(xy[0]) <= 6.1
        and 0.0 <= float(xy[1]) <= 13.4
    )


def _frame_court_end(frame: Dict[str, Any]) -> Optional[str]:
    end = frame.get("court_end")
    if end in ("end_a", "end_b"):
        return str(end)
    xy = frame.get("xy")
    if _valid_court_xy(xy):
        return "end_b" if float(xy[1]) > 6.7 else "end_a"
    return None


def _frame_segment_key(frame: Dict[str, Any]) -> Tuple[str, int, str]:
    scope = str(frame.get("identity_scope") or "")
    seg = frame.get("occupancy_segment") if type(frame.get("occupancy_segment")) is int else 0
    end = _frame_court_end(frame) or ""
    return (scope, seg, end)


def _has_multiple_occupants(frames: Sequence[Dict[str, Any]]) -> bool:
    segments = {
        f.get("occupancy_segment")
        for f in frames
        if isinstance(f, dict) and type(f.get("occupancy_segment")) is int and f.get("occupancy_segment") >= 1
    }
    return len(segments) > 1


def _contiguous_xy_runs(
    frames: Sequence[Dict[str, Any]],
    *,
    max_gap_sec: float = MAX_TRAJECTORY_JOIN_GAP_SEC,
) -> List[List[Dict[str, Any]]]:
    valid = [
        f for f in frames
        if isinstance(f, dict) and _finite_num(f.get("time_sec")) and _valid_court_xy(f.get("xy"))
    ]
    valid.sort(key=lambda item: float(item["time_sec"]))
    runs: List[List[Dict[str, Any]]] = []
    for f in valid:
        if not runs:
            runs.append([f])
            continue
        prev = runs[-1][-1]
        dt = float(f["time_sec"]) - float(prev["time_sec"])
        if dt <= 0.0:
            continue
        if _frame_segment_key(f) != _frame_segment_key(prev) or dt > max_gap_sec:
            runs.append([f])
        else:
            runs[-1].append(f)
    return runs


def _is_verified_hit_event(ev: Any) -> bool:
    if not isinstance(ev, dict):
        return False
    if ev.get("verified") is False:
        return False
    if ev.get("status") not in (None, "verified", "confirmed", "observed"):
        return False
    if ev.get("contact_status") not in (None, "verified", "confirmed", "observed"):
        return False
    if ev.get("source") in CANDIDATE_HIT_SOURCES:
        return False
    if (
        ev.get("identity_scope") == "half_court_slots.v1"
        or ev.get("actor_identity_scope") == "half_court_slots.v1"
        or ev.get("receiver_identity_scope") == "half_court_slots.v1"
    ):
        return False
    for key in ("actor_identity_verified", "receiver_identity_verified", "opponent_verified", "calibration_verified"):
        if key in ev and ev.get(key) is not True:
            return False
    t_val = ev.get("time_sec") if ev.get("time_sec") is not None else ev.get("source_time_sec")
    return _finite_num(t_val) and float(t_val) >= 0.0


def _verified_player_shots(
    player_track_id: str,
    hit_events: Sequence[Dict[str, Any]],
    *,
    min_separation_sec: float = 0.35,
) -> List[Dict[str, Any]]:
    raw = [
        ev for ev in (hit_events or [])
        if _is_verified_hit_event(ev)
        and str(ev.get("track_id") or ev.get("hitter_track_id") or ev.get("actor_track_id")) == str(player_track_id)
    ]
    raw.sort(key=lambda ev: float(ev.get("time_sec") if ev.get("time_sec") is not None else ev.get("source_time_sec") or 0.0))
    deduped: List[Dict[str, Any]] = []
    for ev in raw:
        t_hit = float(ev.get("time_sec") if ev.get("time_sec") is not None else ev.get("source_time_sec") or 0.0)
        if deduped:
            prev_t = float(deduped[-1].get("time_sec") if deduped[-1].get("time_sec") is not None else deduped[-1].get("source_time_sec") or 0.0)
            if t_hit - prev_t < min_separation_sec:
                continue
        deduped.append(ev)
    return deduped


def _shot_time(shot: Dict[str, Any]) -> float:
    t_val = shot.get("time_sec") if shot.get("time_sec") is not None else shot.get("source_time_sec")
    return float(t_val) if _finite_num(t_val) else 0.0


def _resolve_end_in_window(
    frames: Sequence[Dict[str, Any]],
    t_start: float,
    t_end: float,
) -> Optional[str]:
    """Return unique court_end ('end_a'/'end_b') in [t_start, t_end] if unambiguous and single-occupant."""
    ends = set()
    occupants = set()
    for f in frames:
        if not isinstance(f, dict) or not _finite_num(f.get("time_sec")):
            continue
        t_val = float(f["time_sec"])
        if not (t_start <= t_val <= t_end):
            continue
        f_end = _frame_court_end(f)
        if f_end in ("end_a", "end_b"):
            ends.add(f_end)
        occ = f.get("occupancy_segment")
        if type(occ) is int and occ >= 1:
            occupants.add(occ)
    if len(ends) != 1 or len(occupants) > 1:
        return None
    return next(iter(ends))


def _hitter_court_end_at(
    player_track_id: str,
    t_hit: float,
    track_series: Dict[str, List[Dict[str, Any]]],
    shot: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    traj_frames = track_series.get(player_track_id) or []
    traj_end = _resolve_end_in_window(traj_frames, t_hit - 0.5, t_hit + 0.5) if traj_frames else None
    shot_end = str(shot["court_end"]) if isinstance(shot, dict) and shot.get("court_end") in ("end_a", "end_b") else None
    if traj_frames:
        # If player has trajectory frames around t_hit and switched sides/was ambiguous, return None.
        near_any = any(
            isinstance(f, dict) and _finite_num(f.get("time_sec")) and abs(float(f["time_sec"]) - t_hit) <= 0.5
            for f in traj_frames
        )
        if near_any and traj_end is None:
            return None
        if traj_end and shot_end and traj_end != shot_end:
            return None
        return traj_end or shot_end
    return shot_end


def _shot_opponent_displacement(
    player_track_id: str,
    shot: Dict[str, Any],
    track_series: Dict[str, List[Dict[str, Any]]],
    opponent_tracks: Sequence[str],
    *,
    require_temporal_end: bool = False,
) -> Optional[float]:
    t_hit = _shot_time(shot)
    hitter_end = _hitter_court_end_at(player_track_id, t_hit, track_series, shot)
    if require_temporal_end and hitter_end not in ("end_a", "end_b"):
        return None
    candidate_opps = list(opponent_tracks) if opponent_tracks else [
        tid for tid in track_series if str(tid) != str(player_track_id)
    ]
    shot_opp_ds: List[float] = []
    for opp_id in candidate_opps:
        if str(opp_id) == str(player_track_id):
            continue
        window = [
            f for f in (track_series.get(opp_id) or [])
            if isinstance(f, dict)
            and _finite_num(f.get("time_sec"))
            and t_hit <= float(f["time_sec"]) <= t_hit + 1.2
            and _valid_court_xy(f.get("xy"))
        ]
        if len(window) < 2:
            continue
        for run in _contiguous_xy_runs(window, max_gap_sec=MAX_TRAJECTORY_JOIN_GAP_SEC):
            if len(run) < 2:
                continue
            if float(run[0]["time_sec"]) - t_hit > MAX_TRAJECTORY_JOIN_GAP_SEC:
                continue
            run_ends = {_frame_court_end(f) for f in run if _frame_court_end(f) in ("end_a", "end_b")}
            if len(run_ends) > 1:
                continue
            opp_end = next(iter(run_ends)) if len(run_ends) == 1 else None
            if hitter_end in ("end_a", "end_b"):
                if opp_end not in ("end_a", "end_b") or opp_end == hitter_end:
                    continue
            elif require_temporal_end:
                continue
            p_start = run[0]["xy"]
            max_d = max(math.hypot(float(p_start[0]) - float(f["xy"][0]), float(p_start[1]) - float(f["xy"][1])) for f in run)
            shot_opp_ds.append(max_d)
            break
    return max(shot_opp_ds) if shot_opp_ds else None


def compute_tactical_suppression(
    player_track_id: str,
    hit_events: Sequence[Dict[str, Any]],
    track_series: Dict[str, List[Dict[str, Any]]],
    opponent_tracks: Sequence[str],
    *,
    require_temporal_end: bool = False,
) -> Dict[str, Any]:
    """1. 战术压制力: Opponent displacement imposed in [t_hit, t_hit + 1.2s] on verified shots."""
    player_frames = track_series.get(player_track_id) or []
    if _has_multiple_occupants(player_frames):
        return {
            "id": "tactical_suppression",
            "label": "战术压制力",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "m_per_shot",
            "display_value": "样本不足",
            "detail": "半场席位存在多次换人分段，暂不合并评估个人战术压制力。",
            "sample_count": 0,
        }
    player_shots = _verified_player_shots(player_track_id, hit_events)
    shot_displacements: List[float] = []
    for shot in player_shots:
        disp = _shot_opponent_displacement(
            player_track_id, shot, track_series, opponent_tracks,
            require_temporal_end=require_temporal_end,
        )
        if disp is not None:
            shot_displacements.append(disp)

    if len(shot_displacements) >= 3:
        mean_d = _mean(shot_displacements)
        max_d = max(shot_displacements)
        score = round(min(100.0, max(0.0, (mean_d / 1.5) * 100.0)), 1)
        return {
            "id": "tactical_suppression",
            "label": "战术压制力",
            "status": "observed",
            "evidence_status": "verified",
            "score": score,
            "raw_value": round(mean_d, 3),
            "unit": "m_per_shot",
            "display_value": f"{round(mean_d, 2)}m/拍",
            "detail": (
                f"出球施加压制：平均每拍迫使对手移动 {round(mean_d, 2)} 米"
                f"（单拍最高调度 {round(max_d, 2)} 米，有效样本 {len(shot_displacements)} 拍）。"
                f"数值越高说明出球落点越刁钻、调度对手跑动范围越大。"
            ),
            "sample_count": len(shot_displacements),
        }
    has_candidates = any(
        isinstance(ev, dict)
        and str(ev.get("track_id") or ev.get("hitter_track_id") or ev.get("actor_track_id")) == str(player_track_id)
        and not _is_verified_hit_event(ev)
        for ev in (hit_events or [])
    )
    return {
        "id": "tactical_suppression",
        "label": "战术压制力",
        "status": "insufficient_measurements",
        "evidence_status": "candidate_only" if has_candidates and not player_shots else "insufficient",
        "score": None,
        "raw_value": round(_mean(shot_displacements), 3) if shot_displacements else None,
        "unit": "m_per_shot",
        "display_value": "样本不足",
        "detail": (
            "当前仅有未校验候选击球记录，未经逐拍复核不计入对手受迫位移。"
            if has_candidates and not player_shots
            else "已验证出球与对手受迫位移样本不足 3 拍，暂不评估战术压制力。"
        ),
        "sample_count": len(shot_displacements),
    }


def compute_overhead_offensive_power(
    player_track_id: str,
    player_frames: Sequence[Dict[str, Any]],
    hit_events: Sequence[Dict[str, Any]],
    track_series: Dict[str, List[Dict[str, Any]]],
    opponent_tracks: Sequence[str],
    *,
    require_temporal_end: bool = False,
) -> Dict[str, Any]:
    """2. 高点进攻威力: Verified hit-bound overhead stroke ratio + verified opponent displacement."""
    if _has_multiple_occupants(player_frames):
        return {
            "id": "overhead_power",
            "label": "高点进攻威力",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "ratio",
            "display_value": "样本不足",
            "detail": "半场席位存在多次换人分段，暂不合并评估个人高点进攻威力。",
            "sample_count": 0,
        }

    player_shots = _verified_player_shots(player_track_id, hit_events)
    if len(player_shots) < 3:
        return {
            "id": "overhead_power",
            "label": "高点进攻威力",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "ratio",
            "display_value": "样本不足",
            "detail": "已验证击球事件样本不足 3 拍，连续举手或无击球姿态帧不计为高点进攻。",
            "sample_count": len(player_shots),
        }

    paired_zones: List[str] = []
    opp_displacements: List[float] = []
    for shot in player_shots:
        t_hit = float(shot.get("time_sec") if shot.get("time_sec") is not None else shot.get("source_time_sec") or 0.0)
        window = [
            f for f in player_frames
            if isinstance(f, dict)
            and _finite_num(f.get("time_sec"))
            and abs(float(f["time_sec"]) - t_hit) <= 0.15
            and isinstance(f.get("pose_metrics"), dict)
            and f["pose_metrics"].get("stroke_zone") in ("overhead", "drive", "underhand")
        ]
        if not window:
            continue
        # Ensure window does not cross occupant/side segments
        anchor = min(window, key=lambda f: abs(float(f["time_sec"]) - t_hit))
        anchor_seg = _frame_segment_key(anchor)
        same_seg_window = [f for f in window if _frame_segment_key(f) == anchor_seg]
        peak_frame = max(
            same_seg_window,
            key=lambda f: (
                float(f["pose_metrics"].get("rel_y"))
                if _finite_num(f["pose_metrics"].get("rel_y"))
                else (1.0 if f["pose_metrics"].get("stroke_zone") == "overhead" else 0.0),
                -abs(float(f["time_sec"]) - t_hit),
            ),
        )
        zone = str(peak_frame["pose_metrics"]["stroke_zone"])
        disp = _shot_opponent_displacement(
            player_track_id, shot, track_series, opponent_tracks,
            require_temporal_end=require_temporal_end,
        )
        if disp is not None:
            paired_zones.append(zone)
            opp_displacements.append(disp)

    total_strokes = len(paired_zones)
    max_opp_d: Optional[float] = max(opp_displacements) if opp_displacements else None

    if total_strokes >= 3 and len(opp_displacements) >= 3 and max_opp_d is not None:
        overhead_count = paired_zones.count("overhead")
        overhead_ratio = overhead_count / total_strokes
        score_ratio = min(100.0, (overhead_ratio / 0.40) * 100.0)
        score_impact = min(100.0, (min(2.0, max_opp_d) / 2.0) * 100.0)
        score = round(0.50 * score_ratio + 0.50 * score_impact, 1)
        overhead_pct = round(overhead_ratio * 100.0, 1)
        return {
            "id": "overhead_power",
            "label": "高点进攻威力",
            "status": "observed",
            "evidence_status": "verified",
            "score": score,
            "raw_value": round(overhead_ratio, 3),
            "unit": "ratio",
            "display_value": f"{overhead_pct}% 高点",
            "detail": (
                f"高点进攻威力：上手高点动作占比 {overhead_pct}%，"
                f"下压造成对手最大位移 {round(max_opp_d, 2)} 米。衡量主动抢占高点击球以及下压突击的威力。"
            ),
            "sample_count": total_strokes,
        }
    return {
        "id": "overhead_power",
        "label": "高点进攻威力",
        "status": "insufficient_measurements",
        "evidence_status": "insufficient",
        "score": None,
        "raw_value": None,
        "unit": "ratio",
        "display_value": "样本不足",
        "detail": "有效已验证击球姿态与对手受迫位移配对样本不足 3 拍，未知对手位移不计为 0。",
        "sample_count": total_strokes,
    }


def compute_court_dominance(
    player_frames: Sequence[Dict[str, Any]],
    is_near: bool,
    mean_opp_displacement: Optional[float],
    match_duration_sec: float,
    total_match_shots: int,
) -> Dict[str, Any]:
    """3. 控场能力: Mid-court (2.87m~4.79m) occupancy with triple anti-cheat gating."""
    if _has_multiple_occupants(player_frames):
        return {
            "id": "court_control",
            "label": "控场能力",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "score",
            "display_value": "样本不足",
            "detail": "半场席位存在多次换人分段，暂不合并评估个人控场能力。",
            "sample_count": 0,
        }
    valid_xy_frames = [f for f in player_frames if isinstance(f, dict) and _valid_court_xy(f.get("xy"))]
    if len(valid_xy_frames) < 10:
        return {
            "id": "court_control",
            "label": "控场能力",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "score",
            "display_value": "样本不足",
            "detail": "有效场地定位轨迹不足，暂不评估控场能力。",
            "sample_count": len(valid_xy_frames),
        }

    if mean_opp_displacement is None or not _finite_num(mean_opp_displacement) or total_match_shots < 2:
        return {
            "id": "court_control",
            "label": "控场能力",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "score",
            "display_value": "样本不足",
            "detail": "缺少对抗击球或对手受迫位移证据，暂不评估控场能力。",
            "sample_count": len(valid_xy_frames),
        }

    mid_count = 0
    for f in valid_xy_frames:
        y = float(f["xy"][1])
        f_end = _frame_court_end(f)
        if f_end in ("end_a", "end_b"):
            frame_is_near = (f_end == "end_b")
        else:
            frame_is_near = is_near
        depth = (13.4 - y) if frame_is_near else y
        if 2.87 <= depth <= 4.79:
            mid_count += 1

    mid_court_pct = (mid_count / len(valid_xy_frames)) * 100.0
    duration_min = max(0.2, match_duration_sec / 60.0)
    pace = total_match_shots / duration_min

    # Triple Anti-Cheat Guardrails
    opp_mult = min(1.0, max(0.3, 0.3 + 0.7 * (float(mean_opp_displacement) / 1.0)))
    raw_score = mid_court_pct * opp_mult
    if pace < 25.0 and total_match_shots >= 2:
        final_score = min(40.0, raw_score)
        guard_note = f"（对抗拍频 {round(pace, 1)} 拍/分低于 25 拍/分门槛，触发防刷分封顶 40 分）"
    else:
        final_score = min(100.0, max(0.0, raw_score))
        guard_note = f"（对抗拍频 {round(pace, 1)} 拍/分，对手受迫乘数 {round(opp_mult, 2)}）"

    final_score = round(final_score, 1)
    return {
        "id": "court_control",
        "label": "控场能力",
        "status": "observed",
        "evidence_status": "verified",
        "score": final_score,
        "raw_value": round(mid_court_pct, 1),
        "unit": "score",
        "display_value": f"{round(final_score)} /100",
        "detail": (
            f"控场能力：中场黄金攻防区（2.87m~4.79m）驻留率 {round(mid_court_pct, 1)}%{guard_note}。"
            f"结合对抗拍频与对手受迫位移综合判定，杜绝低速对推刷分，衡量在中枢区域的掌控力。"
        ),
        "sample_count": len(valid_xy_frames),
    }


def compute_launch_explosiveness(
    player_frames: Sequence[Dict[str, Any]],
    opponent_shots: Sequence[Dict[str, Any]],
    *,
    player_track_id: Optional[str] = None,
    track_series: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    require_temporal_end: bool = False,
) -> Dict[str, Any]:
    """4. 启动爆发力: 5-frame smoothed Center of Mass (CoM) launch acceleration (m/s²) on verified opponent shots."""
    if _has_multiple_occupants(player_frames):
        return {
            "id": "launch_explosiveness",
            "label": "启动爆发力",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "mps2",
            "display_value": "样本不足",
            "detail": "半场席位存在多次换人分段，暂不合并评估个人启动爆发力。",
            "sample_count": 0,
        }
    verified_opp_shots = [ev for ev in (opponent_shots or []) if _is_verified_hit_event(ev)]
    runs = _contiguous_xy_runs(player_frames, max_gap_sec=0.25)
    run_vels: List[List[Dict[str, Any]]] = []
    for run in runs:
        if len(run) < 5:
            continue
        smoothed = []
        for i in range(2, len(run) - 2):
            t_i = float(run[i]["time_sec"])
            xs = [float(run[k]["xy"][0]) for k in range(i - 2, i + 3)]
            ys = [float(run[k]["xy"][1]) for k in range(i - 2, i + 3)]
            smoothed.append({"t": t_i, "xy": [sum(xs) / 5.0, sum(ys) / 5.0], "seg": _frame_segment_key(run[i])})
        vels = []
        for i in range(1, len(smoothed)):
            dt = smoothed[i]["t"] - smoothed[i - 1]["t"]
            if 0.01 < dt < 0.25 and smoothed[i]["seg"] == smoothed[i - 1]["seg"]:
                v = math.hypot(
                    smoothed[i]["xy"][0] - smoothed[i - 1]["xy"][0],
                    smoothed[i]["xy"][1] - smoothed[i - 1]["xy"][1],
                ) / dt
                vels.append({"t": (smoothed[i]["t"] + smoothed[i - 1]["t"]) / 2.0, "v": v, "seg": smoothed[i]["seg"]})
        if vels:
            run_vels.append(vels)

    if not run_vels:
        return {
            "id": "launch_explosiveness",
            "label": "启动爆发力",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "mps2",
            "display_value": "样本不足",
            "detail": "连续平滑轨迹点不足，暂不评估启动爆发力。",
            "sample_count": 0,
        }

    launch_accels = []
    for shot in verified_opp_shots:
        t_hit = _shot_time(shot)
        hitter_id = str(shot.get("track_id") or shot.get("hitter_track_id") or shot.get("actor_track_id") or "")
        if player_track_id is not None and hitter_id == str(player_track_id):
            continue
        if require_temporal_end or track_series is not None or (isinstance(shot, dict) and shot.get("court_end") in ("end_a", "end_b")):
            player_end = _resolve_end_in_window(player_frames, t_hit - 0.15, t_hit + 1.2)
            hitter_end = (
                _hitter_court_end_at(hitter_id, t_hit, track_series, shot)
                if track_series is not None and hitter_id
                else (str(shot["court_end"]) if isinstance(shot, dict) and shot.get("court_end") in ("end_a", "end_b") else None)
            )
            if require_temporal_end:
                if player_end not in ("end_a", "end_b") or hitter_end not in ("end_a", "end_b") or player_end == hitter_end:
                    continue
            elif player_end in ("end_a", "end_b") and hitter_end in ("end_a", "end_b") and player_end == hitter_end:
                continue
            elif track_series is not None and (player_end not in ("end_a", "end_b") or hitter_end not in ("end_a", "end_b")):
                continue
        for vels in run_vels:
            sub_vels = [p for p in vels if t_hit - 0.1 <= p["t"] <= t_hit + 1.2]
            if len(sub_vels) < 4:
                continue
            pre_hit = [p["v"] for p in sub_vels if abs(p["t"] - t_hit) <= 0.12]
            v_base = min(pre_hit) if pre_hit else sub_vels[0]["v"]

            launch_time = None
            launch_idx = None
            for idx, p in enumerate(sub_vels):
                if p["t"] >= t_hit + 0.08 and p["v"] >= max(0.8, v_base + 0.35):
                    k = idx
                    while k > 0 and sub_vels[k - 1]["t"] >= t_hit:
                        if sub_vels[k - 1]["v"] <= sub_vels[k]["v"]:
                            k -= 1
                        else:
                            break
                    launch_time = sub_vels[k]["t"]
                    launch_idx = k
                    break

            if launch_time is not None and launch_idx is not None:
                reaction_sec = launch_time - t_hit
                if 0.10 <= reaction_sec <= 0.85:
                    post = [p["v"] for p in sub_vels[launch_idx:] if launch_time <= p["t"] <= launch_time + 0.28]
                    if post:
                        peak_v = max(post)
                        v_start = sub_vels[launch_idx]["v"]
                        accel = (peak_v - v_start) / 0.25
                        if accel > 0:
                            launch_accels.append(accel)
                            break

    if len(launch_accels) >= 2:
        mean_a = _mean(launch_accels)
        peak_a = max(launch_accels)
        score = round(min(100.0, max(0.0, (mean_a / 3.5) * 100.0)), 1)
        return {
            "id": "launch_explosiveness",
            "label": "启动爆发力",
            "status": "observed",
            "evidence_status": "verified",
            "score": score,
            "raw_value": round(mean_a, 2),
            "unit": "mps2",
            "display_value": f"{round(mean_a, 2)} m/s²",
            "detail": (
                f"启动爆发力：防守蹬地起动 0.25 秒内爆发加速度均值 {round(mean_a, 2)} m/s²"
                f"（峰值 {round(peak_a, 2)} m/s²，样本 {len(launch_accels)} 拍）。"
                f"采用 5 帧滤波质心轨迹，杜绝 10 FPS 欠采样毫秒伪影。"
            ),
            "sample_count": len(launch_accels),
        }
    return {
        "id": "launch_explosiveness",
        "label": "启动爆发力",
        "status": "insufficient_measurements",
        "evidence_status": "insufficient",
        "score": None,
        "raw_value": round(_mean(launch_accels), 2) if launch_accels else None,
        "unit": "mps2",
        "display_value": "样本不足",
        "detail": "有效防守启动对抗样本不足 2 拍，暂不评估启动爆发力。",
        "sample_count": len(launch_accels),
    }


def compute_core_stability(
    torso_tilts: Sequence[float],
    knee_flexions: Sequence[Union[Tuple[float, float], Tuple[float, float, Any]]],
    *,
    torso_samples: Optional[Sequence[Tuple[float, float, Any]]] = None,
) -> Dict[str, Any]:
    """5. 核心稳定性: Requires >= 30 contiguous torso samples (>=3.0s within one segment) AND >= 3 matched lunge recoveries."""
    raw_valid_tilts = [
        float(v) for v in (torso_tilts or [])
        if _finite_num(v) and 0.0 <= float(v) <= 90.0
    ]
    if torso_samples is None:
        return {
            "id": "core_stability",
            "label": "核心稳定性",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "score",
            "display_value": "样本不足",
            "detail": "缺少带时间戳的连续躯干姿态序列，无法证明连续覆盖时长，暂不评估核心稳定性。",
            "sample_count": len(raw_valid_tilts),
        }

    torso_occupant_segs = {
        item[2][1] for item in torso_samples
        if isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[2], tuple)
        and len(item[2]) >= 2 and type(item[2][1]) is int and item[2][1] >= 1
    }
    knee_occupant_segs = {
        item[2][1] for item in (knee_flexions or [])
        if isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[2], tuple)
        and len(item[2]) >= 2 and type(item[2][1]) is int and item[2][1] >= 1
    }
    if len(torso_occupant_segs | knee_occupant_segs) > 1:
        return {
            "id": "core_stability",
            "label": "核心稳定性",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "score",
            "display_value": "样本不足",
            "detail": "半场席位存在多次换人分段，不跨段合并评估个人核心稳定性。",
            "sample_count": 0,
        }

    by_seg: Dict[Any, Dict[float, float]] = defaultdict(dict)
    for item in torso_samples:
        if not isinstance(item, (list, tuple)) or len(item) < 2 or not _finite_num(item[0]) or not _finite_num(item[1]):
            continue
        t_val, tilt_val = float(item[0]), float(item[1])
        if t_val < 0.0 or not (0.0 <= tilt_val <= 90.0):
            continue
        seg_key = item[2] if len(item) >= 3 else ("", 0, "")
        by_seg[seg_key].setdefault(t_val, tilt_val)

    torso_runs: List[Tuple[Any, List[Tuple[float, float]], float, float, float]] = []
    for seg_key, seg_map in by_seg.items():
        if not seg_map:
            continue
        times_sorted = sorted(seg_map.keys())
        cur: List[Tuple[float, float]] = []
        for t_val in times_sorted:
            if not cur:
                cur = [(t_val, seg_map[t_val])]
            elif t_val - cur[-1][0] <= MAX_POSE_CONTINUITY_GAP_SEC:
                cur.append((t_val, seg_map[t_val]))
            else:
                torso_runs.append((seg_key, cur, cur[0][0], cur[-1][0], cur[-1][0] - cur[0][0]))
                cur = [(t_val, seg_map[t_val])]
        if cur:
            torso_runs.append((seg_key, cur, cur[0][0], cur[-1][0], cur[-1][0] - cur[0][0]))

    cleaned_knees: List[Tuple[float, float, Any]] = []
    seen_knee_times = set()
    for item in (knee_flexions or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t_val, k_val = item[0], item[1]
        seg_key = item[2] if len(item) >= 3 else ("", 0, "")
        if not _finite_num(t_val) or not _finite_num(k_val) or float(t_val) < 0.0:
            continue
        if not (DEEP_LUNGE_MIN_PHYSIOLOGICAL_ANGLE_DEG <= float(k_val) <= 180.0):
            continue
        key = (seg_key, float(t_val))
        if key in seen_knee_times:
            continue
        seen_knee_times.add(key)
        cleaned_knees.append((float(t_val), float(k_val), seg_key))
    cleaned_knees.sort(key=lambda x: (repr(x[2]), x[0]))

    all_recoveries: List[Tuple[Any, float, float, float]] = []
    # State machine per contiguous (seg_key, dt <= MAX_POSE_CONTINUITY_GAP_SEC) sequence:
    # must observe upright posture (>= 140.0) -> deep lunge entry (<= 125.0) -> stand-up recovery (>= 140.0).
    # Each complete stand-up cycle emits at most one recovery regardless of flat bottom plateaus.
    cur_seg: Any = None
    prev_t: Optional[float] = None
    seen_upright = False
    in_lunge = False
    nadir_t: Optional[float] = None
    nadir_k: Optional[float] = None
    for t_k, knee, seg_k in cleaned_knees:
        if prev_t is None or seg_k != cur_seg or (t_k - prev_t) > MAX_POSE_CONTINUITY_GAP_SEC:
            cur_seg = seg_k
            seen_upright = False
            in_lunge = False
            nadir_t = None
            nadir_k = None
        prev_t = t_k
        if not in_lunge:
            if knee >= 140.0:
                seen_upright = True
            elif seen_upright and knee <= 125.0:
                in_lunge = True
                nadir_t = t_k
                nadir_k = knee
        else:
            if knee < 140.0:
                if nadir_k is None or knee < nadir_k:
                    nadir_k = knee
                    nadir_t = t_k
            else:
                if nadir_t is not None and 0.0 < (t_k - nadir_t) <= 1.8:
                    all_recoveries.append((seg_k, nadir_t, t_k, t_k - nadir_t))
                in_lunge = False
                nadir_t = None
                nadir_k = None
                seen_upright = True

    torso_qualified_runs = [
        run_tuple for run_tuple in torso_runs
        if len(run_tuple[1]) >= MIN_CORE_STABILITY_TORSO_SAMPLES and run_tuple[4] >= MIN_CORE_STABILITY_SPAN_SEC
    ]
    if not torso_qualified_runs:
        max_contig = max((len(r[1]) for r in torso_runs), default=0)
        return {
            "id": "core_stability",
            "label": "核心稳定性",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "score",
            "display_value": "样本不足",
            "detail": (
                f"有效连续躯干姿态片段不足（最长连续段 {max_contig} 帧，需单一连续段至少 {MIN_CORE_STABILITY_TORSO_SAMPLES} 帧且跨度 ≥ {MIN_CORE_STABILITY_SPAN_SEC} 秒），"
                f"断续散点或短片段不评估核心稳定性。"
            ),
            "sample_count": max_contig,
        }

    scored_candidates: List[Tuple[List[float], List[float], float, float]] = []
    best_unpaired_count = 0
    best_unpaired_recs: List[float] = []
    for seg_key, pts, run_t0, run_t1, run_span in torso_qualified_runs:
        matched_recs = [
            dt for (r_seg, t_lunge, t_rec, dt) in all_recoveries
            if r_seg == seg_key and run_t0 <= t_lunge and t_rec <= run_t1
        ]
        tilts = [p[1] for p in pts]
        if len(tilts) > best_unpaired_count or (len(tilts) == best_unpaired_count and len(matched_recs) > len(best_unpaired_recs)):
            best_unpaired_count = len(tilts)
            best_unpaired_recs = matched_recs
        if len(matched_recs) >= MIN_CORE_STABILITY_LUNGE_RECOVERIES:
            scored_candidates.append((tilts, matched_recs, run_span, float(len(tilts))))

    if not scored_candidates:
        return {
            "id": "core_stability",
            "label": "核心稳定性",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "score",
            "display_value": "样本不足",
            "detail": (
                f"同一连续有效片段与相同场地侧深弓步制动样本少于 {MIN_CORE_STABILITY_LUNGE_RECOVERIES} 次（当前 {len(best_unpaired_recs)} 次），"
                f"缺失同段弓步恢复证据时不生成核心稳定性分值。"
            ),
            "sample_count": best_unpaired_count,
        }

    valid_tilts, recs, _, _ = max(scored_candidates, key=lambda item: (item[3], len(item[1]), item[2]))

    sigma_tilt = _std(valid_tilts)
    s_torso = max(0.0, 100.0 - sigma_tilt * 4.0)
    var_rec = (_std(recs) ** 2)
    s_lunge = max(0.0, 100.0 - var_rec * 300.0)
    final_score = round(0.50 * s_torso + 0.50 * s_lunge, 1)
    detail = (
        f"核心稳定性：躯干倾角波动 ±{round(sigma_tilt, 1)}°，"
        f"深弓步制动恢复用时方差 {round(var_rec, 4)}（样本 {len(recs)} 次）。"
        f"综合上身挺拔度与下肢制动回弹节奏稳定性。"
    )
    return {
        "id": "core_stability",
        "label": "核心稳定性",
        "status": "observed",
        "evidence_status": "verified",
        "score": final_score,
        "raw_value": round(sigma_tilt, 1),
        "unit": "score",
        "display_value": f"{round(final_score)} /100",
        "detail": detail,
        "sample_count": len(valid_tilts),
    }


def compute_footwork_coherence(
    player_track_id: str,
    hit_events: Sequence[Dict[str, Any]],
    player_frames: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """6. 步法连贯性: Post-hit momentum reversal reset rate on verified shots within a single segment."""
    if _has_multiple_occupants(player_frames):
        return {
            "id": "footwork_coherence",
            "label": "步法连贯性",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "ratio",
            "display_value": "样本不足",
            "detail": "半场席位存在多次换人分段，暂不合并评估个人步法连贯性。",
            "sample_count": 0,
        }
    runs = _contiguous_xy_runs(player_frames, max_gap_sec=MAX_TRAJECTORY_JOIN_GAP_SEC)
    if sum(len(r) for r in runs) < 5:
        return {
            "id": "footwork_coherence",
            "label": "步法连贯性",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "ratio",
            "display_value": "样本不足",
            "detail": "有效击球步法轨迹样本不足，暂不评估步法连贯性。",
            "sample_count": 0,
        }

    player_shots = _verified_player_shots(player_track_id, hit_events)
    if len(player_shots) < 2:
        return {
            "id": "footwork_coherence",
            "label": "步法连贯性",
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": "ratio",
            "display_value": "样本不足",
            "detail": "本场已验证击球样本不足 2 拍，暂不评估击球后步法连贯性。",
            "sample_count": len(player_shots),
        }

    reset_count = 0
    approach_count = 0
    for shot in player_shots:
        t_hit = _shot_time(shot)
        shot_end = shot.get("court_end") if isinstance(shot, dict) and shot.get("court_end") in ("end_a", "end_b") else None
        for run in runs:
            run_end = _frame_court_end(run[0])
            if shot_end in ("end_a", "end_b") and run_end in ("end_a", "end_b") and shot_end != run_end:
                continue
            pre_frames = [f for f in run if t_hit - 0.5 <= float(f["time_sec"]) <= t_hit]
            post_frames = [f for f in run if t_hit + 0.3 <= float(f["time_sec"]) <= t_hit + 0.8]
            if len(post_frames) >= 2:
                v_post = (
                    float(post_frames[-1]["xy"][0]) - float(post_frames[0]["xy"][0]),
                    float(post_frames[-1]["xy"][1]) - float(post_frames[0]["xy"][1]),
                )
                d_post = math.hypot(v_post[0], v_post[1])
                if d_post >= 0.25:
                    if len(pre_frames) >= 2:
                        v_pre = (
                            float(pre_frames[-1]["xy"][0]) - float(pre_frames[0]["xy"][0]),
                            float(pre_frames[-1]["xy"][1]) - float(pre_frames[0]["xy"][1]),
                        )
                        d_pre = math.hypot(v_pre[0], v_pre[1])
                        if d_pre >= 0.15:
                            dot = v_pre[0] * v_post[0] + v_pre[1] * v_post[1]
                            cos_ang = max(-1.0, min(1.0, dot / (d_pre * d_post)))
                            if math.degrees(math.acos(cos_ang)) >= 110.0:
                                reset_count += 1
                        elif d_post >= 0.35:
                            reset_count += 1
                    elif d_post >= 0.35:
                        reset_count += 1
                approach_count += 1
                break

    if approach_count >= 2:
        reset_rate = reset_count / approach_count
        score = round(min(100.0, max(0.0, reset_rate * 100.0)), 1)
        reset_pct = round(reset_rate * 100.0, 1)
        return {
            "id": "footwork_coherence",
            "label": "步法连贯性",
            "status": "observed",
            "evidence_status": "verified",
            "score": score,
            "raw_value": round(reset_rate, 3),
            "unit": "ratio",
            "display_value": f"{reset_pct}%",
            "detail": (
                f"步法连贯性：击球后反向回动架拍连贯率 {reset_pct}%（有效击球 {approach_count} 拍）。"
                f"免除死板单打中心点设定，全场景通用；衡量出球后能否立即反向制动、完成回位架拍准备下一拍。"
            ),
            "sample_count": approach_count,
        }
    return {
        "id": "footwork_coherence",
        "label": "步法连贯性",
        "status": "insufficient_measurements",
        "evidence_status": "insufficient",
        "score": None,
        "raw_value": None,
        "unit": "ratio",
        "display_value": "样本不足",
        "detail": "有效出球后步法跟进样本不足，暂不评估步法连贯性。",
        "sample_count": approach_count,
    }


def build_empty_combat_radar_profile(track_id: str) -> Dict[str, Any]:
    dim_defs = [
        ("tactical_suppression", "战术压制力", "m_per_shot"),
        ("overhead_power", "高点进攻威力", "ratio"),
        ("court_control", "控场能力", "score"),
        ("launch_explosiveness", "启动爆发力", "mps2"),
        ("core_stability", "核心稳定性", "score"),
        ("footwork_coherence", "步法连贯性", "ratio"),
    ]
    dimensions = [
        {
            "id": dim_id,
            "label": label,
            "status": "insufficient_measurements",
            "evidence_status": "insufficient",
            "score": None,
            "raw_value": None,
            "unit": unit,
            "display_value": "样本不足",
            "detail": f"{label}样本不足，暂不生成分值。",
            "sample_count": 0,
        }
        for dim_id, label, unit in dim_defs
    ]
    return {
        "schema_version": COMBAT_RADAR_SCHEMA_VERSION,
        "track_id": track_id,
        "status": "insufficient_measurements",
        "evidence_status": "insufficient",
        "observed_dimension_count": 0,
        "observed_dimension_ids": [],
        "missing_dimension_ids": [dim_id for dim_id, _, _ in dim_defs],
        "dimensions": dimensions,
        "summary_note": "样本不足，暂无有效雷达图数据。",
    }


def build_six_dimensional_combat_radar(
    track_id: str,
    *,
    track_series: Dict[str, List[Dict[str, Any]]],
    hit_events: List[Dict[str, Any]],
    team_near: List[str],
    team_far: List[str],
    match_duration_sec: float = 0.0,
    total_match_shots: int = 0,
) -> Dict[str, Any]:
    """Assemble 6 orthogonal combat radar dimensions."""
    player_frames = track_series.get(track_id) or []
    verified_hits = [ev for ev in (hit_events or []) if _is_verified_hit_event(ev)]
    verified_shot_count = len(verified_hits) if hit_events is not None else 0
    if not hit_events and total_match_shots > 0:
        verified_shot_count = 0
    elif verified_hits and total_match_shots > len(verified_hits) and all(_is_verified_hit_event(ev) for ev in hit_events):
        verified_shot_count = total_match_shots

    is_near = (track_id in team_near)
    all_other_tracks = [str(tid) for tid in track_series if str(tid) != str(track_id)]
    candidate_opp_shots = [
        ev for ev in verified_hits
        if str(ev.get("track_id") or ev.get("hitter_track_id") or ev.get("actor_track_id") or "") != str(track_id)
    ]

    dim1 = compute_tactical_suppression(
        track_id, verified_hits, track_series, all_other_tracks,
        require_temporal_end=True,
    )
    dim2 = compute_overhead_offensive_power(
        track_id, player_frames, verified_hits, track_series, all_other_tracks,
        require_temporal_end=True,
    )
    mean_opp_disp = (
        float(dim1["raw_value"])
        if dim1.get("status") == "observed" and dim1.get("raw_value") is not None
        else None
    )
    dim3 = compute_court_dominance(player_frames, is_near, mean_opp_disp, match_duration_sec, verified_shot_count)
    dim4 = compute_launch_explosiveness(
        player_frames,
        candidate_opp_shots,
        player_track_id=track_id,
        track_series=track_series,
        require_temporal_end=True,
    )

    torso_tilts = [
        f["pose_metrics"]["torso_tilt"]
        for f in player_frames
        if isinstance(f, dict) and f.get("pose_metrics") and f["pose_metrics"].get("torso_tilt") is not None
    ]
    torso_samples = [
        (float(f["time_sec"]), float(f["pose_metrics"]["torso_tilt"]), _frame_segment_key(f))
        for f in player_frames
        if isinstance(f, dict)
        and _finite_num(f.get("time_sec"))
        and f.get("pose_metrics")
        and _finite_num(f["pose_metrics"].get("torso_tilt"))
    ]
    knee_flexions = [
        (float(f["time_sec"]), float(f["pose_metrics"]["min_knee"]), _frame_segment_key(f))
        for f in player_frames
        if isinstance(f, dict)
        and _finite_num(f.get("time_sec"))
        and f.get("pose_metrics")
        and _finite_num(f["pose_metrics"].get("min_knee"))
    ]
    dim5 = compute_core_stability(torso_tilts, knee_flexions, torso_samples=torso_samples)
    dim6 = compute_footwork_coherence(track_id, verified_hits, player_frames)

    dimensions = [dim1, dim2, dim3, dim4, dim5, dim6]
    observed_ids = [d["id"] for d in dimensions if d["status"] == "observed"]
    missing_ids = [d["id"] for d in dimensions if d["status"] != "observed"]
    observed_count = len(observed_ids)
    status = "observed" if observed_count >= 1 else "insufficient_measurements"
    scopes = {f.get("identity_scope") for f in player_frames if isinstance(f, dict) and f.get("identity_scope")}
    identity_scope = (
        "half_court_slots.v1"
        if "half_court_slots.v1" in scopes
        else (next(iter(sorted(scopes)), None) if len(scopes) == 1 else None)
    )

    return {
        "schema_version": COMBAT_RADAR_SCHEMA_VERSION,
        "track_id": track_id,
        "status": status,
        "evidence_status": "verified" if observed_count >= 1 else ("candidate_only" if hit_events and not verified_hits else "insufficient"),
        "identity_scope": identity_scope,
        "observed_dimension_count": observed_count,
        "observed_dimension_ids": observed_ids,
        "missing_dimension_ids": missing_ids,
        "dimensions": dimensions,
        "summary_note": (
            "基于真实空间轨迹与 COCO-17 骨骼关节点连续量提取，杜绝欠采样毫秒伪影与代数线性重复。"
            if observed_count >= 1
            else "当前缺少已验证逐拍击球与足够连续姿态样本，暂不生成战术雷达分值。"
        ),
    }


def _temporal_opponent_ids(
    track_id: str,
    track_radars: Dict[str, Dict[str, Any]],
    track_series: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """Identify verified opponents strictly from overlapping opposite-side court ends."""
    def _track_buckets(frames: Sequence[Dict[str, Any]]) -> Dict[int, Optional[str]]:
        raw: Dict[int, set] = defaultdict(set)
        occs: Dict[int, set] = defaultdict(set)
        for f in frames:
            if not isinstance(f, dict) or not _finite_num(f.get("time_sec")):
                continue
            f_end = _frame_court_end(f)
            if f_end not in ("end_a", "end_b"):
                continue
            b = int(round(float(f["time_sec"]) * 2.0))
            raw[b].add(f_end)
            occ = f.get("occupancy_segment")
            if type(occ) is int and occ >= 1:
                occs[b].add(occ)
        out: Dict[int, Optional[str]] = {}
        for b, ends in raw.items():
            if len(ends) == 1 and len(occs.get(b, set())) <= 1:
                out[b] = next(iter(ends))
            else:
                out[b] = None
        return out

    self_frames = track_series.get(track_id) or []
    if _has_multiple_occupants(self_frames):
        return []
    self_buckets = _track_buckets(self_frames)
    if not self_buckets:
        return []

    opp_ids: List[str] = []
    for other_id in track_radars:
        if str(other_id) == str(track_id):
            continue
        other_frames = track_series.get(other_id) or []
        if _has_multiple_occupants(other_frames):
            continue
        other_buckets = _track_buckets(other_frames)
        if not other_buckets:
            continue
        shared = set(self_buckets.keys()) & set(other_buckets.keys())
        if not shared:
            continue
        same_side = False
        opp_count = 0
        for b in shared:
            e1, e2 = self_buckets[b], other_buckets[b]
            if e1 not in ("end_a", "end_b") or e2 not in ("end_a", "end_b"):
                same_side = True
                break
            if e1 == e2:
                same_side = True
                break
            opp_count += 1
        if not same_side and opp_count >= 1:
            opp_ids.append(other_id)
    return opp_ids


def compute_opponent_comparison(
    track_id: str,
    track_radars: Dict[str, Dict[str, Any]],
    team_near: Sequence[str],
    team_far: Sequence[str],
    *,
    track_series: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute opponent team average across all 6 dimensions and calculate relative advantage."""
    if track_series is not None:
        opp_ids = _temporal_opponent_ids(track_id, track_radars, track_series)
    else:
        is_near = (track_id in team_near)
        opp_team = team_far if is_near else team_near
        opp_ids = [t for t in opp_team if t in track_radars]
    if not opp_ids:
        return None

    player_radar = track_radars.get(track_id) or {}
    player_dims = {d["id"]: d for d in player_radar.get("dimensions", [])}

    dim_ids = [
        "tactical_suppression",
        "overhead_power",
        "court_control",
        "launch_explosiveness",
        "core_stability",
        "footwork_coherence",
    ]
    dim_labels = {
        "tactical_suppression": "战术压制力",
        "overhead_power": "高点进攻威力",
        "court_control": "控场能力",
        "launch_explosiveness": "启动爆发力",
        "core_stability": "核心稳定性",
        "footwork_coherence": "步法连贯性",
    }
    dim_units = {
        "tactical_suppression": "m_per_shot",
        "overhead_power": "ratio",
        "court_control": "score",
        "launch_explosiveness": "mps2",
        "core_stability": "score",
        "footwork_coherence": "ratio",
    }

    comp_dims = []
    for d_id in dim_ids:
        opp_scores = []
        opp_raws = []
        for o_id in opp_ids:
            radar = track_radars[o_id]
            for dim in radar.get("dimensions", []):
                if dim.get("id") == d_id and dim.get("status") == "observed":
                    if dim.get("score") is not None:
                        opp_scores.append(dim["score"])
                    if dim.get("raw_value") is not None:
                        opp_raws.append(dim["raw_value"])

        avg_score = round(_mean(opp_scores), 1) if opp_scores else None
        avg_raw = round(_mean(opp_raws), 3) if opp_raws else None

        p_dim = player_dims.get(d_id) or {}
        p_score = p_dim.get("score") if p_dim.get("status") == "observed" else None

        diff_score = None
        diff_display = ""
        advantage = "balanced"

        if p_score is not None and avg_score is not None:
            diff_score = round(p_score - avg_score, 1)
            if diff_score >= 5.0:
                advantage = "superior"
                diff_display = f"+{diff_score}分"
            elif diff_score <= -5.0:
                advantage = "inferior"
                diff_display = f"{diff_score}分"
            else:
                advantage = "balanced"
                diff_display = "≈持平"

        if avg_score is None:
            display_val = "样本不足"
        elif d_id == "tactical_suppression":
            display_val = f"{round(avg_raw, 2)}m/拍" if avg_raw is not None else f"{round(avg_score)}分"
        elif d_id == "overhead_power":
            display_val = f"{round((avg_raw or 0) * 100, 1)}% 高点"
        elif d_id == "court_control":
            display_val = f"{round(avg_score)} /100"
        elif d_id == "launch_explosiveness":
            display_val = f"{round(avg_raw, 2)} m/s²" if avg_raw is not None else f"{round(avg_score)}分"
        elif d_id == "core_stability":
            display_val = f"{round(avg_score)} /100"
        elif d_id == "footwork_coherence":
            display_val = f"{round((avg_raw or 0) * 100, 1)}%"
        else:
            display_val = f"{round(avg_score)}分"

        comp_dims.append({
            "id": d_id,
            "label": dim_labels[d_id],
            "score": avg_score,
            "raw_value": avg_raw,
            "unit": dim_units[d_id],
            "display_value": display_val,
            "diff_score": diff_score,
            "diff_display": diff_display,
            "advantage": advantage,
        })

    return {
        "target": "opponents",
        "label": "同场对手均值",
        "sample_track_count": len(opp_ids),
        "dimensions": comp_dims,
    }


def collect_detections_pose_and_shot_evidence(
    detections_path: Any,
    *,
    candidate_routes_path: Any = None,
    fps: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """Read per-track pose biomechanics and candidate hit heights from detections.jsonl."""
    from pathlib import Path
    import json

    det_file = Path(detections_path) if detections_path else None
    if not det_file or not det_file.is_file():
        return {}

    candidate_times_by_track: Dict[str, List[float]] = defaultdict(list)
    all_candidate_hit_events: List[Dict[str, Any]] = []

    def _ingest_route_payload(payload: Any, default_source: str) -> None:
        if not isinstance(payload, dict):
            return
        doc_source = str(payload.get("source") or default_source)
        doc_schema = str(payload.get("schema_version") or "")
        for rec in payload.get("records") or payload.get("events") or []:
            if not isinstance(rec, dict):
                continue
            tid = rec.get("actor_track_id") or rec.get("hitter_track_id") or rec.get("track_id")
            t_val = rec.get("time_sec") if rec.get("time_sec") is not None else rec.get("source_time_sec")
            if not tid or not _finite_num(t_val) or float(t_val) < 0.0:
                continue
            actor_scope = rec.get("actor_identity_scope") or rec.get("identity_scope")
            receiver_scope = rec.get("receiver_identity_scope")
            actor_seg = rec.get("actor_occupancy_segment") if type(rec.get("actor_occupancy_segment")) is int else (
                rec.get("occupancy_segment") if type(rec.get("occupancy_segment")) is int else 0
            )
            receiver_seg = rec.get("receiver_occupancy_segment") if type(rec.get("receiver_occupancy_segment")) is int else 0
            rec_source = str(rec.get("source") or doc_source)
            is_verified = (
                doc_schema == "personal-shot-route-events.v1"
                and doc_source == "paired_contacts_receiving_region"
                and rec_source == "paired_contacts_receiving_region"
                and all(
                    rec.get(k) is True
                    for k in ("actor_identity_verified", "receiver_identity_verified", "opponent_verified", "calibration_verified")
                )
                and rec.get("contact_status") == "confirmed"
                and rec.get("receiver_contact_status") == "confirmed"
                and rec.get("target_kind") == "receiver_contact_position"
                and actor_scope != "half_court_slots.v1"
                and receiver_scope != "half_court_slots.v1"
            )
            candidate_times_by_track[str(tid)].append(float(t_val))
            all_candidate_hit_events.append({
                "track_id": str(tid),
                "actor_track_id": str(tid),
                "receiver_track_id": str(rec.get("receiver_track_id") or ""),
                "time_sec": float(t_val),
                "source_time_sec": float(t_val),
                "status": "verified" if is_verified else "candidate",
                "contact_status": "confirmed" if is_verified else str(rec.get("contact_status") or "candidate"),
                "verified": bool(is_verified),
                "source": rec_source,
                "identity_scope": actor_scope,
                "actor_identity_scope": actor_scope,
                "receiver_identity_scope": receiver_scope,
                "occupancy_segment": actor_seg,
                "actor_occupancy_segment": actor_seg,
                "receiver_occupancy_segment": receiver_seg,
                "court_end": rec.get("court_end") or rec.get("actor_court_end"),
            })

    if candidate_routes_path and Path(candidate_routes_path).is_file():
        try:
            routes_payload = json.loads(Path(candidate_routes_path).read_text(encoding="utf-8"))
            _ingest_route_payload(routes_payload, Path(candidate_routes_path).name)
        except (OSError, ValueError):
            pass

    if not all_candidate_hit_events and det_file:
        for alt_rel in ("derived/receiving_route_estimates_v1.json", "derived/shot_events_v2.jsonl", "candidate-receiving-routes.v1.json"):
            alt_path = det_file.parent / alt_rel
            if not alt_path.is_file():
                continue
            try:
                if alt_path.suffix == ".jsonl":
                    with alt_path.open("r", encoding="utf-8") as sf:
                        for line in sf:
                            line = line.strip()
                            if not line:
                                continue
                            srow = json.loads(line)
                            htid = (srow.get("hitter") or {}).get("track_id")
                            ht_sec = srow.get("hit_time_sec")
                            if htid and _finite_num(ht_sec) and float(ht_sec) >= 0.0:
                                candidate_times_by_track[str(htid)].append(float(ht_sec))
                                all_candidate_hit_events.append({
                                    "track_id": str(htid),
                                    "actor_track_id": str(htid),
                                    "time_sec": float(ht_sec),
                                    "source_time_sec": float(ht_sec),
                                    "status": "candidate",
                                    "contact_status": "candidate",
                                    "verified": False,
                                    "source": "shot_events_v2",
                                })
                else:
                    alt_payload = json.loads(alt_path.read_text(encoding="utf-8"))
                    _ingest_route_payload(alt_payload, alt_path.name)
                if all_candidate_hit_events:
                    break
            except (OSError, ValueError):
                pass

    track_frames: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_samples = set()
    with det_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            t_sec = row.get("time_sec")
            frame = row.get("frame")
            if not _finite_num(t_sec):
                if type(frame) is int and fps and fps > 0:
                    t_sec = frame / float(fps)
                else:
                    continue
            if float(t_sec) < 0.0:
                continue
            key = ("time", float(t_sec))
            if key in seen_samples:
                continue
            seen_samples.add(key)
            for tr in (row.get("spatial") or {}).get("tracks") or []:
                if not isinstance(tr, dict) or tr.get("status") != "detected" or not tr.get("track_id"):
                    continue
                if "confidence" in tr and (not _finite_num(tr.get("confidence")) or float(tr["confidence"]) < 0.5):
                    continue
                loc_ev = tr.get("location_evidence")
                if isinstance(loc_ev, dict) and "confidence" in loc_ev:
                    if not _finite_num(loc_ev.get("confidence")) or float(loc_ev["confidence"]) < 0.5:
                        continue
                assoc = tr.get("association")
                scope = tr.get("identity_scope")
                occ_seg = tr.get("occupancy_segment") if type(tr.get("occupancy_segment")) is int else 0
                if isinstance(assoc, dict) or scope is not None:
                    assoc_dict = assoc if isinstance(assoc, dict) else {}
                    src = assoc_dict.get("source")
                    if "identity_confidence" in assoc_dict:
                        if not _finite_num(assoc_dict.get("identity_confidence")) or float(assoc_dict["identity_confidence"]) < 0.7:
                            continue
                    if scope == "half_court_slots.v1" or src in HALF_SLOT_SOURCES:
                        if (
                            scope != "half_court_slots.v1"
                            or src not in HALF_SLOT_SOURCES
                            or type(tr.get("occupancy_segment")) is not int
                            or tr.get("occupancy_segment") < 1
                            or tr.get("court_end") not in ("end_a", "end_b")
                            or tr.get("slot_court_end") != tr.get("court_end")
                        ):
                            continue
                    elif src is not None and src not in TRUSTED_PERSON_SOURCES:
                        continue
                tid = str(tr["track_id"])
                pose = tr.get("pose") or {}
                if (pose.get('is_current_measurement') is not True
                        or pose.get('format') != 'coco17_image_v1'):
                    continue
                raw_xy = tr.get("court_xy_m")
                xy = [float(raw_xy[0]), float(raw_xy[1])] if _valid_court_xy(raw_xy) else None
                c_end = tr.get("court_end") if tr.get("court_end") in ("end_a", "end_b") else (
                    ("end_b" if xy[1] > 6.7 else "end_a") if xy else None
                )
                pm = _analyze_coco17_frame(pose.get("keypoints_image"), pose.get("keypoint_scores"))
                track_frames[tid].append({
                    "time_sec": float(t_sec),
                    "xy": xy,
                    "pose_metrics": pm,
                    "identity_scope": scope,
                    "occupancy_segment": occ_seg,
                    "court_end": c_end,
                })

    summary_by_track: Dict[str, Dict[str, Any]] = {}
    for tid, frames in track_frames.items():
        frames.sort(key=lambda item: item["time_sec"])
        # Debounced deep lunge count (knee in [60.0, 110.0] deg, >=1.0s separation within same segment)
        deep_lunge_count = 0
        last_lunge_t = -999.0
        last_lunge_seg = None
        min_knee = None
        torso_tilts = []
        active_zones = Counter()
        for item in frames:
            pm = item.get("pose_metrics")
            if not pm:
                continue
            item_seg = _frame_segment_key(item)
            if item_seg != last_lunge_seg:
                last_lunge_t = -999.0
                last_lunge_seg = item_seg
            if pm["torso_tilt"] is not None:
                torso_tilts.append(pm["torso_tilt"])
            if pm["min_knee"] is not None:
                if pm["min_knee"] >= DEEP_LUNGE_MIN_PHYSIOLOGICAL_ANGLE_DEG:
                    min_knee = pm["min_knee"] if min_knee is None else min(min_knee, pm["min_knee"])
                if DEEP_LUNGE_MIN_PHYSIOLOGICAL_ANGLE_DEG <= pm["min_knee"] <= DEEP_LUNGE_MAX_KNEE_ANGLE_DEG:
                    if item["time_sec"] - last_lunge_t >= DEEP_LUNGE_DEBOUNCE_GAP_SEC:
                        deep_lunge_count += 1
                    last_lunge_t = item["time_sec"]
            if pm["stroke_zone"]:
                active_zones[pm["stroke_zone"]] += 1

        # Candidate hit window evaluation if candidate timestamps exist (single segment per window)
        cand_times = candidate_times_by_track.get(tid) or []
        hit_zones = Counter()
        hit_depths = Counter()
        for ct in cand_times:
            near = [f for f in frames if abs(f["time_sec"] - ct) <= 0.15]
            if not near:
                continue
            best_f = min(near, key=lambda f: abs(f["time_sec"] - ct))
            best_seg = _frame_segment_key(best_f)
            near = [f for f in near if _frame_segment_key(f) == best_seg]
            xy = best_f.get("xy")
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                d_net = abs(float(xy[1]) - 6.7)
                hit_depths["front" if d_net < 2.23 else ("mid" if d_net < 4.47 else "rear")] += 1
            best_rel = -999.0
            best_zone = None
            for wf in near:
                pm = wf.get("pose_metrics")
                if pm and pm["rel_y"] is not None and pm["rel_y"] > best_rel:
                    best_rel = pm["rel_y"]
                    best_zone = pm["stroke_zone"]
            if best_zone:
                hit_zones[best_zone] += 1

        scopes = {f.get("identity_scope") for f in frames if f.get("identity_scope")}
        occ_segs = {f.get("occupancy_segment") for f in frames if type(f.get("occupancy_segment")) is int and f.get("occupancy_segment") >= 1}
        summary_by_track[tid] = {
            "identity_scope": "half_court_slots.v1" if "half_court_slots.v1" in scopes else (next(iter(sorted(scopes)), None) if len(scopes) == 1 else None),
            "occupancy_segment_count": len(occ_segs),
            "pose_valid_frames": len(torso_tilts),
            "deep_lunge_count": deep_lunge_count,
            "min_knee_flexion_deg": round(min_knee, 1) if min_knee is not None else None,
            "mean_torso_tilt_deg": round(sum(torso_tilts) / len(torso_tilts), 1) if torso_tilts else None,
            "candidate_shot_count": len(cand_times),
            "candidate_hit_zones": dict(hit_zones),
            "candidate_hit_depths": dict(hit_depths),
            "continuous_pose_zones": dict(active_zones),
        }

    track_y_coords = defaultdict(list)
    for tid, frames in track_frames.items():
        for f in frames:
            xy = f.get("xy")
            if xy and len(xy) >= 2:
                track_y_coords[tid].append(float(xy[1]))
    team_near = [tid for tid, ys in track_y_coords.items() if _mean(ys) > 6.7]
    team_far = [tid for tid, ys in track_y_coords.items() if _mean(ys) <= 6.7]

    all_times = [f["time_sec"] for frames in track_frames.values() for f in frames]
    match_duration = (max(all_times) - min(all_times)) if all_times else 0.0
    verified_hit_events = [ev for ev in all_candidate_hit_events if _is_verified_hit_event(ev)]

    track_radars = {}
    for tid in summary_by_track:
        combat_radar = build_six_dimensional_combat_radar(
            tid,
            track_series=track_frames,
            hit_events=all_candidate_hit_events,
            team_near=team_near,
            team_far=team_far,
            match_duration_sec=match_duration,
            total_match_shots=len(verified_hit_events),
        )
        track_radars[tid] = combat_radar

    for tid, radar in track_radars.items():
        comp = compute_opponent_comparison(tid, track_radars, team_near, team_far, track_series=track_frames)
        if comp:
            radar["opponent_comparison"] = comp
        summary_by_track[tid]["combat_radar_profile"] = radar

    return summary_by_track


def build_tactical_and_joint_profile(
    track_id: str,
    *,
    performance_charts_data: Optional[Dict[str, Any]],
    movement_data: Optional[Dict[str, Any]],
    ability_scores_data: Optional[Dict[str, Any]] = None,
    shot_route_data: Optional[Dict[str, Any]] = None,
    duration_sec: float = 0.0,
    pose_evidence: Optional[Dict[str, Any]] = None,
    upstream_references: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build deterministic tactical archetype and duration-gated joint/fatigue guard."""
    duration = max(0.0, float(duration_sec or 0.0))
    mov = movement_data or {}
    moving_sec = float(mov.get("moving_time_sec") or 0.0)
    decel_count = int(mov.get("deceleration_event_count") or 0)
    dir_change_count = int(mov.get("direction_change_count") or 0)

    # 1. Net-folded half-court depth distribution (from 14x6 position_heatmap)
    hm = (performance_charts_data or {}).get("position_heatmap") or {}
    obs_sec = float(hm.get("observed_seconds") or 0.0)
    depth_sec = [0.0] * 7
    if obs_sec > 0 and isinstance(hm.get("cells"), list):
        for cell in hm["cells"]:
            if not isinstance(cell, dict):
                continue
            r = int(cell.get("row") or 0)
            s = float(cell.get("seconds") or 0.0)
            if 0 <= r < 14 and s > 0:
                depth_idx = (6 - r) if r < 7 else (r - 7)
                depth_sec[depth_idx] += s

    if obs_sec > 0:
        net_zone_pct = round(sum(depth_sec[0:2]) / obs_sec * 100.0, 1)
        service_line_pct = round(depth_sec[2] / obs_sec * 100.0, 1)
        front_total_pct = round(sum(depth_sec[0:3]) / obs_sec * 100.0, 1)
        mid_court_pct = round(sum(depth_sec[3:5]) / obs_sec * 100.0, 1)
        rear_court_pct = round(sum(depth_sec[5:7]) / obs_sec * 100.0, 1)
        depth_status = "observed"
    else:
        net_zone_pct = service_line_pct = front_total_pct = mid_court_pct = rear_court_pct = 0.0
        depth_status = "insufficient_measurements"

    # 2. Stroke height distribution & evidence status
    pe = pose_evidence or {}
    cand_zones = pe.get("candidate_hit_zones") or {}
    cand_total = sum(cand_zones.values())
    shot_prof = shot_route_data or {}
    shot_source = str(shot_prof.get("source") or "")
    if cand_total > 0:
        overhead_pct = round(cand_zones.get("overhead", 0) / cand_total * 100.0, 1)
        drive_pct = round(cand_zones.get("drive", 0) / cand_total * 100.0, 1)
        underhand_pct = round(cand_zones.get("underhand", 0) / cand_total * 100.0, 1)
        stroke_sample_count = cand_total
        stroke_evidence_status = (
            "observed"
            if shot_source == "paired_contacts_receiving_region" and shot_prof.get("status") == "observed"
            else "candidate_estimated"
        )
    else:
        cont_zones = pe.get("continuous_pose_zones") or {}
        cont_total = sum(cont_zones.values())
        if cont_total > 0:
            overhead_pct = round(cont_zones.get("overhead", 0) / cont_total * 100.0, 1)
            drive_pct = round(cont_zones.get("drive", 0) / cont_total * 100.0, 1)
            underhand_pct = round(cont_zones.get("underhand", 0) / cont_total * 100.0, 1)
            stroke_sample_count = int(shot_prof.get("hit_candidate_count") or shot_prof.get("sample_count") or 0)
            stroke_evidence_status = "candidate_estimated"
        else:
            overhead_pct = drive_pct = underhand_pct = 0.0
            stroke_sample_count = 0
            stroke_evidence_status = "position_only"

    # 3. Two-factor Tactical Archetype classification
    if depth_status != "observed":
        archetype_id = "insufficient_data"
        archetype_label = "样本不足 · 暂未定型"
        summary_copy = "有效球场定位时间不足，暂不判定球风画像。"
    elif front_total_pct >= 55.0 and rear_court_pct < 10.0:
        archetype_id = "front_court_net_drive"
        archetype_label = "中前场封网 · 平抽快挡型"
        summary_copy = (
            f"{front_total_pct}% 时间驻守在距球网 2.87 米内的中前场（纯网前占 {net_zone_pct}%），"
            f"后场底线停留仅 {rear_court_pct}%；以平抽快挡与抢网封压为主。"
        )
    elif rear_court_pct >= 28.0 and overhead_pct >= 35.0:
        archetype_id = "rear_court_power_attack"
        archetype_label = "后场突击 · 进攻重炮型"
        summary_copy = (
            f"后场底线区驻留达 {rear_court_pct}%，配合 {overhead_pct}% 的上手高点挥拍占比，"
            f"承担主要后场下压与劈杀突击任务。"
        )
    elif (mid_court_pct + rear_court_pct) >= 85.0 and rear_court_pct >= 12.0:
        archetype_id = "mid_rear_counter_rally"
        archetype_label = "中后场防守反击 · 拉吊周旋型"
        summary_copy = (
            f"中场（{mid_court_pct}%）与后场底线（{rear_court_pct}%）合计驻留达 "
            f"{round(mid_court_pct + rear_court_pct, 1)}%，高频后退接杀与中后场平抽拉吊周旋。"
        )
    else:
        archetype_id = "all_round_mid_transition"
        archetype_label = "中场均衡衔接 · 攻防轮转型"
        summary_copy = (
            f"中前场（{front_total_pct}%）与中场（{mid_court_pct}%）分布均衡，"
            f"在前后场轮转衔接与平抽过渡中承担枢纽角色。"
        )

    # 4. Joint Mechanical Load & Duration-Gated Fatigue Guard
    deep_lunge_count = int(pe.get("deep_lunge_count") or 0)
    min_knee_deg = pe.get("min_knee_flexion_deg")
    combined_impacts = decel_count + deep_lunge_count
    if combined_impacts >= 18:
        load_level = "high"
        load_label = "下肢制动冲击偏高"
    elif combined_impacts >= 8:
        load_level = "moderate"
        load_label = "下肢制动冲击适中"
    else:
        load_level = "low"
        load_label = "下肢制动冲击温和"

    eligible_for_fatigue = (
        duration >= MIN_FATIGUE_VIDEO_DURATION_SEC
        and moving_sec >= MIN_FATIGUE_MOVING_TIME_SEC
    )
    endurance_raw = (ability_scores_data or {}).get("endurance") or {}
    if not eligible_for_fatigue:
        fatigue_diagnosis = {
            "status": "insufficient_duration_for_fatigue",
            "required_video_sec": MIN_FATIGUE_VIDEO_DURATION_SEC,
            "required_moving_sec": MIN_FATIGUE_MOVING_TIME_SEC,
            "actual_video_sec": round(duration, 1),
            "actual_moving_sec": round(moving_sec, 1),
            "speed_decline_ratio": None,
            "copy": (
                f"当前视频时长 {int(duration // 60)}分{int(round(duration % 60))}秒"
                f"（净跑动 {round(moving_sec, 1)} 秒），未满 10 分钟生理疲劳评估基准，"
                f"仅统计本场瞬时急停与跨步冲击，不判定前后程体能衰减或核心力竭。"
            ),
        }
    elif (type(endurance_raw.get('speed_decline_ratio')) not in (int, float)
          or not math.isfinite(endurance_raw['speed_decline_ratio'])):
        fatigue_diagnosis = {
            'status': 'insufficient_measurements',
            'required_video_sec': MIN_FATIGUE_VIDEO_DURATION_SEC,
            'required_moving_sec': MIN_FATIGUE_MOVING_TIME_SEC,
            'actual_video_sec': round(duration, 1),
            'actual_moving_sec': round(moving_sec, 1),
            'speed_decline_ratio': None,
            'copy': '视频时长达标，但缺少有效的前后程速度对比，暂不判定体能衰减。',
        }
    else:
        decline = float(endurance_raw['speed_decline_ratio'])
        fatigue_diagnosis = {
            "status": "evaluated",
            "required_video_sec": MIN_FATIGUE_VIDEO_DURATION_SEC,
            "required_moving_sec": MIN_FATIGUE_MOVING_TIME_SEC,
            "actual_video_sec": round(duration, 1),
            "actual_moving_sec": round(moving_sec, 1),
            "speed_decline_ratio": round(decline, 4),
            "copy": (
                f"对局已满生理耐力评估基准，后程移动速度衰减率为 {round(decline * 100, 1)}%。"
            ),
        }

    return {
        "schema_version": TACTICAL_JOINT_SCHEMA_VERSION,
        "track_id": track_id,
        "upstream_version": upstream_references or {"algorithm_version": TACTICAL_JOINT_SCHEMA_VERSION},
        "tactical_archetype": {
            "status": depth_status,
            "archetype_id": archetype_id,
            "archetype_label": archetype_label,
            "evidence_status": stroke_evidence_status,
            "half_court_depth_distribution": {
                "net_zone_0_to_191cm_pct": net_zone_pct,
                "service_line_191_to_287cm_pct": service_line_pct,
                "front_court_total_pct": front_total_pct,
                "mid_court_287_to_479cm_pct": mid_court_pct,
                "rear_court_479_to_670cm_pct": rear_court_pct,
            },
            "stroke_height_distribution": {
                "evidence_status": stroke_evidence_status,
                "sample_count": stroke_sample_count,
                "overhead_pct": overhead_pct,
                "drive_flat_pct": drive_pct,
                "underhand_net_pct": underhand_pct,
            },
            "summary_copy": summary_copy,
        },
        "joint_and_fatigue_guard": {
            "instant_mechanical_load": {
                "status": "observed" if moving_sec > 0 else "insufficient_measurements",
                "load_level": load_level,
                "load_label": load_label,
                "hard_deceleration_count": decel_count,
                "direction_change_count": dir_change_count,
                "deep_lunge_count": deep_lunge_count,
                "min_knee_flexion_deg": min_knee_deg,
                "copy": (
                    f"本场实测急停制动 {decel_count} 次、深跨步救球（膝角<{int(DEEP_LUNGE_MAX_KNEE_ANGLE_DEG)}°）{deep_lunge_count} 次"
                    + (f"（最小屈膝角 {min_knee_deg}°）" if min_knee_deg is not None else "")
                    + f"，{load_label}。"
                ),
            },
            "fatigue_decay_diagnosis": fatigue_diagnosis,
        },
        "combat_radar_profile": (
            pe.get("combat_radar_profile")
            if isinstance(pe.get("combat_radar_profile"), dict)
            else build_empty_combat_radar_profile(track_id)
        ),
    }
