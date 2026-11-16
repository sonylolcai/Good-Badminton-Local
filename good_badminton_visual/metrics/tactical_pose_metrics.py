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

    # Knee flexion stats (with debounced deep lunge event counting, gap > 0.5s)
    knee_angle_vals = [a for _, a in knee_flexions]
    deep_lunge_ratio = round(sum(1 for a in knee_angle_vals if a <= 135.0) / len(knee_angle_vals), 3) if knee_angle_vals else 0.0
    mean_knee_flex = round(_mean(knee_angle_vals), 2) if knee_angle_vals else 0.0
    min_knee_flex = round(float(min(knee_angle_vals)), 1) if knee_angle_vals else None
    deep_lunge_count = 0
    last_lunge_t = -999.0
    for t_val, k_ang in knee_flexions:
        if k_ang < 135.0:
            if t_val - last_lunge_t > 0.5:
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
    if candidate_routes_path and Path(candidate_routes_path).is_file():
        try:
            routes_payload = json.loads(Path(candidate_routes_path).read_text(encoding="utf-8"))
            for rec in routes_payload.get("records") or routes_payload.get("events") or []:
                tid = rec.get("actor_track_id") or rec.get("hitter_track_id")
                t_val = rec.get("time_sec") if rec.get("time_sec") is not None else rec.get("source_time_sec")
                if tid and isinstance(t_val, (int, float)) and math.isfinite(t_val):
                    candidate_times_by_track[str(tid)].append(float(t_val))
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
            if not isinstance(t_sec, (int, float)) or not math.isfinite(t_sec):
                if isinstance(frame, int) and fps and fps > 0:
                    t_sec = frame / float(fps)
                else:
                    continue
            key = ("time", float(t_sec))
            if key in seen_samples:
                continue
            seen_samples.add(key)
            for tr in (row.get("spatial") or {}).get("tracks") or []:
                if not isinstance(tr, dict) or tr.get("status") != "detected" or not tr.get("track_id"):
                    continue
                tid = str(tr["track_id"])
                pose = tr.get("pose") or {}
                if (pose.get('is_current_measurement') is not True
                        or pose.get('format') != 'coco17_image_v1'):
                    continue
                pm = _analyze_coco17_frame(pose.get("keypoints_image"), pose.get("keypoint_scores"))
                track_frames[tid].append({
                    "time_sec": float(t_sec),
                    "xy": tr.get("court_xy_m"),
                    "pose_metrics": pm,
                })

    summary_by_track: Dict[str, Dict[str, Any]] = {}
    for tid, frames in track_frames.items():
        frames.sort(key=lambda item: item["time_sec"])
        # Debounced deep lunge count (knee < 135 deg, >=0.5s separation)
        deep_lunge_count = 0
        last_lunge_t = -999.0
        min_knee = None
        torso_tilts = []
        active_zones = Counter()
        for item in frames:
            pm = item.get("pose_metrics")
            if not pm:
                continue
            if pm["torso_tilt"] is not None:
                torso_tilts.append(pm["torso_tilt"])
            if pm["min_knee"] is not None:
                min_knee = pm["min_knee"] if min_knee is None else min(min_knee, pm["min_knee"])
                if pm["min_knee"] < 135.0:
                    if item["time_sec"] - last_lunge_t > 0.5:
                        deep_lunge_count += 1
                    last_lunge_t = item["time_sec"]
            if pm["stroke_zone"]:
                active_zones[pm["stroke_zone"]] += 1

        # Candidate hit window evaluation if candidate timestamps exist
        cand_times = candidate_times_by_track.get(tid) or []
        hit_zones = Counter()
        hit_depths = Counter()
        for ct in cand_times:
            near = [f for f in frames if abs(f["time_sec"] - ct) <= 0.15]
            if not near:
                continue
            best_f = min(near, key=lambda f: abs(f["time_sec"] - ct))
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

        summary_by_track[tid] = {
            "pose_valid_frames": len(torso_tilts),
            "deep_lunge_count": deep_lunge_count,
            "min_knee_flexion_deg": round(min_knee, 1) if min_knee is not None else None,
            "mean_torso_tilt_deg": round(sum(torso_tilts) / len(torso_tilts), 1) if torso_tilts else None,
            "candidate_shot_count": len(cand_times),
            "candidate_hit_zones": dict(hit_zones),
            "candidate_hit_depths": dict(hit_depths),
            "continuous_pose_zones": dict(active_zones),
        }
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
                    f"本场实测急停制动 {decel_count} 次、深跨步救球（膝角<135°）{deep_lunge_count} 次"
                    + (f"（最小屈膝角 {min_knee_deg}°）" if min_knee_deg is not None else "")
                    + f"，{load_label}。"
                ),
            },
            "fatigue_decay_diagnosis": fatigue_diagnosis,
        },
    }
