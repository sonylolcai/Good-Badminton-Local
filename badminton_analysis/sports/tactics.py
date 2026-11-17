"""Badminton Tactical Spatial Analysis and Biomechanical Shot Classification.

This module provides:
1. 3D Court Spatial Discretization (CourtZone: Net, Midcourt, Rearcourt x Left, Center, Right x High, Mid, Low).
2. Biomechanical Pose Analysis (Event-driven Keyframe Pose via YOLO-Pose: Forehand vs Backhand for Left/Right Handed).
3. Spatial Transfer Matrix Shot Classifier (BadmintonShotClassifier: Serve, Lift, Smash, Half-smash, Push, Drive, Drop, Net Shot, Cross Net, Block, Net Error).
4. Tactical Narrative Engine (Translating shot sequences and terminals into natural language match commentary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class DepthZone(str, Enum):
    UNKNOWN = "unknown"
    NET = "net"            # 前场网前
    MIDCOURT = "midcourt"  # 中场腰部
    REARCOURT = "rearcourt"# 后场底线


class LateralZone(str, Enum):
    UNKNOWN = "unknown"
    LEFT = "left"          # 左半区
    CENTER = "center"      # 中路
    RIGHT = "right"        # 右半区


class HeightZone(str, Enum):
    UNKNOWN = "unknown"
    OVERHEAD = "overhead"  # 高位/头顶
    WAIST = "waist"        # 中位/腰部
    UNDERHAND = "underhand"# 低位/下手被动


class HandSide(str, Enum):
    FOREHAND = "forehand"  # 正手
    BACKHAND = "backhand"  # 反手
    UNKNOWN = "unknown"


@dataclass
class CourtZone:
    court_side: str        # "near" | "far"
    depth: DepthZone
    lateral: LateralZone
    height: HeightZone = HeightZone.UNKNOWN
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.court_side}_{self.depth.value}_{self.lateral.value}"


@dataclass
class TacticalShot:
    """Represents a single tactical stroke in a rally."""
    shot_index: int
    hitter: str                  # "near_team" | "far_team"
    dominant_hand: str           # "left" | "right"
    hand_side: HandSide          # Forehand | Backhand
    shot_type: str               # "serve", "lift", "smash", "half_smash", "push", "drive", "drop", "net_shot", "cross_net", "block", "net_error"
    shot_type_cn: str            # "发球", "挑球", "重杀", "软压", "推球", "平抽", "吊球", "放网", "勾对角", "接杀挡网", "挂网"
    start_time_s: float
    end_time_s: float
    duration_s: float
    peak_speed_kmh: Optional[float]
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    start_zone: CourtZone
    target_zone: CourtZone
    flight_direction: str        # "near_to_far" | "far_to_near"
    tactical_line: str           # "straight" | "cross_court"
    is_terminal: bool = False
    terminal_result: Optional[str] = None  # "out_of_bounds", "in_court_landing", "net_error", None
    description: str = ""


class BadmintonSpatialZoning:
    """Maps pixel or metric coordinates into 3D court spatial zones."""

    def __init__(
        self,
        net_y: Optional[float] = None,
        court_x_bounds: Optional[Tuple[float, float]] = None,
        near_bounds_y: Optional[Tuple[float, float]] = None,
        far_bounds_y: Optional[Tuple[float, float]] = None,
    ):
        self.net_y = net_y
        self.calibrated = all(v is not None for v in (net_y, court_x_bounds, near_bounds_y, far_bounds_y))
        if not self.calibrated:
            return
        self.court_x_min, self.court_x_max = court_x_bounds
        self.near_y_min, self.near_y_max = near_bounds_y
        self.far_y_min, self.far_y_max = far_bounds_y

    def get_zone(self, x: float, y: float, height_level: HeightZone = HeightZone.UNKNOWN) -> CourtZone:
        """Discretize 2D coordinates into a structured CourtZone."""
        if not self.calibrated:
            return CourtZone("unknown", DepthZone.UNKNOWN, LateralZone.UNKNOWN, height_level, "unknown")
        is_near = (y >= self.net_y)
        court_side = "near" if is_near else "far"

        # Lateral partition (Left / Center / Right)
        x_span = max(1.0, self.court_x_max - self.court_x_min)
        norm_x = (x - self.court_x_min) / x_span
        if norm_x < 0.35:
            lat = LateralZone.LEFT
        elif norm_x > 0.65:
            lat = LateralZone.RIGHT
        else:
            lat = LateralZone.CENTER

        # Depth partition (Net / Midcourt / Rearcourt)
        if is_near:
            # Use supplied near-court calibration.
            y_span = max(1.0, self.near_y_max - self.near_y_min)
            rel_y = (y - self.near_y_min) / y_span
            if rel_y < 0.28:
                dep = DepthZone.NET
            elif rel_y < 0.68:
                dep = DepthZone.MIDCOURT
            else:
                dep = DepthZone.REARCOURT
        else:
            # Use supplied far-court calibration.
            y_span = max(1.0, self.net_y - self.far_y_min)
            rel_y = (self.net_y - y) / y_span
            if rel_y < 0.28:
                dep = DepthZone.NET
            elif rel_y < 0.68:
                dep = DepthZone.MIDCOURT
            else:
                dep = DepthZone.REARCOURT

        return CourtZone(
            court_side=court_side,
            depth=dep,
            lateral=lat,
            height=height_level,
        )


class BiomechanicalPoseAnalyzer:
    """Event-driven pose analyzer for determining Forehand vs Backhand."""

    def __init__(
        self,
        model_path: str = "weights/yolo11n-pose.pt",
        device: str = "cpu",
        conf: float = 0.25,
    ):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)

    def analyze_stroke_hand(
        self,
        frame: np.ndarray,
        ball_xy: Tuple[float, float],
        is_near_player: bool = True,
        dominant_hand: str = "unknown",
        net_y: Optional[float] = None,
    ) -> Tuple[HandSide, HeightZone, Optional[np.ndarray]]:
        """Determine Forehand vs Backhand and hitting height using player keypoints."""
        if net_y is None or dominant_hand not in ("left", "right"):
            return HandSide.UNKNOWN, HeightZone.UNKNOWN, None
        try:
            self._ensure_model()
            # Run inference on 640px for ultra-fast response
            results = self._model(frame, imgsz=640, device=self.device, conf=self.conf, verbose=False)
            res = results[0]
            if len(res.boxes) == 0:
                return HandSide.UNKNOWN, HeightZone.UNKNOWN, None

            best_person_idx = -1
            min_dist = float("inf")
            target_kp = None

            # Filter persons matching near/far half and find the one closest to ball
            bx, by = ball_xy
            for idx, box in enumerate(res.boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                person_near = (xyxy[3] > net_y)
                if person_near != is_near_player:
                    continue
                center_x = (xyxy[0] + xyxy[2]) / 2.0
                center_y = (xyxy[1] + xyxy[3]) / 2.0
                dist = math.hypot(center_x - bx, center_y - by)
                if dist < min_dist:
                    min_dist = dist
                    best_person_idx = idx

            if best_person_idx == -1:
                return HandSide.UNKNOWN, HeightZone.UNKNOWN, None

            kp = res.keypoints.xy[best_person_idx].cpu().numpy()  # (17, 2)
            target_kp = kp

            # Keypoints:
            # 5: left_shoulder, 6: right_shoulder
            # 7: left_elbow, 8: right_elbow
            # 9: left_wrist, 10: right_wrist
            ls, rs = kp[5], kp[6]
            lw, rw = kp[9], kp[10]

            # If shoulders missing or invalid
            if ls[0] == 0 or rs[0] == 0:
                return HandSide.UNKNOWN, HeightZone.UNKNOWN, target_kp

            torso_cx = (ls[0] + rs[0]) / 2.0
            torso_cy = (ls[1] + rs[1]) / 2.0
            shoulder_width = abs(rs[0] - ls[0])

            # Determine hitting hand keypoints
            is_left = (dominant_hand.lower() == "left")
            hitting_wrist = lw if is_left else rw
            hitting_shoulder = ls if is_left else rs

            if not np.isfinite(hitting_wrist).all() or np.any(hitting_wrist <= 0) or shoulder_width <= 0:
                return HandSide.UNKNOWN, HeightZone.UNKNOWN, target_kp

            # Height zone determination
            if hitting_wrist[1] < hitting_shoulder[1] - 0.2 * shoulder_width:
                height = HeightZone.OVERHEAD
            elif hitting_wrist[1] > torso_cy + 1.2 * shoulder_width:
                height = HeightZone.UNDERHAND
            else:
                height = HeightZone.WAIST

            # Forehand vs Backhand biomechanical lateral projection:
            # When camera looks from behind near players towards net:
            # For Near Player (back mostly to camera):
            #   Left Shoulder is screen left (smaller X), Right Shoulder is screen right (larger X).
            #   Left-handed player:
            #     Left hand on player's left side (X <= torso_cx) -> Forehand!
            #     Left hand crosses body to player's right side (X > torso_cx + 0.15*W) -> Backhand!
            #   Right-handed player:
            #     Right hand on player's right side (X >= torso_cx) -> Forehand!
            #     Right hand crosses body to player's left side (X < torso_cx - 0.15*W) -> Backhand!
            # For Far Player (facing towards camera):
            #   Left Shoulder is screen right (larger X), Right Shoulder is screen left (smaller X).
            #   Left-handed player:
            #     Left hand on player's left side (X >= torso_cx) -> Forehand!
            #     Left hand crosses body to player's right side (X < torso_cx) -> Backhand!
            #   Right-handed player:
            #     Right hand on player's right side (X <= torso_cx) -> Forehand!
            #     Right hand crosses body to player's left side (X > torso_cx) -> Backhand!

            if is_near_player:
                if is_left:
                    hand_side = HandSide.FOREHAND if hitting_wrist[0] <= torso_cx + 0.15 * shoulder_width else HandSide.BACKHAND
                else:
                    hand_side = HandSide.FOREHAND if hitting_wrist[0] >= torso_cx - 0.15 * shoulder_width else HandSide.BACKHAND
            else:
                if is_left:
                    hand_side = HandSide.FOREHAND if hitting_wrist[0] >= torso_cx - 0.15 * shoulder_width else HandSide.BACKHAND
                else:
                    hand_side = HandSide.FOREHAND if hitting_wrist[0] <= torso_cx + 0.15 * shoulder_width else HandSide.BACKHAND

            return hand_side, height, target_kp

        except Exception:
            return HandSide.UNKNOWN, HeightZone.UNKNOWN, None


class BadmintonShotClassifier:
    """Classifies stroke type based on 3D spatial transfer matrix and kinematics."""

    def __init__(self, zoning: Optional[BadmintonSpatialZoning] = None):
        self.zoning = zoning or BadmintonSpatialZoning()

    def classify_shot(
        self,
        shot_index: int,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
        duration_s: float,
        peak_speed_kmh: float,
        tactical_line: str,
        hand_side: HandSide,
        height_zone: HeightZone,
        dominant_hand: str = "right",
        is_rally_starter: bool = False,
        is_rally_ender: bool = False,
        terminal_type: Optional[str] = None,
        serve_observed: bool = False,
    ) -> Tuple[str, str]:
        """Classify stroke into (shot_type, shot_type_cn).

        Returns:
            Tuple of (english_code, chinese_name).
        """
        start_zone = self.zoning.get_zone(start_xy[0], start_xy[1], height_zone)
        target_zone = self.zoning.get_zone(end_xy[0], end_xy[1])
        if (start_zone.depth == DepthZone.UNKNOWN or target_zone.depth == DepthZone.UNKNOWN
                or peak_speed_kmh is None):
            return "unknown", "击球类型未知"
        hand_prefix = "左手" if dominant_hand == "left" else "右手"
        hand_str = "正手" if hand_side == HandSide.FOREHAND else "反手" if hand_side == HandSide.BACKHAND else ""

        # 1. Terminal Net Error
        if is_rally_ender and terminal_type == "net_error":
            return "net_error", f"{hand_str}击球挂网"

        # 2. Serve Detection
        if serve_observed:
            if start_zone.depth in (DepthZone.NET, DepthZone.MIDCOURT):
                if peak_speed_kmh < 55.0 and duration_s > 0.8:
                    return "low_serve", f"{hand_str}低发球" if hand_str else "发球"
                elif peak_speed_kmh >= 55.0 or duration_s > 1.2:
                    return "high_serve", f"{hand_str}高远发球" if hand_str else "发球"
                return "serve", f"{hand_str}发球" if hand_str else "发球"

        # 3. Smash vs Half-Smash vs Drop (Overhead from Rearcourt/Midcourt)
        if start_zone.depth in (DepthZone.REARCOURT, DepthZone.MIDCOURT):
            if height_zone == HeightZone.OVERHEAD or peak_speed_kmh >= 110.0:
                if peak_speed_kmh >= 135.0 or (is_rally_ender and peak_speed_kmh >= 90.0):
                    return "smash", f"{hand_str}重杀" if hand_str else "杀球"
                elif 70.0 <= peak_speed_kmh < 135.0:
                    if target_zone.depth == DepthZone.NET:
                        return "drop", f"{hand_str}吊球" if hand_str else "吊球"
                    else:
                        return "half_smash", f"{hand_str}软压" if hand_str else "软压"
                elif target_zone.depth == DepthZone.REARCOURT and duration_s >= 1.2:
                    return "clear", f"{hand_str}高远球" if hand_str else "高远球"
                elif target_zone.depth == DepthZone.NET:
                    return "drop", f"{hand_str}劈吊" if hand_str else "吊球"

        # 4. Lift vs Push vs Net Shot from Front Court
        if start_zone.depth == DepthZone.NET:
            if target_zone.depth == DepthZone.REARCOURT:
                if duration_s >= 1.05:
                    return "lift", f"{hand_str}挑球" if hand_str else "挑球"
                else:
                    return "push", f"{hand_str}推球" if hand_str else "推球"
            elif target_zone.depth == DepthZone.NET:
                dx = abs(end_xy[0] - start_xy[0])
                if dx > 600.0 or tactical_line == "cross_court":
                    return "cross_net", f"{hand_str}勾对角" if hand_str else "勾对角"
                else:
                    return "net_shot", f"{hand_str}放网" if hand_str else "放网"
            else:
                return "push", f"{hand_str}推中场" if hand_str else "推球"

        # 5. Drive from Midcourt
        if start_zone.depth == DepthZone.MIDCOURT and target_zone.depth in (DepthZone.MIDCOURT, DepthZone.REARCOURT):
            if duration_s < 0.85 and peak_speed_kmh >= 50.0:
                return "drive", f"{hand_str}平抽" if hand_str else "平抽"
            elif duration_s >= 1.1:
                return "lift", f"{hand_str}挑后场" if hand_str else "挑球"

        # 6. Defense Block
        if start_zone.depth in (DepthZone.MIDCOURT, DepthZone.REARCOURT) and target_zone.depth == DepthZone.NET:
            return "block", f"{hand_str}挡网" if hand_str else "接杀挡网"

        # Default fallback
        if target_zone.depth == DepthZone.REARCOURT:
            return "lift", f"{hand_str}挑球" if hand_str else "挑球"
        elif peak_speed_kmh >= 90.0:
            return "attack", f"{hand_str}进攻" if hand_str else "进攻"
        return "push", f"{hand_str}回球" if hand_str else "回球"


class TacticalNarrativeEngine:
    """Translates tactical events and shot sequences into professional commentary."""

    @staticmethod
    def generate_rally_narrative(
        rally_id: int,
        shots: List[TacticalShot],
        terminal: Optional[dict] = None,
    ) -> str:
        """Generate structured and natural-language narrative for a rally."""
        if not shots:
            return f"【第{rally_id}回合】数据不足，未形成有效回合。"

        parts = []
        for s in shots:
            hitter_name = {"near_team": "近端选手", "far_team": "远端选手"}.get(s.hitter, "选手未知：")
            shot_desc = s.shot_type_cn
            parts.append(f"{hitter_name}{shot_desc}")

        narrative_flow = " -> ".join(parts)

        term_desc = "（结束信号未知）"
        if terminal:
            label = {
                "dead_ball": "死球",
                "out_of_bounds": "界外落地",
                "in_court_landing": "界内落地",
            }.get(terminal.get("terminal_type"), "结束信号未知")
            term_desc = f"（{label}）"
        return f"【运动片段{rally_id} ({len(shots)}段轨迹)】{narrative_flow}{term_desc}"
