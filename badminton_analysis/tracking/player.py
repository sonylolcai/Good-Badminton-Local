from collections import deque
import time

import numpy as np

from ..court.mapper import CourtMapper


class PlayerTracker:
    """
    Player tracking system.

    Tracks player positions, court coordinates, movement statistics, and writes
    one structured detection record per processed court frame.
    """

    def __init__(self, corners, threshold=680, history_size=50, detection_writer=None, fps=30,
                 court_dimensions=(6.1, 13.4), world_points_m=None):
        self.threshold = threshold
        self.fps = fps
        self.detection_writer = detection_writer
        self.max_frame_distance = 8.0 / self.fps

        self.players = {
            "upper": None,
            "lower": None,
        }
        self.history = {
            "upper": deque(maxlen=history_size),
            "lower": deque(maxlen=history_size),
        }
        self.court_history = {
            "upper": deque(maxlen=history_size),
            "lower": deque(maxlen=history_size),
        }

        self.match_stats = {
            "upper": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
            "lower": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
        }
        self.rally_stats = {
            "upper": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
            "lower": {"total_distance": 0, "max_speed": 0, "total_frames": 0},
        }
        self.current_speed = {
            "upper": 0,
            "lower": 0,
        }

        self.court_mapper = CourtMapper(
            corners,
            court_dimensions=tuple(float(value) for value in court_dimensions),
            world_points_m=world_points_m,
        )
        self.last_update_timing = {
            "player_tracking_seconds": 0.0,
            "jsonl_write_seconds": 0.0,
        }

    def _empty_player_record(self):
        return {
            "image": None,
            "court": None,
            "speed": None,
            "hands": {
                "left": None,
                "right": None,
            },
            "position_evidence": {
                "status": "missing",
                "method": None,
                "confidence": 0.0,
                "degraded": False,
                "source": None,
            },
        }

    def _initialize_player_record(self):
        return {
            "upper": self._empty_player_record(),
            "lower": self._empty_player_record(),
        }

    def _point_or_none(self, point, zero_is_none=False):
        if point is None:
            return None
        try:
            x, y = point[0], point[1]
        except (TypeError, IndexError):
            return None
        if x is None or y is None:
            return None
        if zero_is_none and float(x) == 0.0 and float(y) == 0.0:
            return None
        return [float(x), float(y)]

    def write_detection_record(self, frame_index, players_record, ball_image_position, detect_frame_count,
                               ball_detection=None, spatial_state=None):
        started = time.perf_counter()
        if self.detection_writer is None:
            return 0.0

        shuttlecock_record = {
            "image": self._point_or_none(ball_image_position, zero_is_none=True),
            "status": "missing",
            "confidence": None,
            "source": None,
            "measurement_kind": None,
            "confidence_status": None,
            "gap_frames": 0,
            "accepted": False,
        }
        if ball_detection:
            shuttlecock_record.update(
                {
                    "status": ball_detection.get("status", "missing"),
                    "confidence": ball_detection.get("confidence"),
                    "source": ball_detection.get("source"),
                    "measurement_kind": ball_detection.get("measurement_kind"),
                    "confidence_status": ball_detection.get("confidence_status"),
                    "gap_frames": int(ball_detection.get("gap_frames", 0)),
                    "accepted": bool(ball_detection.get("accepted", False)),
                    "visible": bool(ball_detection.get("visible", False)),
                    "candidate_count": int(ball_detection.get("candidate_count", 0)),
                    "raw_candidate_count": int(ball_detection.get("raw_candidate_count", 0)),
                    "filtered_rejections": dict(ball_detection.get("filtered_rejections", {})),
                    "rejection_reason": ball_detection.get("rejection_reason"),
                }
            )

        record = {
            "schema_version": "2.0",
            "frame": int(frame_index),
            "time_sec": round(frame_index / self.fps, 6) if self.fps else None,
            "detect_frame": int(detect_frame_count),
            "players": players_record,
            "shuttlecock": shuttlecock_record,
        }
        if spatial_state is not None:
            # The v1 upper/lower records are retained for existing consumers.
            # All new tracking and spatial analytics must read this field:
            # persistent identity is track_id and zone_id is only instantaneous.
            record["spatial"] = spatial_state
        self.detection_writer.write(record)
        return time.perf_counter() - started

    def update(self, frame_index, centroids, ball_image_position, left_hand_positions, right_hand_positions,
               detect_frame_count, pose_detections=None, ball_detection=None, spatial_state=None):
        started = time.perf_counter()
        players_record = self._initialize_player_record()
        pose_detections = pose_detections or []

        for region in ["upper", "lower"]:
            if self.players[region] is not None:
                self.match_stats[region]["total_frames"] += 1
                self.rally_stats[region]["total_frames"] += 1

        upper_court_centroids = []
        lower_court_centroids = []
        for centroid in centroids:
            if centroid[1] < self.threshold:
                upper_court_centroids.append(centroid)
            else:
                lower_court_centroids.append(centroid)

        # Keep one observation for each legacy slot.  Applying this policy only
        # to the upper slot made the lower slot depend on YOLO result ordering:
        # the last incidental person could overwrite the tracked player.
        upper_court_centroids = self._select_region_candidate(
            "upper", upper_court_centroids
        )
        lower_court_centroids = self._select_region_candidate(
            "lower", lower_court_centroids
        )

        filtered_centroids = upper_court_centroids + lower_court_centroids

        for centroid in filtered_centroids:
            try:
                region = "upper" if centroid[1] < self.threshold else "lower"
                left_hand = left_hand_positions.get(centroid[1])
                right_hand = right_hand_positions.get(centroid[1])
                pose_detection = self._find_pose_detection(centroid, pose_detections)
                self._update_player_position(
                    region, centroid, left_hand, right_hand, players_record, pose_detection
                )
            except Exception as exc:
                print(f"Error processing player position: {exc}")
                import traceback
                traceback.print_exc()

        jsonl_write_seconds = self.write_detection_record(
            frame_index,
            players_record,
            ball_image_position,
            detect_frame_count,
            ball_detection=ball_detection,
            spatial_state=spatial_state,
        )
        self.last_update_timing = {
            "player_tracking_seconds": max(0.0, time.perf_counter() - started - jsonl_write_seconds),
            "jsonl_write_seconds": float(jsonl_write_seconds),
        }
        return self.players

    def _select_region_candidate(self, region, candidates):
        """Return one deterministic candidate for a temporary legacy slot.

        Existing fixed-camera processing still exposes ``upper`` and ``lower``
        output fields.  Until it is migrated to persistent track IDs, retain
        temporal continuity when a prior player location exists.  On the first
        frame, choose the candidate furthest toward the camera in image space;
        this preserves the former upper-slot behaviour and applies it equally
        to the lower slot.
        """
        if not candidates:
            return []

        previous = self.players[region]
        if previous is None:
            selected = max(candidates, key=lambda point: point[1])
        else:
            selected = min(
                candidates,
                key=lambda point: float(
                    np.hypot(point[0] - previous[0], point[1] - previous[1])
                ),
            )
        return [selected]

    @staticmethod
    def _find_pose_detection(centroid, pose_detections):
        best = None
        best_distance = float("inf")
        for detection in pose_detections:
            location = detection.get("location")
            if not location or len(location) < 2:
                continue
            distance = float(np.hypot(float(location[0]) - centroid[0], float(location[1]) - centroid[1]))
            if distance < best_distance:
                best = detection
                best_distance = distance
        return best if best_distance <= 2.0 else None

    @staticmethod
    def _serializable_list(value):
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        return array.tolist()

    def _position_evidence(self, pose_detection):
        if not pose_detection:
            return {
                "status": "detected",
                "method": "legacy_pose_location",
                "confidence": None,
                "degraded": False,
                "source": "pose_model",
            }
        degraded = bool(pose_detection.get("location_degraded", False))
        inference = pose_detection.get("inference") or {}
        return {
            "status": "detected_degraded" if degraded else "detected",
            "method": pose_detection.get("location_method", "unknown"),
            "confidence": (
                float(pose_detection["location_confidence"])
                if pose_detection.get("location_confidence") is not None
                else None
            ),
            "degraded": degraded,
            "source": pose_detection.get("source", "pose_model"),
            "merged_sources": list(pose_detection.get("merged_sources", [])),
            "person_confidence": (
                float(pose_detection["confidence"])
                if pose_detection.get("confidence") is not None
                else None
            ),
            "bbox_xyxy": self._serializable_list(pose_detection.get("bbox")),
            "inference": {
                "model": inference.get("model"),
                "imgsz": inference.get("imgsz"),
                "conf": inference.get("conf"),
                "device": inference.get("device"),
                "roi": list(inference.get("roi", [])),
                "input_shape": list(inference.get("input_shape", [])),
            },
        }

    def _update_player_position(self, region, centroid, left_hand_pos, right_hand_pos, players_record,
                                pose_detection=None):
        self.players[region] = centroid
        self.history[region].append(centroid)

        court_position = self.court_mapper.image_to_court(centroid)
        self.court_history[region].append(court_position)

        player_record = players_record[region]
        player_record["image"] = self._point_or_none(centroid)
        player_record["court"] = self._point_or_none(court_position)
        player_record["speed"] = float(self.current_speed[region])
        player_record["position_evidence"] = self._position_evidence(pose_detection)
        if left_hand_pos:
            player_record["hands"]["left"] = self._point_or_none(left_hand_pos)
        if right_hand_pos:
            player_record["hands"]["right"] = self._point_or_none(right_hand_pos)

    def _update_rally_and_match_stats(self, region, distance, speed):
        capped_speed = round(min(speed, 8.0), 2)

        self.rally_stats[region]["total_distance"] += distance
        self.rally_stats[region]["max_speed"] = max(self.rally_stats[region]["max_speed"], capped_speed)
        self.current_speed[region] = capped_speed

        self.match_stats[region]["total_distance"] += distance
        self.match_stats[region]["max_speed"] = max(self.match_stats[region]["max_speed"], capped_speed)
        self.current_speed[region] = capped_speed

    def start_new_rally(self):
        for region in ["upper", "lower"]:
            self.rally_stats[region]["total_distance"] = 0
            self.rally_stats[region]["max_speed"] = 0
            self.rally_stats[region]["total_frames"] = 0

    def get_player_movement_stats(self):
        stats = {}
        for region in ["upper", "lower"]:
            history = [pos for pos in list(self.court_history[region]) if pos is not None]
            region_stats = {
                "current_speed": 0,
                "rally_avg_speed": 0,
                "rally_max_speed": 0,
                "rally_distance": 0,
                "match_avg_speed": 0,
                "match_max_speed": 0,
                "match_distance": 0,
                "position_count": len(history),
            }

            if len(history) < 2:
                stats[region] = region_stats
                continue

            current_time = len(history) - 1
            window_start = max(0, current_time - int(self.fps / 2))
            half_second_total_distance = 0
            valid_frames = 0
            actual_time_span = 0
            sample_interval = 5

            if current_time - window_start < sample_interval:
                sample_points = [window_start, current_time]
            else:
                sample_points = list(range(window_start, current_time + 1, sample_interval))
                if current_time not in sample_points:
                    sample_points.append(current_time)

            for i in range(len(sample_points) - 1):
                idx1 = sample_points[i]
                idx2 = sample_points[i + 1]
                p1 = np.array(history[idx1])
                p2 = np.array(history[idx2])
                distance = np.linalg.norm(p2 - p1)
                time_span = (idx2 - idx1) / self.fps
                max_possible_distance = self.max_frame_distance * (idx2 - idx1)

                if distance > 0.05 and distance < max_possible_distance:
                    half_second_total_distance += distance
                    valid_frames += 1
                    actual_time_span += time_span

            current_speed = 0
            if valid_frames > 0 and actual_time_span > 0:
                current_speed = half_second_total_distance / actual_time_span
                self._update_rally_and_match_stats(region, half_second_total_distance, current_speed)

            current_speed = min(current_speed, 8.0)

            rally_distance = self.rally_stats[region]["total_distance"]
            rally_max_speed = self.rally_stats[region]["max_speed"]
            rally_frames = self.rally_stats[region]["total_frames"]
            if rally_frames > 1 and self.fps > 0:
                rally_time = rally_frames / self.fps
                rally_avg_speed = rally_distance / rally_time if rally_time > 0 else 0
            else:
                rally_avg_speed = 0

            match_distance = self.match_stats[region]["total_distance"]
            match_max_speed = self.match_stats[region]["max_speed"]
            match_frames = self.match_stats[region]["total_frames"]
            if match_frames > 1 and self.fps > 0:
                match_time = match_frames / self.fps
                match_avg_speed = match_distance / match_time if match_time > 0 else 0
            else:
                match_avg_speed = 0

            region_stats["current_speed"] = round(current_speed, 2)
            region_stats["rally_avg_speed"] = round(rally_avg_speed, 2)
            region_stats["rally_max_speed"] = round(rally_max_speed, 2)
            region_stats["rally_distance"] = round(rally_distance, 2)
            region_stats["match_avg_speed"] = round(match_avg_speed, 2)
            region_stats["match_max_speed"] = round(match_max_speed, 2)
            region_stats["match_distance"] = round(match_distance, 2)
            stats[region] = region_stats

        return stats

    def get_player_trajectories(self):
        return {region: list(history) for region, history in self.history.items()}

    def close(self):
        if self.detection_writer is not None:
            self.detection_writer.close()
