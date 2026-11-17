import time

import cv2
import numpy as np

from ..detection.rtmpose import RTMPoseProcessor


class PlayerPoseVisualizer:
    """Detect, filter, and draw player pose keypoints."""

    def __init__(
        self,
        rtmpose_processor=None,
        show_skeletons=True,
        show_player_trajectories=True,
        show_performance_stats=False,
        court_filter_margin=0.75,
        far_baseline_margin=3.0,
        near_baseline_margin=None,
        court_dimensions=(6.1, 13.4),
        keypoint_conf_threshold=0.25,
    ):
        self.rtmpose_processor = rtmpose_processor or RTMPoseProcessor()
        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_performance_stats = show_performance_stats
        self.current_pose_data = None
        self.court_mapper = None
        self.court_filter_margin = float(court_filter_margin)
        self.far_baseline_margin = float(far_baseline_margin)
        self.near_baseline_margin = (
            self.court_filter_margin
            if near_baseline_margin is None
            else float(near_baseline_margin)
        )
        self.court_width_m, self.court_length_m = (
            float(value) for value in court_dimensions
        )
        self.keypoint_conf_threshold = float(keypoint_conf_threshold)

        self.skeleton_connections = [
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

    def detect_players(self, roi, x1, y1, court_mapper=None):
        centroids = []
        point_left_hands = {}
        point_right_hands = {}

        t0 = time.time()
        keypoints_all, confidence_scores = self.rtmpose_processor.process_frame(roi)
        if self.show_performance_stats:
            inference_name = getattr(self.rtmpose_processor, "inference_name", "Pose")
            print(f"{inference_name} inference took {time.time() - t0:.2f} sec")

        if keypoints_all is None:
            self.current_pose_data = None
            return centroids, point_left_hands, point_right_hands

        persons = self._normalize_people(keypoints_all)
        detailed_detections = self._get_detailed_detections()
        filtered_people = []
        filtered_detections = []
        locations = []
        active_court_mapper = court_mapper or self.court_mapper

        for index, kp in enumerate(persons):
            kp_arr = np.asarray(kp)
            if kp_arr.ndim != 2 or kp_arr.shape[0] < 17 or kp_arr.shape[1] < 2:
                continue

            detail = detailed_detections[index] if index < len(detailed_detections) else {}
            scores = detail.get("keypoint_scores")
            if scores is None and confidence_scores is not None and index < len(confidence_scores):
                scores = np.asarray(confidence_scores[index])
            bbox = detail.get("bbox")
            if bbox is None:
                bbox = self._bbox_from_keypoints(kp_arr)
            location = self._select_ground_point(
                kp_arr,
                keypoint_scores=scores,
                bbox=bbox,
                person_confidence=detail.get("confidence", 1.0),
            )
            if location is None:
                continue

            local_point = location["point"]
            mid_point = (float(local_point[0] + x1), float(local_point[1] + y1))
            if not self._is_on_court(mid_point, active_court_mapper):
                continue

            filtered_people.append(kp_arr)
            centroids.append(mid_point)

            global_detail = self._globalize_detection(detail, kp_arr, bbox, x1, y1)
            global_detail.update(
                {
                    "location": [mid_point[0], mid_point[1]],
                    "location_method": location["method"],
                    "location_confidence": location["confidence"],
                    "location_degraded": location["degraded"],
                }
            )
            filtered_detections.append(global_detail)
            locations.append(
                {
                    "point": [mid_point[0], mid_point[1]],
                    "method": location["method"],
                    "confidence": location["confidence"],
                    "degraded": location["degraded"],
                }
            )

            lh = kp_arr[9]
            rh = kp_arr[10]
            if self._keypoint_is_visible(lh, scores, 9):
                point_left_hands[mid_point[1]] = (int(lh[0] + x1), int(lh[1] + y1))
            if self._keypoint_is_visible(rh, scores, 10):
                point_right_hands[mid_point[1]] = (int(rh[0] + x1), int(rh[1] + y1))

        if filtered_people:
            self.current_pose_data = {
                "keypoints": np.asarray(filtered_people),
                "offset_x": x1,
                "offset_y": y1,
                "detections": filtered_detections,
                "locations": locations,
            }
        else:
            self.current_pose_data = None

        return centroids, point_left_hands, point_right_hands

    def _get_detailed_detections(self):
        getter = getattr(self.rtmpose_processor, "get_last_detections", None)
        if not callable(getter):
            return []
        detections = getter()
        return detections if isinstance(detections, list) else []

    def clear_current_pose_data(self):
        """Clear evidence for a source frame deliberately skipped by sampling."""
        self.current_pose_data = None

    @staticmethod
    def _bbox_from_keypoints(keypoints):
        points = np.asarray(keypoints, dtype=float)
        valid = np.isfinite(points[:, :2]).all(axis=1)
        valid &= (points[:, 0] > 1) & (points[:, 1] > 1)
        if not np.any(valid):
            return None
        visible = points[valid, :2]
        return np.asarray(
            [visible[:, 0].min(), visible[:, 1].min(), visible[:, 0].max(), visible[:, 1].max()],
            dtype=float,
        )

    def _keypoint_is_visible(self, point, scores, index):
        point = np.asarray(point, dtype=float)
        if point.size < 2 or not np.isfinite(point[:2]).all() or point[0] <= 1 or point[1] <= 1:
            return False
        if scores is None or index >= len(scores):
            return True
        score = float(scores[index])
        return np.isfinite(score) and score >= self.keypoint_conf_threshold

    def _select_ground_point(self, keypoints, keypoint_scores=None, bbox=None, person_confidence=1.0):
        """Select a court contact point and expose whether it used degraded evidence."""
        points = np.asarray(keypoints, dtype=float)
        scores = None if keypoint_scores is None else np.asarray(keypoint_scores, dtype=float).reshape(-1)
        try:
            base_confidence = float(np.clip(person_confidence, 0.0, 1.0))
        except (TypeError, ValueError):
            base_confidence = 0.0

        left_visible = self._keypoint_is_visible(points[15], scores, 15)
        right_visible = self._keypoint_is_visible(points[16], scores, 16)
        if left_visible and right_visible:
            point = (points[15, :2] + points[16, :2]) / 2.0
            ankle_confidence = 1.0 if scores is None else min(float(scores[15]), float(scores[16]))
            return {
                "point": point,
                "method": "ankles_midpoint",
                "confidence": float(np.clip(base_confidence * ankle_confidence, 0.0, 1.0)),
                "degraded": False,
            }

        if left_visible or right_visible:
            ankle_index = 15 if left_visible else 16
            ankle_confidence = 1.0 if scores is None else float(scores[ankle_index])
            return {
                "point": points[ankle_index, :2].copy(),
                "method": "single_ankle",
                "confidence": float(np.clip(base_confidence * ankle_confidence * 0.75, 0.0, 1.0)),
                "degraded": True,
            }

        if bbox is None:
            return None
        bbox = np.asarray(bbox, dtype=float).reshape(-1)
        if bbox.size < 4 or not np.isfinite(bbox[:4]).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return {
            "point": np.asarray([(bbox[0] + bbox[2]) / 2.0, bbox[3]], dtype=float),
            "method": "bbox_bottom_center",
            "confidence": float(np.clip(base_confidence * 0.35, 0.0, 1.0)),
            "degraded": True,
        }

    @staticmethod
    def _globalize_detection(detail, keypoints, bbox, offset_x, offset_y):
        global_detail = dict(detail)
        global_keypoints = np.asarray(keypoints, dtype=float).copy()
        global_keypoints[:, 0] += offset_x
        global_keypoints[:, 1] += offset_y
        global_detail["keypoints"] = global_keypoints
        if bbox is not None:
            global_bbox = np.asarray(bbox, dtype=float).copy()
            global_bbox[[0, 2]] += offset_x
            global_bbox[[1, 3]] += offset_y
            global_detail["bbox"] = global_bbox
        return global_detail

    def _normalize_people(self, keypoints):
        if isinstance(keypoints, np.ndarray):
            if keypoints.ndim == 2:
                return [keypoints]
            if keypoints.ndim == 3:
                return [keypoints[i] for i in range(keypoints.shape[0])]
            return []
        if isinstance(keypoints, (list, tuple)):
            return list(keypoints)
        return []

    def _is_on_court(self, image_point, court_mapper):
        if court_mapper is None:
            return True
        court_position = court_mapper.image_to_court(image_point)
        if court_position is None or len(court_position) < 2:
            return False
        x, y = float(court_position[0]), float(court_position[1])
        lateral_margin = self.court_filter_margin
        return (
            -lateral_margin <= x <= self.court_width_m + lateral_margin
            and -self.far_baseline_margin <= y <= self.court_length_m + self.near_baseline_margin
        )

    def draw_players(
        self,
        frame,
        player_tracker,
        cached_movement_stats,
        stats_visualizer=None,
        rally_count=0,
        spatial_tracks=None,
        unassigned_detections=None,
    ):
        if self.show_skeletons and self.current_pose_data is not None:
            t0 = time.time()
            self._draw_skeleton_on_frame(
                frame,
                self.current_pose_data["keypoints"],
                self.current_pose_data["offset_x"],
                self.current_pose_data["offset_y"],
            )
            if self.show_performance_stats:
                print(f"Drawing skeleton took {time.time() - t0:.2f} sec")

        t0 = time.time()
        if spatial_tracks is not None:
            self._draw_spatial_tracks(
                frame,
                spatial_tracks,
                draw_trajectory=self.show_player_trajectories,
            )
            self._draw_unassigned_detections(frame, unassigned_detections)
        else:
            # Compatibility rendering for older callers. New analysis passes
            # spatial tracks and never uses upper/lower as an identity source.
            for position in ["upper", "lower"]:
                if player_tracker.players[position] is None:
                    continue

                color = (0, 255, 255) if position == "upper" else (255, 0, 255)
                cv2.circle(frame, tuple(map(int, player_tracker.players[position])), 5, color, -1, cv2.LINE_AA)

                if self.show_player_trajectories:
                    history = list(player_tracker.history[position])
                    for i, pos in enumerate(history):
                        if pos is None:
                            continue
                        radius = int(2 + (i / len(history)) * 3) if history else 2
                        cv2.circle(frame, tuple(map(int, pos)), radius, color, -1, cv2.LINE_AA)

        if self.show_performance_stats:
            print(f"Drawing players and trajectories took {time.time() - t0:.2f} sec")

        if stats_visualizer is not None:
            t0 = time.time()
            if spatial_tracks is not None:
                # ``upper/lower`` has no durable identity meaning in doubles
                # and can disagree with the green spatial-track boxes. The
                # active analysis supplies spatial tracks, so the on-video
                # statistics now share exactly the same evidence source.
                stats_visualizer.draw_spatial_track_stats(frame, spatial_tracks, rally_count)
            else:
                stats_visualizer.draw_player_stats(frame, cached_movement_stats, rally_count)
            if self.show_performance_stats:
                print(f"Drawing player stats took {time.time() - t0:.2f} sec")

    @staticmethod
    def _draw_spatial_tracks(frame, tracks, draw_trajectory=True):
        """Draw roster-backed player boxes, IDs, and evidence-aware movement.

        A detected box is a real pose measurement. Predicted tracks use a
        dashed last-known box and missing tracks deliberately have no box, so
        the annotated video never presents an inferred location as a person
        that the detector actually observed.
        """
        colors = {
            "detected": (0, 255, 0),
            "predicted": (0, 210, 255),
            "missing": (70, 70, 255),
        }
        for track in tracks or []:
            image_xy = track.get("image_xy")
            if not image_xy or len(image_xy) < 2:
                continue
            status = track.get("status", "missing")
            color = colors.get(status, colors["missing"])
            position = (int(image_xy[0]), int(image_xy[1]))
            trajectory = [
                (int(point[0]), int(point[1]))
                for point in (track.get("trajectory_image") or [])
                if point is not None and len(point) >= 2
            ]
            if draw_trajectory and len(trajectory) >= 2:
                cv2.polylines(frame, [np.asarray(trajectory, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

            bbox = (track.get("location_evidence") or {}).get("bbox_xyxy")
            if bbox is not None and len(bbox) >= 4:
                try:
                    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox[:4])
                except (TypeError, ValueError):
                    x1 = y1 = x2 = y2 = 0
                if x2 > x1 and y2 > y1:
                    if status == "detected":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    elif status == "predicted":
                        PlayerPoseVisualizer._draw_dashed_rectangle(
                            frame, (x1, y1), (x2, y2), color
                        )
            cv2.circle(frame, position, 6, color, -1, cv2.LINE_AA)
            label = str(track.get("track_id", "track"))
            if track.get("team_id"):
                label = f"{label} {track['team_id']}"
            if status != "detected":
                label = f"{label} ({status})"
            association = track.get("association") or {}
            try:
                identity_confidence = float(association.get("identity_confidence", 1.0))
            except (TypeError, ValueError):
                identity_confidence = 1.0
            if status == "detected" and identity_confidence < 0.7:
                # The box is real, but the long-occlusion association is not
                # strong enough for individual statistics until it stabilizes.
                label = f"{label} (recovered {identity_confidence:.0%})"
            # The court is green and detection colours are bright, so a
            # colour-matched label is easy to lose. Keep the durable ID at the
            # player foot point in black, with a light outline for dark shoes
            # or a dark background behind the court.
            PlayerPoseVisualizer._draw_foot_label(frame, label, position)

    @staticmethod
    def _draw_foot_label(frame, label, position):
        """Draw a high-contrast black track label just below a foot point."""
        frame_height, frame_width = frame.shape[:2]
        text = str(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = min(max(2, int(position[0]) + 7), max(2, frame_width - text_width - 2))
        y = min(max(text_height + 2, int(position[1]) + text_height + 8), max(text_height + 2, frame_height - baseline - 2))
        origin = (x, y)
        cv2.putText(frame, text, origin, font, scale, (235, 235, 235), 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_unassigned_detections(frame, detections):
        """Show real candidates that the fixed roster could not safely name."""
        color = (0, 110, 255)
        for detection in detections or []:
            bbox = detection.get("bbox_xyxy")
            if bbox is None or len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = (int(round(float(value))) for value in bbox[:4])
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            PlayerPoseVisualizer._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color)
            try:
                confidence = float(detection.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            cv2.putText(
                frame,
                f"candidate / unassigned {confidence:.0%}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_dashed_rectangle(frame, top_left, bottom_right, color, segment=8, gap=5):
        """Render a predicted box without making it look like a measurement."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        for start, end in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = max(abs(dx), abs(dy))
            if length <= 0:
                continue
            for offset in range(0, length, segment + gap):
                next_offset = min(length, offset + segment)
                p1 = (int(start[0] + dx * offset / length), int(start[1] + dy * offset / length))
                p2 = (int(start[0] + dx * next_offset / length), int(start[1] + dy * next_offset / length))
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

    def _draw_skeleton_on_frame(self, frame, keypoints, offset_x, offset_y):
        for person in self._normalize_people(keypoints):
            person_arr = np.asarray(person)
            if person_arr.ndim != 2 or person_arr.shape[1] < 2:
                continue

            keypoint_count = person_arr.shape[0]
            for a, b in self.skeleton_connections:
                if a >= keypoint_count or b >= keypoint_count:
                    continue
                x1, y1 = float(person_arr[a, 0]), float(person_arr[a, 1])
                x2, y2 = float(person_arr[b, 0]), float(person_arr[b, 1])
                if x1 > 1 and y1 > 1 and x2 > 1 and y2 > 1:
                    pt1 = (int(x1 + offset_x), int(y1 + offset_y))
                    pt2 = (int(x2 + offset_x), int(y2 + offset_y))
                    cv2.line(frame, pt1, pt2, (255, 191, 0), 2, cv2.LINE_AA)

            for i in range(keypoint_count):
                x_raw, y_raw = float(person_arr[i, 0]), float(person_arr[i, 1])
                if x_raw > 1 and y_raw > 1:
                    cv2.circle(frame, (int(x_raw + offset_x), int(y_raw + offset_y)), 3, (255, 128, 0), -1, cv2.LINE_AA)

    def get_current_pose_data(self):
        return self.current_pose_data
