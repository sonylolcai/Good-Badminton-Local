import os
import tempfile
import json
from tkinter import filedialog
import tkinter as tk
import time
import argparse

from .cancellation import raise_if_cancelled


def load_runtime_dependencies():
    """Load heavy runtime dependencies after argparse has handled --help."""
    global cv2, np, YOLO, CourtMapper, annotate_court, compute_expanded_roi, PlayerTracker
    global CourtTrajectoryVisualizer, ShuttlecockTracker, FixedCameraMatchPipeline
    global PlayerPoseVisualizer, StatsVisualizer, RTMPoseProcessor, YOLOPoseProcessor, vap
    global JsonlDetectionWriter, write_json, SCHEMA_VERSION, TrackNetV3RawMeasurements

    yolo_config_dir = os.path.join(tempfile.gettempdir(), "good-badminton-ultralytics")
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)

    try:
        import cv2 as _cv2
        import numpy as _np
        from ultralytics import YOLO as _YOLO
        from .court.mapper import CourtMapper as _CourtMapper, annotate_court as _annotate_court
        from .court.mapper import compute_expanded_roi as _compute_expanded_roi
        from .tracking.player import PlayerTracker as _PlayerTracker
        from .visualization.court_trajectory import CourtTrajectoryVisualizer as _CourtTrajectoryVisualizer
        from .detection.shuttlecock import ShuttlecockTracker as _ShuttlecockTracker
        from .detection.tracknet_v3 import TrackNetV3RawMeasurements as _TrackNetV3RawMeasurements
        from .analysis.fixed_camera_match import FixedCameraMatchPipeline as _FixedCameraMatchPipeline
        from .visualization.player_pose import PlayerPoseVisualizer as _PlayerPoseVisualizer
        from .visualization.stats import StatsVisualizer as _StatsVisualizer
        from .detection.rtmpose import RTMPoseProcessor as _RTMPoseProcessor
        from .detection.yolo_pose import YOLOPoseProcessor as _YOLOPoseProcessor
        from .media import video_audio as _vap
        from .data.writer import JsonlDetectionWriter as _JsonlDetectionWriter
        from .data.writer import write_json as _write_json
        from .data.writer import SCHEMA_VERSION as _SCHEMA_VERSION
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name}. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    cv2 = _cv2
    np = _np
    YOLO = _YOLO
    CourtMapper = _CourtMapper
    annotate_court = _annotate_court
    compute_expanded_roi = _compute_expanded_roi
    PlayerTracker = _PlayerTracker
    CourtTrajectoryVisualizer = _CourtTrajectoryVisualizer
    ShuttlecockTracker = _ShuttlecockTracker
    TrackNetV3RawMeasurements = _TrackNetV3RawMeasurements
    FixedCameraMatchPipeline = _FixedCameraMatchPipeline
    PlayerPoseVisualizer = _PlayerPoseVisualizer
    StatsVisualizer = _StatsVisualizer
    RTMPoseProcessor = _RTMPoseProcessor
    YOLOPoseProcessor = _YOLOPoseProcessor
    vap = _vap
    JsonlDetectionWriter = _JsonlDetectionWriter
    write_json = _write_json
    SCHEMA_VERSION = _SCHEMA_VERSION

class BadmintonAnalysisSystem:
    def __init__(self, video_path, show_display=False,
                 show_skeletons=True, show_player_trajectories=True, 
                 show_court_trajectory=True, show_shuttlecock_trajectory=True,
                 show_player_stats=True, show_performance_stats=False, 
                 save_images=False, language='zh', output_dir=None,
                 ball_model_path='weights/yolo11s-ball.pt', template_path=None,
                 pose_mode='balanced', pose_family='rtmpose',
                 yolo_pose_model='yolo11n-pose.pt', show_pose_roi=True,
                 output_video_style='annotated', pose_imgsz=960,
                 pose_sample_hz=0.0,
                 pose_conf=0.15, far_player_enhancement=False,
                 far_pose_roi=(0.12, 0.30, 0.86, 0.82), net_image_line=None,
                 match_mode='singles', tracker_backend='court_association',
                 enable_bytetrack=False, lock_match_roster=True,
                 roster_stable_frames=2, shuttle_detector='yolo',
                 tracknet_measurements_path=None, huji_action_model=None,
                 huji_sample_hz=6.0, generate_annotated_video=False,
                 browser_video_reencode=False):
        self.video_path = video_path
        self.show_display = show_display
        self.language = language
        self.template_path = template_path
        self.ball_model_path = ball_model_path
        if shuttle_detector not in {'none', 'yolo', 'tracknet_v3'}:
            raise ValueError("shuttle_detector must be 'none', 'yolo', or 'tracknet_v3'.")
        self.shuttle_detector = shuttle_detector
        self.tracknet_measurements_path = tracknet_measurements_path
        # Huji-compatible scene classification is an optional, coarse
        # post-processing prior.  It is intentionally opt-in because the
        # published Huji repository does not bundle a reviewed badminton
        # classifier checkpoint with this project.
        self.huji_action_model = huji_action_model or os.environ.get("GOOD_BADMINTON_HUJI_ACTION_MODEL")
        self.huji_sample_hz = float(os.environ.get("GOOD_BADMINTON_HUJI_SAMPLE_HZ", huji_sample_hz))
        if self.huji_sample_hz <= 0:
            raise ValueError("huji_sample_hz must be greater than 0")
        self.pose_mode = pose_mode
        self.pose_family = pose_family
        self.yolo_pose_model = yolo_pose_model
        self.show_pose_roi = show_pose_roi
        self.pose_imgsz = int(pose_imgsz)
        self.pose_sample_hz = float(pose_sample_hz)
        if self.pose_sample_hz < 0.0 or (0.0 < self.pose_sample_hz < 1.0):
            raise ValueError("pose_sample_hz must be 0 (every source frame) or at least 1")
        self.pose_conf = float(pose_conf)
        if not 0.0 < self.pose_conf <= 1.0:
            raise ValueError("pose_conf must be greater than 0 and no more than 1")
        self.far_player_enhancement = bool(far_player_enhancement)
        self.far_pose_roi = tuple(float(value) for value in far_pose_roi)
        self.net_image_line = net_image_line
        if match_mode not in {'singles', 'doubles'}:
            raise ValueError("match_mode must be 'singles' or 'doubles'.")
        if tracker_backend not in {'court_association', 'bytetrack'}:
            raise ValueError("tracker_backend must be 'court_association' or 'bytetrack'.")
        self.match_mode = match_mode
        self.tracker_backend = tracker_backend
        self.enable_bytetrack = bool(enable_bytetrack)
        self.lock_match_roster = bool(lock_match_roster)
        self.roster_stable_frames = max(1, int(roster_stable_frames))
        if output_video_style not in {'annotated', 'skeleton'}:
            raise ValueError(
                "output_video_style must be 'annotated' or 'skeleton'."
            )
        self.output_video_style = output_video_style
        # Video rendering is optional. The detector, court mapping, persistent
        # tracks, JSONL evidence, and reports do not require a rendered MP4.
        self.generate_annotated_video = bool(generate_annotated_video)
        self.browser_video_reencode = bool(
            browser_video_reencode and self.generate_annotated_video
        )


        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_court_trajectory = show_court_trajectory
        self.show_shuttlecock_trajectory = show_shuttlecock_trajectory
        self.show_player_stats = show_player_stats
        self.show_performance_stats = show_performance_stats
        self.save_images = save_images  

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Input video not found: {self.video_path}\n"
                "Pass a valid video file with --video-path."
            )
        if self.shuttle_detector == 'yolo' and not os.path.exists(self.ball_model_path):
            raise FileNotFoundError(
                f"Ball detection model not found: {self.ball_model_path}\n"
                "Download or train a YOLO shuttlecock model and place it at "
                "weights/yolo11s-ball.pt, or pass its path with --ball-model."
            )
        
        if self.pose_family == 'yolo-pose':
            self.rtmpose_processor = YOLOPoseProcessor(
                model_path=self.yolo_pose_model,
                imgsz=self.pose_imgsz,
                conf=self.pose_conf,
                asymmetric=self.far_player_enhancement,
                far_roi=self.far_pose_roi,
            )
        else:
            self.rtmpose_processor = RTMPoseProcessor(mode=self.pose_mode, pose_family=self.pose_family)
        self.yolo_ball_model = YOLO(self.ball_model_path) if self.shuttle_detector == 'yolo' else None
        if self.shuttle_detector == 'tracknet_v3' and not self.tracknet_measurements_path:
            raise ValueError("tracknet_measurements_path is required when shuttle_detector='tracknet_v3'.")
        self.tracknet_measurements = (
            TrackNetV3RawMeasurements(self.tracknet_measurements_path)
            if self.shuttle_detector == 'tracknet_v3'
            else None
        )

        self.last_stats_update_frame = 0


        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path)[:-4]
        self.save_dir = output_dir or os.path.join('outputs', self.video_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.images_save_dir = os.path.join(self.save_dir, 'detect_images')
        os.makedirs(self.images_save_dir, exist_ok=True)
        

        self.metadata_path = os.path.join(self.save_dir, "metadata.json")
        self.detections_path = os.path.join(self.save_dir, "detections.jsonl")
        self.spatial_match_summary_path = os.path.join(self.save_dir, "spatial_match_summary.json")
        output_prefix = "skeleton" if self.output_video_style == "skeleton" else "detect"
        self.output_video_path = (
            os.path.join(self.save_dir, f"{output_prefix}_{self.video_name}.mp4")
            if self.generate_annotated_video else None
        )
        self.temp_output_video_path = None
        self.video_writer = None
        self.detection_writer = None
        

        self.player_1_hand = "right"  
        self.player_2_hand = "right"  
        self.start_time = None
        self.end_time = None
        

        self.shuttlecock_tracker = ShuttlecockTracker(
            yolo_ball_model=self.yolo_ball_model,
            trajectory_length=30,
            show_trajectory=self.show_shuttlecock_trajectory,
            show_performance_stats=False
        )
        
        self.player_pose_visualizer = PlayerPoseVisualizer(
            rtmpose_processor=self.rtmpose_processor,
            show_skeletons=self.show_skeletons,
            show_player_trajectories=self.show_player_trajectories,
            show_performance_stats=False
        )
        

        self.court_trajectory_visualizer = CourtTrajectoryVisualizer()
        

        self.stats_update_interval_frames = 0
        self.cached_movement_stats = {}

        self.is_court_view_count = 0
        self.consecutive_non_court_frames = 0
        self.rally_active = False
        self.rally_count = 0  
        self.fps = 30  
        self.court_view_frames_threshold = 5
        self.non_court_frames_threshold = 5

        self.frame_width = 0
        self.frame_height = 0
        self.performance_log_interval_frames = 150
        self.pose_processed_frames = 0
        self.performance_report = None
    def process_video(self, progress_callback=None, state_callback=None, cancel_callback=None):
        """Process the input video.

        Args:
            progress_callback: Optional callable(frame_count, total_frames)
                invoked after each frame so callers can report progress.
            cancel_callback: Optional callable() returning True after an
                operator interrupts the task. Cancellation is checked between
                frames so model inference and media writers are not torn down
                halfway through a frame.
        """
        self.start_time = time.time()

        if state_callback is not None:
            state_callback({"phase": "human_tracking", "stage": "video_open_and_calibration"})
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
        video_duration = total_frames / fps
        

        self.fps = fps
        self.performance_log_interval_frames = max(1, int(fps * 5))
        

        template_path = self._get_template_path()
        template_gray, template_color = self._load_template(template_path, cap)
        

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = self._setup_video_writer(self.frame_width, self.frame_height, fps)


        corners, roi_corners, mid_height = self._setup_court_annotation(template_color)
        self.court_corners = corners
        self.court_roi_corners = roi_corners

        self.detection_writer = JsonlDetectionWriter(self.detections_path)
        

        self.court_mapper = CourtMapper(corners)
        self.player_pose_visualizer.court_mapper = self.court_mapper
        self.player_tracker = PlayerTracker(corners=corners, threshold=mid_height, history_size=30,
                                          detection_writer=self.detection_writer, fps=fps)
        self.fixed_camera_match = FixedCameraMatchPipeline(
            corners,
            fps=fps,
            net_image_line=self.net_image_line,
            match_mode=self.match_mode,
            tracker_backend=self.tracker_backend,
            enable_bytetrack=self.enable_bytetrack,
            lock_match_roster=self.lock_match_roster,
            roster_stable_frames=self.roster_stable_frames,
        )
        self._write_metadata(fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height)
        

        self.stats_visualizer = StatsVisualizer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            language=self.language
        )
        
        frame_count = 0
        detect_frame_count = 0

        if state_callback is not None:
            state_callback({
                "phase": "human_tracking",
                "stage": "human_frame_processing",
                "stage_detail": {"total_frames": total_frames, "fps": float(fps)},
            })

        while cap.isOpened():
            raise_if_cancelled(cancel_callback)
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame, detect_frame_count = self._process_frame(
                frame, template_gray, corners, roi_corners, frame_count, out,
                detect_frame_count, state_callback=state_callback,
            )
            if progress_callback is not None:
                progress_callback(frame_count, total_frames)
            raise_if_cancelled(cancel_callback)

        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        print(f"\n处理完成:")
        print(f"原始视频时长: {video_duration:.2f} 秒")
        print(f"处理耗时: {processing_time:.2f} 秒")
        print(f"处理速度比: {processing_time/video_duration:.2f}x")
        
        self._cleanup(cap, state_callback=state_callback)

    def _write_metadata(self, fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height):
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "video": {
                "path": self.video_path,
                "name": self.video_name,
                "fps": float(fps),
                "total_frames": int(total_frames),
                "duration_sec": float(video_duration),
                "width": int(self.frame_width),
                "height": int(self.frame_height),
            },
            "models": {
                "shuttlecock": self.ball_model_path if self.shuttle_detector == 'yolo' else None,
                "shuttlecock_detection": {
                    "primary_source": self.shuttle_detector,
                    "enabled": self.shuttle_detector != 'none',
                    "raw_measurements_path": self.tracknet_measurements_path if self.shuttle_detector == 'tracknet_v3' else None,
                    "measurement_kind": (
                        "not_requested" if self.shuttle_detector == 'none' else
                        "temporal_heatmap" if self.shuttle_detector == 'tracknet_v3' else
                        "yolo_box_center"
                    ),
                    "confidence_policy": (
                        "not_applicable" if self.shuttle_detector == 'none' else
                        "uncalibrated_binary_visibility_threshold_0.5"
                        if self.shuttle_detector == 'tracknet_v3'
                        else "yolo_model_confidence"
                    ),
                },
                "play_state": {
                    "enabled": bool(self.huji_action_model),
                    "model": self.huji_action_model,
                    "sample_hz": self.huji_sample_hz if self.huji_action_model else None,
                    "policy": (
                        "optional Huji-compatible scene classification; only a conservative "
                        "conflict prior for direction-reversal-after-gap candidates, never score proof"
                    ),
                },
                "pose": {
                    "family": self.pose_family,
                    "model": self.yolo_pose_model if self.pose_family == "yolo-pose" else self.pose_mode,
                    "imgsz": self.pose_imgsz if self.pose_family == "yolo-pose" else None,
                    "conf": self.pose_conf if self.pose_family == "yolo-pose" else None,
                    "detection_plan": (
                        "full_640+far_roi_640"
                        if self.pose_family == "yolo-pose" and self.far_player_enhancement
                        else f"full_{self.pose_imgsz}"
                        if self.pose_family == "yolo-pose"
                        else "native"
                    ),
                    "far_roi_normalized": list(self.far_pose_roi) if self.far_player_enhancement else None,
                    # 0 is the explicit full-evidence mode: run pose on every
                    # source frame, then persist raw COCO17 joints for every
                    # detected track observation.  A positive value is an
                    # opt-in performance/storage trade-off.
                    "requested_sample_hz": self.pose_sample_hz,
                    "effective_sample_hz": float(fps) if self.pose_sample_hz == 0 else min(float(fps), self.pose_sample_hz),
                    "processed_frame_count": 0,
                    "raw_keypoint_contract": self.fixed_camera_match.tracker.pose_keypoint_contract(),
                    "sampling_policy": (
                        "every_source_frame" if self.pose_sample_hz == 0 else
                        "timestamp buckets retain source timestamps; skipped frames carry explicit "
                        "predicted/missing track state and are never pose measurements"
                    ),
                    "court_filter_margins_m": {
                        "lateral": self.player_pose_visualizer.court_filter_margin,
                        "far_baseline": self.player_pose_visualizer.far_baseline_margin,
                        "near_baseline": self.player_pose_visualizer.near_baseline_margin,
                    },
                },
            },
            "court": {
                "template_path": template_path,
                "corners": corners,
                "roi_corners": roi_corners,
                "mid_height": mid_height,
                "net_image_line": self.fixed_camera_match.net_image_line if hasattr(self, "fixed_camera_match") else self.net_image_line,
                "coordinate_system": {
                    "unit": "meter",
                    "width": 6.1,
                    "length": 13.4,
                },
            },
            "outputs": {
                "video": self.output_video_path,
                "detections": self.detections_path,
                "tracknet_raw_csv": self.tracknet_measurements_path if self.shuttle_detector == 'tracknet_v3' else None,
                "video_style": self.output_video_style if self.generate_annotated_video else None,
                "video_generation": {
                    "enabled": self.generate_annotated_video,
                    "annotation_drawing": self.generate_annotated_video,
                    "initial_h264_export": self.generate_annotated_video,
                    "browser_reencode_requested": self.browser_video_reencode,
                    "policy": (
                        "disabled; data artifacts remain complete"
                        if not self.generate_annotated_video else
                        "enabled; browser reencode is handled by the pipeline"
                    ),
                },
                "spatial_match_summary": self.spatial_match_summary_path,
            },
            "temporal_tracking": {
                "players": {
                    "identity_key": "track_id",
                    "zone_policy": "zone_id is transient court space; never identity, team, or side",
                    "legacy_compatibility": "players.upper/lower retained for existing consumers only",
                    "match_mode": self.match_mode,
                    "max_players_per_team": 1 if self.match_mode == 'singles' else 2,
                    "backend": self.tracker_backend,
                    "bytetrack_enabled": self.tracker_backend == 'bytetrack',
                    "match_roster": (
                        self.fixed_camera_match.tracker.roster_summary()
                        if hasattr(self, "fixed_camera_match")
                        else {
                            "enabled": self.lock_match_roster,
                            "status": "not_initialized",
                            "expected_player_count": 2 if self.match_mode == "singles" else 4,
                        }
                    ),
                    "state_policy": "detected is a measurement; predicted/missing are explicit temporal states and not detector facts",
                },
                "shuttlecock": {
                    "enabled": self.shuttle_detector != 'none',
                    "max_prediction_frames": self.shuttlecock_tracker.max_prediction_frames,
                    "prediction_confidence_decay": self.shuttlecock_tracker.prediction_confidence_decay,
                    "prediction_usage": (
                        "not_requested; no shuttle, hit, or rally evidence is generated"
                        if self.shuttle_detector == 'none' else
                        "visualization_and_export_only; exclude from hit/error ground truth"
                    ),
                },
            },
        }
        write_json(self.metadata_path, metadata)

    def _process_frame(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count,
                       state_callback=None):

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame = self.draw_court_roi(frame, corners, roi_corners)

        is_court = self.is_court_view(gray_frame, template_gray)
        
        if is_court:
            self.is_court_view_count += 1
            self.consecutive_non_court_frames = 0
        else:
            self.consecutive_non_court_frames += 1
            self.is_court_view_count = 0
            

        if self.is_court_view_count >= self.court_view_frames_threshold and not self.rally_active:
            self.rally_active = True

            self.rally_count += 1

            self.player_tracker.start_new_rally()
            

        if self.consecutive_non_court_frames >= self.non_court_frames_threshold and self.rally_active:
            self.rally_active = False

            self.shuttlecock_tracker.clear_trajectory()


        if not is_court:
            output_frame = self._create_output_frame(frame) if self._needs_visual_output() else frame
            self._write_output_frame(output_frame, frame_count, out)
            return output_frame, detect_frame_count

        detect_frame_count += 1

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        roi = frame[y1:y2, x1:x2]
        pose_t0 = time.time()
        pose_was_sampled = self._should_sample_pose(frame_count)
        if pose_was_sampled:
            centroids, point_left_hands, point_right_hands = self.player_pose_visualizer.detect_players(roi, x1, y1)
            self.pose_processed_frames += 1
        else:
            # Never carry a preceding pose forward as a new measurement.
            # The multi-object tracker will emit an explicit predicted/missing
            # state for this source timestamp instead.
            self.player_pose_visualizer.clear_current_pose_data()
            centroids, point_left_hands, point_right_hands = [], {}, {}
        pose_elapsed = time.time() - pose_t0

        ball_t0 = time.time()
        if self.shuttle_detector == 'tracknet_v3':
            detected_ball_position = None
            ball_position = self.shuttlecock_tracker.update_external_measurement(
                self.tracknet_measurements.measurement_for_frame(frame_count - 1),
                roi_corners=roi_corners,
            )
        elif self.shuttle_detector == 'yolo':
            detected_ball_position = self.shuttlecock_tracker.detect_ball(frame, roi_corners=roi_corners)
            ball_position = self.shuttlecock_tracker.update_trajectory(detected_ball_position, roi_corners)
        else:
            # An operator opt-out is not a missing/predicted ball track.
            detected_ball_position = None
            ball_position = self.shuttlecock_tracker.mark_not_requested()
        ball_elapsed = time.time() - ball_t0
        

        pose_data = self.player_pose_visualizer.get_current_pose_data() or {}
        spatial_state = self.fixed_camera_match.update(
            frame_index=frame_count,
            observations=self._spatial_observations(pose_data.get("detections", [])),
            shuttlecock=self._spatial_shuttle_input(
                ball_position, self.shuttlecock_tracker.get_last_detection()
            ),
        )
        self._emit_tracking_state(state_callback, frame_count, spatial_state)
        players = self.player_tracker.update(
            frame_count,
            centroids,
            ball_position,
            point_left_hands,
            point_right_hands,
            detect_frame_count,
            pose_detections=pose_data.get("detections"),
            ball_detection=self.shuttlecock_tracker.get_last_detection(),
            spatial_state=spatial_state,
        )
        

        if frame_count == 1 or not self.cached_movement_stats:
            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.stats_update_interval_frames = int(self.player_tracker.fps * 0.5)

        if frame_count - self.last_stats_update_frame >= self.stats_update_interval_frames:

            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.last_stats_update_frame = frame_count


        should_log_performance = (
            self.show_performance_stats
            and self.performance_log_interval_frames > 0
            and frame_count % self.performance_log_interval_frames == 0
        )

        output_frame = frame
        shuttle_draw_elapsed = 0.0
        players_draw_elapsed = 0.0
        court_draw_elapsed = 0.0
        if self._needs_visual_output():
            output_frame = self._create_output_frame(frame)
            if self.shuttle_detector == 'tracknet_v3':
                cv2.putText(
                    output_frame,
                    "Shuttle: TrackNetV3 raw",
                    (max(12, self.frame_width - 290), 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )
            if self.output_video_style == "annotated" and self.show_pose_roi:
                cv2.rectangle(output_frame, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
                cv2.putText(output_frame, "Pose ROI", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            shuttle_draw_t0 = time.time()
            if self.shuttle_detector != 'none':
                self.shuttlecock_tracker.handle_visualization(output_frame)
            shuttle_draw_elapsed = time.time() - shuttle_draw_t0

            t0 = time.time()
            self.player_pose_visualizer.draw_players(
                frame=output_frame,
                player_tracker=self.player_tracker,
                cached_movement_stats=self.cached_movement_stats,
                stats_visualizer=(
                    self.stats_visualizer
                    if self.show_player_stats and self.output_video_style == "annotated"
                    else None
                ),
                rally_count=None if self.shuttle_detector == 'none' else self.rally_count,
                spatial_tracks=spatial_state.get("tracks", []),
                unassigned_detections=(spatial_state.get("match_roster") or {}).get("unassigned_observations", []),
            )
            players_draw_elapsed = time.time() - t0

            if self.show_court_trajectory and self.output_video_style == "annotated":
                t0 = time.time()
                output_frame = self.court_trajectory_visualizer.draw_overlay(output_frame, self.player_tracker.court_history)
                court_draw_elapsed = time.time() - t0

        if should_log_performance:
            print(
                f"Frame {frame_count}: pose {pose_elapsed:.2f}s "
                f"({'sampled' if pose_was_sampled else 'skipped'}), "
                f"shuttlecock {ball_elapsed:.2f}s, "
                f"shuttle draw {shuttle_draw_elapsed:.2f}s, "
                f"players draw {players_draw_elapsed:.2f}s, "
                f"court draw {court_draw_elapsed:.2f}s"
            )
        

        self._write_output_frame(output_frame, frame_count, out)
        return output_frame, detect_frame_count

    def _emit_tracking_state(self, callback, frame_count, spatial_state):
        """Publish bounded roster candidates while the main job still runs."""
        if callback is None:
            return
        last_frame = getattr(self, "_last_tracking_status_frame", 0)
        roster = spatial_state.get("match_roster") or {}
        status_changed = roster.get("status") != getattr(self, "_last_roster_status", None)
        cadence_frames = max(1, int(round(self.fps)))
        if not status_changed and int(frame_count) - last_frame < cadence_frames:
            return
        self._last_tracking_status_frame = int(frame_count)
        self._last_roster_status = roster.get("status")
        callback(
            {
                "phase": "human_tracking",
                "stage": "human_frame_processing",
                "frame": int(frame_count),
                "time_sec": round((int(frame_count) - 1) / self.fps, 4),
                "match_roster": roster,
                "track_candidates": [
                    {
                        "track_id": track.get("track_id"),
                        "status": track.get("status"),
                        "confidence": track.get("confidence"),
                    }
                    for track in spatial_state.get("tracks", [])
                ],
            }
        )

    def _should_sample_pose(self, frame_count):
        """Return whether this source frame carries a new pose measurement.

        ``pose_sample_hz=0`` is the data-complete default and means every
        source frame.  A positive rate is an explicit timestamp-based sampling
        policy; skipped frames have no fresh pose keypoints and must stay null.
        """
        if self.pose_sample_hz == 0 or self.pose_sample_hz >= self.fps:
            return True
        index = max(0, int(frame_count) - 1)
        if index == 0:
            return True
        previous_bucket = int(((index - 1) * self.pose_sample_hz) // self.fps)
        current_bucket = int((index * self.pose_sample_hz) // self.fps)
        return current_bucket != previous_bucket

    def _spatial_observations(self, pose_detections):
        """Translate pose evidence to court coordinates for the new tracker."""
        observations = []
        for detection in pose_detections:
            image_xy = detection.get("location")
            if not image_xy or len(image_xy) < 2:
                continue
            court_xy = self.court_mapper.image_to_court(image_xy)
            if len(court_xy) != 2:
                continue
            bbox = detection.get("bbox")
            bbox_xyxy = None
            if bbox is not None:
                try:
                    bbox_xyxy = [float(value) for value in bbox[:4]]
                except (TypeError, ValueError):
                    bbox_xyxy = None
            hands_image = self._pose_hands(detection.get("keypoints"))
            keypoints_image = self._pose_keypoints(detection.get("keypoints"))
            observations.append(
                {
                    "image_xy": image_xy,
                    "court_xy": [float(court_xy[0]), float(court_xy[1])],
                    "confidence": detection.get("confidence", 0.0),
                    "location_method": detection.get("location_method"),
                    "location_confidence": detection.get("location_confidence"),
                    "location_degraded": detection.get("location_degraded", False),
                    "source": detection.get("source"),
                    "bbox_xyxy": bbox_xyxy,
                    "hands_image": hands_image,
                    # The keypoints are a real YOLO Pose measurement for this
                    # sampled frame only. Court tracking may predict a player
                    # location between samples, but it never predicts joints.
                    "keypoints_image": keypoints_image,
                    "keypoint_scores": self._pose_scores(detection.get("keypoint_scores")),
                }
            )
        return observations

    @staticmethod
    def _pose_keypoints(keypoints):
        if keypoints is None:
            return None
        try:
            values = []
            for point in keypoints:
                x, y = float(point[0]), float(point[1])
                values.append([x, y] if x > 1 and y > 1 else None)
            return values if len(values) >= 17 else None
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _pose_scores(scores):
        if scores is None:
            return None
        try:
            values = [float(item) for item in scores]
            return values if len(values) >= 17 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pose_hands(keypoints):
        """Extract visible wrist locations as optional hit-attribution evidence."""
        if keypoints is None:
            return {"left": None, "right": None}
        try:
            values = list(keypoints)
        except TypeError:
            return {"left": None, "right": None}

        def point_at(index):
            if len(values) <= index:
                return None
            try:
                point = values[index]
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                return None
            return [x, y] if x > 1 and y > 1 else None

        return {"left": point_at(9), "right": point_at(10)}

    def _spatial_shuttle_input(self, ball_position, ball_detection):
        """Only actual ball observations may contribute to analytic evidence."""
        if ball_position is None or not ball_detection:
            return None
        detected = ball_detection.get("status") == "detected"
        return {
            "image_xy": ball_position,
            "confidence": ball_detection.get("confidence", 0.0),
            "detected": detected,
        }

    def _create_output_frame(self, source_frame):
        """Create either the normal annotation canvas or an anonymous line canvas."""
        if self.output_video_style != "skeleton":
            return source_frame

        canvas = np.zeros_like(source_frame)
        if hasattr(self, "court_mapper"):
            canvas, _ = self.court_mapper.draw_court_overlay(canvas)
        return canvas

    def _needs_visual_output(self):
        """Whether this run needs an annotated frame for any media sink."""
        return bool(self.generate_annotated_video or self.show_display or self.save_images)

    def _write_output_frame(self, frame, frame_count, out):
        """Write every source frame so the exported video keeps its full timeline."""
        if frame is None:
            return
        if self.show_display:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        if out is not None:
            out.write(frame)
        if self.save_images:
            cv2.imwrite(os.path.join(self.images_save_dir, f"{frame_count}.png"), frame)

    def _get_template_path(self):
        """Get the court template image path."""
        if self.template_path:
            if not os.path.exists(self.template_path):
                raise FileNotFoundError(
                    f"Court template image not found: {self.template_path}"
                )
            return self.template_path

        try:
            root = tk.Tk()
            root.withdraw()
            template_path = filedialog.askopenfilename(
                title="Select court template image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            root.destroy()
        except Exception as exc:
            raise RuntimeError(
                "Unable to open the template picker. In headless environments, "
                "pass a court template image path with --template-path."
            ) from exc

        if not template_path:
            raise RuntimeError(
                "No court template image selected. Pass --template-path to run "
                "without the file picker."
            )
        return template_path

    def _load_template(self, template_path, cap):
        """Load and resize the court template image."""
        template_gray = cv2.imread(template_path, 0)
        template_color = cv2.imread(template_path)
        if template_gray is None or template_color is None:
            raise RuntimeError(f"Unable to read court template image: {template_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        template_gray = cv2.resize(template_gray, (frame_width, frame_height))
        template_color = cv2.resize(template_color, (frame_width, frame_height))
        
        return template_gray, template_color

    def _setup_video_writer(self, frame_width, frame_height, fps):
        if not self.generate_annotated_video:
            return None
        self.temp_output_video_path = os.path.join(self.save_dir, f"temp_detect_{self.video_name}.mp4")
        

        self.video_writer = vap.setup_video_writer(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            temp_output_path=self.temp_output_video_path
        )
        
        return self.video_writer

    def _setup_court_annotation(self, template_color):
        """Set up court annotation."""

        if os.path.exists(os.path.join(self.save_dir, 'court_annotations.txt')):
            with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'r') as f:
                corners = eval(f.readline().split('=')[1])
                f.readline()
                mid_height = eval(f.readline().split('=')[1])
                roi_corners = compute_expanded_roi(corners, template_color.shape)
        else:
            auto_preview_path = os.path.join(self.save_dir, 'auto_court_preview.png')
            corners, roi_corners, mid_height = annotate_court(template_color, auto_preview_path=auto_preview_path)
       
        if not corners or not roi_corners or len(corners) != 4 or len(roi_corners) != 2:
            raise RuntimeError("Court annotation is incomplete: click 4 court corners in order. ROI is generated automatically.")

        with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'w') as f:
            f.write(f"corners={corners}\n")
            f.write(f"roi_corners={roi_corners}\n")
            f.write(f"mid_height={mid_height}\n")
        return corners, roi_corners, mid_height

    def _cleanup(self, cap, state_callback=None):
        """Clean up resources and merge audio when needed."""
        if hasattr(self, "fixed_camera_match"):
            if state_callback is not None:
                state_callback({"phase": "post_processing", "stage": "spatial_finalize"})
            spatial_summary = self.fixed_camera_match.finalize()
            write_json(self.spatial_match_summary_path, spatial_summary)
            if os.path.isfile(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as source:
                    metadata = json.load(source)
                metadata.setdefault("temporal_tracking", {}).setdefault("players", {})[
                    "match_roster"
                ] = spatial_summary.get("match_roster")
                metadata.setdefault("models", {}).setdefault("pose", {})[
                    "processed_frame_count"
                ] = int(self.pose_processed_frames)
                write_json(self.metadata_path, metadata)
        if self.detection_writer is not None:
            self.detection_writer.close()
            self.detection_writer = None
        if self.shuttle_detector == 'none':
            # Keep raw person evidence usable without fabricating an empty
            # shot/rally artifact from a run that opted out of ball evidence.
            self.offline_artifacts = {
                "status": "not_requested",
                "reason": "shuttle_detector=none",
                "policy": "ball trajectories, hit candidates, and rally derivation were not requested",
            }
            if os.path.isfile(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as source:
                    metadata = json.load(source)
                metadata["derived"] = self.offline_artifacts
                write_json(self.metadata_path, metadata)
        else:
            try:
                if state_callback is not None:
                    state_callback({"phase": "post_processing", "stage": "offline_shot_reconstruction"})
                from .analysis.offline_shot_reconstruction import generate_offline_artifacts
                from .analysis.huji_play_state import HUJI_PLAY_STATE_FILENAME, generate_huji_play_state

                play_state_path = None
                if self.huji_action_model:
                    play_state_path = os.path.join(self.save_dir, "derived", HUJI_PLAY_STATE_FILENAME)
                    try:
                        if state_callback is not None:
                            state_callback({"phase": "post_processing", "stage": "huji_play_state"})
                        play_state_result = generate_huji_play_state(
                            self.video_path,
                            self.huji_action_model,
                            play_state_path,
                            sample_hz=self.huji_sample_hz,
                        )
                        print(
                            "Huji-compatible play-state: "
                            f"{play_state_result['sample_count']} samples at "
                            f"{play_state_result['sample_hz']:.2f}Hz"
                        )
                    except Exception as play_state_error:
                        # The core detector output remains valid when an optional
                        # scene model is absent, invalid, or not yet deployed.
                        print(f"Huji-compatible play-state unavailable: {play_state_error}")
                        play_state_path = None

                self.offline_artifacts = generate_offline_artifacts(
                    self.detections_path,
                    fps=getattr(self, "fps", None),
                    play_state_path=play_state_path,
                )
                # ``metadata.json`` is the WebUI's compact result manifest. Keep
                # derived paths/counts here without changing raw detections.
                if os.path.isfile(self.metadata_path):
                    with open(self.metadata_path, "r", encoding="utf-8") as source:
                        metadata = json.load(source)
                    metadata["derived"] = self.offline_artifacts
                    write_json(self.metadata_path, metadata)
                print(
                    "Offline shuttle reconstruction: "
                    f"{self.offline_artifacts['frame_count']} frames, "
                    f"{self.offline_artifacts['event_count']} candidate shots, "
                    f"{self.offline_artifacts.get('rally_count', 0)} candidate rallies, "
                    f"{self.offline_artifacts.get('inferred_shot_count', 0)} motion-inferred shots"
                )
            except Exception as exc:
                # The annotated video and immutable detections are still usable if
                # a post-processing artifact fails. Do not silently claim derived
                # shot data exists; leave a visible console diagnostic instead.
                self.offline_artifacts = {"status": "failed", "error": str(exc)}
                print(f"Offline shuttle reconstruction failed: {exc}")

        # Start the single bounded report request before the optional audio
        # remux/export stage. Report generation depends on the finalized
        # detections and spatial summary, but should not wait behind a costly
        # browser-video encode at the end of a match.
        try:
            if state_callback is not None:
                state_callback({"phase": "post_processing", "stage": "performance_report"})
            from .analysis.performance_report import generate_performance_report

            self.performance_report = generate_performance_report(
                output_dir=self.save_dir,
                metadata_path=self.metadata_path,
                spatial_summary_path=self.spatial_match_summary_path,
            )
            if os.path.isfile(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as source:
                    metadata = json.load(source)
                metadata.setdefault("derived", {})["performance_report"] = {
                    "status": self.performance_report.get("status"),
                    "report_path": self.performance_report.get("report_path"),
                    "evidence_path": self.performance_report.get("evidence_path"),
                }
                write_json(self.metadata_path, metadata)
        except Exception as exc:
            self.performance_report = {
                "status": "failed",
                "reason": f"Unable to generate performance report: {exc}",
            }
            print(f"Performance report generation failed: {exc}")

        if self.generate_annotated_video:
            if state_callback is not None:
                state_callback({"phase": "post_processing", "stage": "video_writer_finalize"})
            if self.video_writer is not None:
                self.video_writer.release()
                time.sleep(1)

            cap.release()
            if self.show_display:
                cv2.destroyAllWindows()

            if state_callback is not None:
                state_callback({"phase": "post_processing", "stage": "annotated_video_export"})
            if hasattr(self, 'keep_audio') and self.keep_audio:
                export_ok = vap.process_video_with_audio(
                    video_path=self.video_path,
                    temp_video_path=self.temp_output_video_path,
                    output_path=self.output_video_path,
                    save_dir=self.save_dir
                )
            else:
                export_ok = vap.process_video_without_audio(
                    temp_video_path=self.temp_output_video_path,
                    output_path=self.output_video_path
                )
            if not export_ok:
                raise RuntimeError(
                    "Annotated video export failed. Open the backend console for the FFmpeg error."
                )
            return

        # Data-only mode avoids the temporary writer and both FFmpeg export
        # passes. Detector and analytics artifacts above are already final.
        cap.release()
        if self.show_display:
            cv2.destroyAllWindows()
        if state_callback is not None:
            state_callback({
                "phase": "post_processing",
                "stage": "annotated_video_export_skipped",
                "stage_detail": {"reason": "generate_annotated_video=false"},
            })

    def analyze_shuttlecock(self, roi_corners, corners):
        """Hit-point analysis is currently disabled."""
        raise RuntimeError(
            "Hit-point analysis is disabled until it is migrated to detections.jsonl."
        )

    def is_court_view(self, frame, template_gray, threshold=0.75):
        """Return whether the frame matches the court template."""
        result = cv2.matchTemplate(frame, template_gray, cv2.TM_CCOEFF_NORMED)
        # print("match score: ", result)
        return np.max(result) >= threshold

    def draw_court_roi(self, frame, corners, roi_corners):
        self.court_mapper = CourtMapper(corners)
        overlay, mid_height_int = self.court_mapper.draw_court_overlay(frame)
        cv2.rectangle(overlay, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
        return overlay
