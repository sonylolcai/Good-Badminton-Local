"""Tests for stream candidate shot routes (High #1/#2/#4 fixes & contracts)."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from api.stream_sessions import StreamSessionManager
from api.visual_result import (
    LocalEvidenceClient,
    VisualResultService,
    compute_visual_result,
    input_identity,
)
from badminton_analysis.streaming import (
    AnalysisEngine,
    FramePacket,
    FrameSegment,
    ProcessorEvent,
    SegmentDescriptor,
)
from good_badminton_visual.derivation import derive_stream_movement_metrics
from good_badminton_visual.metrics.stream_candidate_routes import (
    SOURCE,
    derive_stream_candidate_shot_routes,
)
from good_badminton_visual.result_contract import ResultContractError, validate_metrics
from tests.stream_test_utils import counting_processor_factory, create_request, segment_metadata, write_video_segment_bytes


CORNERS = [[200.0, 200.0], [1720.0, 200.0], [1720.0, 980.0], [200.0, 980.0]]
VIDEO = {"width": 1920, "height": 1080, "fps": 30.0}


def _pose_with_wrist_peak(foot_xy, hand_xy, *, wrist_dx=0.0):
    points = [[float(foot_xy[0]), float(foot_xy[1]) - 195.0] for _ in range(17)]
    # Nose (0), Shoulders (5, 6), hips (11, 12), wrists (9, 10), knees (13, 14), ankles (15, 16)
    points[0] = [float(foot_xy[0]), float(foot_xy[1]) - 195.0]
    points[5] = [float(foot_xy[0]) - 20.0, float(foot_xy[1]) - 170.0]
    points[6] = [float(foot_xy[0]) + 20.0, float(foot_xy[1]) - 170.0]
    points[11] = [float(foot_xy[0]) - 15.0, float(foot_xy[1]) - 100.0]
    points[12] = [float(foot_xy[0]) + 15.0, float(foot_xy[1]) - 100.0]
    points[13] = [float(foot_xy[0]) - 15.0, float(foot_xy[1]) - 50.0]
    points[14] = [float(foot_xy[0]) + 15.0, float(foot_xy[1]) - 50.0]
    points[9] = [float(hand_xy[0]) - 10.0, float(hand_xy[1])]
    points[10] = [float(hand_xy[0]) + wrist_dx, float(hand_xy[1])]
    points[15] = [float(foot_xy[0]) - 25.0, float(foot_xy[1])]
    points[16] = [float(foot_xy[0]) + 25.0, float(foot_xy[1])]
    return {
        "format": "coco17_image_v1",
        "coordinate_system": "full_source_image_pixels",
        "is_current_measurement": True,
        "keypoints_image": points,
        "keypoint_scores": [0.95] * 17,
    }


def _person_event(
    sid: str,
    event_id: str,
    frame_idx: int,
    time_sec: float,
    *,
    track_id: str,
    court_xy_m: list[float],
    image_xy: list[float],
    hands_image: dict[str, list[float]] | None,
    court_end: str,
    epoch: int = 0,
    occupancy_segment: int = 1,
    assoc_source: str = "half_court_slot_continuity",
    status: str = "detected",
    is_current_measurement: bool = True,
    wrist_dx: float = 0.0,
    measurement_bucket: int | None = None,
) -> dict:
    right_hand = (hands_image or {}).get("right") or [image_xy[0] + 15.0, image_xy[1] - 90.0]
    return {
        "schema_version": "stream-session.v1",
        "event_id": event_id,
        "event_type": "person_observation",
        "analysis_session_id": sid,
        "source_time_sec": round(time_sec, 6),
        "segment_index": 0,
        "confidence": 0.92 if status == "detected" else 0.0,
        "evidence_state": status,
        "data": {
            "analysis_mode": "person_only",
            "continuity_epoch": epoch,
            "track": {
                "track_id": track_id,
                "status": status,
                "confidence": 0.92 if status == "detected" else 0.0,
                "court_xy_m": list(court_xy_m),
                "image_xy": list(image_xy),
                "court_end": court_end,
                "slot_court_end": court_end,
                "identity_scope": "half_court_slots.v1",
                "occupancy_segment": occupancy_segment,
                "zone_id": "mid_center",
                "source_frame_index": frame_idx,
                "measurement_bucket": frame_idx if measurement_bucket is None else measurement_bucket,
                "association": {"source": assoc_source, "identity_confidence": 0.95},
                "location_evidence": {
                    "confidence": 0.90 if status == "detected" else 0.40,
                    "hands_image": hands_image,
                    "bbox_xyxy": [
                        float(image_xy[0]) - 35.0,
                        float(image_xy[1]) - 200.0,
                        float(image_xy[0]) + 35.0,
                        float(image_xy[1]),
                    ],
                    "is_current_measurement": is_current_measurement,
                },
                "pose": _pose_with_wrist_peak(image_xy, right_hand, wrist_dx=wrist_dx),
            },
        },
    }


def _shuttle_event(
    sid: str,
    event_id: str,
    frame_idx: int,
    time_sec: float,
    image_xy: list[float] | None,
    *,
    epoch: int = 0,
    status: str = "detected",
) -> dict:
    ok = status == "detected" and image_xy is not None
    return {
        "schema_version": "stream-session.v1",
        "event_id": event_id,
        "event_type": "shuttle_observation",
        "analysis_session_id": sid,
        "source_time_sec": round(time_sec, 6),
        "segment_index": 0,
        "confidence": 0.85 if ok else 0.0,
        "evidence_state": "detected" if ok else "missing",
        "data": {
            "detector": "tracknet_v3",
            "source_frame_index": frame_idx,
            "measurement_bucket": frame_idx,
            "continuity_epoch": epoch,
            "measurement": {
                "status": "detected" if ok else "missing",
                "visible": ok,
                "accepted": ok,
                "image": list(image_xy) if ok else None,
                "confidence": 0.85 if ok else None,
            },
        },
    }


def _build_two_shot_rally_events(
    sid: str = "ssn_candidate_test",
    *,
    person_stride: int = 1,
    reappear_occupancy_segment: int = 1,
    second_hit_epoch: int = 0,
    drop_hit_ball_gap: bool = False,
    omit_all_shuttle: bool = False,
    outside_court_hitter: bool = False,
) -> list[dict]:
    """Synthetic 30fps rally with two opposite-side hits at t=0.60s and t=1.20s."""
    events: list[dict] = []
    # 0.00..2.00s (frames 0..60 at 30fps)
    # Near player (track_003, end_b) at x=800, y=820 (court [2.5, 10.5])
    # Far player (track_001, end_a) at x=960, y=350 (court [3.1, 2.5])
    for f in range(61):
        t = f / 30.0
        epoch = second_hit_epoch if f >= 27 else 0
        # Shuttle trajectory:
        # f 0..17: flies from far (960, 260) toward near hand (820, 720)
        # f 18 (t=0.60s): hits near player's right hand (820, 720) -> sharp reversal up-right!
        # f 19..35: flies up toward far hand (980, 260)
        # f 36 (t=1.20s): hits far player's right hand (980, 260) -> sharp reversal downward!
        # f 37..55: flies down toward (900, 700)
        if f <= 18:
            ratio = f / 18.0
            sx = 960.0 + (820.0 - 960.0) * ratio
            sy = 260.0 + (720.0 - 260.0) * ratio
        elif f <= 36:
            ratio = (f - 18) / 18.0
            sx = 820.0 + (980.0 - 820.0) * ratio
            sy = 720.0 + (260.0 - 720.0) * ratio
        else:
            ratio = (f - 36) / 24.0
            sx = 980.0 + (900.0 - 980.0) * ratio
            sy = 260.0 + (700.0 - 260.0) * ratio

        if not omit_all_shuttle:
            # Drop ball around f in [19..21] for short reconstructible gap test (<=0.35s)
            # or drop long gap [15..45] if drop_hit_ball_gap
            missing = drop_hit_ball_gap and (15 <= f <= 45)
            events.append(
                _shuttle_event(
                    sid,
                    f"evt_s_{f:03d}",
                    f,
                    t,
                    None if missing else [sx, sy],
                    epoch=epoch,
                    status="missing" if missing else "detected",
                )
            )

        if f % person_stride == 0:
            # Smooth arm sweep centered on hits so both 30Hz and 10Hz observe >=2 arm-reaction samples
            near_hand_dx = max(0.0, 90.0 - 15.0 * abs(f - 18)) if 12 <= f <= 24 else 0.0
            far_hand_dx = max(0.0, 90.0 - 15.0 * abs(f - 36)) if 30 <= f <= 42 else 0.0
            occ = reappear_occupancy_segment if f >= 27 else 1
            assoc = "half_court_slot_replacement" if occ > 1 else "half_court_slot_continuity"
            near_foot = [100.0, 820.0] if outside_court_hitter else [800.0, 820.0]
            events.append(
                _person_event(
                    sid,
                    f"evt_p3_{f:03d}",
                    f,
                    t,
                    track_id="track_003",
                    court_xy_m=[2.5, 10.5],
                    image_xy=near_foot,
                    hands_image={"left": [780.0, 740.0], "right": [820.0 + near_hand_dx, 720.0]},
                    court_end="end_b",
                    epoch=epoch,
                    occupancy_segment=occ,
                    assoc_source=assoc,
                    wrist_dx=near_hand_dx,
                    measurement_bucket=f // person_stride,
                )
            )
            events.append(
                _person_event(
                    sid,
                    f"evt_p1_{f:03d}",
                    f,
                    t,
                    track_id="track_001",
                    court_xy_m=[3.1, 2.5],
                    image_xy=[960.0, 350.0],
                    hands_image={"left": [940.0, 270.0], "right": [980.0 + far_hand_dx, 260.0]},
                    court_end="end_a",
                    epoch=epoch,
                    occupancy_segment=occ,
                    assoc_source=assoc,
                    wrist_dx=far_hand_dx,
                    measurement_bucket=f // person_stride,
                )
            )
    return events


class StreamCandidateShotRoutesTests(unittest.TestCase):
    def test_geometry_marker_survives_forward_recovery_restart_and_repair(self):
        normal = write_video_segment_bytes(size=(64, 64))
        resized = write_video_segment_bytes(size=(80, 64))
        scenarios = (
            (normal, normal, False, None),
            (normal, resized, False, "calibration_frame_size_mismatch"),
            (normal, normal, True, "calibration_frame_size_unverified"),
            (b"invalid video", normal, False, "calibration_frame_size_unverified"),
        )
        for initial, subsequent, hide_shape, expected in scenarios:
            with self.subTest(initial_valid=initial == normal, resized=subsequent == resized, hide_shape=hide_shape):
                with tempfile.TemporaryDirectory() as directory:
                    manager = StreamSessionManager(
                        Path(directory), processor_factory=counting_processor_factory(), start_worker=False)
                    request = create_request()
                    request["configuration"]["shuttle_detector"] = "tracknet_v3"
                    _, created = manager.create_session(request, "business-geometry-recovery")
                    sid = created["analysis_session_id"]
                    manager.receive_segment(sid, 0, segment_metadata(0, initial), initial)
                    manager.receive_segment(sid, 2, segment_metadata(2, subsequent, source_start=2.0), subsequent)
                    manager.declare_gap(sid, 1, {"schema_version": "stream-session.v1", "source_start_time_sec": 1.0,
                                                  "duration_sec": 1.0, "reason": "source_segment_missing"})
                    decoder = manager.decoder.decode

                    def decode(path, descriptor):
                        segment = decoder(path, descriptor)
                        if hide_shape and descriptor.segment_index != 0:
                            return replace(segment, frames=(replace(packet, frame=None) for packet in segment.frames))
                        return segment

                    with patch.object(manager.decoder, "decode", side_effect=decode):
                        manager.drain()
                    saved = manager.get_session(sid)
                    self.assertEqual(saved.get("calibration_frame_size"), [64, 64] if initial == normal else None)
                    manager = StreamSessionManager(
                        Path(directory), processor_factory=counting_processor_factory(), start_worker=False)
                    _, grant = manager.request_repairs(sid, {
                        "schema_version": "stream-session.v1", "reason": "business_gap_inventory",
                        "ttl_seconds": 600, "ranges": [{"segment_index": 1, "kind": "missing"}],
                    }, "geometry-repair-grant")
                    metadata = segment_metadata(1, subsequent, source_start=1.0)
                    metadata["repair_request_id"] = grant["request_id"]
                    manager.repair_segment(sid, 1, metadata, subsequent, grant["ranges"][0]["grant_token"])
                    with patch.object(manager.decoder, "decode", side_effect=decode):
                        manager.drain()
                    geometry = {(event["segment_index"], event["data"].get("candidate_route_geometry"))
                                for event in manager._load_events(sid) if event["data"].get("candidate_route_geometry")}
                    self.assertEqual(geometry, {(1, expected), (2, expected)} if expected else set())
                    self.assertTrue(manager.get_session(sid)["segments"]["1"]["processed"])

    def test_failed_middle_fragment_recovers_against_original_size(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = StreamSessionManager(
                Path(directory), processor_factory=counting_processor_factory(), start_worker=False)
            request = create_request()
            request["configuration"]["shuttle_detector"] = "tracknet_v3"
            _, created = manager.create_session(request, "business-geometry-decode-failure")
            sid = created["analysis_session_id"]
            original = write_video_segment_bytes(size=(64, 64))
            changed = write_video_segment_bytes(size=(80, 64))
            for index, raw in enumerate((original, b"invalid video", changed, original)):
                manager.receive_segment(sid, index, segment_metadata(index, raw, source_start=float(index)), raw)
            manager.drain()
            session = manager.get_session(sid)
            self.assertTrue(session["segments"]["1"]["processing_failed"])
            self.assertEqual(session["calibration_frame_size"], [64, 64])
            self.assertEqual((session["video"]["width"], session["video"]["height"]), (64, 64))
            self.assertEqual(
                [(ev["segment_index"], ev["data"]["candidate_route_geometry"])
                 for ev in manager._load_events(sid) if "candidate_route_geometry" in ev["data"]],
                [(2, "calibration_frame_size_mismatch")],
            )

    def test_rejected_geometry_never_enters_ball_or_person_windows(self):
        from good_badminton_visual.metrics import stream_candidate_routes as routes

        events = _build_two_shot_rally_events()
        for event in events:
            event["segment_index"] = 1
        inputs = events + [{
            "event_type": "session_status", "event_id": "geometry_rejected",
            "source_time_sec": 0.0, "segment_index": 1,
            "data": {"continuity_epoch": 0, "candidate_route_geometry": "calibration_frame_size_mismatch"},
        }]
        shuttle, people, _, _, _ = routes._partition_by_continuity(
            inputs, video_fps=30, match_mode="doubles", roster_identity_policy="half_court_slots.v1")
        self.assertEqual((shuttle, people), ({}, {}))
        result = derive_stream_candidate_shot_routes(
            inputs, court_corners=CORNERS, video_width=VIDEO["width"], video_height=VIDEO["height"],
            video_fps=30, roster_identity_policy="half_court_slots.v1")
        self.assertEqual(result["included_count"], 0)

    def test_same_timestamp_boundary_uses_ball_and_person_source_frames(self):
        from good_badminton_visual.metrics import stream_candidate_routes as routes

        events = _build_two_shot_rally_events()
        for event in events:
            if routes._event_source_frame(event) in (27, 28):
                event["source_time_sec"] = 0.9
        ball = next(event for event in events if event["event_type"] == "shuttle_observation"
                    and routes._event_source_frame(event) == 28)
        segment = ball["segment_index"]
        marker = {
            "event_type": "session_status", "event_id": "same_time_boundary",
            "source_time_sec": 0.9, "segment_index": segment,
            "data": {"continuity_epoch": 0, "ball_continuity": "unknown",
                     "reason": "source_time_not_increasing", "source_frame_index": 28},
        }
        missing_ball = copy.deepcopy(ball)
        missing_ball["event_id"] = "ball_without_frame"
        missing_ball["data"].pop("source_frame_index")
        missing_person = copy.deepcopy(next(event for event in events
            if event["event_type"] == "person_observation" and routes._event_source_frame(event) == 28))
        missing_person["event_id"] = "person_without_frame"
        missing_person["data"]["track"].pop("source_frame_index")
        missing_person["data"]["track"]["track_id"] = "ambiguous"
        inputs = [missing_ball, missing_person, marker, *reversed(events)]
        for reason, frame in (("source_time_not_increasing", 28), ("source_resolution_changed", 28),
                              ("source_frame_index_gap", None)):
            with self.subTest(reason=reason, boundary_frame=frame):
                marker["data"].update(reason=reason, source_frame_index=frame)
                shuttle, people, roster, detected, _ = routes._partition_by_continuity(
                    inputs, video_fps=30, match_mode="doubles", roster_identity_policy="half_court_slots.v1")
                self.assertNotIn("ambiguous", roster)
                self.assertEqual(detected, sum(len(rows) for rows in shuttle.values()))
                before, after = (0, 0), (0, 1)
                for groups in (shuttle, people):
                    self.assertEqual([row["frame"] for row in groups[before] if row["time_sec"] == 0.9],
                                     [] if frame is None else [27])
                    if reason == "source_resolution_changed":
                        self.assertNotIn(after, groups)
                        self.assertTrue(all(row["time_sec"] <= 0.9 for row in groups[before]))
                    else:
                        self.assertEqual([row["frame"] for row in groups[after] if row["time_sec"] == 0.9],
                                         [] if frame is None else [28])
                result = derive_stream_candidate_shot_routes(
                    inputs, court_corners=CORNERS, video_width=VIDEO["width"], video_height=VIDEO["height"],
                    video_fps=30, roster_identity_policy="half_court_slots.v1")
                self.assertEqual(result["included_count"], 0)

    def test_explicit_ball_boundary_splits_same_epoch_before_reconstruction(self):
        events = _build_two_shot_rally_events()
        kwargs = {
            "court_corners": CORNERS,
            "video_width": VIDEO["width"],
            "video_height": VIDEO["height"],
            "video_fps": VIDEO["fps"],
            "roster_identity_policy": "half_court_slots.v1",
        }
        baseline = derive_stream_candidate_shot_routes(events, **kwargs)
        self.assertEqual(baseline["included_count"], 1)
        for reason in ("source_frame_index_gap", "source_timestamp_gap", "source_time_not_increasing",
                       "source_frame_identity_unverified", "source_resolution_changed"):
            with self.subTest(reason=reason):
                marker = {
                    "event_type": "session_status",
                    "event_id": "evt_continuity_boundary",
                    "source_time_sec": 0.9,
                    "segment_index": 0,
                    "data": {"continuity_epoch": 0, "ball_continuity": "unknown", "reason": reason},
                }
                from good_badminton_visual.metrics import stream_candidate_routes as routes

                with patch.object(routes, "reconstruct_shuttle_track", wraps=routes.reconstruct_shuttle_track) as reconstruct:
                    result = derive_stream_candidate_shot_routes([marker, *reversed(events), marker], **kwargs)
                self.assertEqual(result["included_count"], 0)
                windows = [call.args[0] for call in reconstruct.call_args_list]
                self.assertEqual(len(windows), 1 if reason == "source_resolution_changed" else 2)
                self.assertTrue(all(row["time_sec"] < 0.9 for row in windows[0]))
                if len(windows) == 2:
                    self.assertTrue(all(row["time_sec"] >= 0.9 for row in windows[1]))
                for rally in result["rallies"]:
                    self.assertFalse(rally["start_time_sec"] < 0.9 < rally["end_time_sec"])

    def test_slot_occupancy_and_epoch_discontinuities_cut_pairing(self):
        # 1) Same epoch, continuous occupancy_segment=1 pairs normally
        same_occ_events = _build_two_shot_rally_events(reappear_occupancy_segment=1, second_hit_epoch=0)
        same_occ_res = derive_stream_candidate_shot_routes(
            same_occ_events,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        self.assertGreaterEqual(same_occ_res["included_count"], 1)
        p3_ok = same_occ_res["players"]["track_003"]
        self.assertEqual(p3_ok["source"], SOURCE)
        self.assertEqual(p3_ok["status"], "estimated")
        self.assertGreaterEqual(p3_ok["sample_count"], 1)
        self.assertEqual(p3_ok["value"], 0.0)
        self.assertEqual(p3_ok["normalized_value"], 0.0)
        self.assertEqual(p3_ok["receiving_heatmap"]["sample_count"], p3_ok["sample_count"])

        # 2) Same epoch, occupancy_segment increments 1 -> 2 across the shot: must NOT pair across substitution boundary
        events = _build_two_shot_rally_events(reappear_occupancy_segment=2, second_hit_epoch=0)
        res = derive_stream_candidate_shot_routes(
            events,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        self.assertEqual(res["included_count"], 0)
        p3 = res["players"]["track_003"]
        self.assertEqual(p3["status"], "insufficient_candidate_pairs")
        self.assertEqual(p3["sample_count"], 0)
        self.assertIsNone(p3["value"])
        self.assertIsNone(p3["normalized_value"])
        self.assertGreaterEqual(p3["excluded"].get("occupancy_discontinuity", 0), 1)
        self.assertEqual(p3["receiving_heatmap"]["sample_count"], 0)

        # 3) Receiver absent during hit1 then replaced with occ=2 before hit2 must also cut pairing
        recv_gap_events = _build_two_shot_rally_events(reappear_occupancy_segment=1, second_hit_epoch=0)
        kept = []
        for ev in recv_gap_events:
            if ev.get("event_type") == "person_observation" and ev["data"]["track"]["track_id"] == "track_001":
                f = ev["data"]["track"]["source_frame_index"]
                if 15 <= f <= 26:
                    continue
                if f >= 27:
                    ev["data"]["track"]["occupancy_segment"] = 2
                    ev["data"]["track"]["association"]["source"] = "half_court_slot_replacement" if f == 27 else "half_court_slot_continuity"
            kept.append(ev)
        recv_gap_res = derive_stream_candidate_shot_routes(
            kept,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        self.assertEqual(recv_gap_res["included_count"], 0)
        self.assertGreaterEqual(recv_gap_res["players"]["track_003"]["excluded"].get("occupancy_discontinuity", 0), 1)

        # 4) Once both players are stably in occupancy_segment=2 before Hit 1, intra-segment shots pair normally
        post_sub_events = _build_two_shot_rally_events(reappear_occupancy_segment=1, second_hit_epoch=0)
        for ev in post_sub_events:
            if ev.get("event_type") == "person_observation":
                f = ev["data"]["track"]["source_frame_index"]
                if f >= 12:
                    ev["data"]["track"]["occupancy_segment"] = 2
                    ev["data"]["track"]["association"]["source"] = "half_court_slot_replacement" if f == 12 else "half_court_slot_continuity"
        post_sub_res = derive_stream_candidate_shot_routes(
            post_sub_events,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        self.assertGreaterEqual(post_sub_res["included_count"], 1)

        # 5) Across continuity_epoch 0 -> 1, must NOT pair across stream discontinuity
        split_events = _build_two_shot_rally_events(reappear_occupancy_segment=1, second_hit_epoch=1)
        split_res = derive_stream_candidate_shot_routes(
            split_events,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        self.assertEqual(split_res["included_count"], 0)
        self.assertEqual(split_res["players"]["track_003"]["status"], "insufficient_candidate_pairs")
        self.assertIsNone(split_res["players"]["track_003"]["value"])

    def test_stale_hands_on_missing_track_and_outside_court_are_blocked(self):
        events = _build_two_shot_rally_events(outside_court_hitter=True)
        res = derive_stream_candidate_shot_routes(
            events,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        # track_003 foot is at x=100 outside CORNERS[0].x=200 -> cannot produce an included pair
        self.assertEqual(res["players"]["track_003"]["sample_count"], 0)

        # Also test stale hands_image on a track with status='missing'
        events2 = _build_two_shot_rally_events()
        for ev in events2:
            if ev["event_type"] == "person_observation" and ev["data"]["track"]["track_id"] == "track_003":
                ev["data"]["track"]["status"] = "missing"
                ev["data"]["track"]["location_evidence"]["is_current_measurement"] = False
        res2 = derive_stream_candidate_shot_routes(
            events2,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        self.assertEqual(res2["included_count"], 0)

    def test_30hz_sampling_does_not_skip_rounded_frames_and_records_video_metadata(self):
        class _DummyProc:
            def process_frame(self, frame, ctx):
                return [
                    ProcessorEvent(
                        event_type="person_observation",
                        evidence_state="detected",
                        confidence=0.9,
                        data={"bucket": ctx.measurement_bucket},
                    )
                ]

            def finalize(self, ctx):
                return []

            def snapshot_state(self):
                return {}

            def restore_state(self, state):
                pass

        descriptor = SegmentDescriptor(
            segment_index=0,
            source_start_time_sec=0.0,
            duration_sec=1.0,
            sha256="a" * 64,
            idempotency_key="seg-0",
            content_length_bytes=30,
        )
        # Simulate 6-decimal rounded 30fps timestamps (e.g. 0.333333 at frame 10)
        packets = [
            FramePacket(
                frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
                source_frame_index=i,
                source_time_sec=round(i / 30.0, 6),
                segment_index=0,
            )
            for i in range(30)
        ]
        engine = AnalysisEngine("ssn_30hz", 30, _DummyProc())
        out = engine.process_segment(FrameSegment(descriptor=descriptor, frames=packets))
        self.assertEqual(out.source_frames, 30)
        self.assertEqual(out.measurement_frames, 30)
        self.assertEqual(out.checkpoint["video"], {"width": 1920, "height": 1080, "fps": 30.0})

    def test_missing_context_fails_closed_without_defaults_and_preserves_movement(self):
        events = _build_two_shot_rally_events()
        for kwargs, expected_reason in [
            ({"court_corners": None, "video_width": 1920, "video_height": 1080, "video_fps": 30.0}, "missing_or_invalid_four_corners"),
            ({"court_corners": CORNERS, "video_width": None, "video_height": 1080, "video_fps": 30.0}, "missing_video_dimensions"),
            ({"court_corners": CORNERS, "video_width": 1920, "video_height": 1080, "video_fps": None}, "missing_video_fps"),
        ]:
            with self.subTest(expected_reason=expected_reason):
                res = derive_stream_candidate_shot_routes(events, **kwargs)
                self.assertEqual(res["missing_reason"], expected_reason)
                self.assertEqual(res["players"]["track_003"]["status"], "insufficient_candidate_pairs")
                self.assertIsNone(res["players"]["track_003"]["value"])

    def test_missing_shuttle_never_falls_back_to_wrist_estimate_and_isolates_other_metrics(self):
        # Events have strong wrist swing peaks (wrist_dx=60) but NO shuttle detections
        events = _build_two_shot_rally_events(omit_all_shuttle=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session"
            root.mkdir(parents=True)
            from api.visual_result import implementation

            sid = "ssn_candidate_test"
            session = {
                "analysis_session_id": sid,
                "status": "finalized",
                "sealed": True,
                "sealed_at": "2026-11-18T00:00:00Z",
                "events_revision": 1,
                "continuity_epoch": 0,
                "court_corners": CORNERS,
                "video": VIDEO,
                "visual_result": {"schema_version": "stream-visual-result.v1", "implementation_sha256": implementation()},
                "expected_last_segment_index": 0,
                "configuration": {
                    "analysis_sample_hz": 30,
                    "pose_imgsz": 960,
                    "shuttle_detector": "tracknet_v3",
                    "roster_identity_policy": "half_court_slots.v1",
                    "match_mode": "doubles",
                },
                "segments": {"0": {"processed": True, "duration_sec": 2.0}},
            }
            (root / "manifest.json").write_text(json.dumps(session))
            (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
            progress = {
                "received_segments": 1,
                "processed_segments": 1,
                "failed_segments": 0,
                "received_duration_sec": 2.0,
                "processed_source_time_sec": 2.0,
                "missing_segment_indexes": [],
                "failed_segment_indexes": [],
                "backlog_duration_sec": 0,
            }
            terminal = {"analysis_session_id": sid, "status": "finalized", "progress": progress}
            identity = input_identity(root, session)
            self.assertEqual(compute_visual_result(str(root), identity, terminal), {"ready": True})

            manager = unittest.mock.Mock()
            manager._session_dir.return_value = root
            manager._compute_progress.return_value = progress
            code, payload = VisualResultService(manager, active=True).read(session)
            self.assertEqual(code, 200)
            validate_metrics(payload["metrics"])
            for p in payload["metrics"]["players"]:
                sr = p["shot_route_profile"]
                self.assertEqual(sr["source"], SOURCE)
                self.assertEqual(sr["status"], "insufficient_candidate_pairs")
                self.assertEqual(sr["missing_reason"], "insufficient_shuttle_evidence")
                self.assertIsNone(sr["value"])
                self.assertIsNone(sr["normalized_value"])
                self.assertNotIn("events", sr)
                self.assertNotIn("event_ids", sr)
            self.assertEqual(
                set(payload["completeness"]["missing_stages"]),
                {"rally_segmentation", "hit_moments", "hitter_attribution"},
            )

    def test_10hz_person_and_30hz_shuttle_both_produce_candidate_routes_without_polluting_movement(self):
        events_10hz = _build_two_shot_rally_events(person_stride=3)
        events_no_ball = _build_two_shot_rally_events(person_stride=3, omit_all_shuttle=True)
        with tempfile.TemporaryDirectory() as tmp:
            t1 = Path(tmp) / "with_ball"
            t2 = Path(tmp) / "no_ball"
            sid = "ssn_candidate_test"
            term = {
                "status": "finalized",
                "analysis_session_id": sid,
                "progress": {"processed_source_time_sec": 2.0},
            }
            req = {
                "configuration": {
                    "analysis_sample_hz": 10,
                    "shuttle_detector": "tracknet_v3",
                    "roster_identity_policy": "half_court_slots.v1",
                    "match_mode": "doubles",
                },
                "court_corners": CORNERS,
                "video": VIDEO,
            }
            r1 = derive_stream_movement_metrics(
                t1, client=LocalEvidenceClient(events_10hz, sid, 1), terminal_status=term, create_request=req
            )
            r2 = derive_stream_movement_metrics(
                t2,
                client=LocalEvidenceClient(events_no_ball, sid, 1),
                terminal_status=term,
                create_request={**req, "configuration": {"analysis_sample_hz": 10, "shuttle_detector": "none", "roster_identity_policy": "half_court_slots.v1"}},
            )
            p1 = {p["track_id"]: p for p in r1["metrics"]["players"]}
            p2 = {p["track_id"]: p for p in r2["metrics"]["players"]}
            self.assertEqual(p1["track_003"]["movement"], p2["track_003"]["movement"])
            self.assertEqual(p1["track_003"]["movement_profile"], p2["track_003"]["movement_profile"])
            self.assertEqual(p1["track_003"]["tactical_archetype"], p2["track_003"]["tactical_archetype"])
            self.assertEqual(p1["track_003"]["receiving_profile"], p2["track_003"]["receiving_profile"])
            self.assertEqual(p1["track_003"]["shot_route_profile"]["source"], SOURCE)
            self.assertGreaterEqual(p1["track_003"]["shot_route_profile"]["sample_count"], 1)

    def test_short_ball_gap_reconstructs_flight_while_long_gap_gives_up_without_wrist_fallback(self):
        # Short gap (3 frames = 0.10s <= 0.35s) mid-flight reconstructs cleanly
        short_gap = _build_two_shot_rally_events()
        for ev in short_gap:
            if ev["event_type"] == "shuttle_observation" and ev["data"]["source_frame_index"] in (24, 25, 26):
                ev["evidence_state"] = "missing"
                ev["confidence"] = 0.0
                ev["data"]["measurement"].update(status="missing", visible=False, accepted=False, image=None, confidence=None)
        res_short = derive_stream_candidate_shot_routes(
            short_gap,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        self.assertGreaterEqual(res_short["included_count"], 1)

        # Long gap covering hit region (15..45 = 1.03s > 0.35s) -> unknown_gap, gives up until ball reappears
        long_gap = _build_two_shot_rally_events(drop_hit_ball_gap=True)
        res_long = derive_stream_candidate_shot_routes(
            long_gap,
            court_corners=CORNERS,
            video_width=VIDEO["width"],
            video_height=VIDEO["height"],
            video_fps=VIDEO["fps"],
            roster_identity_policy="half_court_slots.v1",
        )
        self.assertEqual(res_long["included_count"], 0)
        self.assertEqual(res_long["players"]["track_003"]["source"], SOURCE)
        self.assertEqual(res_long["players"]["track_003"]["status"], "insufficient_candidate_pairs")
        self.assertIsNone(res_long["players"]["track_003"]["value"])

    def test_saved_stage_a_events_jsonl_fails_closed_without_corners(self):
        sample_dir = Path(__file__).resolve().parents[2] / "outputs" / "shot-route-compatibility-20261116" / "coordinated-upgrade"
        events_path = sample_dir / "events.jsonl"
        manifest_path = sample_dir / "manifest.json"
        if not events_path.is_file() or not manifest_path.is_file():
            self.skipTest("saved Stage A evidence sample not present")
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            res = derive_stream_movement_metrics(
                tmp,
                client=LocalEvidenceClient(events, manifest["analysis_session_id"], manifest["events_revision"]),
                terminal_status={
                    "status": "finalized",
                    "analysis_session_id": manifest["analysis_session_id"],
                    "progress": {"processed_source_time_sec": 301.97},
                },
                create_request={"configuration": manifest["configuration"]},
            )
            validate_metrics(res["metrics"])
            for p in res["metrics"]["players"]:
                sr = p["shot_route_profile"]
                self.assertEqual(sr["source"], SOURCE)
                self.assertEqual(sr["status"], "insufficient_candidate_pairs")
                self.assertEqual(sr["missing_reason"], "missing_or_invalid_four_corners")
                self.assertIsNone(sr["value"])

    def test_gpu_visual_result_aggregation_is_complete_for_30hz_finalized_and_partial_sessions(self):
        import os
        from fastapi.testclient import TestClient
        from api.app import create_app
        from api.visual_result import MAX_RESULT_BYTES, encode, implementation
        from tests.stream_test_utils import counting_processor_factory

        events = _build_two_shot_rally_events()
        required_player_blocks = (
            "movement",
            "measurement_coverage",
            "ability_scores",
            "movement_profile",
            "performance_charts",
            "shot_route_profile",
            "receiving_profile",
            "tactical_archetype",
            "joint_and_fatigue_guard",
            "combat_radar_profile",
            "match_load",
        )
        for status in ("finalized", "partial"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "session"
                root.mkdir(parents=True)
                sid = f"ssn_complete_{status}"
                session_events = [{**ev, "analysis_session_id": sid} for ev in events]
                session = {
                    "analysis_session_id": sid,
                    "status": status,
                    "sealed": True,
                    "sealed_at": "2026-11-18T00:00:00Z",
                    "events_revision": 1,
                    "continuity_epoch": 0,
                    "court_corners": CORNERS,
                    "video": VIDEO,
                    "visual_result": {"schema_version": "stream-visual-result.v1", "implementation_sha256": implementation()},
                    "expected_last_segment_index": 0,
                    "configuration": {
                        "analysis_sample_hz": 30,
                        "pose_imgsz": 960,
                        "shuttle_detector": "tracknet_v3",
                        "roster_identity_policy": "half_court_slots.v1",
                        "match_mode": "doubles",
                    },
                    "segments": {"0": {"processed": True, "duration_sec": 2.0}},
                }
                (root / "manifest.json").write_text(json.dumps(session))
                (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in session_events))
                progress = {
                    "received_segments": 1,
                    "processed_segments": 1,
                    "failed_segments": 0,
                    "received_duration_sec": 2.0,
                    "processed_source_time_sec": 2.0,
                    "missing_segment_indexes": [],
                    "failed_segment_indexes": [],
                    "backlog_duration_sec": 0,
                }
                terminal = {"analysis_session_id": sid, "status": status, "progress": progress}
                identity = input_identity(root, session)
                self.assertEqual(compute_visual_result(str(root), identity, terminal), {"ready": True})

                manager = unittest.mock.Mock()
                manager._session_dir.return_value = root
                manager._compute_progress.return_value = progress
                code, payload = VisualResultService(manager, active=True).read(session)
                self.assertEqual(code, 200)
                self.assertEqual(payload["state"], "ready")
                self.assertEqual(payload["gpu_status"], status)
                self.assertLess(len(encode(payload)), MAX_RESULT_BYTES)
                validate_metrics(payload["metrics"])
                self.assertEqual(payload["metrics"]["identity_scope"], "half_court_slots.v1")
                self.assertTrue(payload["track_candidates"])
                for p in payload["metrics"]["players"]:
                    for block in required_player_blocks:
                        self.assertIn(block, p, f"missing player block {block}")
                    sr = p["shot_route_profile"]
                    self.assertEqual(sr["source"], SOURCE)
                    self.assertNotIn("events", sr)
                    self.assertNotIn("event_ids", sr)
                by_id = {p["track_id"]: p for p in payload["metrics"]["players"]}
                self.assertIn("track_001", by_id)
                self.assertIn("track_003", by_id)
                self.assertGreater(by_id["track_003"]["shot_route_profile"]["hit_candidate_count"], 0)
                self.assertEqual(
                    set(payload["completeness"]["missing_stages"]),
                    {"rally_segmentation", "hit_moments", "hitter_attribution"},
                )

        # Verify pure GPU health endpoint advertises visual-result capability by default.
        with tempfile.TemporaryDirectory() as tmp:
            old_pure = os.environ.get("GOOD_BADMINTON_GPU_PURE_STREAM_ONLY")
            old_vr = os.environ.pop("GOOD_BADMINTON_VISUAL_RESULT_ENABLED", None)
            try:
                os.environ["GOOD_BADMINTON_GPU_PURE_STREAM_ONLY"] = "1"
                app = create_app(Path(tmp), start_worker=False, stream_processor_factory=counting_processor_factory())
                health = TestClient(app).get("/api/v1/health").json()
                self.assertEqual(health["visual_result_version"], "stream-visual-result.v1")
                self.assertEqual(health["visual_metrics_implementation_sha256"], implementation())
                self.assertEqual(health["visual_result_identity_policies"], ["individual_tracks.v1", "half_court_slots.v1", "anchored_roster.v1"])
            finally:
                if old_pure is None:
                    os.environ.pop("GOOD_BADMINTON_GPU_PURE_STREAM_ONLY", None)
                else:
                    os.environ["GOOD_BADMINTON_GPU_PURE_STREAM_ONLY"] = old_pure
                if old_vr is not None:
                    os.environ["GOOD_BADMINTON_VISUAL_RESULT_ENABLED"] = old_vr


if __name__ == "__main__":
    unittest.main()
