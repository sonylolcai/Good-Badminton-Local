import unittest
from types import SimpleNamespace

import numpy as np

from badminton_analysis.detection.yolo_pose import YOLOPoseProcessor, derive_far_court_roi


class FakeModel:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def __call__(self, frame, **kwargs):
        if isinstance(frame, list):
            self.calls.append({"batch_size": len(frame), "shape": frame[0].shape, **kwargs})
            out = []
            for _ in frame:
                batch = self.batches.pop(0)
                keypoints = SimpleNamespace(xy=batch["keypoints"], conf=batch["keypoint_scores"])
                boxes = SimpleNamespace(xyxy=batch["boxes"], conf=batch["box_scores"])
                out.append(SimpleNamespace(keypoints=keypoints, boxes=boxes))
            return out
        self.calls.append({"shape": frame.shape, **kwargs})
        batch = self.batches.pop(0)
        keypoints = SimpleNamespace(xy=batch["keypoints"], conf=batch["keypoint_scores"])
        boxes = SimpleNamespace(xyxy=batch["boxes"], conf=batch["box_scores"])
        return [SimpleNamespace(keypoints=keypoints, boxes=boxes)]


def make_person(bbox, ankle_y=40, score=0.8):
    x1, y1, x2, y2 = bbox
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[:, 0] = (x1 + x2) / 2
    keypoints[:, 1] = np.linspace(y1 + 2, ankle_y, 17)
    keypoints[15] = (x1 + 5, ankle_y)
    keypoints[16] = (x2 - 5, ankle_y)
    return keypoints, np.full(17, score, dtype=float)


def make_batch(people):
    if not people:
        return {
            "keypoints": np.empty((0, 17, 2), dtype=float),
            "keypoint_scores": np.empty((0, 17), dtype=float),
            "boxes": np.empty((0, 4), dtype=float),
            "box_scores": np.empty((0,), dtype=float),
        }
    return {
        "keypoints": np.stack([person[0] for person in people]),
        "keypoint_scores": np.stack([person[1] for person in people]),
        "boxes": np.asarray([person[2] for person in people], dtype=float),
        "box_scores": np.asarray([person[3] for person in people], dtype=float),
    }


class YOLOPoseProcessorTests(unittest.TestCase):
    def test_accepts_all_supported_image_sizes(self):
        for imgsz in (640, 960, 1280):
            with self.subTest(imgsz=imgsz):
                processor = YOLOPoseProcessor(imgsz=imgsz, model=FakeModel([]), device="cpu")
                self.assertEqual(processor.imgsz, imgsz)

    def test_rejects_unsupported_imgsz(self):
        with self.assertRaises(ValueError):
            YOLOPoseProcessor(imgsz=800, model=FakeModel([]))

    def test_configurable_imgsz_and_provenance(self):
        keypoints, scores = make_person((10, 10, 40, 70), ankle_y=65)
        model = FakeModel([make_batch([(keypoints, scores, (10, 10, 40, 70), 0.91)])])
        processor = YOLOPoseProcessor(imgsz=960, model=model, device="cpu")

        legacy_keypoints, legacy_scores = processor.process_frame(np.zeros((100, 200, 3), dtype=np.uint8))
        detection = processor.get_last_detections()[0]

        self.assertEqual(model.calls[0]["imgsz"], 960)
        self.assertEqual(legacy_keypoints.shape, (1, 17, 2))
        self.assertEqual(legacy_scores.shape, (1, 17))
        self.assertEqual(detection["source"], "full_frame")
        self.assertEqual(detection["inference"]["imgsz"], 960)
        self.assertEqual(detection["inference"]["roi"], [0, 0, 200, 100])
        self.assertAlmostEqual(detection["confidence"], 0.91)

    def test_fixed_camera_offsets_roi_merges_duplicate_and_honors_selected_size(self):
        full_kp, full_scores = make_person((50, 10, 90, 48), ankle_y=46, score=0.70)
        roi_kp, roi_scores = make_person((49, 9, 91, 49), ankle_y=47, score=0.92)
        full_batch = make_batch([(full_kp, full_scores, (50, 10, 90, 48), 0.80)])
        roi_batch = make_batch([(roi_kp, roi_scores, (49, 9, 91, 49), 0.88)])
        model = FakeModel([full_batch, roi_batch])
        processor = YOLOPoseProcessor(model=model, device="cpu", merge_iou=0.45)

        detections = processor.process_fixed_camera(
            np.zeros((100, 200, 3), dtype=np.uint8),
            far_roi=(0.0, 0.0, 1.0, 0.5),
            imgsz=960,
        )

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["imgsz"], 960)
        self.assertEqual(model.calls[1]["imgsz"], 960)
        self.assertEqual(model.calls[1]["shape"], (50, 200, 3))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["source"], "far_roi")
        self.assertEqual(set(detections[0]["merged_sources"]), {"full_frame", "far_roi"})
        self.assertEqual(detections[0]["inference"]["roi"], [0, 0, 200, 50])
        self.assertEqual(len(detections[0]["supporting_detections"]), 1)

    def test_pixel_roi_offsets_coordinates_to_full_frame(self):
        keypoints, scores = make_person((10, 5, 30, 35), ankle_y=33)
        model = FakeModel(
            [
                make_batch([]),
                make_batch([(keypoints, scores, (10, 5, 30, 35), 0.9)]),
            ]
        )
        processor = YOLOPoseProcessor(model=model, device="cpu")

        detections = processor.process_fixed_camera(
            np.zeros((100, 200, 3), dtype=np.uint8), far_roi=(20, 10, 180, 60)
        )

        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(detections[0]["bbox"], [30, 15, 50, 45])
        np.testing.assert_allclose(detections[0]["keypoints"][15], [35, 43])

    def test_fixed_camera_batch_runs_one_full_and_one_roi_batch_call(self):
        kp1, sc1 = make_person((50, 10, 90, 48), ankle_y=46, score=0.70)
        kp2, sc2 = make_person((49, 9, 91, 49), ankle_y=47, score=0.92)
        model = FakeModel(
            [
                make_batch([(kp1, sc1, (50, 10, 90, 48), 0.80)]),
                make_batch([(kp1, sc1, (50, 10, 90, 48), 0.80)]),
                make_batch([(kp2, sc2, (49, 9, 91, 49), 0.88)]),
                make_batch([(kp2, sc2, (49, 9, 91, 49), 0.88)]),
            ]
        )
        processor = YOLOPoseProcessor(model=model, device="cpu", merge_iou=0.45)
        frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(2)]
        batched = processor.process_fixed_camera_batch(
            frames,
            far_roi=(0.0, 0.0, 1.0, 0.5),
            imgsz=960,
        )

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["batch_size"], 2)
        self.assertEqual(model.calls[0]["shape"], (100, 200, 3))
        self.assertEqual(model.calls[1]["batch_size"], 2)
        self.assertEqual(model.calls[1]["shape"], (50, 200, 3))
        self.assertEqual(len(batched), 2)
        for detections in batched:
            self.assertEqual(len(detections), 1)
            self.assertEqual(detections[0]["source"], "far_roi")
            self.assertEqual(set(detections[0]["merged_sources"]), {"full_frame", "far_roi"})

    def test_fixed_camera_batch_splits_large_batches_by_max_batch_size(self):
        kp, sc = make_person((50, 10, 90, 48), ankle_y=46, score=0.8)
        frame_count = 35
        batches = [make_batch([(kp, sc, (50, 10, 90, 48), 0.85)]) for _ in range(frame_count * 2)]
        model = FakeModel(batches)
        processor = YOLOPoseProcessor(model=model, device="cpu", merge_iou=0.45)
        frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(frame_count)]

        result = processor.process_fixed_camera_batch(
            frames,
            far_roi=(0.0, 0.0, 1.0, 0.5),
            imgsz=960,
        )

        self.assertEqual(len(result), frame_count)
        self.assertEqual(
            [call["batch_size"] for call in model.calls],
            [16, 16, 3, 16, 16, 3],
        )

    def test_derive_far_court_roi_uses_ground_net_line_not_image_midpoint(self):
        corners = [[706.0, 326.0], [1173.0, 336.0], [1558.0, 920.0], [298.0, 892.0]]
        roi = derive_far_court_roi(corners, (1080, 1920, 3), include_upper_body=True)
        x1, y1, x2, y2 = roi
        # Upper edge leaves headroom above the far baseline; bottom edge is
        # anchored to the projected ground-plane net line (~494px + pad), well
        # above the image midpoint (~609px + pad).
        self.assertLessEqual(y1, 200)
        self.assertGreaterEqual(y2, 500)
        self.assertLessEqual(y2, 540)
        self.assertLess(x1, 706)
        self.assertGreater(x2, 1173)
        self.assertEqual(corners[0], [706.0, 326.0])

    def test_merge_detections_deduplicates_top_clipped_far_roi_and_keeps_full_frame(self):
        full_kp, full_scores = make_person((976.5, 225.0, 1095.2, 362.8), ankle_y=360.0, score=0.80)
        roi_kp, roi_scores = make_person((978.8, 324.0, 1094.0, 364.6), ankle_y=362.0, score=0.92)
        processor = YOLOPoseProcessor(model=FakeModel([]), device="cpu", merge_iou=0.45)

        detections = processor.merge_detections(
            [
                {
                    "bbox": np.asarray([978.8, 324.0, 1094.0, 364.6], dtype=float),
                    "keypoints": roi_kp,
                    "keypoint_scores": roi_scores,
                    "confidence": 0.90,
                    "source": "far_roi",
                    "inference": {"roi": [230, 324, 1651, 886]},
                },
                {
                    "bbox": np.asarray([976.5, 225.0, 1095.2, 362.8], dtype=float),
                    "keypoints": full_kp,
                    "keypoint_scores": full_scores,
                    "confidence": 0.85,
                    "source": "full_frame",
                    "inference": {"roi": [0, 0, 1920, 1080]},
                },
            ]
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["source"], "full_frame")
        self.assertEqual(set(detections[0]["merged_sources"]), {"full_frame", "far_roi"})

    def test_merge_detections_deduplicates_bottom_clipped_near_player_at_net(self):
        full_kp, full_scores = make_person((800.0, 380.0, 900.0, 620.0), ankle_y=612.0, score=0.75)
        roi_kp, roi_scores = make_person((802.0, 382.0, 898.0, 514.0), ankle_y=510.0, score=0.95)
        processor = YOLOPoseProcessor(model=FakeModel([]), device="cpu", merge_iou=0.45)

        detections = processor.merge_detections(
            [
                {
                    "bbox": np.asarray([802.0, 382.0, 898.0, 514.0], dtype=float),
                    "keypoints": roi_kp,
                    "keypoint_scores": roi_scores,
                    "confidence": 0.95,
                    "source": "far_roi",
                    "inference": {"roi": [514, 164, 1359, 521]},
                },
                {
                    "bbox": np.asarray([800.0, 380.0, 900.0, 620.0], dtype=float),
                    "keypoints": full_kp,
                    "keypoint_scores": full_scores,
                    "confidence": 0.82,
                    "source": "full_frame",
                    "inference": {"roi": [0, 0, 1920, 1080]},
                },
            ]
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["source"], "full_frame")
        self.assertEqual(set(detections[0]["merged_sources"]), {"full_frame", "far_roi"})

    def test_merge_detections_does_not_ios_merge_same_source_overlapping_players(self):
        near_kp, near_scores = make_person((700.0, 300.0, 950.0, 900.0), ankle_y=890.0, score=0.90)
        far_kp, far_scores = make_person((780.0, 310.0, 840.0, 420.0), ankle_y=415.0, score=0.80)
        processor = YOLOPoseProcessor(model=FakeModel([]), device="cpu", merge_iou=0.45)

        detections = processor.merge_detections(
            [
                {
                    "bbox": np.asarray([700.0, 300.0, 950.0, 900.0], dtype=float),
                    "keypoints": near_kp,
                    "keypoint_scores": near_scores,
                    "confidence": 0.90,
                    "source": "full_frame",
                    "inference": {"roi": [0, 0, 1920, 1080]},
                },
                {
                    "bbox": np.asarray([780.0, 310.0, 840.0, 420.0], dtype=float),
                    "keypoints": far_kp,
                    "keypoint_scores": far_scores,
                    "confidence": 0.80,
                    "source": "full_frame",
                    "inference": {"roi": [0, 0, 1920, 1080]},
                },
            ]
        )

        self.assertEqual(len(detections), 2)


if __name__ == "__main__":
    unittest.main()
