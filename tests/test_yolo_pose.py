import unittest
from types import SimpleNamespace

import numpy as np

from badminton_analysis.detection.yolo_pose import YOLOPoseProcessor


class FakeModel:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def __call__(self, frame, **kwargs):
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

    def test_fixed_camera_offsets_roi_and_merges_duplicate(self):
        full_kp, full_scores = make_person((50, 10, 90, 48), ankle_y=46, score=0.70)
        roi_kp, roi_scores = make_person((49, 9, 91, 49), ankle_y=47, score=0.92)
        full_batch = make_batch([(full_kp, full_scores, (50, 10, 90, 48), 0.80)])
        roi_batch = make_batch([(roi_kp, roi_scores, (49, 9, 91, 49), 0.88)])
        model = FakeModel([full_batch, roi_batch])
        processor = YOLOPoseProcessor(model=model, device="cpu", merge_iou=0.45)

        detections = processor.process_fixed_camera(
            np.zeros((100, 200, 3), dtype=np.uint8), far_roi=(0.0, 0.0, 1.0, 0.5)
        )

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["imgsz"], 640)
        self.assertEqual(model.calls[1]["imgsz"], 640)
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


if __name__ == "__main__":
    unittest.main()
