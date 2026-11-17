import json
import tempfile
from pathlib import Path
from unittest import TestCase

from scripts.racket.prepare_yolo_pose_dataset import prepare_dataset, source_video_name


class RacketPoseDatasetTests(TestCase):
    def _write_label(self, directory: Path, image_name: str, groups: list[list[tuple[float, float]]]):
        (directory / image_name).write_bytes(b"not-a-real-jpeg")
        shapes = []
        for group_id, points in enumerate(groups, start=1):
            for point in points:
                shapes.append(
                    {
                        "label": "racket",
                        "points": [list(point)],
                        "group_id": group_id,
                        "shape_type": "point",
                    }
                )
        (directory / f"{Path(image_name).stem}.json").write_text(
            json.dumps(
                {
                    "imagePath": image_name,
                    "imageWidth": 200,
                    "imageHeight": 100,
                    "shapes": shapes,
                }
            ),
            encoding="utf-8",
        )

    def test_uses_complete_groups_and_splits_by_whole_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            points = [(10, 20), (20, 30), (30, 40), (25, 45), (35, 45)]
            self._write_label(raw, "match_a__0010.00s.jpg", [points, points[:3]])
            self._write_label(raw, "match_b__0010.00s.jpg", [points])

            summary = prepare_dataset(raw, root / "dataset", {"match_b"})

            self.assertEqual(summary["images"], {"train": 1, "val": 1})
            self.assertEqual(summary["racket_instances"], {"train": 1, "val": 1})
            self.assertEqual(summary["skipped_groups"], 1)
            train_label = (root / "dataset" / "labels" / "train" / "match_a__0010.00s.txt")
            fields = train_label.read_text(encoding="utf-8").strip().split()
            self.assertEqual(len(fields), 20)  # class + box + 5 * (x, y, visibility)
            self.assertEqual(fields[0], "0")
            self.assertEqual(fields[-1], "2")
            yaml_text = (root / "dataset" / "racket-pose.yaml").read_text(encoding="utf-8")
            self.assertNotIn(str(root), yaml_text)
            self.assertIn("train: images/train", yaml_text)

    def test_extracts_source_video_from_sampled_frame_name(self):
        self.assertEqual(source_video_name("match__0012.50s.jpg"), "match")
