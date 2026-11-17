"""Convert X-AnyLabeling racket points into a reproducible YOLO Pose dataset.

The source labels stay untouched.  A usable racket is one ``group_id`` with
exactly five ``point`` shapes, in the project's fixed click order:
grip, throat, head_top, head_left, head_right.  The source format has no
rectangle, so this tool derives a padded training box from the five points.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


KEYPOINT_COUNT = 5
POSE_CLASS_ID = 0
SOURCE_SUFFIX = re.compile(r"__(\d+(?:\.\d+)?)s$")


@dataclass(frozen=True)
class PoseInstance:
    source_video: str
    image_path: Path
    json_path: Path
    group_id: str
    width: int
    height: int
    keypoints: tuple[tuple[float, float], ...]


def source_video_name(image_name: str) -> str:
    """Return the video identifier embedded by the frame-sampling scripts."""
    return SOURCE_SUFFIX.sub("", Path(image_name).stem)


def _group_id(shape: dict) -> str | None:
    value = shape.get("group_id")
    return None if value is None else str(value)


def load_instances(json_path: Path) -> tuple[list[PoseInstance], list[dict[str, str]]]:
    """Read one LabelMe-compatible JSON file without modifying it.

    Incomplete groups are recorded for audit and excluded only at group level;
    complete rackets in the same image are still useful training data.
    """
    document = json.loads(json_path.read_text(encoding="utf-8"))
    image_name = document.get("imagePath")
    width = int(document.get("imageWidth") or 0)
    height = int(document.get("imageHeight") or 0)
    if not image_name or width <= 0 or height <= 0:
        raise ValueError(f"missing image metadata: {json_path}")

    grouped_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    skipped: list[dict[str, str]] = []
    for index, shape in enumerate(document.get("shapes", [])):
        if shape.get("shape_type") != "point":
            continue
        group_id = _group_id(shape)
        points = shape.get("points") or []
        if group_id is None:
            skipped.append(
                {
                    "json": json_path.name,
                    "image": str(image_name),
                    "group_id": "",
                    "reason": f"point_at_index_{index}_has_no_group_id",
                }
            )
            continue
        if len(points) != 1 or len(points[0]) < 2:
            skipped.append(
                {
                    "json": json_path.name,
                    "image": str(image_name),
                    "group_id": group_id,
                    "reason": f"point_at_index_{index}_has_invalid_coordinates",
                }
            )
            continue
        grouped_points[group_id].append((float(points[0][0]), float(points[0][1])))

    source_video = source_video_name(str(image_name))
    image_path = json_path.parent / image_name
    instances: list[PoseInstance] = []
    for group_id, keypoints in sorted(grouped_points.items(), key=lambda item: item[0]):
        if len(keypoints) != KEYPOINT_COUNT:
            skipped.append(
                {
                    "json": json_path.name,
                    "image": str(image_name),
                    "group_id": group_id,
                    "reason": f"expected_{KEYPOINT_COUNT}_points_found_{len(keypoints)}",
                }
            )
            continue
        instances.append(
            PoseInstance(
                source_video=source_video,
                image_path=image_path,
                json_path=json_path,
                group_id=group_id,
                width=width,
                height=height,
                keypoints=tuple(keypoints),
            )
        )
    return instances, skipped


def yolo_pose_line(
    instance: PoseInstance,
    *,
    padding_fraction: float = 0.15,
    min_padding_px: float = 8.0,
) -> str:
    """Return one normalized YOLO Pose label line for an annotated racket."""
    if padding_fraction < 0 or min_padding_px < 0:
        raise ValueError("padding values must be non-negative")
    xs = [point[0] for point in instance.keypoints]
    ys = [point[1] for point in instance.keypoints]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    padding = max(min_padding_px, span * padding_fraction)
    left = max(0.0, min(xs) - padding)
    top = max(0.0, min(ys) - padding)
    right = min(float(instance.width), max(xs) + padding)
    bottom = min(float(instance.height), max(ys) + padding)
    if right <= left or bottom <= top:
        raise ValueError(f"invalid derived box for {instance.json_path} group {instance.group_id}")

    values = [
        POSE_CLASS_ID,
        ((left + right) / 2) / instance.width,
        ((top + bottom) / 2) / instance.height,
        (right - left) / instance.width,
        (bottom - top) / instance.height,
    ]
    fields = [str(values[0])] + [f"{value:.8f}" for value in values[1:]]
    for x, y in instance.keypoints:
        fields.extend((f"{min(max(x, 0.0), instance.width) / instance.width:.8f}",
                       f"{min(max(y, 0.0), instance.height) / instance.height:.8f}", "2"))
    return " ".join(fields)


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    validation_videos: set[str],
    *,
    padding_fraction: float = 0.15,
    min_padding_px: float = 8.0,
) -> dict:
    """Copy valid source pairs and write YOLO labels split by complete video."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    if not validation_videos:
        raise ValueError("at least one complete source video must be held out for validation")
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}. Choose a new versioned output directory."
        )

    all_instances: list[PoseInstance] = []
    skipped: list[dict[str, str]] = []
    json_files = sorted(source_dir.glob("*.json"))
    for json_path in json_files:
        instances, group_skips = load_instances(json_path)
        skipped.extend(group_skips)
        if any(not instance.image_path.is_file() for instance in instances):
            missing = [str(instance.image_path) for instance in instances if not instance.image_path.is_file()]
            skipped.append(
                {
                    "json": json_path.name,
                    "image": "; ".join(missing),
                    "group_id": "",
                    "reason": "image_file_missing",
                }
            )
            continue
        all_instances.extend(instances)

    sources = {instance.source_video for instance in all_instances}
    unknown_validation = sorted(validation_videos - sources)
    if unknown_validation:
        raise ValueError(f"validation video(s) not found: {unknown_validation}")
    if sources <= validation_videos:
        raise ValueError("validation selection leaves no training video")

    grouped_instances: dict[Path, list[PoseInstance]] = defaultdict(list)
    for instance in all_instances:
        grouped_instances[instance.image_path].append(instance)

    output_dir.mkdir(parents=True)
    manifest: list[dict[str, str]] = []
    image_counts: Counter[str] = Counter()
    racket_counts: Counter[str] = Counter()
    for image_path, instances in sorted(grouped_instances.items(), key=lambda item: item[0].name.lower()):
        split = "val" if instances[0].source_video in validation_videos else "train"
        image_target = output_dir / "images" / split / image_path.name
        label_target = output_dir / "labels" / split / f"{image_path.stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_target)
        label_target.write_text(
            "\n".join(
                yolo_pose_line(
                    instance,
                    padding_fraction=padding_fraction,
                    min_padding_px=min_padding_px,
                )
                for instance in instances
            )
            + "\n",
            encoding="utf-8",
        )
        image_counts[split] += 1
        racket_counts[split] += len(instances)
        manifest.append(
            {
                "split": split,
                "source_video": instances[0].source_video,
                "image": image_path.name,
                "json": instances[0].json_path.name,
                "racket_instances": str(len(instances)),
            }
        )

    yaml_path = output_dir / "racket-pose.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# Dataset root is this YAML file's directory; keep this file beside images/ and labels/.",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: racket",
                "kpt_shape: [5, 3]",
                "flip_idx: [0, 1, 2, 4, 3]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "manifest.csv", manifest, ["split", "source_video", "image", "json", "racket_instances"])
    _write_csv(output_dir / "skipped_groups.csv", skipped, ["json", "image", "group_id", "reason"])
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "json_files_scanned": len(json_files),
        "source_videos": sorted(sources),
        "validation_videos": sorted(validation_videos),
        "images": dict(sorted(image_counts.items())),
        "racket_instances": dict(sorted(racket_counts.items())),
        "skipped_groups": len(skipped),
        "keypoint_order": ["grip", "throat", "head_top", "head_left", "head_right"],
        "box_padding_fraction": padding_fraction,
        "min_box_padding_px": min_padding_px,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert grouped X-AnyLabeling racket points to YOLO Pose")
    parser.add_argument("source_dir", type=Path, help="directory containing paired JPG and X-AnyLabeling JSON files")
    parser.add_argument("--output", type=Path, required=True, help="new versioned YOLO dataset directory")
    parser.add_argument(
        "--val-video",
        action="append",
        required=True,
        help="complete source-video name to reserve for validation; repeat for multiple videos",
    )
    parser.add_argument("--padding-fraction", type=float, default=0.15)
    parser.add_argument("--min-padding-px", type=float, default=8.0)
    args = parser.parse_args()
    summary = prepare_dataset(
        args.source_dir,
        args.output,
        set(args.val_video),
        padding_fraction=args.padding_fraction,
        min_padding_px=args.min_padding_px,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
