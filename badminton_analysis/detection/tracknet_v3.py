"""Read immutable raw TrackNetV3 measurements for the primary video pipeline.

TrackNet exposes a binary heatmap visibility decision in ``Frame,Visibility,X,Y``.
It does *not* provide a calibrated probability, so callers must retain the
``uncalibrated_binary_visibility_threshold`` provenance rather than treating
the value as a YOLO confidence.
"""

from __future__ import annotations

import csv
from pathlib import Path


class TrackNetV3RawMeasurements:
    """Indexed, validated raw measurements from one TrackNetV3 CSV file."""

    source = "tracknet_v3_raw"
    measurement_kind = "temporal_heatmap"
    confidence_status = "uncalibrated_binary_visibility_threshold_0.5"

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"TrackNetV3 raw CSV not found: {self.csv_path}")
        self._measurements = self._load()

    def _load(self):
        measurements = {}
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Frame", "Visibility", "X", "Y"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"TrackNetV3 CSV missing columns {sorted(missing)}: {self.csv_path}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    frame = int(float(row["Frame"]))
                    visible = int(float(row["Visibility"])) != 0
                    x, y = float(row["X"]), float(row["Y"])
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid TrackNetV3 measurement at {self.csv_path}:{line_number}"
                    ) from exc
                if frame < 0 or frame in measurements:
                    raise ValueError(
                        f"Invalid or duplicate TrackNetV3 frame {frame} at {self.csv_path}:{line_number}"
                    )
                measurements[frame] = (visible, x, y)
        if not measurements:
            raise ValueError(f"TrackNetV3 CSV has no measurements: {self.csv_path}")
        return measurements

    @property
    def frame_count(self):
        return len(self._measurements)

    def measurement_for_frame(self, frame_index):
        """Return one raw observation without filling an absent TrackNet frame."""
        visible, x, y = self._measurements.get(int(frame_index), (False, None, None))
        return {
            "frame": int(frame_index),
            "visible": bool(visible),
            "image": [x, y] if visible else None,
            "source": self.source,
            "measurement_kind": self.measurement_kind,
            "confidence": 0.5 if visible else None,
            "confidence_status": self.confidence_status if visible else "not_visible",
        }
