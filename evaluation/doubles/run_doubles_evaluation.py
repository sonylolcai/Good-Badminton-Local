"""CLI for explicit fixed-camera doubles JSONL evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.doubles.metrics import evaluate_doubles_tracking


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate four-player fixed-camera tracking")
    parser.add_argument("--annotations", required=True, help="Human-reviewed doubles annotation JSONL")
    parser.add_argument("--detections", required=True, help="Analysis detections.jsonl with spatial.tracks")
    parser.add_argument("--output", required=True, help="Output report JSON")
    parser.add_argument("--match-distance-m", type=float, default=1.25)
    args = parser.parse_args()
    report = {
        "schema_version": "1.0",
        "annotations": str(Path(args.annotations).resolve()),
        "detections": str(Path(args.detections).resolve()),
        "match_distance_m": args.match_distance_m,
        "metrics": evaluate_doubles_tracking(
            load_jsonl(args.annotations), load_jsonl(args.detections), args.match_distance_m
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
