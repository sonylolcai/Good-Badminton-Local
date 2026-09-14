"""Command-line entry for the local analysis and evaluation platform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .app_service import (
    compare_run_reports,
    create_run,
    execute_local_run,
    publish_dataset_version,
    verify_published_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analysis_platform")
    commands = parser.add_subparsers(dest="command", required=True)

    publish = commands.add_parser("publish-dataset", help="publish an immutable Dataset Version")
    _store_argument(publish)
    publish.add_argument("--manifest", required=True, type=Path)

    create = commands.add_parser("create-run", help="freeze an analysis Run input")
    _store_argument(create)
    create.add_argument("--task", required=True, type=Path)

    run = commands.add_parser("run-local", help="run analysis, evaluation and optional comparison")
    _store_argument(run)
    run.add_argument("--dataset-version", required=True, type=Path)
    run.add_argument("--task", required=True, type=Path)
    run.add_argument("--runner-config", required=True, type=Path)
    run.add_argument("--evaluator", required=True, choices=(
        "far_player", "doubles", "shuttle_tracknet_ab", "performance_streaming"
    ))
    run.add_argument("--evaluator-arguments", required=True, type=Path, help="JSON array of existing evaluator CLI arguments")
    run.add_argument("--input-root", type=Path, default=Path.cwd())
    run.add_argument("--baseline", type=Path)
    run.add_argument("--gate-profile", type=Path)

    compare = commands.add_parser("compare", help="compare two compatible Run reports")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--gate-profile", type=Path)
    compare.add_argument("--output", type=Path)

    verify = commands.add_parser("verify", help="verify an immutable published manifest")
    _store_argument(verify)
    verify.add_argument("--kind", required=True, choices=(
        "dataset", "dataset_version", "case", "annotation_version", "experiment", "metric_result", "artifact"
    ))
    verify.add_argument("--id", required=True)

    args = parser.parse_args(argv)
    if args.command == "publish-dataset":
        result = publish_dataset_version(args.store, _json_object(args.manifest))
    elif args.command == "create-run":
        result = create_run(args.store, _json_object(args.task))
    elif args.command == "run-local":
        evaluator_arguments = _json(args.evaluator_arguments)
        if not isinstance(evaluator_arguments, list) or not all(isinstance(value, str) for value in evaluator_arguments):
            parser.error("--evaluator-arguments must contain a JSON array of strings")
        result = execute_local_run(
            args.store,
            _json_object(args.dataset_version),
            _json_object(args.task),
            _json_object(args.runner_config),
            args.evaluator,
            evaluator_arguments,
            input_root=args.input_root,
            baseline_report=_json_object(args.baseline) if args.baseline else None,
            gate_profile=_json_object(args.gate_profile) if args.gate_profile else None,
        )
    elif args.command == "compare":
        result = compare_run_reports(
            _json_object(args.baseline),
            _json_object(args.candidate),
            gate_profile=_json_object(args.gate_profile) if args.gate_profile else None,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(_render(result), encoding="utf-8")
    else:
        result = verify_published_manifest(args.store, args.kind, args.id)

    print(_render(result), end="")
    return 0


def _store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", required=True, type=Path)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_object(path: Path) -> dict[str, Any]:
    value = _json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
