from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import load_project


PIPELINE_CHOICES = ("hog_svm", "small_cnn", "mobilenet_v3_small")


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/project_config.json")


def _device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pv-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor")
    _config_argument(doctor)
    _device_argument(doctor)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    for name in ("acquire", "build-manifest", "validate"):
        child = data_commands.add_parser(name)
        _config_argument(child)

    smoke = commands.add_parser("smoke-test")
    _config_argument(smoke)
    _device_argument(smoke)

    train = commands.add_parser("train")
    train.add_argument("pipeline", choices=PIPELINE_CHOICES)
    _config_argument(train)
    _device_argument(train)

    gate = commands.add_parser("freeze-test-gate")
    _config_argument(gate)
    gate.add_argument("--created-by", default=os.environ.get("USERNAME", "operator"))

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--pipeline", choices=("all", *PIPELINE_CHOICES), required=True)
    evaluate.add_argument("--split", choices=("val", "test"), required=True)
    _config_argument(evaluate)
    _device_argument(evaluate)

    robustness = commands.add_parser("robustness")
    robustness.add_argument("--pipeline", choices=("all", *PIPELINE_CHOICES), required=True)
    _config_argument(robustness)
    _device_argument(robustness)

    compare = commands.add_parser("compare")
    _config_argument(compare)

    verify = commands.add_parser("verify-run")
    _config_argument(verify)

    predict = commands.add_parser("predict")
    predict.add_argument("--pipeline", choices=PIPELINE_CHOICES, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    _config_argument(predict)
    _device_argument(predict)

    report = commands.add_parser("report")
    report_commands = report.add_subparsers(dest="report_command", required=True)
    build_assets = report_commands.add_parser("build-assets")
    _config_argument(build_assets)
    return parser


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False))


def dispatch(args: argparse.Namespace) -> tuple[Any, bool]:
    context = load_project(args.config)
    if args.command == "doctor":
        from .doctor import run_doctor

        result = run_doctor(context, args.device)
        return result, result["status"] == "pass"
    if args.command == "data":
        if args.data_command == "acquire":
            from .data.acquire import acquire_dataset
            from .state import update_run_state

            result = acquire_dataset(context)
            update_run_state(
                context,
                phase="P0",
                status="complete",
                artifacts=("data/source_snapshot/dataset_lock.json",),
                command=sys.argv,
            )
            return result, True
        if args.data_command == "build-manifest":
            from .data.manifest import build_manifest

            return build_manifest(context), True
        from .data.manifest import validate_manifest
        from .state import update_run_state

        result = validate_manifest(context)
        update_run_state(
            context,
            phase="P1",
            status="complete",
            artifacts=("data/manifests/manifest.jsonl", "data/manifests/validate_manifest.json"),
            command=sys.argv,
        )
        return result, True
    if args.command == "smoke-test":
        from .smoke import run_smoke_test
        from .state import update_run_state

        result = run_smoke_test(context, args.device)
        update_run_state(
            context,
            phase="P3",
            status="complete" if result["status"] == "pass" else "failed",
            artifacts=("artifacts/smoke/smoke_test.json",),
            command=sys.argv,
            note="P2 human montage review remains pending.",
        )
        return result, result["status"] == "pass"
    if args.command == "train":
        if args.pipeline == "hog_svm":
            from .training.hog_svm import train_hog_svm

            path = train_hog_svm(context, sys.argv)
        elif args.pipeline == "small_cnn":
            from .training.deep_pipelines import train_small_cnn

            path = train_small_cnn(
                context, device_name=args.device, command=sys.argv
            )
        else:
            from .training.deep_pipelines import train_mobilenet_v3

            path = train_mobilenet_v3(
                context, device_name=args.device, command=sys.argv
            )
        from .state import update_run_state

        phase = {"hog_svm": "P4", "small_cnn": "P5", "mobilenet_v3_small": "P6"}[
            args.pipeline
        ]
        update_run_state(
            context,
            phase=phase,
            status="complete",
            artifacts=(path,),
            command=sys.argv,
        )
        return {"status": "pass", "artifact": str(path)}, True
    if args.command == "freeze-test-gate":
        from .evaluation.runner import freeze_test_gate
        from .state import update_run_state

        result = freeze_test_gate(context, args.created_by)
        update_run_state(
            context,
            phase="P7",
            status="complete",
            artifacts=("artifacts/test_gate.json",),
            command=sys.argv,
        )
        return result, True
    if args.command == "evaluate":
        from .evaluation.runner import evaluate_many

        paths = evaluate_many(
            context,
            pipeline=args.pipeline,
            split=args.split,
            device_name=args.device,
            command=sys.argv,
        )
        return {"status": "pass", "predictions": [str(path) for path in paths]}, True
    if args.command == "robustness":
        from .evaluation.robustness import run_robustness

        path = run_robustness(
            context, pipeline=args.pipeline, device_name=args.device
        )
        return {"status": "pass", "summary": str(path)}, True
    if args.command == "compare":
        from .evaluation.compare import compare_models
        from .state import update_run_state

        path = compare_models(context)
        update_run_state(
            context,
            phase="P8",
            status="complete",
            artifacts=(path,),
            command=sys.argv,
        )
        return {"status": "pass", "artifacts": str(path)}, True
    if args.command == "verify-run":
        from .verify_run import verify_complete_run
        from .state import update_run_state

        result = verify_complete_run(context)
        update_run_state(
            context,
            phase="P9",
            status="complete",
            artifacts=("artifacts/verify_run.json",),
            command=sys.argv,
        )
        return result, result["status"] == "pass"
    if args.command == "predict":
        from .predict import predict_image

        result = predict_image(
            context,
            pipeline_id=args.pipeline,
            image_path=args.image.resolve(),
            model_artifact=args.model.resolve(),
            device_name=args.device,
        )
        return result, True
    if args.command == "report" and args.report_command == "build-assets":
        from .reporting.assets import build_report_assets

        path = build_report_assets(context)
        return {"status": "pass", "report_tables": str(path)}, True
    raise RuntimeError(f"Unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result, passed = dispatch(args)
        _emit(result)
        return 0 if passed else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.environ.get("PV_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)
        return 1
