"""Commandes explicites du moteur : aucun entraînement au simple import du paquet."""

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

from smartsite_ia.continuation import extend_training
from smartsite_ia.inference import DEFAULT_PROFILE, PROFILES
from smartsite_ia.learning import runtime, train_model
from smartsite_ia.model_assets import CLASS_NAMES, PRETRAINED
from smartsite_ia.prediction import predict_photo
from smartsite_ia.source import download_archive
from smartsite_ia.threshold_study import DEFAULT_THRESHOLDS, study_thresholds
from smartsite_ia.training_data import load_training_config, prepare_training_inputs
from smartsite_ia.validation import compare_validations, evaluate_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and inspect experimental SmartSite models")
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("weights", help="Download the pinned official pretrained weights")
    weights.add_argument("--output", type=Path, required=True)
    doctor = commands.add_parser("doctor", help="Check installed engine and requested hardware")
    prepare = commands.add_parser(
        "prepare", help="Prepare class-specific train/valid data without a model"
    )
    prepare.add_argument("corpus", type=Path)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser(
        "train", help="Train on prepared train/valid data; leave test reserved"
    )
    train.add_argument("corpus", type=Path)
    train.add_argument("--config", type=Path, required=True)
    origin = train.add_mutually_exclusive_group(required=True)
    origin.add_argument("--weights", type=Path)
    origin.add_argument(
        "--from-run", type=Path, help="Fine-tune verified weights with a new optimizer"
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true", help="Resume a recorded interrupted run")
    extend = commands.add_parser("extend", help="Continue full state and evaluate each extra epoch")
    extend.add_argument("corpus", type=Path)
    extend.add_argument("--parent", type=Path, required=True)
    extend.add_argument("--plan", type=Path, required=True)
    extend.add_argument("--output", type=Path, required=True)
    extend.add_argument("--check", action="store_true", help="Check readiness without training")
    extend.add_argument("--resume", action="store_true", help="Resume this extension after a stop")
    predict = commands.add_parser(
        "predict", help="Show predictions from a completed experimental run"
    )
    predict.add_argument("image", type=Path)
    predict.add_argument("--run", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--threshold", type=float, default=0.3)
    evaluate = commands.add_parser("evaluate", help="Evaluate all pinned validation images")
    evaluate.add_argument("corpus", type=Path)
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--classes", nargs="+", choices=CLASS_NAMES, help="Compare a shared subset of model classes"
    )
    for command in (predict, evaluate):
        command.add_argument("--preprocessing", choices=PROFILES, default=DEFAULT_PROFILE)
        command.add_argument("--mask-threshold", type=float, default=0.5)
    compare = commands.add_parser("compare", help="Compare two identical validation protocols")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    study = commands.add_parser(
        "study-thresholds", help="Compare saved validation scores; no model required"
    )
    study.add_argument("corpus", type=Path)
    study.add_argument("--run", type=Path, required=True)
    study.add_argument("--evaluation", type=Path, required=True)
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--class", dest="target_class", choices=CLASS_NAMES, default="surface_loss")
    study.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    for command in (doctor, train, extend, predict, evaluate):
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    args = parser.parse_args()
    result: dict[str, Any]
    try:
        if args.command == "weights":
            result = {"weights": str(download_archive(args.output, PRETRAINED))}
        elif args.command == "doctor":
            result = runtime(args.device)
        elif args.command == "prepare":
            config, _ = load_training_config(args.config)
            selection = prepare_training_inputs(args.corpus, args.output, config)
            result = {
                "class_names": selection["class_names"],
                "report": str(args.output / "selection.json"),
                "splits": {
                    split: {
                        key: value
                        for key, value in details.items()
                        if key not in ("records", "without_target_annotation_ids")
                    }
                    for split, details in selection["splits"].items()
                },
            }
        elif args.command == "train":
            options: dict[str, Any] = {"resume": True} if args.resume else {}
            if args.from_run:
                options["from_run"] = args.from_run
            report = train_model(
                args.corpus, args.output, args.config, args.weights, args.device, **options
            )
            result = {"status": report["status"], "run": str(args.output / "run.json")}
        elif args.command == "extend":
            report = extend_training(
                args.corpus,
                args.parent,
                args.output,
                args.plan,
                args.device,
                check=args.check,
                resume=args.resume,
            )
            result = {
                "status": report["status"],
                "training_started": not args.check,
                "report": str(args.output / "index.html") if not args.check else None,
            }
        elif args.command == "evaluate":
            report = evaluate_validation(
                args.run,
                args.corpus,
                args.output,
                args.device,
                args.preprocessing,
                args.mask_threshold,
                tuple(args.classes) if args.classes is not None else None,
            )
            result = {"images": report["images"], "report": str(args.output / "index.html")}
        elif args.command == "compare":
            compare_validations(args.before, args.after, args.output)
            result = {"report": str(args.output / "index.html")}
        elif args.command == "study-thresholds":
            report = study_thresholds(
                args.run,
                args.corpus,
                args.evaluation,
                args.output,
                args.target_class,
                tuple(args.thresholds),
            )
            result = {
                "images": report["images"],
                "report": str(args.output / "index.html"),
                "best_observed_f1_thresholds": report["best_observed_f1_thresholds"],
                "selected_threshold": None,
            }
        else:
            report = predict_photo(
                args.run,
                args.image,
                args.output,
                args.device,
                args.threshold,
                args.preprocessing,
                args.mask_threshold,
            )
            result = {
                "predictions": len(report["predictions"]),
                "report": str(args.output / "index.html"),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ImportError, PackageNotFoundError) as error:
        print(
            f"Moteur indisponible : {error}. Installer : uv sync --locked --extra training",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Échec du moteur : {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
