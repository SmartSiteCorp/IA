"""Commandes explicites du moteur : aucun entraînement au simple import du paquet."""

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

from smartsite_ia.learning import runtime, train_model
from smartsite_ia.model_assets import PRETRAINED
from smartsite_ia.prediction import predict_photo
from smartsite_ia.source import download_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and inspect experimental SmartSite models")
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("weights", help="Download the pinned official pretrained weights")
    weights.add_argument("--output", type=Path, required=True)
    doctor = commands.add_parser("doctor", help="Check installed engine and requested hardware")
    train = commands.add_parser(
        "train", help="Train on prepared train/valid data; leave test reserved"
    )
    train.add_argument("corpus", type=Path)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--weights", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser(
        "predict", help="Show predictions from a completed experimental run"
    )
    predict.add_argument("image", type=Path)
    predict.add_argument("--run", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--threshold", type=float, default=0.3)
    for command in (doctor, train, predict):
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    args = parser.parse_args()
    result: dict[str, Any]
    try:
        if args.command == "weights":
            result = {"weights": str(download_archive(args.output, PRETRAINED))}
        elif args.command == "doctor":
            result = runtime(args.device)
        elif args.command == "train":
            report = train_model(args.corpus, args.output, args.config, args.weights, args.device)
            result = {"status": report["status"], "run": str(args.output / "run.json")}
        else:
            report = predict_photo(args.run, args.image, args.output, args.device, args.threshold)
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
