"""Commandes du pilote par rectangles, séparées de l'ancien modèle à masques."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from smartsite_ia.box_data import load_box_config, prepare_box_inputs
from smartsite_ia.box_review import review_boxes
from smartsite_ia.box_training import check_box_training, train_boxes
from smartsite_ia.model_assets import BOX_PRETRAINED
from smartsite_ia.source import download_archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("weights", help="Download only the pinned official Nano weights")
    weights.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Check and copy reviewed train/validation boxes")
    check = commands.add_parser("check", help="Check inputs and hardware without training")
    train = commands.add_parser("train", help="Run the short box pilot; preserve existing models")
    for command in (prepare, check, train):
        command.add_argument("corpus", type=Path)
        command.add_argument("--config", type=Path, required=True)
    for command in (check, train):
        command.add_argument("--weights", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    for command in (prepare, train):
        command.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    review = commands.add_parser("review", help="Compare validation boxes and predictions locally")
    review.add_argument("run", type=Path)
    review.add_argument("--corpus", type=Path, required=True)
    review.add_argument("--config", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    review.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args(argv)
    result: dict[str, Any]
    try:
        if args.command == "weights":
            result = {"weights": str(download_archive(args.output, BOX_PRETRAINED))}
        elif args.command == "prepare":
            config, _ = load_box_config(args.config)
            selection = prepare_box_inputs(args.corpus, args.output, config)
            result = {
                "status": "prepared",
                "splits": {
                    s: {k: v for k, v in d.items() if k != "records"}
                    for s, d in selection["splits"].items()
                },
            }
        elif args.command == "check":
            result = check_box_training(args.corpus, args.config, args.weights, args.device)
        elif args.command == "review":
            report = review_boxes(
                args.run, args.corpus, args.config, args.output, args.device, args.threshold
            )
            result = {
                "status": report["status"],
                "gallery": str(args.output / "index.html"),
                "summary": report["summary"],
            }
        else:
            report = train_boxes(
                args.corpus, args.config, args.weights, args.output, args.device, resume=args.resume
            )
            result = {
                "status": report["status"],
                "run": str(args.output / "run.json"),
                "metrics": report["last_epoch_metrics"],
            }
    except KeyboardInterrupt:
        print("Opération interrompue ; les entrées restent conservées.", file=sys.stderr)
        return 130
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Pilote par boîtes interrompu : {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
