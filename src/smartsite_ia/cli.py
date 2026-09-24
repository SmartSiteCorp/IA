"""Explicit download and import commands; no training side effects."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile

from smartsite_ia.external import prepare_external
from smartsite_ia.importer import import_archive
from smartsite_ia.prepare import prepare_corpus
from smartsite_ia.review import review_corpus
from smartsite_ia.sdnet_sample import SURFACES, prepare_sample
from smartsite_ia.source import SOURCES, download_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and audit SmartSite defect datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify a pinned source archive")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--dataset", choices=SOURCES, default="damsegment_v1")
    convert = commands.add_parser("import", help="Validate and convert to an unsplit audit corpus")
    convert.add_argument("archive", type=Path)
    convert.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review", help="Review annotations and search for similar images")
    review.add_argument("corpus", type=Path)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--max-hamming-distance", type=int, default=8)
    review.add_argument("--max-pixel-error", type=float, default=20.0)
    prepare = commands.add_parser(
        "prepare", help="Apply reviewed curation and export grouped COCO splits"
    )
    prepare.add_argument("corpus", type=Path)
    prepare.add_argument("--policy", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20260918)
    external = commands.add_parser(
        "prepare-external", help="Normalize verified external image/mask pairs"
    )
    external.add_argument(
        "source", type=Path, help="Source directory containing extracted/rgb and extracted/BW"
    )
    external.add_argument("--policy", type=Path, required=True)
    external.add_argument("--output", type=Path, required=True)
    patches = commands.add_parser(
        "prepare-patches", help="Sample labelled SDNET2018 patches, clear ones and cracked ones"
    )
    patches.add_argument("archive", type=Path, help="Manually downloaded SDNET2018.zip")
    patches.add_argument("--output", type=Path, required=True)
    patches.add_argument("--surface", default="W", choices=sorted(SURFACES))
    patches.add_argument("--per-photo", type=int, default=10)
    patches.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "prepare-patches":
            report = prepare_sample(
                args.archive, args.output, args.surface, args.per_photo, args.seed
            )
            print(
                json.dumps(
                    {
                        "counts": report["counts"],
                        "scene_groups": report["scene_groups"],
                        "selection": report["selection"],
                        "approved_for_training": report["approved_for_training"],
                    },
                    indent=2,
                )
            )
        elif args.command == "download":
            print(download_archive(args.output, SOURCES[args.dataset]))
        elif args.command in ("prepare", "prepare-external"):
            report = (
                prepare_corpus(args.corpus, args.output, args.policy, args.seed)
                if args.command == "prepare"
                else prepare_external(args.source, args.output, args.policy)
            )
            print(
                json.dumps(
                    {
                        "images": report["images"],
                        "usage": report["usage"],
                        "approved_for_training": report["approved_for_training"],
                        "report": str(args.output / "index.html"),
                        "splits": report.get("splits", report.get("split_counts")),
                    },
                    indent=2,
                )
            )
        elif args.command == "review":
            report = review_corpus(
                args.corpus, args.output, args.max_hamming_distance, args.max_pixel_error
            )
            print(
                json.dumps(
                    {
                        "images": report["images"],
                        "flag_counts": report["flag_counts"],
                        "similar_pairs": len(report["similar_pairs"]),
                        "approved_for_training": report["approved_for_training"],
                        "review_page": str(args.output / "index.html"),
                    },
                    indent=2,
                )
            )
        else:
            report = import_archive(args.archive, args.output)
            print(
                json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "images",
                            "annotations",
                            "approved_for_training",
                            "training_blockers",
                        )
                    },
                    indent=2,
                )
            )
    except (OSError, ValueError, BadZipFile) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
