"""Explicit download and import commands; no training side effects."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile

from smartsite_ia.importer import import_archive
from smartsite_ia.review import review_corpus
from smartsite_ia.source import SOURCES, download_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and audit DamSegment v1")
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
    args = parser.parse_args()
    try:
        if args.command == "download":
            print(download_archive(args.output, SOURCES[args.dataset]))
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
