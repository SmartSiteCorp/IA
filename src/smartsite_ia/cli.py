"""Explicit download and import commands; no training side effects."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile

from smartsite_ia.importer import import_archive
from smartsite_ia.source import download_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and audit DamSegment v1")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="Download and verify the pinned segmentation ZIP"
    )
    download.add_argument("--output", type=Path, required=True)
    convert = commands.add_parser("import", help="Validate and convert to an unsplit audit corpus")
    convert.add_argument("archive", type=Path)
    convert.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "download":
            print(download_archive(args.output))
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
