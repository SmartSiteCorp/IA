"""Reconstituer le petit lot figé depuis l'archive MBDD2025, sans tout extraire."""

import argparse
import re
import sys
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from smartsite_ia.archive import checked_members, read_member
from smartsite_ia.curation import digest, read_document, staged_output
from smartsite_ia.review import MAX_JSON_BYTES, read_local
from smartsite_ia.zero_shot import load_sample


def prepare_sample(archive: Path, selection: Path, output: Path) -> None:
    """Les empreintes figent les photos et XML exacts ; aucune nouvelle sélection."""
    document, checksum = read_document(selection)
    records = document.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 64:
        raise ValueError("Expected between 1 and 64 pinned sample records")
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size > 3_000_000_000:
        raise ValueError("Expected an unlinked ZIP smaller than 3 GB")
    with staged_output(output, [archive, selection]) as stage, ZipFile(archive) as source:
        members = checked_members(source, max_members=50_000, max_expanded_bytes=4_000_000_000)
        for row in records:
            if not isinstance(row, dict) or not re.fullmatch(
                r"Hefei[0-9]{1,6}", str(row.get("id", ""))
            ):
                raise ValueError("Invalid MBDD2025 identifier")
            sample_id = row["id"]
            if row.get("file") != f"images/{sample_id}.jpg":
                raise ValueError("Unexpected image destination")
            for source_dir, suffix, target_dir, key in (
                ("JPEGImages", ".jpg", "images", "sha256"),
                ("Annotations", ".xml", "annotations", "source_annotation_sha256"),
            ):
                name = f"MBDD2025/{source_dir}/{sample_id}{suffix}"
                if name not in members:
                    raise ValueError(f"Missing selected archive entry: {name}")
                raw = read_member(source, members[name])
                if digest(raw) != row.get(key):
                    raise ValueError("Selected file differs from the pinned sample")
                target = stage / target_dir / f"{sample_id}{suffix}"
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(raw)
        raw_manifest = read_local(selection.parent, selection.name, MAX_JSON_BYTES)
        if digest(raw_manifest) != checksum:
            raise ValueError("Sample selection changed during preparation")
        (stage / "manifest.json").write_bytes(raw_manifest)
        load_sample(stage / "manifest.json", 64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a pinned MBDD2025 probe sample")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare_sample(args.archive, args.selection, args.output)
    except (ValueError, OSError, BadZipFile) as error:
        print(f"Sample preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Prepared sample: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
