"""Préparer une collecte à annoter, sans la faire passer pour un corpus validé."""

import argparse
import io
import json
import math
import re
import sys
import time
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image, ImageDraw

from smartsite_ia.categories import COLLECTION_COLORS, COLLECTION_LABELS
from smartsite_ia.collection_assets import Asset, fetch_asset
from smartsite_ia.collection_html import write_collection_html
from smartsite_ia.curation import content_groups, digest, read_document, require_text, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.prediction import decode_photo
from smartsite_ia.review import MAX_FILE_BYTES, read_local
from smartsite_ia.similarity import candidate_groups, find_similar_pairs, fingerprint

CLASSES = COLLECTION_LABELS


def safe_link(value: object) -> str:
    """Les liens des sources s'affichent dans le rapport ; aucun script n'y est accepté."""
    text = require_text(value)
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Source and license links must be public HTTPS URLs")
    return text


def load_collection(path: Path) -> tuple[dict[str, Any], str]:
    document, sha = read_document(path)
    records, sources = document.get("records"), document.get("sources")
    if not isinstance(records, list) or not 1 <= len(records) <= 200:
        raise ValueError("A collection must contain between 1 and 200 photos")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Collection sources are required")
    if any(not isinstance(source, dict) for source in sources.values()):
        raise ValueError("Invalid collection source metadata")
    excluded = document.get("excluded_candidates", [])
    if not isinstance(excluded, list) or len(excluded) > 200:
        raise ValueError("Invalid excluded candidate list")
    for candidate in excluded:
        if not isinstance(candidate, dict):
            raise ValueError("Invalid excluded candidate")
        require_text(candidate.get("id"))
        require_text(candidate.get("reason"))
    ids = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Invalid collection record")
        sample_id = require_text(row.get("id"))
        if re.fullmatch(r"[a-z0-9_]{1,80}", sample_id) is None or sample_id in ids:
            raise ValueError("Unsafe or duplicate sample ID")
        ids.add(sample_id)
        if row.get("source_id") not in sources:
            raise ValueError("Unknown collection source")
        if row.get("source_split") not in {"train", "val", "unassigned"}:
            raise ValueError("Reserved test photos are not accepted in this working collection")
        if not isinstance(row.get("asset"), dict):
            raise ValueError("Missing image asset")
        Asset.parse(row["asset"])
        if row.get("image_policy", "single_frame") not in {"single_frame", "mpo_primary_of_two"}:
            raise ValueError("Unknown image frame policy")
        credit = row.get("credit")
        if not isinstance(credit, dict):
            raise ValueError("Missing image attribution")
        for key in ("title", "author", "license"):
            require_text(credit.get(key))
        for key in ("source_url", "license_url"):
            safe_link(credit.get(key))
        if row.get("decision") not in {"review", "quarantine"}:
            raise ValueError("Collection records must remain pending review or quarantined")
        require_text(row.get("note"))
        # Les catégories de la source restent distinctes des diagnostics terrain.
        boxes_for(row, sources[row["source_id"]])
    content_groups(ids, document.get("groups", []))
    require_text(document.get("limitations"))
    return document, sha


def boxes_for(row: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    """Lire les boîtes YOLO source ; une photo sans annotation reste à annoter."""
    raw = row.get("source_label_text")
    if raw is None:
        return []
    if not isinstance(raw, str) or len(raw) > 65536:
        raise ValueError("Source annotations exceed the text limit")
    if digest(raw.encode()) != row.get("source_label_sha256"):
        raise ValueError("Source annotation SHA-256 mismatch")
    if source.get("coordinate_frame") != "display_oriented":
        raise ValueError("Source box coordinate frame has not been reviewed")
    names, mapping = source.get("class_names"), source.get("class_map")
    if not isinstance(names, list) or not isinstance(mapping, dict):
        raise ValueError("Missing source class mapping")
    result = []
    lines = raw.splitlines()
    if len(lines) > 200:
        raise ValueError("Too many source annotations")
    for line in lines:
        values = line.split()
        if len(values) != 5 or not values[0].isdigit():
            raise ValueError("Expected a YOLO class and four coordinates")
        class_id = int(values[0])
        if not 0 <= class_id < len(names):
            raise ValueError("Unknown source class")
        x, y, width, height = (float(v) for v in values[1:])
        box = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
        if not all(math.isfinite(v) for v in box) or width <= 0 or height <= 0:
            raise ValueError("Nonfinite or empty source box")
        # Les arrondis YOLO à six décimales peuvent dépasser d'un demi-millionième.
        if min(box) < -0.000001 or max(box) > 1.000001:
            raise ValueError("Source box is outside the image")
        category = mapping.get(str(class_id))
        if category is None:
            continue
        if category not in CLASSES:
            raise ValueError("Unsupported semantic mapping")
        result.append(
            {
                "class": category,
                "source_class_id": class_id,
                "xyxy_normalized": [min(1.0, max(0.0, v)) for v in box],
            }
        )
    return result


def fetch_collection(selection: Path, cache: Path) -> dict[str, int]:
    document, _ = load_collection(selection)
    transferred, reused = 0, 0
    for row in document["records"]:
        asset = Asset.parse(row["asset"])
        exists = (cache / asset.sha256).exists()
        if not exists and transferred:
            time.sleep(3)
        fetch_asset(asset, cache)
        reused += int(exists)
        transferred += int(not exists)
    return {"downloaded": transferred, "reused": reused}


def prepare_collection(selection: Path, cache: Path, output: Path) -> dict[str, Any]:
    """Créer un lot de travail atomique : originaux, photos orientées, boîtes et revue."""
    document, sha = load_collection(selection)
    records, fingerprints = [], []
    with staged_output(output, [cache, selection.parent]) as stage:
        (stage / "originals").mkdir()
        (stage / "images").mkdir()
        (stage / "previews").mkdir()
        (stage / "source_labels").mkdir()
        for row in document["records"]:
            asset = Asset.parse(row["asset"])
            raw = read_local(cache, asset.sha256, MAX_FILE_BYTES)
            if len(raw) != asset.size or digest(raw) != asset.sha256:
                raise ValueError("Photo differs from the pinned manifest")
            image = decode_photo(
                raw, allow_primary_mpo=row.get("image_policy") == "mpo_primary_of_two"
            )
            with Image.open(io.BytesIO(raw)) as original:
                orientation = original.getexif().get(274, 1)
                original_size = list(original.size)
            sample_id = row["id"]
            pixel_sha = digest(f"{image.size}:RGB:".encode() + image.tobytes())
            fingerprints.append(fingerprint(sample_id, image, pixel_sha))
            boxes = boxes_for(row, document["sources"][row["source_id"]])
            (stage / "originals" / f"{sample_id}.image").write_bytes(raw)
            image.save(stage / "images" / f"{sample_id}.jpg", quality=95)
            if row.get("source_label_text") is not None:
                (stage / "source_labels" / f"{sample_id}.txt").write_bytes(
                    row["source_label_text"].encode()
                )
            preview = image.copy()
            preview.thumbnail((960, 960))
            draw = ImageDraw.Draw(preview)
            for box in boxes:
                x1, y1, x2, y2 = box["xyxy_normalized"]
                color = COLLECTION_COLORS[box["class"]]
                draw.rectangle(
                    (
                        x1 * preview.width,
                        y1 * preview.height,
                        x2 * preview.width,
                        y2 * preview.height,
                    ),
                    outline=color,
                    width=3,
                )
            preview.save(stage / "previews" / f"{sample_id}.jpg", quality=90)
            records.append(
                {
                    **row,
                    "width": image.width,
                    "height": image.height,
                    "original_size": original_size,
                    "exif_orientation": orientation,
                    "pixel_sha256": pixel_sha,
                    "boxes": boxes,
                    "annotation_status": "source_boxes_to_review" if boxes else "to_annotate",
                    "partition": "unassigned",
                    "approved_for_training": False,
                }
            )
        pairs = find_similar_pairs(fingerprints)
        # Regrouper prudemment les ressemblances ne prouve pas leur origine
        # Aucun de ces groupes n'est séparé aléatoirement en train/test ici
        groups = content_groups(
            {r["id"] for r in records}, document.get("groups", []) + candidate_groups(pairs)
        )
        for row in records:
            row["group"] = groups[row["id"]]
        split_conflicts = []
        for group in sorted(set(groups.values())):
            members = [r for r in records if r["group"] == group]
            splits = {r["source_split"] for r in members}
            if {"train", "val"} <= splits:
                split_conflicts.append({"group": group, "ids": [r["id"] for r in members]})
        summary = {
            "images": len(records),
            "sources": dict(Counter(r["source_id"] for r in records)),
            "with_source_boxes": sum(bool(r["boxes"]) for r in records),
            "to_annotate": sum(not r["boxes"] for r in records),
            "quarantined": sum(r["decision"] == "quarantine" for r in records),
            "orientation_corrected": sum(r["exif_orientation"] != 1 for r in records),
            "groups": len(set(groups.values())),
            "approved_for_training": 0,
            "excluded_before_preparation": len(document.get("excluded_candidates", [])),
        }
        report = {
            "schema_version": 1,
            "selection_sha256": sha,
            "sources": document["sources"],
            "limitations": document["limitations"],
            "records": records,
            "similar_pairs": [asdict(p) for p in pairs],
            "source_split_conflicts": split_conflicts,
            "excluded_candidates": document.get("excluded_candidates", []),
            "summary": summary,
        }
        write_json(stage / "report.json", report)
        write_json(stage / "selection.json", document)
        write_json(
            stage / "review_queue.json",
            {
                "schema_version": 1,
                "selection_sha256": sha,
                "instructions": "Revoir zones et confusions ; jamais de négatif implicite.",
                "records": [
                    {
                        "id": r["id"],
                        "group": r["group"],
                        "reviewer": None,
                        "decision": "pending",
                        "exhaustive_annotation": False,
                        "annotations": r["boxes"],
                        "note": "",
                    }
                    for r in records
                ],
            },
        )
        write_collection_html(stage, report, CLASSES)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "prepare"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare" and args.output is None:
        parser.error("prepare requires --output")
    try:
        result = (
            fetch_collection(args.selection, args.cache)
            if args.action == "fetch"
            else prepare_collection(args.selection, args.cache, args.output)
        )
    except (OSError, ValueError, zlib.error) as exc:
        print(f"Collection interrompue : {type(exc).__name__}. {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
