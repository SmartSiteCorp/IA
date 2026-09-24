"""Appliquer une revue visuelle traçable et exporter un petit corpus de boîtes."""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from smartsite_ia.categories import COLLECTION_LABELS, DEFECT_FAMILIES
from smartsite_ia.collection import load_collection, prepare_collection
from smartsite_ia.collection_review_html import write_review_gallery
from smartsite_ia.curation import digest, read_document, require_text, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.review import MAX_JSON_BYTES, read_local


def rectangle(value: object) -> list[float]:
    """On garde un seul repère : la photo remise à l'endroit, entre zéro et un."""
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError("Expected four finite box coordinates")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Empty or out-of-bounds review box")
    return [float(v) for v in value]


def source_audit(row: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    """Chaque ancien cadre doit être conservé, remplacé ou retiré avec une raison."""
    boxes, removed = row.get("boxes"), row.get("removed_source_boxes")
    if not isinstance(boxes, list) or len(boxes) > 200 or not isinstance(removed, list):
        raise ValueError("Invalid reviewed box list")
    originals = source["boxes"]
    links: dict[int, list[int]] = {}
    seen = set()
    for i, box in enumerate(boxes):
        if not isinstance(box, dict) or set(box) != {"class", "xyxy_normalized", "source_indices"}:
            raise ValueError("Unknown or missing reviewed box fields")
        if not isinstance(box["class"], str) or box["class"] not in COLLECTION_LABELS:
            raise ValueError("Unknown reviewed class")
        coords = rectangle(box["xyxy_normalized"])
        identity = (box["class"], *coords)
        if identity in seen:
            raise ValueError("Duplicate reviewed box")
        seen.add(identity)
        refs = box["source_indices"]
        if (
            not isinstance(refs, list)
            or any(type(n) is not int or not 0 <= n < len(originals) for n in refs)
            or len(set(refs)) != len(refs)
        ):
            raise ValueError("Invalid source box reference")
        for index in refs:
            links.setdefault(index, []).append(i)
    rejected = {}
    for item in removed:
        if not isinstance(item, dict) or set(item) != {"index", "reason"}:
            raise ValueError("Invalid removed source box")
        index = item["index"]
        if type(index) is not int or index in rejected or not 0 <= index < len(originals):
            raise ValueError("Invalid removed source box index")
        rejected[index] = require_text(item["reason"])
    if set(links) & set(rejected) or set(links) | set(rejected) != set(range(len(originals))):
        raise ValueError("Every source box needs exactly one review outcome")
    audit = []
    for i, original in enumerate(originals):
        targets = links.get(i, [])
        kept = len(targets) == 1 and all(
            boxes[targets[0]][key] == original[key] for key in ("class", "xyxy_normalized")
        )
        audit.append(
            {
                "source_index": i,
                "outcome": "removed" if i in rejected else "kept" if kept else "revised",
                "reviewed_indices": targets,
                "reason": rejected.get(i, row["note"]),
            }
        )
    return audit


def validate_review(document: dict[str, Any], source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Une décision explicite par photo ; jamais de validation humaine inventée."""
    expected = {
        "schema_version",
        "selection_sha256",
        "reviewer",
        "human_validation",
        "scope",
        "coordinate_frame",
        "protocol",
        "group_partitions",
        "records",
        "negative_crops",
    }
    if set(document) != expected or document["selection_sha256"] != source["selection_sha256"]:
        raise ValueError("Review differs from the pinned collection or schema")
    if (
        document["reviewer"] != "assistant_visual_review"
        or document["human_validation"] != "pending"
        or document["scope"] != list(COLLECTION_LABELS)
        or document["coordinate_frame"] != "display_oriented"
    ):
        raise ValueError("Unsupported review provenance or scope")
    protocol = document["protocol"]
    if not isinstance(protocol, dict) or protocol.get("purpose") != "pilot_training_only":
        raise ValueError("This review only supports pilot training")
    for field in ("annotation_unit", "coverage", "limits"):
        require_text(protocol.get(field))
    rules = protocol.get("class_rules")
    if not isinstance(rules, dict) or set(rules) != set(COLLECTION_LABELS):
        raise ValueError("Missing visual annotation conventions")
    for rule in rules.values():
        require_text(rule)
    originals = {r["id"]: r for r in source["records"]}
    partitions = document["group_partitions"]
    if (
        not isinstance(partitions, dict)
        or set(partitions) != {r["group"] for r in originals.values()}
        or any(v not in ("train", "validation") for v in partitions.values())
    ):
        raise ValueError("Assign whole source groups to train/validation, never final test")
    records = document["records"]
    if not isinstance(records, list) or len(records) != len(originals):
        raise ValueError("Review must cover every source photo")
    by_id = {}
    for row in records:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "decision",
            "note",
            "reviewed_scope",
            "boxes",
            "removed_source_boxes",
        }:
            raise ValueError("Unknown or missing review fields")
        sample_id = row["id"]
        if not isinstance(sample_id, str) or sample_id not in originals or sample_id in by_id:
            raise ValueError("Unknown or duplicate reviewed photo")
        require_text(row["note"])
        if row["decision"] not in ("candidate", "negative", "excluded"):
            raise ValueError("Unknown review decision")
        candidate = row["decision"] == "candidate"
        eligible = row["decision"] != "excluded"
        # Une photo vide ne devient du fond qu'après une revue explicite des trois cibles.
        # Les photos ambiguës restent exclues, même si aucun rectangle n'a été retenu.
        if (
            row["reviewed_scope"] is not eligible
            or candidate
            and not row["boxes"]
            or row["decision"] == "negative"
            and row["boxes"]
        ):
            raise ValueError("Review scope and boxes must match the explicit decision")
        source_audit(row, originals[sample_id])
        by_id[sample_id] = row
    validate_crops(document, originals, by_id)
    return by_id


def validate_crops(
    document: dict[str, Any], originals: dict[str, Any], reviewed: dict[str, Any]
) -> None:
    """Les négatifs sont des recadrages revus, pas des zones non annotées par défaut."""
    crops = document["negative_crops"]
    if not isinstance(crops, list) or len(crops) > 100:
        raise ValueError("Invalid negative crop list")
    seen = set(originals)
    for crop in crops:
        if not isinstance(crop, dict) or set(crop) != {
            "id",
            "parent_id",
            "crop_xyxy_normalized",
            "note",
            "reviewed_scope",
        }:
            raise ValueError("Unknown or missing crop fields")
        name, parent = crop["id"], crop["parent_id"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9_]{1,80}", name) is None
            or name in seen
            or not isinstance(parent, str)
            or parent not in originals
        ):
            raise ValueError("Invalid crop identity or parent")
        seen.add(name)
        require_text(crop["note"])
        if (
            crop["reviewed_scope"] is not True
            or reviewed[parent]["decision"] != "candidate"
            or document["group_partitions"][originals[parent]["group"]] != "train"
        ):
            raise ValueError(
                "Negative crops require an eligible training parent and explicit review"
            )
        x1, y1, x2, y2 = rectangle(crop["crop_xyxy_normalized"])
        for box in reviewed[parent]["boxes"]:
            a, b, c, d = box["xyxy_normalized"]
            if max(x1, a) < min(x2, c) and max(y1, b) < min(y2, d):
                raise ValueError("Negative crop intersects a reviewed target")


def box_coco_document(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Un seul calcul des rectangles sert à l'export et au contrôle avant apprentissage."""
    categories = [
        {"id": i, "name": name, "supercategory": DEFECT_FAMILIES[name]}
        for i, name in enumerate(COLLECTION_LABELS, 1)
    ]
    ids = {c["name"]: c["id"] for c in categories}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for row in records:
        image_id = len(images) + 1
        width, height = row["width"], row["height"]
        images.append(
            {
                "id": image_id,
                "file_name": f"{row['id']}.jpg",
                "width": width,
                "height": height,
                "group": row["group"],
                "credit": row["credit"],
            }
        )
        for box in row["boxes"]:
            x1, y1, x2, y2 = box["xyxy_normalized"]
            bbox = [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": ids[box["class"]],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
    return {
        "info": {"description": "Assistant-reviewed pilot boxes; no final test or human approval"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def export_coco(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Exporter des rectangles uniquement. L'aire est celle de la boîte, pas du défaut."""
    summaries = {}
    for split in ("train", "validation"):
        selected = [r for r in records if r["partition"] == split]
        document = box_coco_document(selected)
        for row in selected:
            # Le fichier est copié, jamais lié au cache : les originaux restent intacts.
            (root / split / f"{row['id']}.jpg").write_bytes((root / row["image"]).read_bytes())
        counts = Counter(b["class"] for r in selected for b in r["boxes"])
        if set(counts) != set(COLLECTION_LABELS):
            raise ValueError("Each pilot partition must contain every target class")
        summaries[split] = {
            "images": len(document["images"]),
            "groups": len({r["group"] for r in selected}),
            "boxes_by_class": dict(counts),
            "negative_crops": sum(r["is_crop"] for r in selected),
            "negative_photos": sum(r["decision"] == "negative" for r in selected),
            "groups_by_class": {
                name: len(
                    {r["group"] for r in selected if any(b["class"] == name for b in r["boxes"])}
                )
                for name in COLLECTION_LABELS
            },
        }
        write_json(root / split / "_annotations.coco.json", document)
    return summaries


def apply_review(selection: Path, cache: Path, review: Path, output: Path) -> dict[str, Any]:
    """Rejouer depuis les originaux vérifiés ; publier seulement le résultat complet."""
    _, selection_sha = load_collection(selection)
    document, review_sha = read_document(review)
    if document.get("selection_sha256") != selection_sha:
        raise ValueError("Review selection SHA-256 mismatch")
    with staged_output(output, [cache, selection.parent, review.parent]) as stage:
        # On réutilise les contrôles d'import, d'orientation et de regroupement existants.
        prepare_collection(selection, cache, stage / "source")
        source, _ = read_document(stage / "source/report.json")
        reviewed = validate_review(document, source)
        for folder in ("images", "previews", "train", "validation"):
            (stage / folder).mkdir()
        records = []
        for original in source["records"]:
            row = reviewed[original["id"]]
            records.append(
                {
                    **original,
                    **row,
                    "source_boxes": original["boxes"],
                    "source_audit": source_audit(row, original),
                    "is_crop": False,
                    "image": f"source/images/{row['id']}.jpg",
                    "partition": document["group_partitions"][original["group"]]
                    if row["decision"] != "excluded"
                    else "excluded",
                    "approved_for_training": False,
                    "eligible_for_pilot": row["decision"] != "excluded",
                    "human_validation": "pending",
                }
            )
        originals = {r["id"]: r for r in records}
        for crop in document["negative_crops"]:
            parent = originals[crop["parent_id"]]
            with Image.open(stage / parent["image"]) as photo:
                x1, y1, x2, y2 = crop["crop_xyxy_normalized"]
                bounds = [
                    round(x1 * photo.width),
                    round(y1 * photo.height),
                    round(x2 * photo.width),
                    round(y2 * photo.height),
                ]
                cut = photo.crop(tuple(bounds))
                if min(cut.size) < 32:
                    raise ValueError("Negative crop is too small after pixel rounding")
                relative = f"images/{crop['id']}.jpg"
                cut.save(stage / relative, quality=95)
                records.append(
                    {
                        **crop,
                        "width": cut.width,
                        "height": cut.height,
                        "crop_xyxy_pixels": bounds,
                        "group": parent["group"],
                        "credit": parent["credit"],
                        "is_crop": True,
                        "image": relative,
                        "boxes": [],
                        "source_boxes": [],
                        "source_audit": [],
                        "partition": "train",
                        "decision": "candidate",
                        "eligible_for_pilot": True,
                        "approved_for_training": False,
                        "human_validation": "pending",
                    }
                )
        partitions = export_coco(stage, records)
        outcomes = Counter(a["outcome"] for r in records for a in r["source_audit"])
        summary = {
            "reviewed_photos": len(source["records"]),
            "candidate_photos": sum(
                r["decision"] == "candidate" and not r["is_crop"] for r in records
            ),
            "excluded_photos": sum(r["decision"] == "excluded" for r in records),
            "negative_crops": len(document["negative_crops"]),
            "negative_photos": sum(r["decision"] == "negative" for r in records),
            "source_box_outcomes": dict(outcomes),
            "new_boxes": sum(not b["source_indices"] for r in records for b in r["boxes"]),
            "partitions": partitions,
        }
        report = {
            "schema_version": 1,
            "selection_sha256": selection_sha,
            "review_sha256": review_sha,
            "reviewer": document["reviewer"],
            "human_validation": "pending",
            "purpose": "pilot_training_only",
            "protocol": document["protocol"],
            "final_test_created": False,
            "summary": summary,
            "records": records,
        }
        for row in records:
            row["image_sha256"] = digest((stage / row["image"]).read_bytes())
        # L'empreinte désigne le fichier exact, même si son indentation diffère.
        review_raw = read_local(review.parent, review.name, MAX_JSON_BYTES)
        if digest(review_raw) != review_sha:
            raise ValueError("Review changed during preparation")
        (stage / "review.json").write_bytes(review_raw)
        write_json(stage / "report.json", report)
        write_review_gallery(stage, report)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("selection", "cache", "review", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = apply_review(args.selection, args.cache, args.review, args.output)
    except (OSError, ValueError) as exc:
        print(f"Revue interrompue : {type(exc).__name__}. {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
