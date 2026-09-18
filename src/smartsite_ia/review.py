"""Contrôler un import DamSegment et préparer une revue locale reproductible."""

import hashlib
import json
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

from smartsite_ia.annotations import unique_object
from smartsite_ia.importer import MASK_COLORS, convert_sample, decode_image, write_json
from smartsite_ia.review_html import write_review_page
from smartsite_ia.similarity import candidate_groups, find_similar_pairs, fingerprint

SAMPLE_ID = re.compile(r"(easy|medium|hard)_([0-9]{4})")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class Sample:
    sample_id: str
    filename: str
    record: dict[str, Any]


def read_local(root: Path, relative: str, limit: int) -> bytes:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its root")
    if any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents):
        raise ValueError("Linked dataset files are not accepted")
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Dataset file is missing or exceeds its size limit")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Dataset file exceeded its size limit while reading")
    return content


def load_samples(root: Path) -> tuple[list[Sample], str]:
    raw = read_local(root, "manifest.json", MAX_JSON_BYTES)
    manifest = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Expected a version 1 import manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 5000:
        raise ValueError("Expected between 1 and 5000 manifest records")
    samples = []
    seen = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError("Invalid manifest record")
        sample_id = record["id"]
        match = SAMPLE_ID.fullmatch(sample_id)
        if match is None or sample_id in seen:
            raise ValueError("Invalid or duplicate sample ID")
        difficulty, number = match.groups()
        filename = f"{difficulty[0].upper()} ({int(number)}).jpg"
        expected = {
            "image": f"images/{sample_id}.jpg",
            "mask": f"masks/{sample_id}.png",
            "difficulty": difficulty.title(),
            "source_path": f"Damage Segmentaion/{difficulty.title()}/Images/{filename}",
        }
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Manifest paths or identity differ from the import layout")
        if record.get("scene_group") is not None or record.get("split") is not None:
            raise ValueError("Review expects an unsplit import, not edited scene groups or splits")
        samples.append(Sample(sample_id, filename, record))
        seen.add(sample_id)
    return sorted(samples, key=lambda s: s.sample_id), hashlib.sha256(raw).hexdigest()


def sample_parts(root: Path, sample_id: str) -> list[bytes]:
    return [
        read_local(root, path, MAX_FILE_BYTES)
        for path in (
            f"images/{sample_id}.jpg",
            f"masks/{sample_id}.png",
            f"source_annotations/{sample_id}.json",
            f"source_annotations/{sample_id}.txt",
        )
    ]


def inspect_sample(
    root: Path, sample: Sample
) -> tuple[list[bytes], list[dict[str, Any]], dict[str, Any]]:
    parts = sample_parts(root, sample.sample_id)
    annotations, evidence = convert_sample(parts[0], parts[1], parts[2], parts[3], sample.filename)
    if any(sample.record.get(k) != v for k, v in evidence.items()):
        raise ValueError(f"Files or metrics differ from manifest for {sample.sample_id}")
    if type(sample.record.get("annotation_count")) is not int or (
        sample.record["annotation_count"] != len(annotations)
    ):
        raise ValueError("Annotation count differs from manifest")
    return parts, annotations, evidence


def rasterize(
    annotations: list[dict[str, Any]],
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    rendered = np.zeros((640, 640, 3), dtype=np.uint8)
    class_masks = [np.zeros((640, 640), dtype=bool) for _ in MASK_COLORS]
    inclusive = Image.new("RGB", (640, 640))
    draw = ImageDraw.Draw(inclusive)
    for annotation in annotations:
        polygons = annotation["segmentation"]
        color = MASK_COLORS[annotation["source_category_id"]]
        rle = coco_mask.merge(coco_mask.frPyObjects(polygons, 640, 640))
        mask = coco_mask.decode(rle).astype(bool)
        rendered[mask] = color
        class_masks[annotation["source_category_id"]] |= mask
        for polygon in polygons:
            draw.polygon(list(zip(polygon[::2], polygon[1::2], strict=True)), fill=color)
    overlap = int(np.count_nonzero(class_masks[0] & class_masks[1]))
    areas = [int(mask.sum()) for mask in class_masks]
    return (
        Image.fromarray(rendered),
        inclusive,
        {
            "class_union_pixels": {str(i): area for i, area in enumerate(areas)},
            "cross_class_overlap_pixels": overlap,
            "overlap_fraction_of_smaller_class": overlap / min(areas) if min(areas) else None,
        },
    )


def compare_masks(source: Image.Image, coco: Image.Image, inclusive: Image.Image) -> dict[str, Any]:
    original, rendered = np.asarray(source), np.asarray(coco)
    disagreements = np.any(original != rendered, axis=2)
    # Un pixel près du bord peut changer selon la façon de dessiner le polygone.
    boundaries = np.zeros((640, 640), dtype=bool)
    for array in (original, rendered):
        labels = (array[:, :, 0] != 0).astype(np.uint8) + 2 * (array[:, :, 2] != 0)
        padded = np.pad(labels, 1, mode="edge")
        for y in range(3):
            for x in range(3):
                boundaries |= labels != padded[y : y + 640, x : x + 640]
    return {
        "coco_disagreement_pixels": int(disagreements.sum()),
        "pillow_disagreement_pixels": int(np.any(original != np.asarray(inclusive), axis=2).sum()),
        "disagreement_outside_1px_boundaries": int(np.count_nonzero(disagreements & ~boundaries)),
    }


def save_panels(
    destination: Path,
    sample_id: str,
    photo_bytes: bytes,
    image: Image.Image,
    mask: Image.Image,
    coco: Image.Image,
) -> None:
    folder = destination / "samples" / sample_id
    folder.mkdir(parents=True)
    (folder / "photo.jpg").write_bytes(photo_bytes)
    mask.save(folder / "source.png")
    coco.save(folder / "coco.png")
    differences = np.any(np.asarray(mask) != np.asarray(coco), axis=2)
    overlay = np.asarray(image).copy()
    overlay[differences] = (255, 0, 255)
    Image.fromarray(overlay).save(folder / "differences.png")


def review_corpus(
    root: Path, destination: Path, max_distance: int = 8, max_error: float = 20.0
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Review destination already exists; choose a new output directory")
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Review output must be outside the source corpus")
    # On valide les paramètres avant de lire toutes les images.
    find_similar_pairs([], max_distance, max_error)
    samples, manifest_hash = load_samples(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".smartsite-review-", dir=destination.parent) as work:
        stage = Path(work) / "review"
        (stage / "thumbnails").mkdir(parents=True)
        records, fingerprints = [], []
        class_counts: Counter[int] = Counter()
        for sample in samples:
            parts, annotations, evidence = inspect_sample(root, sample)
            image, mask = decode_image(parts[0], "JPEG"), decode_image(parts[1], "PNG")
            coco, inclusive, overlaps = rasterize(annotations)
            measurements = compare_masks(mask, coco, inclusive)
            flags = []
            if not annotations:
                flags.append("empty_annotations")
            if any(
                v is not None and v < 0.85 for v in evidence["mask_iou_by_source_class"].values()
            ):
                flags.append("mask_iou_below_review_threshold")
            if measurements["disagreement_outside_1px_boundaries"]:
                flags.append("mask_difference_away_from_boundary")
            if (overlaps["overlap_fraction_of_smaller_class"] or 0) >= 0.9:
                flags.append("source_classes_almost_fully_overlap")
            record = {
                "id": sample.sample_id,
                "difficulty": sample.record["difficulty"],
                "annotation_count": len(annotations),
                "flags": flags,
                "mask_iou_by_source_class": evidence["mask_iou_by_source_class"],
                **measurements,
                **overlaps,
                "review_status": "pending",
                "scene_group": None,
                "split": None,
            }
            records.append(record)
            fingerprints.append(fingerprint(sample.sample_id, image, evidence["pixel_sha256"]))
            image.resize((160, 160), Image.Resampling.LANCZOS).save(
                stage / "thumbnails" / f"{sample.sample_id}.jpg", quality=85
            )
            class_counts.update(a["source_category_id"] for a in annotations)
        pairs = find_similar_pairs(fingerprints, max_distance, max_error)
        selected = select_panels(records)
        selected.update(sample_id for pair in pairs for sample_id in (pair.left, pair.right))
        for sample in samples:
            if sample.sample_id in selected:
                parts, annotations, _ = inspect_sample(root, sample)
                coco, _, _ = rasterize(annotations)
                save_panels(
                    stage,
                    sample.sample_id,
                    parts[0],
                    decode_image(parts[0], "JPEG"),
                    decode_image(parts[1], "PNG"),
                    coco,
                )
        report = {
            "schema_version": 1,
            "manifest_sha256": manifest_hash,
            "images": len(records),
            "annotations": sum(class_counts.values()),
            "source_class_counts": dict(class_counts),
            "parameters": {
                "hash": "dhash_horizontal_vertical_128",
                "thumbnail_size": 32,
                "transforms": 8,
                "max_hamming_distance": max_distance,
                "max_rgb_mean_absolute_error": max_error,
                "review_iou_threshold": 0.85,
                "review_class_overlap_threshold": 0.9,
            },
            "flag_counts": dict(Counter(flag for r in records for flag in r["flags"])),
            "mask_totals": {
                key: sum(r[key] for r in records)
                for key in (
                    "coco_disagreement_pixels",
                    "pillow_disagreement_pixels",
                    "disagreement_outside_1px_boundaries",
                    "cross_class_overlap_pixels",
                )
            },
            "similar_pairs": [asdict(pair) for pair in pairs],
            "candidate_groups": candidate_groups(pairs),
            "detailed_samples": sorted(selected),
            "records": records,
            "approved_for_training": False,
            "limits": [
                "Similarities are candidates for review, not proof of identical scenes",
                "Hash search can miss crops, adjacent patches, viewpoint and lighting changes",
                "Boundary proximity does not certify annotation quality or semantic correctness",
                "Source class IDs retained; no automatic approval, relabeling, deletion or split",
                "Photos and masks shown are source annotations, not model predictions",
            ],
        }
        write_json(stage / "review.json", report)
        write_json(
            stage / "review-decisions.template.json",
            {
                "schema_version": 1,
                "manifest_sha256": manifest_hash,
                "instructions": "Record evidence and reviewer; pending is not approved",
                "samples": [
                    {"id": r["id"], "decision": "pending", "reviewer": None, "reason": None}
                    for r in records
                    if r["flags"]
                ],
                "pairs": [
                    {
                        "left": p.left,
                        "right": p.right,
                        "decision": "pending",
                        "reviewer": None,
                        "reason": None,
                    }
                    for p in pairs
                ],
            },
        )
        write_review_page(stage, report)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Review destination appeared during preparation")
        stage.rename(destination)
    return report


def select_panels(records: list[dict[str, Any]]) -> set[str]:
    selected = {r["id"] for r in records if "empty_annotations" in r["flags"]}
    selected.update(r["id"] for r in records if "source_classes_almost_fully_overlap" in r["flags"])
    for difficulty in ("Easy", "Medium", "Hard"):
        subset = [r for r in records if r["difficulty"] == difficulty]
        for category in ("0", "1"):
            present = [r for r in subset if r["mask_iou_by_source_class"][category] is not None]
            ordered = sorted(
                present, key=lambda r: (r["mask_iou_by_source_class"][category], r["id"])
            )
            selected.update(r["id"] for r in ordered[:3])
        selected.update(r["id"] for r in sorted(subset, key=lambda r: r["id"])[:2])
    return selected
