"""Transactional conversion to an unsplit COCO corpus, retaining source evidence."""

import hashlib
import io
import json
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from smartsite_ia import __version__
from smartsite_ia.annotations import parse_annotations
from smartsite_ia.archive import checked_members, read_member
from smartsite_ia.source import DAMSEGMENT, PROVENANCE, Source, verify_archive

IMAGE_PATH = re.compile(r"Damage Segmentaion/(Easy|Medium|Hard)/Images/([EMH]) \(([1-9]\d*)\)\.jpg")
MASK_COLORS = ((255, 0, 0), (0, 0, 255))


def decode_image(content: bytes, expected_format: str) -> Image.Image:
    with Image.open(io.BytesIO(content), formats=[expected_format]) as image:
        if image.size != (640, 640) or image.mode != "RGB":
            raise ValueError("Expected an RGB 640x640 image in DamSegment v1")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Image orientation would invalidate annotation coordinates")
        image.load()
        return image.copy()


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def convert_sample(
    image_bytes: bytes, mask_bytes: bytes, annotations: bytes, yolo: bytes, filename: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image = decode_image(image_bytes, "JPEG")
    source_mask = np.asarray(decode_image(mask_bytes, "PNG"))
    instances = parse_annotations(annotations, yolo, filename, image.size)
    allowed = np.all(source_mask == 0, axis=2)
    for color in MASK_COLORS:
        allowed |= np.all(source_mask == color, axis=2)
    if not allowed.all():
        raise ValueError("PNG mask contains colors outside the observed v1 palette")
    rendered = np.zeros_like(source_mask)
    converted = []
    for instance in instances:
        rle = coco_mask.merge(coco_mask.frPyObjects(instance.polygons, image.height, image.width))
        area = float(coco_mask.area(rle))
        if area <= 0:
            raise ValueError("Polygon produces an empty COCO mask")
        rendered[coco_mask.decode(rle).astype(bool)] = MASK_COLORS[instance.source_class]
        converted.append(
            {
                "category_id": instance.source_class + 1,
                "source_category_id": instance.source_class,
                "segmentation": instance.polygons,
                "bbox": coco_mask.toBbox(rle).tolist(),
                "source_bbox": instance.source_bbox,
                "area": area,
                "iscrowd": 0,
            }
        )
    differences = int(np.count_nonzero(np.any(source_mask != rendered, axis=2)))
    intersections = {}
    for category, color in enumerate(MASK_COLORS):
        original = np.all(source_mask == color, axis=2)
        generated = np.all(rendered == color, axis=2)
        union = int(np.count_nonzero(original | generated))
        intersections[str(category)] = (
            float(np.count_nonzero(original & generated) / union) if union else None
        )
    return converted, {
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "annotations_sha256": hashlib.sha256(annotations).hexdigest(),
        "yolo_sha256": hashlib.sha256(yolo).hexdigest(),
        "mask_disagreement_pixels": differences,
        "mask_iou_by_source_class": intersections,
    }


def import_archive(
    archive_path: Path, destination: Path, source: Source = DAMSEGMENT
) -> dict[str, Any]:
    """Validate all samples or leave no published output; never invent scene groups/splits."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Import destination already exists; choose a new output directory")
    verify_archive(archive_path, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        ZipFile(archive_path) as archive,
        tempfile.TemporaryDirectory(prefix=".smartsite-import-", dir=destination.parent) as work,
    ):
        members = checked_members(archive)
        samples = sorted(name for name in members if IMAGE_PATH.fullmatch(name))
        if len(samples) != source.expected_images:
            raise ValueError(f"Expected {source.expected_images} images, found {len(samples)}")
        stage = Path(work) / "corpus"
        for folder in ("images", "masks", "source_annotations"):
            (stage / folder).mkdir(parents=True, exist_ok=True)
        coco: dict[str, Any] = {
            "info": {"description": "Unsplit DamSegment audit corpus; not approved for training"},
            "licenses": [{"id": 1, "name": "CC BY 4.0", "url": PROVENANCE["license_url"]}],
            "categories": [
                {"id": i + 1, "name": f"source_class_{i}", "supercategory": "surface_damage"}
                for i in range(2)
            ],
            "images": [],
            "annotations": [],
        }
        records = []
        consumed: set[str] = set()
        class_counts: Counter[int] = Counter()
        difficulties: Counter[str] = Counter()
        duplicates: dict[str, list[str]] = defaultdict(list)
        for image_id, name in enumerate(samples, 1):
            match = IMAGE_PATH.fullmatch(name)
            assert match is not None
            difficulty, prefix, index = match.groups()
            if prefix != difficulty[0]:
                raise ValueError(f"Image prefix and difficulty differ: {name}")
            stem = f"{prefix} ({index})"
            base = f"Damage Segmentaion/{difficulty}/Labels"
            paths = [
                name,
                f"{base}/Mask/{stem}_mask.png",
                f"{base}/Pascal VOC/{stem}.json",
                f"{base}/Yolo/{stem}.txt",
            ]
            if any(path not in members for path in paths):
                raise ValueError(f"Missing companion file for {name}")
            parts = [read_member(archive, members[path]) for path in paths]
            try:
                annotations, evidence = convert_sample(
                    parts[0], parts[1], parts[2], parts[3], f"{stem}.jpg"
                )
            except (ValueError, OSError) as error:
                raise ValueError(f"Invalid sample {name}: {error}") from error
            output_id = f"{difficulty.lower()}_{int(index):04d}"
            output_image = f"images/{output_id}.jpg"
            output_mask = f"masks/{output_id}.png"
            for content, target in zip(
                parts,
                [
                    output_image,
                    output_mask,
                    f"source_annotations/{output_id}.json",
                    f"source_annotations/{output_id}.txt",
                ],
                strict=True,
            ):
                (stage / target).write_bytes(content)
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": output_image,
                    "width": 640,
                    "height": 640,
                    "license": 1,
                    "difficulty": difficulty,
                    "source_path": name,
                }
            )
            for annotation in annotations:
                annotation.update(id=len(coco["annotations"]) + 1, image_id=image_id)
                coco["annotations"].append(annotation)
                class_counts[annotation["source_category_id"]] += 1
            records.append(
                {
                    "id": output_id,
                    "source_path": name,
                    "image": output_image,
                    "mask": output_mask,
                    "difficulty": difficulty,
                    "scene_group": None,
                    "split": None,
                    "annotation_count": len(annotations),
                    **evidence,
                }
            )
            duplicates[evidence["pixel_sha256"]].append(output_id)
            consumed.update(paths)
            difficulties[difficulty] += 1
        if consumed != set(members):
            raise ValueError("Archive contains unsupported or orphan files")
        report = {
            "schema_version": 1,
            "importer_version": __version__,
            "source": asdict(source),
            "provenance": PROVENANCE,
            "images": len(records),
            "annotations": len(coco["annotations"]),
            "difficulty_counts": dict(difficulties),
            "source_class_counts": dict(class_counts),
            "empty_annotation_images": [r["id"] for r in records if r["annotation_count"] == 0],
            "exact_pixel_duplicate_groups": [ids for ids in duplicates.values() if len(ids) > 1],
            "mask_disagreement_images": sum(r["mask_disagreement_pixels"] > 0 for r in records),
            "mask_disagreement_pixels": sum(r["mask_disagreement_pixels"] for r in records),
            "approved_for_training": False,
            "training_blockers": [
                "Semantic class mapping needs documented confirmation; source IDs retained",
                "Original-photo/scene groups unavailable; no random train/test split generated",
                "Review PNG versus COCO rasterization discrepancies and annotation completeness",
                "An independent representative evaluation set is not yet qualified",
            ],
            "transformations": [
                "Images, PNG masks and source annotations preserved byte-for-byte",
                "Source categories 0/1 remapped to neutral COCO IDs 1/2",
                "COCO areas and bounding boxes recomputed using pycocotools rasterization",
                "No resizing, augmentation, inferred scene group or automatic split",
            ],
        }
        write_json(stage / "annotations.coco.json", coco)
        write_json(stage / "manifest.json", {"schema_version": 1, "records": records})
        write_json(stage / "report.json", report)
        write_json(stage / "ATTRIBUTION.json", PROVENANCE)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Import destination appeared during preparation")
        stage.rename(destination)
    return report
