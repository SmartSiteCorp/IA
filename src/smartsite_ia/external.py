"""Normaliser les photos externes et leurs masques, en gardant leur rôle séparé."""

import io
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from smartsite_ia.annotations import number
from smartsite_ia.curation import content_groups, digest, read_document, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.preparation_html import write_preparation_page
from smartsite_ia.review import MAX_FILE_BYTES, read_local
from smartsite_ia.similarity import find_similar_pairs, fingerprint
from smartsite_ia.source import CONCRETE_CRACK_SEGMENTATION


def validate_external_policy(policy: dict[str, Any]) -> list[dict[str, Any]]:
    """Les empreintes concernent les originaux ; aucune photo voisine ne les remplace."""
    if (
        policy.get("dataset") != "concrete_crack_segmentation_v1"
        or policy.get("archive_sha256") != CONCRETE_CRACK_SEGMENTATION.sha256
    ):
        raise ValueError("Expected the pinned external dataset policy")
    records = policy.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 1000:
        raise ValueError("Expected a bounded external inventory")
    if type(policy.get("expected_images")) is not int or len(records) != policy["expected_images"]:
        raise ValueError("External inventory count mismatch")
    ids = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid external record")
        sample_id = record.get("id")
        if (
            not isinstance(sample_id, str)
            or not re.fullmatch(r"[0-9]{3}", sample_id)
            or sample_id in ids
        ):
            raise ValueError("Invalid or duplicate external ID")
        ids.add(sample_id)
        for kind, folder in (("image", "rgb"), ("mask", "BW")):
            if record.get(kind) not in {
                f"extracted/{folder}/{sample_id}.{ext}" for ext in ("jpg", "JPG")
            }:
                raise ValueError("External path does not match its ID")
            if not isinstance(record.get(f"{kind}_sha256"), str) or not re.fullmatch(
                r"[a-f0-9]{64}", record[f"{kind}_sha256"]
            ):
                raise ValueError("Missing source file SHA-256")
    masks = policy.get("mask_policy")
    if (
        not isinstance(masks, dict)
        or type(masks.get("threshold")) is not int
        or masks["threshold"] != 128
        or masks.get("sensitivity_thresholds") != [96, 160]
    ):
        raise ValueError("Expected the reviewed 128 threshold and 96/160 sensitivity bounds")
    if not isinstance(policy.get("provenance"), dict):
        raise ValueError("Missing external attribution")
    pair_policy = policy.get("pair_policy")
    if not isinstance(pair_policy, dict):
        raise ValueError("Missing pair review thresholds")
    if (
        not 0 <= number(pair_policy.get("max_native_rgb_mae")) <= 255
        or not 0 < number(pair_policy.get("min_binary_mask_iou")) <= 1
    ):
        raise ValueError("Pair review thresholds outside their valid range")
    return sorted(records, key=lambda r: r["id"])


def open_photo(content: bytes) -> tuple[Image.Image, int]:
    """On borne la taille avant le décodage, puis on applique l'orientation une seule fois."""
    with Image.open(io.BytesIO(content), formats=["JPEG"]) as raw:
        if raw.width * raw.height > 16_000_000 or raw.mode not in ("RGB", "L"):
            raise ValueError("Unsupported external image dimensions or mode")
        orientation = raw.getexif().get(274, 1)
        if type(orientation) is not int or orientation not in range(1, 9):
            raise ValueError("Invalid EXIF orientation")
        image = ImageOps.exif_transpose(raw).convert("RGB")
        image.info.clear()
        return image, orientation


def normalize_pair(
    photo: bytes, mask: bytes
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, Any]]:
    """La bande grise mesure l'effet possible du JPEG, pas une incertitude du modèle."""
    image, orientation = open_photo(photo)
    label_image, mask_orientation = open_photo(mask)
    if image.size != label_image.size:
        raise ValueError("External photo and mask dimensions differ after EXIF orientation")
    gray = np.asarray(label_image.convert("L"))
    foreground = gray >= 128
    band = (gray >= 96) & (gray < 160)
    area = int(foreground.sum())
    if area == 0 or area == foreground.size:
        raise ValueError(
            "External mask is empty or completely foreground; explicit review required"
        )
    metrics = {
        "source_exif_orientation": orientation,
        "source_mask_exif_orientation": mask_orientation,
        "width": image.width,
        "height": image.height,
        "foreground_pixels": area,
        "threshold_sensitivity_pixels": int(band.sum()),
        "foreground_pixels_at_96": int(np.count_nonzero(gray >= 96)),
        "foreground_pixels_at_160": int(np.count_nonzero(gray >= 160)),
        "threshold_band_fraction_of_foreground": float(band.sum() / area),
    }
    return (
        image,
        Image.fromarray(foreground.astype(np.uint8) * 255),
        Image.fromarray(band.astype(np.uint8) * 255),
        metrics,
    )


def measure_pair(
    stage: Path, pair: dict[str, Any], max_error: float = 2.0, min_iou: float = 0.98
) -> dict[str, Any]:
    """Comparer les pixels à leur résolution native, après la rotation proposée."""
    transform = (
        None if pair["transform_right"] == "IDENTITY" else Image.Transpose[pair["transform_right"]]
    )
    arrays = []
    for side in ("left", "right"):
        with Image.open(stage / "images" / f"{pair[side]}.png") as image:
            aligned = (
                image.transpose(transform) if side == "right" and transform is not None else image
            )
            arrays.append(np.asarray(aligned, dtype=np.int16))
    if arrays[0].shape != arrays[1].shape:
        return {**pair, "native_dimensions_match": False, "review_required": True}
    error = float(np.mean(np.abs(arrays[0] - arrays[1])))
    del arrays
    masks = []
    for side in ("left", "right"):
        with Image.open(stage / "masks" / f"{pair[side]}.png") as image:
            aligned = (
                image.transpose(transform) if side == "right" and transform is not None else image
            )
            masks.append(np.asarray(aligned) != 0)
    union = int(np.count_nonzero(masks[0] | masks[1]))
    iou = float(np.count_nonzero(masks[0] & masks[1]) / union)
    return {
        **pair,
        "native_dimensions_match": True,
        "native_rgb_mae": error,
        "binary_mask_iou": iou,
        "review_required": error > max_error or iou < min_iou,
    }


def prepare_external(root: Path, destination: Path, policy_path: Path) -> dict[str, Any]:
    """Exporter les masques sémantiques sans inventer des instances de fissures."""
    policy, policy_hash = read_document(policy_path)
    inventory = validate_external_policy(policy)
    records, fingerprints = [], []
    with staged_output(destination, [root]) as stage:
        for folder in ("images", "masks", "threshold_bands", "thumbnails", "overlays"):
            (stage / folder).mkdir()
        for entry in inventory:
            parts = [read_local(root, entry[kind], MAX_FILE_BYTES) for kind in ("image", "mask")]
            if any(
                digest(part) != entry[f"{kind}_sha256"]
                for part, kind in zip(parts, ("image", "mask"), strict=True)
            ):
                raise ValueError(f"External file SHA-256 mismatch for {entry['id']}")
            image, mask, band, metrics = normalize_pair(*parts)
            if metrics["source_exif_orientation"] != entry.get("exif_orientation") or metrics[
                "source_mask_exif_orientation"
            ] != entry.get("mask_exif_orientation"):
                raise ValueError("EXIF metadata differs from the inspected inventory")
            sample_id = entry["id"]
            fingerprints.append(fingerprint(sample_id, image, digest(image.tobytes())))
            paths = {
                "image": f"images/{sample_id}.png",
                "mask": f"masks/{sample_id}.png",
                "threshold_band": f"threshold_bands/{sample_id}.png",
            }
            for value, key in ((image, "image"), (mask, "mask"), (band, "threshold_band")):
                value.save(stage / paths[key], compress_level=3)
            # Les aperçus servent à vérifier le cadrage. Les métriques utilisent
            # toujours les images et masques complets, sans redimensionnement.
            preview = image.copy()
            preview.thumbnail((480, 480))
            preview.save(stage / "thumbnails" / f"{sample_id}.jpg", quality=85)
            overlay = np.asarray(preview).copy()
            visible = np.asarray(mask.resize(preview.size, Image.Resampling.NEAREST)) != 0
            overlay[visible] = (255, 50, 50)
            Image.fromarray(overlay).save(stage / "overlays" / f"{sample_id}.jpg", quality=90)
            records.append(
                {
                    "id": sample_id,
                    **paths,
                    **metrics,
                    "source_image": entry["image"],
                    "source_mask": entry["mask"],
                    "source_image_sha256": entry["image_sha256"],
                    "source_mask_sha256": entry["mask_sha256"],
                    "image_sha256": digest((stage / paths["image"]).read_bytes()),
                    "mask_sha256": digest((stage / paths["mask"]).read_bytes()),
                    "preview": f"overlays/{sample_id}.jpg",
                    "scene_group": None,
                }
            )
        # On ne fait pas confiance à une ancienne liste : elle est recalculée sur
        # les photos orientées. Toute paire nouvelle apparaît dans le rapport.
        pairs = [asdict(pair) for pair in find_similar_pairs(fingerprints)]
        pair_policy = policy["pair_policy"]
        measured = [
            measure_pair(
                stage, pair, pair_policy["max_native_rgb_mae"], pair_policy["min_binary_mask_iou"]
            )
            for pair in pairs
        ]
        groups = content_groups(
            {r["id"] for r in records}, [[p["left"], p["right"]] for p in pairs]
        )
        flagged = {groups[p["left"]] for p in measured if p["review_required"]}
        for record in records:
            group = groups[record["id"]]
            record["content_group"] = group
            record["split"] = (
                "quarantine"
                if group in flagged
                else "external_candidate"
                if record["id"] == group
                else "external_variant"
            )
            record["reason"] = (
                "Pair image/mask disagreement needs review"
                if group in flagged
                else "Lowest ID represents this content group; exploratory evaluation only"
                if record["id"] == group
                else "Similar view kept for robustness checks; not an independent case"
            )
        report = {
            "schema_version": 1,
            "dataset": policy["dataset"],
            "policy_sha256": policy_hash,
            "images": len(records),
            "content_groups": len(set(groups.values())),
            "split_counts": dict(Counter(r["split"] for r in records)),
            "similar_pairs": measured,
            "mask_policy": policy["mask_policy"],
            "pair_policy": pair_policy,
            "approved_for_training": False,
            "approved_for_evaluation": False,
            "usage": "prepared_external_candidate; alignment and protocol qualification pending",
            "limits": [
                "Semantic crack masks only; no fabricated crack instances or surface_loss labels",
                "Content groups are similarity groups, not known building/scene IDs",
                "Do not train on this source or its related 5y9wdsg2zt classification patches",
                "Threshold 128 is a project decision; report sensitivity at 96/160",
                "No SmartSite qualification, drone coverage or representative difficult negatives",
            ],
            "records": records,
        }
        write_json(stage / "manifest.json", {"schema_version": 1, "records": records})
        write_json(stage / "report.json", report)
        write_json(stage / "policy.json", policy)
        write_json(
            stage / "ATTRIBUTION.json",
            {
                **policy["provenance"],
                "changes": "EXIF orientation; RGB PNG; binary masks at 128; similarity grouping",
            },
        )
        write_preparation_page(stage, report)
    return report
