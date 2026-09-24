"""Contrôler le corpus revu avant de le confier au détecteur par rectangles."""

import json
import math
import re
from pathlib import Path
from typing import Any

from smartsite_ia.annotations import unique_object
from smartsite_ia.categories import COLLECTION_LABELS
from smartsite_ia.collection_review import box_coco_document, validate_review
from smartsite_ia.curation import digest, read_document, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.prediction import decode_photo
from smartsite_ia.review import MAX_FILE_BYTES, MAX_JSON_BYTES, read_local

BOX_CORPUS_FILES = (
    "report.json",
    "review.json",
    "source/report.json",
    "train/_annotations.coco.json",
    "validation/_annotations.coco.json",
)
BOX_CLASSES = tuple(COLLECTION_LABELS)


def load_box_config(path: Path) -> tuple[dict[str, Any], str]:
    """Un pilote court et figé ; une option inconnue ne passe pas silencieusement."""
    config, sha = read_document(path)
    bounds = {
        "seed": (0, 2**32 - 1),
        "epochs": (1, 10),
        "batch_size": (1, 4),
        "grad_accum_steps": (1, 8),
    }
    if (
        set(config)
        != {"schema_version", "purpose", "lr", "corpus_sha256", "checkpoint_selection", *bounds}
        or config["purpose"] != "pilot_box_training"
        or config["checkpoint_selection"] != "last_epoch"
    ):
        raise ValueError("Expected a bounded pilot with last-epoch selection")
    for key, (minimum, maximum) in bounds.items():
        if type(config[key]) is not int or not minimum <= config[key] <= maximum:
            raise ValueError(f"Invalid box training parameter: {key}")
    lr = config["lr"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 0.001:
        raise ValueError("Invalid box learning rate")
    hashes = config["corpus_sha256"]
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(BOX_CORPUS_FILES)
        or any(
            not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in hashes.values()
        )
    ):
        raise ValueError("Expected pinned box corpus files")
    return config, sha


def reviewed_records(documents: dict[str, Any]) -> list[dict[str, Any]]:
    """Rattacher chaque image exportée à sa décision, y compris les exclusions."""
    report, review, source = (documents[n] for n in BOX_CORPUS_FILES[:3])
    reviewed = validate_review(review, source)
    if (
        report.get("schema_version") != 1
        or report.get("purpose") != "pilot_training_only"
        or report.get("human_validation") != "pending"
        or report.get("reviewer") != "assistant_visual_review"
        or report.get("final_test_created") is not False
        or report.get("selection_sha256") != review["selection_sha256"]
        or report.get("protocol") != review["protocol"]
    ):
        raise ValueError("Unsupported box corpus provenance")
    originals = {r["id"]: r for r in source["records"]}
    crops = {r["id"]: r for r in review["negative_crops"]}
    records = report.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 300:
        raise ValueError("Invalid box corpus size")
    seen = set()
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Invalid reviewed image")
        name = row["id"]
        if name in seen or name not in originals.keys() | crops.keys():
            raise ValueError("Duplicate or unknown reviewed image")
        seen.add(name)
        crop = name in crops
        decision = crops[name] if crop else reviewed[name]
        original = originals[crops[name]["parent_id"]] if crop else originals[name]
        candidate = crop or decision["decision"] != "excluded"
        partition = review["group_partitions"][original["group"]] if candidate else "excluded"
        expected = {
            **decision,
            "is_crop": crop,
            "group": original["group"],
            "credit": original["credit"],
            "partition": partition,
            "decision": "candidate" if crop else decision["decision"],
            "eligible_for_pilot": candidate,
            "approved_for_training": False,
            "human_validation": "pending",
            "image": f"images/{name}.jpg" if crop else f"source/images/{name}.jpg",
        }
        if crop:
            expected["boxes"] = []
        else:
            expected.update(width=original["width"], height=original["height"])
        if any(row.get(k) != value for k, value in expected.items()):
            raise ValueError("Exported image differs from its review decision")
        if any(
            type(row.get(k)) is not int or not 32 <= row[k] <= 8192 for k in ("width", "height")
        ):
            raise ValueError("Invalid reviewed image dimensions")
        if not isinstance(row.get("image_sha256"), str) or not re.fullmatch(
            r"[a-f0-9]{64}", row["image_sha256"]
        ):
            raise ValueError("Missing reviewed image hash")
    if seen != originals.keys() | crops.keys():
        raise ValueError("Review images are missing from the export")
    return records


def inspect_box_corpus(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Vérifier boîtes, photos et groupes sans charger de moteur ni ouvrir un test."""
    documents = {}
    for name in BOX_CORPUS_FILES:
        raw = read_local(root, name, MAX_JSON_BYTES)
        if digest(raw) != config["corpus_sha256"][name]:
            raise ValueError(f"Box corpus changed: {name}")
        document = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(document, dict):
            raise ValueError("Expected a corpus JSON object")
        documents[name] = document
    if documents["report.json"].get("review_sha256") != config["corpus_sha256"]["review.json"]:
        raise ValueError("Report and review hash disagree")
    try:
        records = reviewed_records(documents)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed reviewed corpus") from exc
    selection: dict[str, Any] = {
        "schema_version": 1,
        "test_used": False,
        "class_names": list(BOX_CLASSES),
        "configuration": config,
        "source_corpus_sha256": config["corpus_sha256"],
        "human_validation": "pending",
        "splits": {},
    }
    groups: dict[str, str] = {}
    pixels: dict[str, str] = {}
    for split, target in (("train", "train"), ("validation", "valid")):
        rows = [r for r in records if r["partition"] == split]
        document = documents[f"{split}/_annotations.coco.json"]
        # Le même calcul que l'export évite deux conventions de coordonnées.
        if not rows or document != box_coco_document(rows):
            raise ValueError("COCO rectangles differ from the reviewed annotations")
        if {b["class"] for r in rows for b in r["boxes"]} != set(BOX_CLASSES):
            raise ValueError("Each pilot split must contain all classes")
        selected = []
        for row in rows:
            group = row["group"]
            if group in groups and groups[group] != split:
                raise ValueError("A reviewed group crosses splits")
            groups[group] = split
            name = f"{split}/{row['id']}.jpg"
            raw = read_local(root, name, MAX_FILE_BYTES)
            if digest(raw) != row["image_sha256"]:
                raise ValueError(f"Reviewed photo changed: {row['id']}")
            photo = decode_photo(raw)
            if list(photo.size) != [row["width"], row["height"]]:
                raise ValueError("Reviewed photo dimensions changed")
            pixel_sha = digest(f"{photo.size}:RGB:".encode() + photo.tobytes())
            if pixel_sha in pixels and pixels[pixel_sha] != split:
                raise ValueError("Identical image pixels cross splits")
            pixels[pixel_sha] = split
            selected.append({**row, "image": f"{target}/{row['id']}.jpg"})
        selection["splits"][target] = {
            "images": len(rows),
            "groups": len({r["group"] for r in rows}),
            "annotations": len(document["annotations"]),
            "negative_crops": sum(r["is_crop"] for r in rows),
            "records": selected,
        }
        # Les anciens essais ont épinglé leur sélection sans ce champ.
        # On garde leur document identique lorsqu'aucune photo négative n'est présente.
        if negative_photos := sum(r["decision"] == "negative" for r in rows):
            selection["splits"][target]["negative_photos"] = negative_photos
    if selection["splits"]["train"]["images"] < config["batch_size"]:
        raise ValueError("Training split is smaller than one batch")
    return selection


def prepare_box_inputs(root: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Copier les lots vérifiés ; seule l'appellation validation devient valid."""
    selection = inspect_box_corpus(root, config)
    with staged_output(output, [root]) as stage:
        for split, target in (("train", "train"), ("validation", "valid")):
            (stage / target).mkdir()
            for row in selection["splits"][target]["records"]:
                raw = read_local(root, f"{split}/{row['id']}.jpg", MAX_FILE_BYTES)
                if digest(raw) != row["image_sha256"]:
                    raise ValueError("Reviewed photo changed during preparation")
                (stage / row["image"]).write_bytes(raw)
            raw = read_local(root, f"{split}/_annotations.coco.json", MAX_JSON_BYTES)
            if digest(raw) != config["corpus_sha256"][f"{split}/_annotations.coco.json"]:
                raise ValueError("Reviewed annotations changed during preparation")
            (stage / target / "_annotations.coco.json").write_bytes(raw)
        write_json(stage / "selection.json", selection)
    return selection
