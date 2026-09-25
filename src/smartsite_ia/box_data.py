"""Contrôler le corpus revu avant de le confier au détecteur par rectangles."""

import io
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

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


CORPUS_KINDS = ("reviewed_collection", "prepared_coco")
DEFAULT_CORPUS_KIND = "reviewed_collection"


MAX_PREPARED_PHOTO_BYTES = 32 * 1024 * 1024
PREPARED_CORPUS_FILES = (
    "report.json",
    "train/_annotations.coco.json",
    "valid/_annotations.coco.json",
)


def corpus_photo_limit(config: Mapping[str, Any]) -> int:
    """La taille maximale d'une photo dépend du corpus, pas du parcours."""
    if config.get("corpus_kind", DEFAULT_CORPUS_KIND) == "prepared_coco":
        return MAX_PREPARED_PHOTO_BYTES
    return MAX_FILE_BYTES


def config_classes(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Les classes de l'essai : celles de la configuration, sinon celles d'humidité."""
    names = config.get("class_names")
    return tuple(names) if names else BOX_CLASSES


def load_box_config(path: Path) -> tuple[dict[str, Any], str]:
    """Un pilote court et figé ; une option inconnue ne passe pas silencieusement.

    Deux champs facultatifs ouvrent la chaîne à d'autres corpus sans toucher aux
    essais déjà mesurés : `class_names` remplace les classes du corpus d'humidité,
    et `corpus_kind` choisit le contrôle d'entrée. Une configuration qui ne les
    mentionne pas garde exactement le comportement d'origine.
    """
    config, sha = read_document(path)
    bounds = {
        "seed": (0, 2**32 - 1),
        "epochs": (1, 10),
        "batch_size": (1, 4),
        "grad_accum_steps": (1, 8),
    }
    required = {"schema_version", "purpose", "lr", "corpus_sha256", "checkpoint_selection", *bounds}
    if (
        not required <= set(config)
        or set(config) - required - {"class_names", "corpus_kind"}
        or config["purpose"] != "pilot_box_training"
        or config["checkpoint_selection"] != "last_epoch"
    ):
        raise ValueError("Expected a bounded pilot with last-epoch selection")
    if config.get("corpus_kind", DEFAULT_CORPUS_KIND) not in CORPUS_KINDS:
        raise ValueError("Unknown box corpus kind")
    names = config.get("class_names")
    if names is not None and (
        not isinstance(names, list)
        or not 1 <= len(names) <= 10
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n or len(n) > 40 for n in names)
    ):
        raise ValueError("Invalid box class names")
    for key, (minimum, maximum) in bounds.items():
        if type(config[key]) is not int or not minimum <= config[key] <= maximum:
            raise ValueError(f"Invalid box training parameter: {key}")
    lr = config["lr"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 0.001:
        raise ValueError("Invalid box learning rate")
    hashes = config["corpus_sha256"]
    expected = (
        PREPARED_CORPUS_FILES
        if config.get("corpus_kind", DEFAULT_CORPUS_KIND) == "prepared_coco"
        else BOX_CORPUS_FILES
    )
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(expected)
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


def prepared_photo_size(raw: bytes) -> tuple[int, int]:
    """Lire les dimensions dans l'en-tête, sans décoder toute l'image.

    Ces photos de drone montent à 48 mégapixels, au-delà de la limite de décodage du
    projet, qui protège les chemins d'inférence. Elles ont déjà été décodées et
    empreintes lors de la préparation, et leur empreinte est revérifiée juste avant :
    ouvrir l'en-tête suffit donc ici, et évite de décoder des milliards de pixels.
    """
    with Image.open(io.BytesIO(raw), formats=["JPEG", "MPO"]) as image:
        width, height = image.size
        frames = getattr(image, "n_frames", 1)

        primary_mpo = image.format == "MPO" and frames == 2
        if (
            image.mode not in ("RGB", "L")
            or (frames != 1 and not primary_mpo)
            or not 32 <= width <= 20000
            or not 32 <= height <= 20000
        ):
            raise ValueError("Unsupported prepared photo")
        return width, height


def inspect_prepared_corpus(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Contrôler un corpus déjà préparé et vérifié ailleurs, sans revalider sa revue.

    Le corpus d'humidité porte toute sa revue avec lui, donc son contrôle rejoue les
    décisions d'annotation. Un corpus comme CUBIT a été préparé et tracé par son
    propre parcours : on vérifie ici ce qui concerne l'apprentissage, c'est-à-dire
    les empreintes, les classes annoncées, les groupes de scène et l'absence de pixels
    partagés entre les deux lots. Le lot de test n'est pas ouvert.
    """
    names = config_classes(config)
    documents = {}
    for name in PREPARED_CORPUS_FILES:
        raw = read_local(root, name, MAX_JSON_BYTES)
        if digest(raw) != config["corpus_sha256"][name]:
            raise ValueError(f"Box corpus changed: {name}")
        documents[name] = json.loads(raw, object_pairs_hook=unique_object)
    report = documents["report.json"]
    if report.get("approved_for_training") is not False or report.get("class_names") != list(names):
        raise ValueError("Prepared corpus classes disagree with the configuration")
    manifest = {row["id"]: row for row in report["records"]}

    selection: dict[str, Any] = {
        "schema_version": 1,
        "test_used": False,
        "class_names": list(names),
        "configuration": config,
        "source_corpus_sha256": config["corpus_sha256"],
        "human_validation": "pending",
        "splits": {},
    }
    groups: dict[str, str] = {}
    pixels: dict[str, str] = {}
    for split in ("train", "valid"):
        document = documents[f"{split}/_annotations.coco.json"]
        categories = [c["name"] for c in sorted(document["categories"], key=lambda c: c["id"])]
        if categories != list(names):
            raise ValueError(f"Unexpected categories in {split}")
        rows = []
        for image in document["images"]:
            sample_id = Path(image["file_name"]).stem
            row = manifest.get(sample_id)
            if row is None or row["split"] != split:
                raise ValueError(f"Photo missing from the corpus manifest: {sample_id}")
            group = row["scene_group"]
            if groups.setdefault(group, split) != split:
                raise ValueError("A scene group crosses splits")
            raw = read_local(root, f"{split}/{sample_id}.jpg", MAX_PREPARED_PHOTO_BYTES)
            fingerprint = digest(raw)
            if fingerprint != row["image_sha256"]:
                raise ValueError(f"Prepared photo changed: {sample_id}")
            if prepared_photo_size(raw) != (image["width"], image["height"]):
                raise ValueError("Prepared photo dimensions changed")

            if pixels.setdefault(fingerprint, split) != split:
                raise ValueError("Identical photos cross splits")
            rows.append({**row, "image": f"{split}/{sample_id}.jpg"})
        if not rows:
            raise ValueError(f"Empty split: {split}")
        selection["splits"][split] = {
            "images": len(rows),
            "groups": len({r["scene_group"] for r in rows}),
            "annotations": len(document["annotations"]),
            "records": rows,
        }
    if selection["splits"]["train"]["images"] < config["batch_size"]:
        raise ValueError("Training split is smaller than one batch")
    return selection


def prepare_prepared_inputs(root: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Recopier les deux lots vérifiés ; le lot de test reste où il est."""
    selection = inspect_prepared_corpus(root, config)
    with staged_output(output, [root]) as stage:
        for split in ("train", "valid"):
            (stage / split).mkdir()
            for row in selection["splits"][split]["records"]:
                raw = read_local(root, row["image"], MAX_PREPARED_PHOTO_BYTES)
                if digest(raw) != row["image_sha256"]:
                    raise ValueError("Prepared photo changed during preparation")
                (stage / row["image"]).write_bytes(raw)
            raw = read_local(root, f"{split}/_annotations.coco.json", MAX_JSON_BYTES)
            if digest(raw) != config["corpus_sha256"][f"{split}/_annotations.coco.json"]:
                raise ValueError("Prepared annotations changed during preparation")
            (stage / split / "_annotations.coco.json").write_bytes(raw)
        write_json(stage / "selection.json", selection)
    return selection


def inspect_corpus(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Aiguiller vers le contrôle correspondant au type de corpus annoncé."""
    if config.get("corpus_kind", DEFAULT_CORPUS_KIND) == "prepared_coco":
        return inspect_prepared_corpus(root, config)
    return inspect_box_corpus(root, config)


def prepare_inputs(root: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Même aiguillage pour la copie des lots d'entrée."""
    if config.get("corpus_kind", DEFAULT_CORPUS_KIND) == "prepared_coco":
        return prepare_prepared_inputs(root, output, config)
    return prepare_box_inputs(root, output, config)
