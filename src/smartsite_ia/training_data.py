"""Préparer un essai à partir du corpus figé, sans ouvrir le test réservé."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from smartsite_ia.annotations import unique_object
from smartsite_ia.categories import get_class_names, project_categories
from smartsite_ia.curation import digest, read_document, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.review import MAX_FILE_BYTES, MAX_JSON_BYTES, read_local
from smartsite_ia.training_data_html import write_training_data_page

CORPUS_FILES = (
    "manifest.json",
    "report.json",
    "train/_annotations.coco.json",
    "valid/_annotations.coco.json",
)


def load_training_config(path: Path) -> tuple[dict[str, Any], str]:
    """On refuse aussi les options inconnues, pour repérer une faute de frappe."""
    config, checksum = read_document(path)
    bounds = {
        "seed": (0, 2**32 - 1),
        "epochs": (1, 1000),
        "train_images": (2, 5000),
        "valid_images": (2, 5000),
        "batch_size": (1, 32),
        "grad_accum_steps": (1, 32),
    }
    expected = {"schema_version", "purpose", "corpus_sha256", "lr", *bounds}
    if "class_names" in config:
        expected.add("class_names")
    get_class_names(config)
    if "checkpoint_selection" in config:
        expected.add("checkpoint_selection")
        if config["checkpoint_selection"] not in ("best_validation", "last_epoch"):
            raise ValueError("Unknown checkpoint selection rule")
    if "initial_checkpoint_sha256" in config:
        expected.add("initial_checkpoint_sha256")
        checksum_value = config["initial_checkpoint_sha256"]
        if not isinstance(checksum_value, str) or not re.fullmatch("[a-f0-9]{64}", checksum_value):
            raise ValueError("Expected a pinned initial checkpoint hash")
    if set(config) != expected or config["purpose"] not in ("smoke", "experiment"):
        raise ValueError("Unknown or missing training configuration fields")
    for key, (minimum, maximum) in bounds.items():
        if type(config[key]) is not int or not minimum <= config[key] <= maximum:
            raise ValueError(f"Invalid training parameter: {key}")
    lr = config["lr"]
    if type(lr) not in (int, float) or not 0 < lr <= 0.01:
        raise ValueError("Learning rate must be finite and between 0 and 0.01")
    hashes = config["corpus_sha256"]
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(CORPUS_FILES)
        or any(
            not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value)
            for value in hashes.values()
        )
    ):
        raise ValueError("Expected pinned hashes for the prepared corpus")
    return config, checksum


def select_groups(records: list[dict[str, Any]], maximum: int, seed: int) -> list[dict[str, Any]]:
    """Prendre des groupes entiers, en alternant difficulté et présence des classes."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["content_group"]].append(record)
    buckets: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        # petit essai ne voie que des fissures. Ce n'est pas un test représentatif
        stratum = str(
            (
                sorted({r["difficulty"] for r in members}),
                sorted({c for r in members for c in r["source_classes"]}),
            )
        )
        buckets[stratum].append(key)
    for values in buckets.values():
        values.sort(key=lambda key: digest(f"{seed}:{key}".encode()))
    selected: list[dict[str, Any]] = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                members = groups[buckets[key].pop(0)]
                if len(selected) + len(members) <= maximum:
                    selected.extend(members)
    if {c for r in selected for c in r["source_classes"]} != {0, 1}:
        raise ValueError("Selected groups must contain both classes; increase the image budget")
    return sorted(selected, key=lambda r: r["id"])


def prepare_training_inputs(root: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Vérifier les fichiers épinglés puis copier seulement train et validation."""
    names = get_class_names(config)
    documents = {}
    for name in CORPUS_FILES:
        raw = read_local(root, name, MAX_JSON_BYTES)
        if digest(raw) != config["corpus_sha256"][name]:
            raise ValueError(f"Prepared corpus changed: {name}")
        documents[name] = json.loads(raw, object_pairs_hook=unique_object)
    report = documents["report.json"]
    if (
        report.get("dataset") != "damsegment_v1"
        or report.get("approved_for_training") is not True
        or report.get("usage") != "experimental_training_only"
    ):
        raise ValueError("Corpus is not approved for this experimental training")
    records = documents["manifest.json"]["records"]
    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= 5000
        or report["records"] != records
    ):
        raise ValueError("Prepared manifest and report disagree")
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for record in records:
        sample_id = record["id"]
        if (
            not isinstance(sample_id, str)
            or not re.fullmatch(r"(easy|medium|hard)_[0-9]{4}", sample_id)
            or sample_id in seen_ids
            or record["split"] not in ("train", "valid", "test", "quarantine")
        ):
            raise ValueError("Invalid prepared image ID or split")
        seen_ids.add(sample_id)
        group = record["content_group"]
        if group in group_splits and group_splits[group] != record["split"]:
            raise ValueError("A content group crosses prepared splits")
        group_splits[group] = record["split"]
    selection: dict[str, Any] = {
        "schema_version": 1,
        "test_used": False,
        "class_names": list(names),
        "configuration": config,
        "source_corpus_sha256": config["corpus_sha256"],
        "limits": [
            "No target annotation does not prove a defect-free surface",
            "Source categories are a project convention, not author-confirmed semantics",
            "Original groups and splits retained; unknown source scenes remain a limitation",
        ],
        "splits": {},
    }
    with staged_output(output, [root]) as stage:
        for split in ("train", "valid"):
            subset = select_groups(
                [r for r in records if r["split"] == split],
                config[f"{split}_images"],
                config["seed"],
            )
            source = documents[f"{split}/_annotations.coco.json"]
            document = project_categories(source, names)
            filenames = {f"{r['id']}.jpg" for r in subset}
            images = [im for im in document["images"] if im["file_name"] in filenames]
            if len(images) != len(subset) or {im["file_name"] for im in images} != filenames:
                raise ValueError("COCO images and manifest disagree")
            ids = {im["id"] for im in images}
            annotations = [a for a in document["annotations"] if a["image_id"] in ids]
            if {a["category_id"] for a in annotations} != set(range(1, len(names) + 1)):
                raise ValueError("All selected COCO classes must occur in the selected split")
            (stage / split).mkdir()
            for record in subset:
                relative = f"{split}/{record['id']}.jpg"
                if record["image"] != relative:
                    raise ValueError("Unexpected image location in prepared corpus")
                raw = read_local(root, relative, MAX_FILE_BYTES)
                if digest(raw) != record["image_sha256"]:
                    raise ValueError(f"Prepared photo changed: {record['id']}")
                (stage / relative).write_bytes(raw)
            write_json(
                stage / split / "_annotations.coco.json",
                {**document, "images": images, "annotations": annotations},
            )
            positive_ids = {a["image_id"] for a in annotations}
            without_target = sorted(
                Path(im["file_name"]).stem for im in images if im["id"] not in positive_ids
            )
            selection["splits"][split] = {
                "images": len(images),
                "annotations": len(annotations),
                "excluded_annotations": sum(a["image_id"] in ids for a in source["annotations"])
                - len(annotations),
                "images_without_target_annotations": len(without_target),
                "without_target_annotation_ids": without_target,
                "records": subset,
            }
        write_json(stage / "selection.json", selection)
        write_training_data_page(stage, selection)
    return selection
