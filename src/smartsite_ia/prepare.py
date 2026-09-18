"""Appliquer la curation DamSegment et exporter les trois partitions COCO."""

from collections import Counter, defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from smartsite_ia import __version__
from smartsite_ia.curation import content_groups, digest, read_document, require_text, staged_output
from smartsite_ia.importer import decode_image, write_json
from smartsite_ia.partition import SPLITS, assign_splits
from smartsite_ia.preparation_html import write_preparation_page
from smartsite_ia.review import MAX_JSON_BYTES, inspect_sample, load_samples, read_local
from smartsite_ia.similarity import find_similar_pairs, fingerprint
from smartsite_ia.source import PROVENANCE


def validate_policy(
    policy: dict[str, Any], manifest_hash: str, ids: set[str]
) -> tuple[dict[str, str], list[list[str]], list[dict[str, Any]]]:
    """Une décision ancienne ou incomplète ne doit pas approuver un autre corpus."""
    if policy.get("dataset") != "damsegment_v1" or policy.get("manifest_sha256") != manifest_hash:
        raise ValueError("Curation policy does not match this DamSegment manifest")
    mapping = policy.get("semantic_mapping")
    if not isinstance(mapping, dict) or mapping.get("status") != "project_visual_decision":
        raise ValueError("Semantic mapping requires a documented project decision")
    require_text(mapping.get("reason"))
    require_text(mapping.get("reviewer"))
    categories = mapping.get("categories")
    expected = [
        {"id": 1, "source_category_id": 0, "name": "crack"},
        {"id": 2, "source_category_id": 1, "name": "surface_loss"},
    ]
    if categories != expected or any(
        type(c.get(k)) is not int for c in categories for k in ("id", "source_category_id")
    ):
        raise ValueError("Expected the reviewed crack/surface_loss category mapping")
    excluded = policy.get("quarantine_for_future_partitions")
    links = policy.get("must_stay_together")
    if not isinstance(excluded, list) or not isinstance(links, list):
        raise ValueError("Missing curation decisions")
    reasons: dict[str, str] = {}
    for entry in excluded:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("Invalid quarantine decision")
        if entry["id"] not in ids or entry["id"] in reasons:
            raise ValueError("Unknown or duplicate quarantine ID")
        reasons[entry["id"]] = require_text(entry.get("reason"))
    groups: list[list[str]] = []
    for entry in links:
        if not isinstance(entry, dict):
            raise ValueError("Invalid grouping decision")
        require_text(entry.get("reason"))
        members = entry.get("members")
        if not isinstance(members, list):
            raise ValueError("Invalid grouping members")
        groups.append(members)
    content_groups(ids, groups)
    return reasons, groups, categories


def prepare_corpus(
    root: Path, destination: Path, policy_path: Path, seed: int = 20260918
) -> dict[str, Any]:
    """Préparer un essai reproductible ; la qualification chantier reste distincte."""
    samples, manifest_hash = load_samples(root)
    policy, policy_hash = read_document(policy_path)
    excluded, links, categories = validate_policy(
        policy, manifest_hash, {s.sample_id for s in samples}
    )
    with staged_output(destination, [root]) as stage:
        # On travaille sur une copie vérifiée. Une source modifiée ensuite ne change
        # pas les photos exportées ni les empreintes enregistrées ici.
        (stage / "pending").mkdir()
        (stage / "thumbnails").mkdir()
        records, fingerprints = [], []
        annotations_by_id = {}
        for sample in samples:
            parts, annotations, evidence = inspect_sample(root, sample)
            image = decode_image(parts[0], "JPEG")
            fingerprints.append(fingerprint(sample.sample_id, image, evidence["pixel_sha256"]))
            (stage / "pending" / f"{sample.sample_id}.jpg").write_bytes(parts[0])
            image.resize((120, 120)).save(stage / "thumbnails" / f"{sample.sample_id}.jpg")
            annotations_by_id[sample.sample_id] = annotations
            records.append(
                {
                    **sample.record,
                    "source_classes": sorted({a["source_category_id"] for a in annotations}),
                    "source_image": sample.record["image"],
                    "source_mask": sample.record["mask"],
                    "preview": f"thumbnails/{sample.sample_id}.jpg",
                }
            )
            if not annotations and sample.sample_id not in excluded:
                raise ValueError("Empty annotations require an explicit quarantine decision")
        # On recalcule les similitudes : le fichier de décisions n'est pas censé
        # cacher un doublon oublié. Tout lien détecté reste dans la même partition.
        pairs = find_similar_pairs(fingerprints)
        links.extend([[pair.left, pair.right] for pair in pairs])
        groups = content_groups({r["id"] for r in records}, links)
        quarantined_groups = {groups[sample_id] for sample_id in excluded}
        for record in records:
            record["content_group"] = groups[record["id"]]
            if record["content_group"] in quarantined_groups and record["id"] not in excluded:
                excluded[record["id"]] = (
                    "Linked to a quarantined image; the whole group is withheld"
                )
        retained = [r for r in records if r["id"] not in excluded]
        assignments = assign_splits(retained, seed)
        documents: dict[str, dict[str, Any]] = {
            split: {
                "info": {
                    "description": "SmartSite DamSegment experimental partition",
                    "version": 1,
                },
                "licenses": [{"id": 1, "name": "CC BY 4.0", "url": PROVENANCE["license_url"]}],
                "categories": categories,
                "images": [],
                "annotations": [],
            }
            for split in SPLITS
        }
        for split in (*SPLITS, "quarantine"):
            (stage / split).mkdir()
        for image_id, record in enumerate(records, 1):
            sample_id = record["id"]
            split = assignments.get(sample_id, "quarantine")
            record.update(
                split=split, image=f"{split}/{sample_id}.jpg", reason=excluded.get(sample_id, "")
            )
            (stage / "pending" / f"{sample_id}.jpg").rename(stage / record["image"])
            # Le masque original reste une référence source, pas un chemin prétendument
            # présent dans cet export d'instances COCO.
            record.pop("mask")
            if split == "quarantine":
                continue
            document = documents[split]
            document["images"].append(
                {
                    "id": image_id,
                    "file_name": f"{sample_id}.jpg",
                    "width": 640,
                    "height": 640,
                    "license": 1,
                }
            )
            for annotation in annotations_by_id[sample_id]:
                document["annotations"].append(
                    {**annotation, "id": len(document["annotations"]) + 1, "image_id": image_id}
                )
        (stage / "pending").rmdir()
        grouped: dict[str, list[str]] = defaultdict(list)
        for record in records:
            grouped[record["content_group"]].append(record["id"])
        summary = {}
        for split, document in documents.items():
            write_json(stage / split / "_annotations.coco.json", document)
            summary[split] = {
                "images": len(document["images"]),
                "annotations": len(document["annotations"]),
                "source_class_counts": dict(
                    Counter(a["source_category_id"] for a in document["annotations"])
                ),
            }
        report = {
            "schema_version": 1,
            "dataset": "damsegment_v1",
            "preparer_version": __version__,
            "dependencies": {name: version(name) for name in ("pillow", "numpy", "pycocotools")},
            "source_manifest_sha256": manifest_hash,
            "policy_sha256": policy_hash,
            "seed": seed,
            "split_method": "grouped_hash_order; strata=difficulty_set+classes; target=80/10/10",
            "images": len(records),
            "retained_images": len(retained),
            "quarantined_images": len(excluded),
            "splits": summary,
            "linked_groups": [members for members in grouped.values() if len(members) > 1],
            "recomputed_similar_pairs": [asdict(pair) for pair in pairs],
            "semantic_mapping": policy["semantic_mapping"],
            "approved_for_training": True,
            "usage": "experimental_training_only",
            "approved_for_independent_evaluation": False,
            "qualified_for_smartsite": False,
            "limits": [
                "Class names are a project interpretation, not an author-confirmed numeric mapping",
                "One dam only; original photo/scene IDs are unavailable",
                "Known similarities stay together; unrecognized related crops can cross partitions",
                "Internal valid/test scores cannot demonstrate new-site generalization",
                "No independent test qualification or representative difficult-negative set",
            ],
            "records": records,
        }
        write_json(stage / "manifest.json", {"schema_version": 1, "records": records})
        write_json(stage / "report.json", report)
        write_json(stage / "policy.json", policy)
        write_json(
            stage / "ATTRIBUTION.json",
            {
                **PROVENANCE,
                "changes": "Grouped splits; project class naming; source polygons unchanged",
            },
        )
        write_preparation_page(stage, report)
        if digest(read_local(root, "manifest.json", MAX_JSON_BYTES)) != manifest_hash:
            raise ValueError("Source manifest changed during preparation")
    return report
