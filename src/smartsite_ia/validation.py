"""Comparer les modèles sur la validation figée, sans ouvrir le test réservé."""

import json
import time
from pathlib import Path
from typing import Any

from smartsite_ia.annotations import unique_object
from smartsite_ia.curation import digest, read_document, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.inference import DEFAULT_PROFILE, inference_metadata
from smartsite_ia.learning import process_peak_rss, runtime, verify_run
from smartsite_ia.model_assets import CLASS_NAMES
from smartsite_ia.prediction import (
    decode_photo,
    encode_predictions,
    load_engine,
    render_predictions,
)
from smartsite_ia.review import MAX_FILE_BYTES, MAX_JSON_BYTES, SAMPLE_ID, read_local
from smartsite_ia.validation_metrics import (
    analyze_masks,
    coco_api,
    coco_scores,
    summarize_counts,
)

# On garde les scores faibles pour calculer l'AP ; le seuil visuel reste séparé.
SCORE_FLOOR = 0.001
DISPLAY_THRESHOLD = 0.3


def load_validation(
    root: Path, run_report: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Les empreintes viennent de l'essai, pas d'une sélection faite après ses résultats."""
    documents = {}
    hashes = run_report["config"]["corpus_sha256"]
    names = ("manifest.json", "report.json", "valid/_annotations.coco.json")
    for name in names:
        raw = read_local(root, name, MAX_JSON_BYTES)
        if digest(raw) != hashes[name]:
            raise ValueError(f"Validation corpus changed: {name}")
        documents[name] = json.loads(raw, object_pairs_hook=unique_object)
    manifest = documents["manifest.json"]["records"]
    if manifest != documents["report.json"]["records"]:
        raise ValueError("Validation manifest and report disagree")
    records = sorted([r for r in manifest if r["split"] == "valid"], key=lambda r: r["id"])
    document = documents["valid/_annotations.coco.json"]
    if not 1 <= len(records) <= 5000 or len(document["images"]) != len(records):
        raise ValueError("Invalid validation image count")
    if [(c["id"], c["name"]) for c in document["categories"]] != list(enumerate(CLASS_NAMES, 1)):
        raise ValueError("Unexpected validation categories")
    expected = {f"{r['id']}.jpg" for r in records if SAMPLE_ID.fullmatch(r["id"])}
    if len(expected) != len(records) or {im["file_name"] for im in document["images"]} != expected:
        raise ValueError("Validation IDs or paths disagree")
    if len({im["id"] for im in document["images"]}) != len(records):
        raise ValueError("Duplicate validation image ID")
    ids = {im["id"] for im in document["images"]}
    seen = set()
    for a in document["annotations"]:
        if (
            a["image_id"] not in ids
            or a["category_id"] not in (1, 2)
            or a.get("iscrowd", 0) != 0
            or a["id"] in seen
        ):
            raise ValueError("Unsupported or inconsistent validation annotation")
        seen.add(a["id"])
    return document, records, hashes["valid/_annotations.coco.json"]


def evaluate_validation(
    run: Path,
    corpus: Path,
    output: Path,
    device: str,
    preprocessing: str = DEFAULT_PROFILE,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """Un chargement du modèle, une photo à la fois, et toutes les images du lot."""
    settings = inference_metadata(preprocessing, mask_threshold)
    report, checkpoint = verify_run(run)
    document, records, annotation_hash = load_validation(corpus, report)
    environment = runtime(device)
    started = time.monotonic()
    gt = coco_api(document)
    images = {im["file_name"]: im for im in document["images"]}
    results, coco_predictions = [], []
    with staged_output(output, [run, corpus]) as stage:
        engine = load_engine(checkpoint, device, preprocessing, mask_threshold)
        for record in records:
            sample_id = record["id"]
            raw = read_local(corpus, f"valid/{sample_id}.jpg", MAX_FILE_BYTES)
            if digest(raw) != record["image_sha256"]:
                raise ValueError(f"Validation photo changed: {sample_id}")
            photo = decode_photo(raw)
            im = images[f"{sample_id}.jpg"]
            if photo.size != (im["width"], im["height"]):
                raise ValueError("Validation photo dimensions disagree")
            before = time.monotonic()
            # À très faible score, le moteur renvoie aussi des boîtes plates au bord.
            # On conserve leurs masques dans l'évaluation, sans retirer des erreurs.
            proposals = encode_predictions(
                engine.predict(photo, threshold=SCORE_FLOOR), photo, allow_empty_boxes=True
            )
            inference_seconds = time.monotonic() - before
            # L'ID réseau commence à zéro, l'ID COCO à un : une seule conversion ici.
            predicted = [
                {
                    "image_id": im["id"],
                    "category_id": p["class_id"] + 1,
                    "score": p["score"],
                    "segmentation": p["mask_rle"],
                }
                for p in proposals
            ]
            references = [{**a, "segmentation": gt.annToRLE(a)} for a in gt.imgToAnns[im["id"]]]
            analysis = analyze_masks(references, predicted, DISPLAY_THRESHOLD)
            counts = {name: row["counts"] for name, row in analysis.items()}
            folder = stage / sample_id
            folder.mkdir()
            # Les vignettes n'altèrent pas les masques utilisés pour mesurer les résultats.
            photo.save(folder / "photo.jpg", quality=90)
            visible = [p for p in proposals if p["score"] >= DISPLAY_THRESHOLD]
            render_predictions(photo, visible, show_boxes=False).save(
                folder / "prediction.jpg", quality=90
            )
            ref_records = []
            for a in references:
                x, y, w, h = a["bbox"]
                ref_records.append(
                    {
                        "class_id": a["category_id"] - 1,
                        "score": 1.0,
                        "bbox_xyxy": [x, y, x + w, y + h],
                        "mask_rle": a["segmentation"],
                    }
                )
            render_predictions(photo, ref_records, show_boxes=False).save(
                folder / "reference.jpg", quality=90
            )
            write_json(folder / "predictions.json", {"schema_version": 1, "predictions": proposals})
            results.append(
                {
                    "id": sample_id,
                    "image_id": im["id"],
                    "difficulty": record["difficulty"],
                    "image_sha256": record["image_sha256"],
                    "counts": counts,
                    "mask_errors": {name: row["errors"] for name, row in analysis.items()},
                    "inference_seconds": inference_seconds,
                    "proposals": len(visible),
                }
            )
            coco_predictions.extend(predicted)
            print(f"Validation {len(results)}/{len(records)} : {sample_id}", flush=True)
        by_difficulty: dict[str, Any] = {}
        for difficulty in sorted({r["difficulty"] for r in results}):
            subset = [r for r in results if r["difficulty"] == difficulty]
            by_difficulty[difficulty] = {
                "images": len(subset),
                "counts": summarize_counts([r["counts"] for r in subset]),
            }
        metrics = coco_scores(document, coco_predictions)
        result = {
            "schema_version": 1,
            "status": "completed",
            "split": "valid",
            "test_used": False,
            "qualified_for_smartsite": False,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "annotations_sha256": annotation_hash,
            "images": len(results),
            "records": results,
            "runtime": environment,
            "inference": settings,
            "protocol": validation_protocol(),
            "coco": metrics,
            "counts": summarize_counts([r["counts"] for r in results]),
            "mask_errors": {
                name: {
                    reason: sum(r["mask_errors"][name][reason] for r in results)
                    for reason in ("duplicate", "insufficient_overlap", "no_overlap")
                }
                for name in CLASS_NAMES
            },
            "by_difficulty": by_difficulty,
            "elapsed_seconds": time.monotonic() - started,
            "process_peak_rss_bytes": process_peak_rss(),
        }
        write_json(stage / "report.json", result)
        write_json(
            stage / "coco_predictions.json", {"schema_version": 1, "predictions": coco_predictions}
        )
        from smartsite_ia.validation_html import write_validation_page

        write_validation_page(stage, result)
    return result


def validation_protocol() -> dict[str, Any]:
    return {
        "version": 1,
        "score_floor": SCORE_FLOOR,
        "display_threshold": DISPLAY_THRESHOLD,
        "threshold_calibrated": False,
        "matching_mask_iou": 0.5,
        "matching": "Same class, descending score, one-to-one, native image masks",
        "coco_max_dets_per_class": 100,
        "engine_max_proposals_per_image": 200,
        "limits": [
            "Internal validation from the same dam, not an independent site test",
            "Counts depend on supplied annotations; false proposals need human review",
            "AP is computed above the recorded score floor; not a percentage of correct photos",
            "No separate reviewed defect-free corpus in this evaluation",
        ],
    }


def compare_validations(before: Path, after: Path, output: Path) -> dict[str, Any]:
    """Refuser une comparaison si les photos, annotations ou règles ont changé."""
    first, first_hash = read_document(before / "report.json")
    second, second_hash = read_document(after / "report.json")
    for report in (first, second):
        if (
            report.get("status") != "completed"
            or report.get("split") != "valid"
            or report.get("test_used") is not False
        ):
            raise ValueError("Comparison requires completed validation reports")
        settings = report.get("inference")
        if settings is not None and (
            not isinstance(settings, dict)
            or settings
            != inference_metadata(
                settings.get("preprocessing", ""), settings.get("mask_probability_threshold", 0)
            )
        ):
            raise ValueError("Invalid inference settings in validation report")
        ids = [r["id"] for r in report["records"]]
        if (
            not 1 <= len(ids) <= 5000
            or len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or not SAMPLE_ID.fullmatch(i) for i in ids)
        ):
            raise ValueError("Invalid comparison image IDs")
    signatures = [
        [(r["id"], r["image_sha256"]) for r in report["records"]] for report in (first, second)
    ]
    if (
        first["protocol"] != second["protocol"]
        or first["annotations_sha256"] != second["annotations_sha256"]
        or signatures[0] != signatures[1]
    ):
        raise ValueError("Cannot compare different validation data or protocols")
    from smartsite_ia.validation_html import write_comparison_page

    with staged_output(output, [before, after]) as stage:
        result = {
            "schema_version": 1,
            "before_report_sha256": first_hash,
            "after_report_sha256": second_hash,
            "before": first,
            "after": second,
            "test_used": False,
        }
        # Copier les visuels rend le rapport autonome, sans chemin local dans le HTML.
        for r in first["records"]:
            folder = stage / r["id"]
            folder.mkdir()
            for src, name, original in (
                (before, "photo.jpg", "photo.jpg"),
                (before, "reference.jpg", "reference.jpg"),
                (before, "before.jpg", "prediction.jpg"),
                (after, "after.jpg", "prediction.jpg"),
            ):
                raw = read_local(src, f"{r['id']}/{original}", MAX_FILE_BYTES)
                (folder / name).write_bytes(raw)
        write_json(stage / "report.json", result)
        write_comparison_page(stage, first, second)
    return result
