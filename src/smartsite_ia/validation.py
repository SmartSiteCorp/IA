"""Comparer les modèles sur la validation figée, sans ouvrir le test réservé."""

import json
import time
from pathlib import Path
from typing import Any

from smartsite_ia.annotations import unique_object
from smartsite_ia.categories import get_class_names, project_categories
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
    root: Path,
    run_report: dict[str, Any],
    class_names: tuple[str, ...] = CLASS_NAMES,
    split: str = "valid",
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Les empreintes viennent de l'essai, pas d'une sélection faite après ses résultats.

    Le manifeste et le rapport du corpus sont scellés par l'entraînement, donc ils
    prouvent aussi que la partie réservée n'a pas bougé. Ses annotations, elles, n'ont
    pas d'empreinte dans l'essai : c'est justement le sens de la réserve. On les
    rapproche alors du manifeste scellé et on publie leur empreinte calculée.
    """
    if split not in ("valid", "test"):
        raise ValueError("Unknown corpus split")
    documents = {}
    hashes = run_report["config"]["corpus_sha256"]
    annotations = f"{split}/_annotations.coco.json"
    found: dict[str, str] = {}
    for name in ("manifest.json", "report.json", annotations):
        raw = read_local(root, name, MAX_JSON_BYTES)
        found[name] = digest(raw)
        if name in hashes and found[name] != hashes[name]:
            raise ValueError(f"Validation corpus changed: {name}")
        documents[name] = json.loads(raw, object_pairs_hook=unique_object)
    manifest = documents["manifest.json"]["records"]
    if manifest != documents["report.json"]["records"]:
        raise ValueError("Validation manifest and report disagree")
    records = sorted([r for r in manifest if r["split"] == split], key=lambda r: r["id"])
    document = project_categories(documents[annotations], class_names)
    if not 1 <= len(records) <= 5000 or len(document["images"]) != len(records):
        raise ValueError("Invalid validation image count")
    expected = {f"{r['id']}.jpg" for r in records if SAMPLE_ID.fullmatch(r["id"])}
    if len(expected) != len(records) or {im["file_name"] for im in document["images"]} != expected:
        raise ValueError("Validation IDs or paths disagree")
    # Même projection pour le spécialiste et le modèle à deux classes, leur
    # comparaison doit porter sur EXACTEMENT les mêmes références de fissure
    checksum = hashes.get(annotations, found[annotations])
    if class_names != CLASS_NAMES:
        checksum = digest(json.dumps(document, sort_keys=True, allow_nan=False).encode())
    return document, records, checksum


def evaluate_validation(
    run: Path,
    corpus: Path,
    output: Path,
    device: str,
    preprocessing: str = DEFAULT_PROFILE,
    mask_threshold: float = 0.5,
    class_names: tuple[str, ...] | None = None,
    split: str = "valid",
    open_reserved_test: bool = False,
) -> dict[str, Any]:
    """Un chargement du modèle, une photo à la fois, et toutes les images du lot.

    Ouvrir la partie réservée exige de le demander explicitement : une fois lue, elle
    ne peut plus servir de juge neutre, et aucun réglage ne doit être choisi ensuite
    en regardant son résultat.
    """
    if split == "test" and not open_reserved_test:
        raise ValueError("Opening the reserved test split must be requested explicitly")
    settings = inference_metadata(preprocessing, mask_threshold)
    report, checkpoint = verify_run(run)
    model_names = get_class_names(report.get("config", {}))
    names = (
        model_names if class_names is None else get_class_names({"class_names": list(class_names)})
    )
    if not set(names).issubset(model_names):
        raise ValueError("Evaluation classes must be included in the model classes")
    document, records, annotation_hash = load_validation(corpus, report, names, split)
    environment = runtime(device)
    started = time.monotonic()
    gt = coco_api(document)
    images = {im["file_name"]: im for im in document["images"]}
    results, coco_predictions = [], []
    with staged_output(output, [run, corpus]) as stage:
        engine = load_engine(checkpoint, device, preprocessing, mask_threshold, model_names)
        for record in records:
            sample_id = record["id"]
            raw = read_local(corpus, f"{split}/{sample_id}.jpg", MAX_FILE_BYTES)
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
                engine.predict(photo, threshold=SCORE_FLOOR),
                photo,
                allow_empty_boxes=True,
                class_names=model_names,
            )
            proposals = [
                {**p, "class_id": names.index(p["class_name"])}
                for p in proposals
                if p["class_name"] in names
            ]
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
            analysis = analyze_masks(references, predicted, DISPLAY_THRESHOLD, names)
            counts = {name: row["counts"] for name, row in analysis.items()}
            folder = stage / sample_id
            folder.mkdir()
            # Les vignettes n'altèrent pas les masques utilisés pour mesurer les résultats.
            photo.save(folder / "photo.jpg", quality=90)
            visible = [p for p in proposals if p["score"] >= DISPLAY_THRESHOLD]
            render_predictions(photo, visible, show_boxes=False, class_names=names).save(
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
            render_predictions(photo, ref_records, show_boxes=False, class_names=names).save(
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
            print(f"{split} {len(results)}/{len(records)} : {sample_id}", flush=True)
        by_difficulty: dict[str, Any] = {}
        for difficulty in sorted({r["difficulty"] for r in results}):
            subset = [r for r in results if r["difficulty"] == difficulty]
            by_difficulty[difficulty] = {
                "images": len(subset),
                "counts": summarize_counts([r["counts"] for r in subset], names),
            }
        metrics = coco_scores(document, coco_predictions)
        result = {
            "schema_version": 1,
            "status": "completed",
            "split": split,
            "test_used": split == "test",
            "qualified_for_smartsite": False,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "class_names": list(names),
            "model_class_names": list(model_names),
            "annotations_sha256": annotation_hash,
            "source_annotations_sha256": report["config"]["corpus_sha256"].get(
                f"{split}/_annotations.coco.json", annotation_hash
            ),
            "images": len(results),
            "records": results,
            "runtime": environment,
            "inference": settings,
            "protocol": validation_protocol(),
            "coco": metrics,
            "counts": summarize_counts([r["counts"] for r in results], names),
            "mask_errors": {
                name: {
                    reason: sum(r["mask_errors"][name][reason] for r in results)
                    for reason in ("duplicate", "insufficient_overlap", "no_overlap")
                }
                for name in names
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
        or get_class_names(first) != get_class_names(second)
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
