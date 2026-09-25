"""Rejouer le pilote sur sa validation et conserver les preuves de chaque rectangle."""

import importlib
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from smartsite_ia.annotations import number
from smartsite_ia.box_data import BOX_CLASSES, inspect_box_corpus, load_box_config
from smartsite_ia.box_metrics import checked_box, match_boxes
from smartsite_ia.box_review_html import write_box_gallery
from smartsite_ia.curation import digest, read_document, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.inference import AlignedPredictor
from smartsite_ia.learning import file_hash, runtime, summarize_metrics, verify_training_data
from smartsite_ia.model_assets import BOX_MODEL_NAME
from smartsite_ia.prediction import decode_photo
from smartsite_ia.review import MAX_FILE_BYTES, read_local
from smartsite_ia.validation_metrics import summarize_counts

MATCH_IOU = 0.5


def review_inputs(
    run: Path, corpus: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    """Relier le modèle terminé aux données revues avant de charger ses poids."""
    report, run_sha = read_document(run / "run.json")
    config, config_sha = load_box_config(config_path)
    if (
        report.get("status") != "completed"
        or report.get("model") != BOX_MODEL_NAME
        or report.get("class_names") != list(BOX_CLASSES)
        or report.get("config") != config
        or report.get("config_sha256") != config_sha
        or report.get("checkpoint_selection") != "last_epoch"
        or report.get("checkpoint") != "checkpoints/checkpoint_best_total.pth"
        or not isinstance(report.get("runtime"), dict)
        or not isinstance(report["runtime"].get("versions"), dict)
    ):
        raise ValueError("Expected a completed box pilot with unchanged configuration")
    selection = inspect_box_corpus(corpus, config)
    stored, _ = read_document(run / "data/selection.json")
    if selection != stored or not 1 <= len(selection["splits"]["valid"]["records"]) <= 64:
        raise ValueError("Review requires the original bounded validation selection")
    verify_training_data(run, report)
    metrics = run / "checkpoints/metrics.csv"
    checkpoint = run / report["checkpoint"]
    if (
        checkpoint.parent.is_symlink()
        or file_hash(checkpoint) != report.get("checkpoint_sha256")
        or file_hash(metrics) != report.get("metrics_sha256")
    ):
        raise ValueError("Completed checkpoint or metrics changed")
    summary = summarize_metrics(metrics, segmentation=False)
    if (
        summary != report.get("last_epoch_metrics")
        or summary["last_validation_epoch"] + 1 != config["epochs"]
    ):
        raise ValueError("Missing or inconsistent final validation")
    return report, selection, checkpoint, run_sha


def load_box_engine(
    checkpoint: Path, device: str, box_classes: tuple[str, ...] = BOX_CLASSES
) -> AlignedPredictor:
    """Le chargement sûr reprend les paramètres des poids, sans entraînement.

    Les classes sont un paramètre : elles doivent correspondre à celles inscrites
    dans le checkpoint, et un pilote entraîné sur un autre corpus en porte d'autres.
    """
    engine = importlib.import_module("rfdetr").RFDETR.from_checkpoint(
        str(checkpoint.resolve()), device=device, trust_checkpoint=False
    )
    if type(engine).__name__ != BOX_MODEL_NAME:
        raise ValueError("Expected the RF-DETR Nano box model")
    # Réutiliser le redimensionnement de validation évite d'introduire
    # un changement de pixels
    return AlignedPredictor(engine, box_classes=box_classes)


def encode_boxes(detections: Any, photo: Image.Image) -> dict[str, Any]:
    """Borner les sorties et conserver les coordonnées brutes si elles débordent."""
    boxes, scores, labels = detections.xyxy, detections.confidence, detections.class_id
    count = len(boxes)
    if (
        count > 500
        or scores is None
        or labels is None
        or detections.mask is not None
        or boxes.shape != (count, 4)
        or scores.shape != (count,)
        or labels.shape != (count,)
    ):
        raise ValueError("Unexpected box prediction dimensions")
    records, discarded = [], []
    names = getattr(detections, "data", {}).get("class_name")
    if names is not None and len(names) != count:
        raise ValueError("Prediction class names differ from labels")
    for index in range(count):
        label = number(float(labels[index]))
        score = number(float(scores[index]))
        raw = [number(float(value)) for value in boxes[index]]
        if not 0 <= score <= 1 or int(label) != label or raw[0] > raw[2] or raw[1] > raw[3]:
            raise ValueError("Invalid prediction class, score or box")
        record: dict[str, Any] = {"id": index + 1, "score": score, "raw_bbox_xyxy": raw}
        # Le model peut exposer la classe sans objet.. Elle reste compté dans
        # les sorties écartées, et ne devient jamais une anomalie
        if label == len(BOX_CLASSES) and names is not None and names[index] == "__background__":
            discarded.append({**record, "reason": "background"})
            continue
        if not 0 <= label < len(BOX_CLASSES):
            raise ValueError("Unknown prediction class")
        name = BOX_CLASSES[int(label)]
        if names is not None and names[index] != name:
            raise ValueError("Prediction class names differ from labels")
        record.update(class_id=int(label), class_name=name)
        clipped = np.clip(raw, 0, [photo.width, photo.height, photo.width, photo.height]).tolist()
        if clipped[0] == clipped[2] or clipped[1] == clipped[3]:
            discarded.append({**record, "reason": "empty_after_clipping"})
            continue
        record.update(
            bbox_xyxy=checked_box(clipped, photo.width, photo.height), clipped=raw != clipped
        )
        records.append(record)
    return {"predictions": records, "discarded": discarded}


def compare_boxes(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Une mauvaise classe ne valide pas une référence, même au bon endroit."""
    per_class, matches = {}, []
    for name in BOX_CLASSES:
        gt = [r for r in references if r["class_name"] == name]
        dt = [p for p in predictions if p["class_name"] == name]
        pairs = match_boxes([r["bbox_xyxy"] for r in gt], dt, MATCH_IOU)
        found = {m["reference_index"] for m in pairs if m["reference_index"] is not None}
        per_class[name] = {"tp": len(found), "fp": len(dt) - len(found), "fn": len(gt) - len(found)}
        for match in pairs:
            reference = match["reference_index"]
            matches.append(
                {
                    **match,
                    "class_name": name,
                    "prediction_id": dt[match["prediction_index"]]["id"],
                    "reference_id": gt[reference]["id"] if reference is not None else None,
                }
            )
    found_ids = {m["reference_id"] for m in matches if m["reference_id"] is not None}
    return {
        "counts": per_class,
        "matches": matches,
        "missed_reference_ids": [r["id"] for r in references if r["id"] not in found_ids],
    }


def review_boxes(
    run: Path, corpus: Path, config_path: Path, output: Path, device: str, threshold: float = 0.3
) -> dict[str, Any]:
    """Traiter uniquement la validation, une photo à la fois, puis publier la galerie complète."""
    if not 0 < number(threshold) < 1:
        raise ValueError("Review score threshold must be between zero and one")
    parent, selection, checkpoint, run_sha = review_inputs(run, corpus, config_path)
    environment = runtime(device)
    if environment["versions"] != parent["runtime"]["versions"]:
        raise ValueError("Review runtime differs from the trained pilot")
    started = time.perf_counter()
    with staged_output(output, [run, corpus, config_path]) as stage:
        engine = load_box_engine(checkpoint, device)
        rows = []
        for source in selection["splits"]["valid"]["records"]:
            raw = read_local(run / "data", source["image"], MAX_FILE_BYTES)
            if digest(raw) != source["image_sha256"]:
                raise ValueError("Validation image changed during review")
            photo = decode_photo(raw)
            encoded = encode_boxes(engine.predict(photo, threshold=threshold), photo)
            if any(p["score"] < threshold for p in encoded["predictions"]):
                raise ValueError("Engine returned a prediction below the requested threshold")
            references = []
            for index, box in enumerate(source["boxes"], 1):
                coords = [
                    v * bound
                    for v, bound in zip(
                        box["xyxy_normalized"],
                        (photo.width, photo.height, photo.width, photo.height),
                        strict=True,
                    )
                ]
                references.append(
                    {
                        "id": index,
                        "class_name": box["class"],
                        "bbox_xyxy": checked_box(coords, photo.width, photo.height),
                    }
                )
            row = {
                "id": source["id"],
                "group": source["group"],
                "credit": source["credit"],
                "annotation_note": source["note"],
                "image_sha256": source["image_sha256"],
                "width": photo.width,
                "height": photo.height,
                "references": references,
                **encoded,
                **compare_boxes(references, encoded["predictions"]),
            }
            folder = stage / source["id"]
            folder.mkdir()
            (folder / "photo.jpg").write_bytes(raw)
            write_json(folder / "result.json", row)
            rows.append(row)
            print(
                f"Photo {len(rows)}/{selection['splits']['valid']['images']} : {source['id']}",
                flush=True,
            )
        del engine
        # Un changement en cours de calcul invalide la livraison entière.
        if (
            file_hash(checkpoint) != parent["checkpoint_sha256"]
            or file_hash(run / "run.json") != run_sha
        ):
            raise ValueError("Training run changed during review")
        verify_training_data(run, parent)
        report = {
            "schema_version": 1,
            "status": "completed",
            "training_started": False,
            "qualified_for_smartsite": False,
            "human_validation": "pending",
            "test_used": False,
            "run_sha256": run_sha,
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "selection_sha256": parent["selection_sha256"],
            "training_epochs": parent["config"]["epochs"],
            "runtime": environment,
            "protocol": {
                "split": "valid",
                "score_threshold": threshold,
                "threshold_calibrated": False,
                "match_iou": MATCH_IOU,
                "matching": "same class, descending score, one-to-one",
                "preprocessing": "training-v1",
                "coordinates": "oriented image pixels",
            },
            "summary": summarize_counts([row["counts"] for row in rows], BOX_CLASSES),
            "records": rows,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(stage / "report.json", report)
        write_box_gallery(stage, report)
    return report
