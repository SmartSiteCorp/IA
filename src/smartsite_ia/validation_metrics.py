"""Mesurer les masques avec COCO et compter les erreurs à un seuil fixé à l'avance."""

import contextlib
import copy
import io
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from smartsite_ia.model_assets import CLASS_NAMES


def coco_api(document: dict[str, Any]) -> Any:
    # L'API imprime beaucoup de détails ; notre commande garde son résumé lisible.
    api = COCO()
    api.dataset = document
    with contextlib.redirect_stdout(io.StringIO()):
        api.createIndex()
    return api


def coco_scores(document: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """AP de segmentation officielle, sans boîtes qui fausseraient l'aire des masques."""
    gt = coco_api(document)
    with contextlib.redirect_stdout(io.StringIO()):
        # COCO.loadRes refuse une liste vide : on crée alors un résultat sans instance.
        dt = (
            gt.loadRes(copy.deepcopy(predictions))
            if predictions
            else coco_api(
                {
                    "images": document["images"],
                    "categories": document["categories"],
                    "annotations": [],
                }
            )
        )
        evaluator = COCOeval(gt, dt, "segm")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    result: dict[str, Any] = {}
    for index, category in enumerate(sorted(document["categories"], key=lambda c: c["id"])):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        recall = evaluator.eval["recall"][:, index, 0, -1]
        ap50 = precision[0]
        result[category["name"]] = {
            "mask_ap_50_95": finite_mean(precision),
            "mask_ap_50": finite_mean(ap50),
            "mask_ar_100": finite_mean(recall),
        }
    return {"mask_map_50_95": float(evaluator.stats[0]), "per_class": result}


def finite_mean(values: Any) -> float | None:
    valid = np.asarray(values)[np.asarray(values) >= 0]
    return float(valid.mean()) if valid.size else None


def counts_for_image(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float
) -> dict[str, dict[str, int]]:
    return {
        name: row["counts"]
        for name, row in analyze_masks(references, predictions, threshold).items()
    }


def analyze_masks(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Un défaut ne peut compter qu'une fois, même si plusieurs propositions le couvrent."""
    result = {}
    for label, name in enumerate(class_names, 1):
        gt = [r["segmentation"] for r in references if r["category_id"] == label]
        dt = [r for r in predictions if r["category_id"] == label and r["score"] >= threshold]
        matches = match_masks(gt, dt)
        errors = {"duplicate": 0, "insufficient_overlap": 0, "no_overlap": 0}
        for match in matches:
            if match["reason"] != "matched":
                errors[match["reason"]] += 1
        tp = len(matches) - sum(errors.values())
        result[name] = {
            "counts": {"tp": tp, "fp": len(dt) - tp, "fn": len(gt) - tp},
            "errors": errors,
        }
    return result


def match_masks(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apparier une seule classe ; garder les indices pour expliquer chaque alerte."""
    # Le tri reste stable en cas de scores égaux, comme dans les anciens rapports
    ordered = sorted(enumerate(predictions), key=lambda item: item[1]["score"], reverse=True)
    overlaps = (
        coco_mask.iou([p["segmentation"] for _, p in ordered], references, [0] * len(references))
        if predictions and references
        else np.zeros((len(predictions), len(references)))
    )
    matched: set[int] = set()
    result = []
    for (index, _), row in zip(ordered, overlaps, strict=True):
        candidates = [i for i in range(len(references)) if i not in matched and row[i] >= 0.5]
        reference = max(candidates, key=lambda i: row[i]) if candidates else None
        best = float(row.max()) if len(row) else 0.0
        if reference is not None:
            matched.add(reference)
            reason = "matched"
        else:
            reason = (
                "duplicate" if best >= 0.5 else "insufficient_overlap" if best > 0 else "no_overlap"
            )
        result.append(
            {
                "prediction_index": index,
                "reference_index": reference,
                "reason": reason,
                "best_iou": best,
            }
        )
    return result


def summarize_counts(
    rows: list[dict[str, Any]], class_names: tuple[str, ...] = CLASS_NAMES
) -> dict[str, Any]:
    result = {}
    for name in class_names:
        counts = {key: sum(row[name][key] for row in rows) for key in ("tp", "fp", "fn")}
        tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
        result[name] = {
            **counts,
            "reference_instances": tp + fn,
            # Une précision sans aucune proposition est indéfinie, pas parfaite.
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_proposals_per_image": fp / len(rows) if rows else None,
        }
    return result


def coverage_for_image(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, dict[str, Any]]:
    """Mesurer le service rendu, en plus de l'appariement un défaut pour un défaut.

    Sur un chantier, l'utilité d'une alerte est d'attirer l'œil sur la bonne zone.
    L'appariement un-à-un compte une fissure annotée en trois morceaux comme deux
    manques, même quand la zone est bien couverte. On mesure donc aussi, par classe,
    si la photo est signalée et quelle part de la zone attendue est couverte.
    Cette mesure complète la mesure stricte ; elle ne la remplace pas.
    """
    result = {}
    for label, name in enumerate(class_names, 1):
        gt = [r["segmentation"] for r in references if r["category_id"] == label]
        dt = [
            r["segmentation"]
            for r in predictions
            if r["category_id"] == label and r["score"] >= threshold
        ]
        result[name] = {
            "expected": bool(gt),
            "signalled": bool(dt),
            "pixels": shared_pixels(gt, dt),
        }
    return result


def shared_pixels(references: list[Any], predictions: list[Any]) -> dict[str, int]:
    """Fusionner puis croiser les masques sans jamais allouer l'image entière."""
    reference = coco_mask.merge(references) if references else None
    proposal = coco_mask.merge(predictions) if predictions else None
    shared = (
        coco_mask.merge([reference, proposal], intersect=1)
        if reference is not None and proposal is not None
        else None
    )
    return {
        "reference": int(coco_mask.area(reference)) if reference is not None else 0,
        "proposal": int(coco_mask.area(proposal)) if proposal is not None else 0,
        "shared": int(coco_mask.area(shared)) if shared is not None else 0,
    }


def summarize_coverage(
    rows: list[dict[str, dict[str, Any]]], class_names: tuple[str, ...] = CLASS_NAMES
) -> dict[str, Any]:
    """Regrouper les photos, en gardant séparés le niveau photo et le niveau zone."""
    result = {}
    for name in class_names:
        per_class = [row[name] for row in rows]
        expected = [row for row in per_class if row["expected"]]
        found = [row for row in expected if row["signalled"]]
        signalled = [row for row in per_class if row["signalled"]]
        without = [row for row in signalled if not row["expected"]]
        pixels = {
            key: sum(row["pixels"][key] for row in per_class)
            for key in ("reference", "proposal", "shared")
        }
        result[name] = {
            "photos": len(per_class),
            "photos_expected": len(expected),
            "photos_found": len(found),
            "photos_signalled": len(signalled),
            "photos_signalled_without_reference": len(without),
            "photo_recall": len(found) / len(expected) if expected else None,
            "photo_precision": len(found) / len(signalled) if signalled else None,
            "reference_pixels": pixels["reference"],
            "proposal_pixels": pixels["proposal"],
            "shared_pixels": pixels["shared"],
            "zone_recall": (
                pixels["shared"] / pixels["reference"] if pixels["reference"] else None
            ),
            "zone_precision": (
                pixels["shared"] / pixels["proposal"] if pixels["proposal"] else None
            ),
        }
    return result
