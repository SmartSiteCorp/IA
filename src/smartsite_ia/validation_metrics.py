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
    for index, name in enumerate(CLASS_NAMES):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        recall = evaluator.eval["recall"][:, index, 0, -1]
        ap50 = precision[0]
        result[name] = {
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
    """Un défaut ne peut compter qu'une fois, même si plusieurs propositions le couvrent."""
    result = {}
    for label, name in enumerate(CLASS_NAMES, 1):
        gt = [r["segmentation"] for r in references if r["category_id"] == label]
        dt = sorted(
            [r for r in predictions if r["category_id"] == label and r["score"] >= threshold],
            key=lambda r: r["score"],
            reverse=True,
        )
        matched: set[int] = set()
        if dt and gt:
            overlaps = coco_mask.iou([r["segmentation"] for r in dt], gt, [0] * len(gt))
            for row in overlaps:
                candidates = [i for i in range(len(gt)) if i not in matched and row[i] >= 0.5]
                if candidates:
                    matched.add(max(candidates, key=lambda i: row[i]))
        tp = len(matched)
        result[name] = {"tp": tp, "fp": len(dt) - tp, "fn": len(gt) - tp}
    return result


def summarize_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in CLASS_NAMES:
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
