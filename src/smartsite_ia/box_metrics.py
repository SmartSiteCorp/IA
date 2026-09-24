"""Comparer des rectangles, en gardant les appariements pour expliquer les erreurs."""

from typing import Any

import numpy as np
from pycocotools import mask as coco_mask

from smartsite_ia.annotations import number


def checked_box(raw: object, width: int, height: int) -> list[float]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("Expected an XYXY box")
    box = [number(value) for value in raw]
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("Box is empty or outside the oriented image")
    return box


def match_boxes(
    references: list[list[float]], predictions: list[dict[str, Any]], overlap: float
) -> list[dict[str, Any]]:
    """Une seule classe à la fois, priorité au score puis meilleur recouvrement libre."""
    if not 0 < number(overlap) <= 1:
        raise ValueError("Invalid matching overlap")
    ordered = sorted(enumerate(predictions), key=lambda item: item[1]["score"], reverse=True)

    def xywh(box: list[float]) -> list[float]:
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    ious = (
        coco_mask.iou(
            [xywh(p["bbox_xyxy"]) for _, p in ordered],
            [xywh(box) for box in references],
            [0] * len(references),
        )
        if references and ordered
        else np.zeros((len(ordered), len(references)))
    )
    used: set[int] = set()
    result = []
    for (index, _), row in zip(ordered, ious, strict=True):
        candidates = [i for i in range(len(references)) if i not in used and row[i] >= overlap]
        reference = max(candidates, key=lambda i: row[i]) if candidates else None
        best = float(row.max()) if len(row) else 0.0
        if reference is not None:
            used.add(reference)
        result.append(
            {
                "prediction_index": index,
                "reference_index": reference,
                "matched_iou": float(row[reference]) if reference is not None else None,
                "best_iou": best,
                "reason": "matched"
                if reference is not None
                else (
                    "duplicate"
                    if best >= overlap
                    else "insufficient_overlap"
                    if best > 0
                    else "no_overlap"
                ),
            }
        )
    return result


def box_counts(
    references: list[list[float]], predictions: list[dict[str, Any]], overlap: float
) -> dict[str, int]:
    """Compter l'accord avec les annotations, pas une vérité terrain confirmée."""
    tp = sum(
        m["reference_index"] is not None for m in match_boxes(references, predictions, overlap)
    )
    return {"tp": tp, "fp": len(predictions) - tp, "fn": len(references) - tp}
