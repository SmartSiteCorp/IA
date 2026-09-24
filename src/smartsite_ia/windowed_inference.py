"""Analyser une photo entière par fenêtres, puis recoller les résultats.

Une façade de douze mégapixels ramenée à 432 pixels perd ses fissures fines. On
découpe donc la photo en fenêtres à la taille des images d'apprentissage, on prédit
sur chacune, et on remet tout dans les coordonnées de la photo d'origine.

Deux sorties séparées, parce qu'elles ne coûtent pas la même chose en mémoire :
la zone couverte par classe, accumulée dans un seul masque, et la liste des
propositions réduite à leur boîte et leur score. Garder un masque pleine taille par
proposition ferait exploser la mémoire sur une grande photo.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

# La taille des images d'apprentissage : une fenêtre reproduit ces conditions.
DEFAULT_WINDOW = 640
# Un défaut coupé par une frontière doit rester entier dans la fenêtre voisine.
DEFAULT_OVERLAP = 128
# Deux propositions qui se recouvrent autant décrivent le même défaut vu deux fois.
MERGE_OVERLAP = 0.3
MAX_WINDOWS = 400


def window_bounds(
    width: int, height: int, window: int = DEFAULT_WINDOW, overlap: int = DEFAULT_OVERLAP
) -> list[tuple[int, int, int, int]]:
    """Couvrir toute la photo, sans fenêtre vide ni débordement hors des bords.

    La dernière fenêtre d'une ligne ou d'une colonne est collée au bord plutôt que
    rognée : elle garde ainsi la taille attendue par le moteur, quitte à recouvrir
    un peu plus que le pas choisi.
    """
    if window < 32 or overlap < 0 or overlap >= window:
        raise ValueError("Window must be at least 32 px, with a smaller overlap")
    step = window - overlap
    bounds = [
        (left, top, min(left + window, width), min(top + window, height))
        for top in offsets(height, window, step)
        for left in offsets(width, window, step)
    ]
    if not 1 <= len(bounds) <= MAX_WINDOWS:
        raise ValueError("Empty or oversized window grid for this photo")
    return bounds


def offsets(size: int, window: int, step: int) -> list[int]:
    """Les positions de départ sur un côté, la dernière calée sur le bord."""
    if size <= window:
        return [0]
    positions = list(range(0, size - window + 1, step))
    if positions[-1] != size - window:
        positions.append(size - window)
    return positions


def windows(photo: Image.Image, bounds: list[tuple[int, int, int, int]]) -> Iterator[Image.Image]:
    """Découper sans modifier la photo d'origine, qui reste la preuve conservée."""
    for box in bounds:
        yield photo.crop(box)


def place_mask(mask_rle: dict[str, Any], box: tuple[int, int, int, int], zone: Any) -> None:
    """Reporter le masque d'une fenêtre dans la zone accumulée de la photo entière."""
    left, top, right, bottom = box
    decoded = coco_mask.decode(normalized(mask_rle)).astype(bool)
    if decoded.shape != (bottom - top, right - left):
        raise ValueError("Window mask does not match its window")
    zone[top:bottom, left:right] |= decoded


def normalized(mask_rle: dict[str, Any]) -> dict[str, Any]:
    """pycocotools accepte les deux formes de `counts` ; on n'en garde qu'une."""
    counts = mask_rle["counts"]
    return {
        "size": mask_rle["size"],
        "counts": counts.encode() if isinstance(counts, str) else counts,
    }


def shift_box(box_xyxy: list[float], window: tuple[int, int, int, int]) -> list[float]:
    """Remettre une boîte de fenêtre dans les coordonnées de la photo d'origine."""
    left, top = window[0], window[1]
    return [box_xyxy[0] + left, box_xyxy[1] + top, box_xyxy[2] + left, box_xyxy[3] + top]


def merge_proposals(
    proposals: list[dict[str, Any]], overlap: float = MERGE_OVERLAP
) -> list[dict[str, Any]]:
    """Regrouper les propositions d'une même classe qui décrivent le même défaut.

    Le critère est la part recouverte de la plus petite des deux boîtes, et non
    l'IoU : quand une fenêtre voit un défaut en entier et sa voisine seulement un
    morceau, l'IoU reste faible alors qu'il s'agit bien du même défaut.
    """
    if not 0 < overlap <= 1:
        raise ValueError("Merge overlap must sit in ]0, 1]")
    kept: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=lambda row: row["score"], reverse=True):
        match = next(
            (
                row
                for row in kept
                if row["class_name"] == proposal["class_name"]
                and small_box_overlap(row["bbox_xyxy"], proposal["bbox_xyxy"]) >= overlap
            ),
            None,
        )
        if match is None:
            kept.append({**proposal, "merged_from": 1})
            continue
        # La boîte gardée est celle du meilleur score, élargie au défaut complet.
        match["bbox_xyxy"] = union_box(match["bbox_xyxy"], proposal["bbox_xyxy"])
        match["merged_from"] += 1
    return kept


def small_box_overlap(first: list[float], second: list[float]) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (first, second)]
    smallest = min(areas)
    return width * height / smallest if smallest > 0 else 0.0


def union_box(first: list[float], second: list[float]) -> list[float]:
    return [
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    ]


def encode_zone(zone: Any) -> dict[str, Any]:
    """Rendre la zone accumulée au même format que les autres masques du projet."""
    encoded: dict[str, Any] = coco_mask.encode(np.asfortranarray(zone.astype(np.uint8)))
    return encoded


def empty_zone(photo: Image.Image) -> Any:
    return np.zeros((photo.height, photo.width), dtype=bool)
