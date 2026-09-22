"""Relire les sorties locales en contrôlant les masques avant tout appel au code natif."""

import json
from pathlib import Path
from typing import Any

from smartsite_ia.annotations import number, unique_object
from smartsite_ia.curation import digest
from smartsite_ia.review import MAX_JSON_BYTES, read_local
from smartsite_ia.validation import SCORE_FLOOR


def validate_rle(value: Any, size: tuple[int, int]) -> None:
    """Compter les pixels codés sans allouer le masque ni faire confiance à sa taille."""
    width, height = size
    if (
        type(width) is not int
        or type(height) is not int
        or min(width, height) < 32
        or width * height > 16_000_000
    ):
        raise ValueError("Invalid saved mask size")
    if not isinstance(value, dict) or value.get("size") != [height, width]:
        raise ValueError("Saved mask dimensions disagree with the photo")
    encoded = value.get("counts")
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 1_000_000:
        raise ValueError("Invalid or oversized compressed mask")
    # Le format COCO code des longueurs signées sur des blocs de cinq bits
    # et des écarts à la longueur située deux positions avant
    previous = [0, 0]
    total = runs = offset = 0
    while offset < len(encoded):
        length = shift = 0
        while True:
            if offset >= len(encoded) or shift >= 30:
                raise ValueError("Truncated or excessive compressed mask run")
            code = ord(encoded[offset]) - 48
            offset += 1
            if not 0 <= code <= 63:
                raise ValueError("Invalid compressed mask character")
            length |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    length -= 1 << shift
                break
        if runs > 2:
            length += previous[runs % 2]
        if length < 0 or (runs > 0 and length == 0) or total + length > width * height:
            raise ValueError("Compressed mask runs exceed the photo")
        previous[runs % 2] = length
        total += length
        runs += 1
    if total != width * height:
        raise ValueError("Compressed mask does not cover the photo")


def load_predictions(
    root: Path, sample_id: str, size: tuple[int, int], names: tuple[str, ...]
) -> tuple[list[dict[str, Any]], str]:
    """Le chemin vient du corpus vérifié ; les scores et coordonnées sont recontrôlés."""
    raw = read_local(root, f"{sample_id}/predictions.json", MAX_JSON_BYTES)
    doc = json.loads(raw, object_pairs_hook=unique_object)
    if (
        not isinstance(doc, dict)
        or type(doc.get("schema_version")) is not int
        or doc["schema_version"] != 1
    ):
        raise ValueError("Expected saved predictions schema version 1")
    records = doc.get("predictions")
    if not isinstance(records, list) or len(records) > 200:
        raise ValueError("Invalid saved prediction count")
    seen = set()
    for p in records:
        if not isinstance(p, dict):
            raise ValueError("Invalid saved prediction")
        label, identifier = p.get("class_id"), p.get("id")
        if (
            type(label) is not int
            or not 0 <= label < len(names)
            or p.get("class_name") != names[label]
            or type(identifier) is not int
            or not 1 <= identifier <= 200
            or identifier in seen
        ):
            raise ValueError("Invalid saved prediction class or ID")
        seen.add(identifier)
        if not SCORE_FLOOR <= number(p.get("score")) <= 1:
            raise ValueError("Saved score is outside the evaluation range")
        box = p.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Invalid saved prediction box")
        left, top, right, bottom = map(number, box)
        if not 0 <= left <= right <= size[0] or not 0 <= top <= bottom <= size[1]:
            raise ValueError("Saved box is outside the photo")
        validate_rle(p.get("mask_rle"), size)
    return records, digest(raw)
