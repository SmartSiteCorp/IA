"""Garder le même sens des catégories, du corpus jusqu'aux résultats du modèle."""

from collections.abc import Mapping
from typing import Any

CLASS_NAMES = ("crack", "surface_loss")
CLASS_LABELS = {"crack": "Fissures", "surface_loss": "Pertes de matière"}


def class_legend(names: tuple[str, ...]) -> str:
    labels = {"crack": "Rouge : fissure", "surface_loss": "Bleu : perte de matière"}
    return " · ".join(labels[name] for name in names)


def get_class_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Les anciens essais gardent leurs deux classes ; les nouveaux peuvent choisir."""
    names = config.get("class_names", list(CLASS_NAMES))
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or name not in CLASS_NAMES for name in names)
        or names != [name for name in CLASS_NAMES if name in names]
    ):
        raise ValueError("Expected nonempty, unique SmartSite class names in canonical order")
    return tuple(names)


def project_categories(document: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """Filtrer les annotations, jamais les photos ni les contours des classes gardées."""
    get_class_names({"class_names": list(names)})
    if any(type(c["id"]) is not int for c in document["categories"]) or [
        (c["id"], c["name"]) for c in document["categories"]
    ] != list(enumerate(CLASS_NAMES, 1)):
        raise ValueError("Unexpected source COCO class mapping")
    # Une catégorie inconnue doit declancher une erreur, pas disparaître du rapport
    if any(
        type(a["category_id"]) is not int or a["category_id"] not in (1, 2)
        for a in document["annotations"]
    ):
        raise ValueError("Unknown source COCO annotation category")
    image_ids = [image["id"] for image in document["images"]]
    if any(type(i) is not int or i < 0 for i in image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("Invalid or duplicate source COCO image ID")
    known_images, seen = set(image_ids), set()
    for annotation in document["annotations"]:
        if (
            type(annotation["id"]) is not int
            or annotation["id"] in seen
            or type(annotation["image_id"]) is not int
            or annotation["image_id"] not in known_images
            or annotation.get("iscrowd", 0) != 0
        ):
            raise ValueError("Unsupported or inconsistent source COCO annotation")
        seen.add(annotation["id"])
    mapping = {CLASS_NAMES.index(name) + 1: index for index, name in enumerate(names, 1)}
    return {
        **document,
        "categories": [
            {**category, "id": mapping[category["id"]]}
            for category in document["categories"]
            if category["id"] in mapping
        ],
        "annotations": [
            {**annotation, "category_id": mapping[annotation["category_id"]]}
            for annotation in document["annotations"]
            if annotation["category_id"] in mapping
        ],
    }
