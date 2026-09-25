"""Lire CUBIT-Det et en tirer des groupes de scène et des partitions propres au projet.

Les auteurs publient leurs propres listes train, val et test, mais la recherche de
quasi-doublons du 24 septembre a montré qu'elles partagent des scènes : la même dalle
apparaît dans les trois. On reconstruit donc les groupes à partir de l'empreinte
perceptuelle, puis les partitions avec la règle déjà utilisée par le projet.

Le numéro de fichier n'est pas utilisé comme indice de scène : la mesure montre que
deux photos qui se suivent ne se ressemblent presque jamais.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from smartsite_ia.curation import digest, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.similarity import close_pairs, groups_from_pairs


CUBIT_CLASSES = ("crack", "surface_loss", "moisture_trace")
AUTHOR_CLASS_NAMES = ("Crack", "Spalling", "Moisture")

SCENE_DISTANCE = 8
MAX_BOXES_PER_IMAGE = 200


def parse_yolo_labels(
    text: str, sample_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Lire un fichier de labels YOLO et séparer les boîtes inutilisables.

    Format attendu : une boîte par ligne, classe puis centre et taille normalisés.
    Un format illisible, une classe inconnue ou une boîte qui sort du cadre arrêtent
    la lecture : on ne corrige pas une annotation en silence.

    Les boîtes de surface nulle sont un cas à part. Le jeu publié en contient trois
    sur 16 792, toutes de largeur exactement zéro, dans des photos qui portent par
    ailleurs des annotations valides. Elles ne décrivent aucun défaut mesurable, donc
    elles sont rendues à part plutôt que gardées ou passées sous silence.
    """
    boxes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Unexpected YOLO line {number} in {sample_id}")
        try:
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError as error:
            raise ValueError(f"Invalid YOLO number on line {number} in {sample_id}") from error
        if not 0 <= class_id < len(CUBIT_CLASSES):
            raise ValueError(f"Unknown CUBIT class {class_id} in {sample_id}")
        cx, cy, width, height = values
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"Box outside the image on line {number} in {sample_id}")
        if width <= 0 or height <= 0:
            rejected.append({"line": number, "reason": "empty_box", "text": line.strip()})
            continue
        if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6:
            raise ValueError(f"Box starts outside the image on line {number} in {sample_id}")
        if cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6:
            raise ValueError(f"Box ends outside the image on line {number} in {sample_id}")
        boxes.append(
            {
                "class_id": class_id,
                "class_name": CUBIT_CLASSES[class_id],
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
            }
        )
    if len(boxes) > MAX_BOXES_PER_IMAGE:
        raise ValueError(f"Too many boxes in {sample_id}")
    return boxes, rejected


def build_scene_groups(
    hashes: Mapping[str, int], max_distance: int = SCENE_DISTANCE
) -> dict[str, str]:
    """Donner à chaque photo son groupe de scène ; une photo isolée a le sien."""
    linked = groups_from_pairs(close_pairs(dict(hashes), max_distance))
    groups: dict[str, str] = {}
    for members in linked:
        for name in members:
            groups[name] = members[0]
    for name in hashes:
        groups.setdefault(name, name)
    return groups


def build_records(
    photos: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, list[dict[str, Any]]],
    groups: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Préparer les fiches attendues par la répartition commune du projet.

    La strate reprend la résolution, parce que CUBIT mêle deux tailles de capteur,
    et l'ensemble des classes présentes, pour ne pas concentrer une classe rare
    dans une seule partition.
    """
    records = []
    for name in sorted(photos):
        if name not in groups:
            raise ValueError(f"Photo without a scene group: {name}")
        boxes = labels.get(name, [])
        photo = photos[name]
        records.append(
            {
                "id": name,
                "content_group": groups[name],
                "difficulty": f"{photo['width']}x{photo['height']}",
                "source_classes": sorted({box["class_name"] for box in boxes}),
                "boxes": len(boxes),
                "author_split": photo.get("split"),
            }
        )
    return records


def summarize_splits(records: list[dict[str, Any]], splits: Mapping[str, str]) -> dict[str, Any]:
    """Publier les effectifs obtenus, y compris ceux qui dérangent."""
    result: dict[str, Any] = {}
    for split in sorted(set(splits.values())):
        members = [r for r in records if splits[r["id"]] == split]
        classes: dict[str, int] = {}
        for record in members:
            for name in record["source_classes"]:
                classes[name] = classes.get(name, 0) + 1
        result[split] = {
            "images": len(members),
            "boxes": sum(r["boxes"] for r in members),
            "scene_groups": len({r["content_group"] for r in members}),
            "images_per_class": {name: classes.get(name, 0) for name in CUBIT_CLASSES},
            "images_without_box": sum(1 for r in members if r["boxes"] == 0),
        }
    crossing = crossing_groups(records, splits)
    result["scene_groups_crossing_splits"] = crossing
    if crossing:
        raise ValueError("A scene group was split across partitions")
    return result


def crossing_groups(records: list[dict[str, Any]], splits: Mapping[str, str]) -> list[str]:
    """Un groupe coupé entre deux partitions annulerait tout l'intérêt du regroupement."""
    seen: dict[str, set[str]] = {}
    for record in records:
        seen.setdefault(record["content_group"], set()).add(splits[record["id"]])
    return sorted(group for group, values in seen.items() if len(values) > 1)


def cubit_coco_document(
    records: list[dict[str, Any]],
    labels: Mapping[str, list[dict[str, Any]]],
    photos: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Convertir les rectangles YOLO en COCO, dans les pixels de la photo.

    YOLO donne un centre et une taille normalisés ; COCO attend le coin supérieur
    gauche puis la largeur et la hauteur en pixels. L'aire publiée est celle du
    rectangle, pas celle du défaut : une fissure en diagonale occupe une petite part
    de sa boîte, et confondre les deux fausserait toute mesure de surface.
    """
    categories = [
        {"id": index, "name": name, "supercategory": "building_defect"}
        for index, name in enumerate(CUBIT_CLASSES, 1)
    ]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for record in records:
        name = record["id"]
        photo = photos[name]
        width, height = photo["width"], photo["height"]
        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": f"{name}.jpg",
                "width": width,
                "height": height,
                "scene_group": record["content_group"],
                "license": 1,
            }
        )
        for box in labels.get(name, []):
            left = (box["cx"] - box["width"] / 2) * width
            top = (box["cy"] - box["height"] / 2) * height
            size = [box["width"] * width, box["height"] * height]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": box["class_id"] + 1,
                    "bbox": [left, top, size[0], size[1]],
                    "area": size[0] * size[1],
                    "iscrowd": 0,
                }
            )
    return {
        "info": {
            "description": (
                "CUBIT-Det boxes converted to COCO; project scene groups and splits, "
                "not the author splits"
            ),
            "source_classes": list(AUTHOR_CLASS_NAMES),
        },
        "licenses": [
            {"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
        ],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def attribution() -> dict[str, Any]:
    """La licence CC BY 4.0 impose de citer les auteurs partout où les photos vont."""
    return {
        "dataset": "CUBIT-Det",
        "authors": [
            "Benyun Zhao",
            "Xunkuai Zhou",
            "Guidong Yang",
            "Junjie Wen",
            "Jihan Zhang",
            "Jia Dou",
            "Guang Li",
            "Xi Chen",
            "Ben M. Chen",
        ],
        "paper": (
            "High-resolution infrastructure defect detection dataset sourced by unmanned "
            "systems and validated with deep learning, Automation in Construction 163 (2024) 105405"
        ),
        "doi": "10.1016/j.autcon.2024.105405",
        "source": "https://github.com/BenyunZhao/CUBIT",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "changes": (
            "Boxes converted from YOLO to COCO; project scene groups and splits replace the "
            "author lists; three zero-width boxes set aside. Photos are copied unchanged."
        ),
    }


MAX_CUBIT_PHOTO_BYTES = 32 * 1024 * 1024
SPLIT_FOLDERS = {"train": "train", "valid": "valid", "test": "test"}


def prepare_cubit_corpus(
    readers: Mapping[str, Any],
    destination: Path,
    photos: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    splits: Mapping[str, str],
    rejected: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Écrire le corpus préparé : photos inchangées, annotations converties, preuves.

    `readers` associe chaque partition d'origine à une fonction qui rend les octets
    d'une photo. Les octets sont recopiés tels quels, sans réencodage, pour que
    l'empreinte du corpus préparé reste celle de l'archive publiée.
    """
    if set(splits.values()) - set(SPLIT_FOLDERS):
        raise ValueError("Unknown split name in the assignment")
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_FOLDERS}
    for record in records:
        by_split[splits[record["id"]]].append(record)
    if any(not members for members in by_split.values()):
        raise ValueError("Every split must hold at least one photo")

    with staged_output(destination, []) as stage:
        manifest = []
        for split, members in by_split.items():
            folder = stage / SPLIT_FOLDERS[split]
            folder.mkdir(parents=True)
            for record in members:
                name = record["id"]
                photo = photos[name]
                raw = readers[photo["split"]](name)
                if len(raw) > MAX_CUBIT_PHOTO_BYTES:
                    raise ValueError(f"Photo above the size limit: {name}")
                if digest(raw) != photo["sha256"]:
                    raise ValueError(f"Photo differs from the inspected archive: {name}")
                (folder / f"{name}.jpg").write_bytes(raw)
                manifest.append(
                    {
                        "id": name,
                        "image": f"{SPLIT_FOLDERS[split]}/{name}.jpg",
                        "split": split,
                        "author_split": photo["split"],
                        "scene_group": record["content_group"],
                        "width": photo["width"],
                        "height": photo["height"],
                        "image_sha256": photo["sha256"],
                        "boxes": record["boxes"],
                    }
                )
            write_json(
                folder / "_annotations.coco.json",
                cubit_coco_document(members, labels, photos),
            )
        report = {
            "schema_version": 1,
            "dataset": "cubit_det_author_release",
            "usage": "prepared_external_candidate; no human approval and no final test yet",
            "approved_for_training": False,
            "approved_for_evaluation": False,
            "class_names": list(CUBIT_CLASSES),
            "images": len(records),
            "split_counts": {s: len(m) for s, m in by_split.items()},
            "scene_groups": len({r["content_group"] for r in records}),
            "rejected_boxes": dict(rejected),
            "limits": [
                "Author train/val/test lists share scenes and are not reused",
                "Scene groups come from a perceptual fingerprint: the same facade seen from "
                "another angle stays in a different group",
                "Boxes only; the area published is the rectangle, not the defect",
                "Author class order confirmed by visual inspection, not by the authors",
            ],
            "records": manifest,
        }
        write_json(stage / "report.json", report)
        write_json(stage / "manifest.json", {"schema_version": 1, "records": manifest})
        write_json(stage / "ATTRIBUTION.json", attribution())
    return report
