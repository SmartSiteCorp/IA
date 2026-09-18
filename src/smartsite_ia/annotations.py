"""Validate the observed DamSegment JSON and YOLO polygon formats."""

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Instance:
    source_class: int
    polygons: list[list[float]]
    source_bbox: list[float]


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite number, not a boolean or string")
    return float(value)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_annotations(
    content: bytes, yolo: bytes, filename: str, size: tuple[int, int]
) -> list[Instance]:
    data = json.loads(content, object_pairs_hook=unique_object)
    if not isinstance(data, dict) or set(data) != {"image", "annotations"}:
        raise ValueError("Unexpected annotation document")
    width, height = size
    image = data["image"]
    if not isinstance(image, dict) or image != {
        "file_name": filename,
        "width": width,
        "height": height,
    }:
        raise ValueError("Annotation image identity or dimensions do not match the image")
    annotations = data["annotations"]
    rows = yolo.decode("utf-8").splitlines()
    if not isinstance(annotations, list) or len(annotations) > 10000:
        raise ValueError("Invalid annotations list")
    if len(rows) != len(annotations):
        raise ValueError("YOLO and JSON instance counts differ")
    result = []
    for annotation, row in zip(annotations, rows, strict=True):
        if not isinstance(annotation, dict) or set(annotation) != {
            "category_id",
            "segmentation",
            "bbox",
            "iscrowd",
        }:
            raise ValueError("Unexpected instance fields")
        category = annotation["category_id"]
        if type(category) is not int or category not in (0, 1):
            raise ValueError("Unknown source class ID")
        if type(annotation["iscrowd"]) is not int or annotation["iscrowd"] != 0:
            raise ValueError("Unexpected crowd annotation")
        raw_polygons = annotation["segmentation"]
        if not isinstance(raw_polygons, list) or len(raw_polygons) != 1:
            raise ValueError("Expected one polygon per source instance")
        raw = raw_polygons[0]
        if not isinstance(raw, list) or len(raw) < 6 or len(raw) % 2 or len(raw) > 20000:
            raise ValueError("Invalid polygon length")
        polygon = [number(v) for v in raw]
        if any(v < 0 or v >= (width if i % 2 == 0 else height) for i, v in enumerate(polygon)):
            raise ValueError("Polygon coordinates outside the image")
        if len(set(zip(polygon[::2], polygon[1::2], strict=True))) < 3:
            raise ValueError("Polygon needs at least three distinct points")
        raw_bbox = annotation["bbox"]
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError("Invalid bounding box")
        bbox = [number(v) for v in raw_bbox]
        x, y, w, h = bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError("Bounding box outside the image or empty")
        xs, ys = polygon[::2], polygon[1::2]
        if min(xs) < x or min(ys) < y or max(xs) > x + w or max(ys) > y + h:
            raise ValueError("Bounding box does not enclose its polygon")
        fields = row.split()
        if len(fields) != len(polygon) + 1 or fields[0] != str(category):
            raise ValueError("YOLO polygon layout or category differs from JSON")
        normalized = [number(float(v)) for v in fields[1:]]
        if any(v < 0 or v > 1 for v in normalized):
            raise ValueError("YOLO coordinates must be normalized")
        if any(
            abs(a - b * (width if i % 2 == 0 else height)) > 1.000001
            for i, (a, b) in enumerate(zip(polygon, normalized, strict=True))
        ):
            raise ValueError("YOLO and JSON polygons differ by more than one pixel")
        result.append(Instance(category, [polygon], bbox))
    return result
