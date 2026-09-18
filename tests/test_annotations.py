import json

import pytest

from smartsite_ia.annotations import number, parse_annotations


def parse(sample):
    return parse_annotations(
        json.dumps(sample["document"]).encode(), sample["yolo"], "E (1).jpg", (640, 640)
    )


def test_valid_polygon_retains_source(sample):
    result = parse(sample)
    assert result[0].source_class == 0
    assert result[0].source_bbox == [10, 10, 20, 10]
    assert result[0].polygons[0] == sample["document"]["annotations"][0]["segmentation"][0]


def test_padded_source_box_is_valid(sample):
    sample["document"]["annotations"][0]["bbox"] = [8, 8, 24, 14]
    assert parse(sample)[0].source_bbox == [8, 8, 24, 14]


@pytest.mark.parametrize(
    "field,value",
    [
        ("category_id", 2),
        ("category_id", True),
        ("iscrowd", 1),
        ("bbox", [20, 10, 20, 10]),
        ("bbox", [0, 0, 0, 4]),
        ("bbox", [-1, 0, 10, 10]),
        ("bbox", [0, 0, 900, 900]),
        ("bbox", []),
        ("segmentation", []),
        ("segmentation", [[1, 2, 3]]),
        ("segmentation", [[1, 1, 1, 1, 1, 1]]),
        ("segmentation", [[0, 0, 640, 0, 20, 20]]),
        ("segmentation", [[0, 0, float("nan"), 0, 20, 20]]),
    ],
)
def test_invalid_annotation_rejected(sample, field, value):
    sample["document"]["annotations"][0][field] = value
    with pytest.raises(ValueError):
        parse(sample)


@pytest.mark.parametrize("value", [True, None, "5", float("inf"), float("nan")])
def test_invalid_numbers(value):
    with pytest.raises(ValueError):
        number(value)


@pytest.mark.parametrize("text", ["", "1 0 0", "0 0 0 1 0 1 1 0 1 0 0", "0 nan 0"])
def test_yolo_inconsistency_rejected(sample, text):
    sample["yolo"] = text.encode()
    with pytest.raises(ValueError):
        parse(sample)


def test_wrong_image_reference(sample):
    sample["document"]["image"]["file_name"] = "other.jpg"
    with pytest.raises(ValueError, match="identity"):
        parse(sample)


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_annotations(b'{"image":{},"image":{}}', b"", "x.jpg", (640, 640))


def test_empty_annotations_preserved(sample):
    sample["document"]["annotations"] = []
    sample["yolo"] = b""
    assert parse(sample) == []
