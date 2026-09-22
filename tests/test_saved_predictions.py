"""Refuser un cache invalide avant de transmettre ses masques à pycocotools."""

import copy

import numpy as np
import pytest
from pycocotools import mask as coco_mask

from smartsite_ia.importer import write_json
from smartsite_ia.saved_predictions import load_predictions, validate_rle


def encoded(pixels):
    result = coco_mask.encode(np.asfortranarray(pixels, dtype=np.uint8))
    result["counts"] = result["counts"].decode("ascii")
    return result


def test_rle_validation_matches_real_encoder_for_empty_full_and_irregular_masks():
    rng = np.random.default_rng(42)
    for pixels in [
        np.zeros((32, 64)),
        np.ones((32, 64)),
        *[rng.integers(0, 2, (32, 64)) for _ in range(10)],
    ]:
        validate_rle(encoded(pixels), (64, 32))
    with pytest.raises(ValueError, match="size"):
        validate_rle({}, (1, 1))


@pytest.mark.parametrize(
    "counts", [None, [], "", "P", "o" * 7, "!", "0", "O", "00", "PPPP1", "0" * 1_000_001]
)
def test_malformed_rle_never_reaches_native_decoder(counts):
    with pytest.raises(ValueError):
        validate_rle({"size": [32, 32], "counts": counts}, (32, 32))


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "list",
        "item",
        "too_many",
        "duplicate",
        "class",
        "bool",
        "name",
        "score",
        "low_score",
        "box",
        "outside",
        "size",
    ],
)
def test_saved_predictions_validate_metadata_and_bounds(tmp_path, change):
    folder = tmp_path / "easy_0001"
    folder.mkdir()
    proposal = {
        "id": 1,
        "class_id": 0,
        "class_name": "crack",
        "score": 0.3,
        "bbox_xyxy": [0, 0, 32, 32],
        "mask_rle": encoded(np.ones((32, 32))),
    }
    doc = {"schema_version": 1, "predictions": [proposal]}
    path = folder / "predictions.json"
    write_json(path, doc)
    assert load_predictions(tmp_path, "easy_0001", (32, 32), ("crack",))[0] == [proposal]
    if change == "schema":
        doc["schema_version"] = True
    elif change == "list":
        doc["predictions"] = None
    elif change == "item":
        doc["predictions"] = [None]
    elif change == "too_many":
        doc["predictions"] *= 201
    elif change == "duplicate":
        doc["predictions"].append(copy.deepcopy(proposal))
    elif change == "class":
        proposal["class_id"] = 1
    elif change == "bool":
        proposal["id"] = True
    elif change == "name":
        proposal["class_name"] = "unknown"
    elif change == "score":
        proposal["score"] = "NaN"
    elif change == "low_score":
        proposal["score"] = 0.0001
    elif change == "box":
        proposal["bbox_xyxy"] = []
    elif change == "outside":
        proposal["bbox_xyxy"] = [0, 0, 64, 32]
    else:
        proposal["mask_rle"]["size"] = [64, 64]
    write_json(path, doc)
    with pytest.raises(ValueError):
        load_predictions(tmp_path, "easy_0001", (32, 32), ("crack",))


def test_saved_cache_rejects_linked_folder_and_duplicate_json_keys(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    (folder / "predictions.json").write_text(
        '{"schema_version":1,"predictions":[],"predictions":[]}'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_predictions(tmp_path, "source", (32, 32), ("crack",))
    (tmp_path / "easy_0001").symlink_to(folder)
    with pytest.raises(ValueError, match="Linked"):
        load_predictions(tmp_path, "easy_0001", (32, 32), ("crack",))
