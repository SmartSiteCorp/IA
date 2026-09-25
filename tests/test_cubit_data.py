"""Vérifier la lecture des labels CUBIT, les groupes de scène et les partitions."""

import json

import pytest

from smartsite_ia import cubit_data
from smartsite_ia.cubit_data import (
    build_records,
    build_scene_groups,
    crossing_groups,
    parse_yolo_labels,
    summarize_splits,
)


def test_labels_are_read_with_their_class_names():
    boxes, rejected = parse_yolo_labels("0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.05 0.05\n", "hk0001")
    assert [b["class_name"] for b in boxes] == ["crack", "moisture_trace"]
    assert boxes[0]["cx"] == 0.5 and boxes[0]["width"] == 0.2
    assert rejected == []


def test_empty_label_file_means_no_box():
    assert parse_yolo_labels("", "hk0001") == ([], [])
    assert parse_yolo_labels("\n  \n", "hk0001") == ([], [])


@pytest.mark.parametrize(
    "line",
    [
        "0 0.5 0.5 0.2",
        "0 0.5 0.5 0.2 0.2 0.2",
        "x 0.5 0.5 0.2 0.2",
        "0 0.5 0.5 0.2 abc",
    ],
)
def test_malformed_line_is_refused(line):
    with pytest.raises(ValueError, match="Unexpected YOLO line|Invalid YOLO number"):
        parse_yolo_labels(line, "hk0001")


def test_unknown_class_is_refused():
    """Une quatrieme classe signalerait un jeu different de celui qui a ete inspecte."""
    with pytest.raises(ValueError, match="Unknown CUBIT class"):
        parse_yolo_labels("3 0.5 0.5 0.2 0.2", "hk0001")


@pytest.mark.parametrize("line", ["0 1.5 0.5 0.2 0.2", "0 0.5 -0.1 0.2 0.2"])
def test_box_outside_the_range_is_refused(line):
    with pytest.raises(ValueError, match="outside the image"):
        parse_yolo_labels(line, "hk0001")


@pytest.mark.parametrize("line", ["0 0.5 0.5 0.0 0.2", "0 0.5 0.5 0.2 0.0"])
def test_empty_box_is_set_aside_not_dropped_silently(line):
    """Trois boites de largeur nulle existent dans le jeu publie : on les trace."""
    boxes, rejected = parse_yolo_labels(f"0 0.5 0.5 0.2 0.2\n{line}", "hk0001")
    assert len(boxes) == 1
    assert rejected == [{"line": 2, "reason": "empty_box", "text": line}]


def test_box_crossing_the_border_is_refused():
    with pytest.raises(ValueError, match="ends outside"):
        parse_yolo_labels("0 0.95 0.5 0.2 0.2", "hk0001")
    with pytest.raises(ValueError, match="starts outside"):
        parse_yolo_labels("0 0.05 0.5 0.2 0.2", "hk0001")


def test_too_many_boxes_is_refused(monkeypatch):
    monkeypatch.setattr(cubit_data, "MAX_BOXES_PER_IMAGE", 2)
    with pytest.raises(ValueError, match="Too many boxes"):
        parse_yolo_labels("0 0.5 0.5 0.2 0.2\n" * 3, "hk0001")


def test_empty_boxes_do_not_count_towards_the_limit(monkeypatch):
    monkeypatch.setattr(cubit_data, "MAX_BOXES_PER_IMAGE", 1)
    boxes, rejected = parse_yolo_labels("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.0 0.2", "hk0001")
    assert len(boxes) == 1 and len(rejected) == 1


def test_close_photos_share_a_scene_group():
    hashes = {"hk0002": 0b1010, "hk0001": 0b1011, "hk0009": (1 << 120)}
    groups = build_scene_groups(hashes, max_distance=2)
    assert groups["hk0001"] == groups["hk0002"] == "hk0001"
    assert groups["hk0009"] == "hk0009"


def test_isolated_photo_keeps_its_own_group():
    groups = build_scene_groups({"a": 0, "b": (1 << 100) - 1}, max_distance=1)
    assert groups == {"a": "a", "b": "b"}


def test_scene_group_name_does_not_depend_on_input_order():
    hashes = {"z": 1, "a": 1, "m": 1}
    assert set(build_scene_groups(hashes, 0).values()) == {"a"}


def photos(**kwargs):
    return {
        name: {"width": 100, "height": 80, "split": "train", **extra}
        for name, extra in kwargs.items()
    }


def test_records_carry_the_group_resolution_and_classes():
    records = build_records(
        photos(hk0001={}, hk0002={"width": 200}),
        {"hk0001": [{"class_name": "crack"}, {"class_name": "crack"}]},
        {"hk0001": "hk0001", "hk0002": "hk0002"},
    )
    assert records[0] == {
        "id": "hk0001",
        "content_group": "hk0001",
        "difficulty": "100x80",
        "source_classes": ["crack"],
        "boxes": 2,
        "author_split": "train",
    }
    assert records[1]["difficulty"] == "200x80"
    assert records[1]["source_classes"] == []


def test_photo_without_group_is_refused():
    with pytest.raises(ValueError, match="without a scene group"):
        build_records(photos(hk0001={}), {}, {})


def test_summary_reports_counts_per_class_and_split():
    records = [
        {"id": "a", "content_group": "a", "boxes": 2, "source_classes": ["crack"]},
        {"id": "b", "content_group": "b", "boxes": 0, "source_classes": []},
        {"id": "c", "content_group": "c", "boxes": 1, "source_classes": ["moisture_trace"]},
    ]
    summary = summarize_splits(records, {"a": "train", "b": "train", "c": "test"})
    assert summary["train"]["images"] == 2 and summary["train"]["boxes"] == 2
    assert summary["train"]["images_without_box"] == 1
    assert summary["train"]["images_per_class"]["crack"] == 1
    assert summary["test"]["images_per_class"]["moisture_trace"] == 1
    assert summary["scene_groups_crossing_splits"] == []


def test_a_group_cut_between_splits_is_refused():
    """Couper un groupe annulerait tout l'interet du regroupement."""
    records = [
        {"id": "a", "content_group": "scene", "boxes": 1, "source_classes": ["crack"]},
        {"id": "b", "content_group": "scene", "boxes": 1, "source_classes": ["crack"]},
    ]
    assert crossing_groups(records, {"a": "train", "b": "test"}) == ["scene"]
    with pytest.raises(ValueError, match="split across partitions"):
        summarize_splits(records, {"a": "train", "b": "test"})


def coco_inputs():
    records = [
        {"id": "hk0001", "content_group": "hk0001", "boxes": 1, "source_classes": ["crack"]},
        {"id": "hk0002", "content_group": "hk0001", "boxes": 0, "source_classes": []},
    ]
    labels = {
        "hk0001": [
            {
                "class_id": 0,
                "class_name": "crack",
                "cx": 0.5,
                "cy": 0.25,
                "width": 0.5,
                "height": 0.5,
            }
        ]
    }
    photos = {
        "hk0001": {"width": 200, "height": 100},
        "hk0002": {"width": 200, "height": 100},
    }
    return records, labels, photos


def test_yolo_centre_becomes_a_coco_corner_in_pixels():
    document = cubit_data.cubit_coco_document(*coco_inputs())
    box = document["annotations"][0]
    # Centre 0.5/0.25 et taille 0.5/0.5 sur 200x100 → coin (50, 0), taille 100x50.
    assert box["bbox"] == [50.0, 0.0, 100.0, 50.0]
    assert box["area"] == 5000.0
    assert box["category_id"] == 1
    assert box["iscrowd"] == 0


def test_document_keeps_images_without_any_box():
    document = cubit_data.cubit_coco_document(*coco_inputs())
    assert [i["file_name"] for i in document["images"]] == ["hk0001.jpg", "hk0002.jpg"]
    assert len(document["annotations"]) == 1


def test_document_carries_the_scene_group_and_licence():
    document = cubit_data.cubit_coco_document(*coco_inputs())
    assert document["images"][1]["scene_group"] == "hk0001"
    assert document["licenses"][0]["name"] == "CC BY 4.0"
    assert [c["name"] for c in document["categories"]] == list(cubit_data.CUBIT_CLASSES)
    assert [c["id"] for c in document["categories"]] == [1, 2, 3]


def test_class_ids_shift_by_one_between_yolo_and_coco():
    """YOLO compte a partir de zero, COCO a partir de un : une seule conversion."""
    records, labels, photos = coco_inputs()
    labels["hk0001"][0].update(class_id=2, class_name="moisture_trace")
    document = cubit_data.cubit_coco_document(records, labels, photos)
    assert document["annotations"][0]["category_id"] == 3
    assert document["categories"][2]["name"] == "moisture_trace"


def test_attribution_names_the_authors_and_the_changes():
    credit = cubit_data.attribution()
    assert credit["license"] == "CC BY 4.0"
    assert "Benyun Zhao" in credit["authors"]
    assert "YOLO to COCO" in credit["changes"]
    assert credit["doi"] == "10.1016/j.autcon.2024.105405"


def test_attribution_is_independent_between_calls():
    cubit_data.attribution()["authors"].append("modifié")
    assert "modifié" not in cubit_data.attribution()["authors"]


def prepare_inputs(tmp_path):
    """Un corpus minimal : trois photos, deux partitions d'auteur, un groupe partagé."""
    import hashlib

    raws = {name: f"photo-{name}".encode() for name in ("hk0001", "hk0002", "hk0003")}
    photos = {
        name: {
            "width": 200,
            "height": 100,
            "split": "train" if name != "hk0003" else "test",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for name, raw in raws.items()
    }
    labels = {
        "hk0001": [
            {
                "class_id": 0,
                "class_name": "crack",
                "cx": 0.5,
                "cy": 0.5,
                "width": 0.4,
                "height": 0.4,
            }
        ]
    }
    records = [
        {"id": "hk0001", "content_group": "hk0001", "boxes": 1, "source_classes": ["crack"]},
        {"id": "hk0002", "content_group": "hk0002", "boxes": 0, "source_classes": []},
        {"id": "hk0003", "content_group": "hk0003", "boxes": 0, "source_classes": []},
    ]
    splits = {"hk0001": "train", "hk0002": "valid", "hk0003": "test"}
    readers = {
        "train": lambda name: raws[name],
        "test": lambda name: raws[name],
    }
    return readers, photos, labels, records, splits, raws


def test_prepared_corpus_keeps_the_original_bytes(tmp_path):
    readers, photos, labels, records, splits, raws = prepare_inputs(tmp_path)
    report = cubit_data.prepare_cubit_corpus(
        readers, tmp_path / "out", photos, labels, records, splits, {}
    )
    assert report["split_counts"] == {"train": 1, "valid": 1, "test": 1}
    assert report["approved_for_training"] is False
    # Les octets doivent etre ceux de l'archive : pas de reencodage silencieux.
    assert (tmp_path / "out" / "train" / "hk0001.jpg").read_bytes() == raws["hk0001"]
    assert (tmp_path / "out" / "ATTRIBUTION.json").exists()
    document = json.loads((tmp_path / "out" / "train" / "_annotations.coco.json").read_text())
    assert len(document["images"]) == 1 and len(document["annotations"]) == 1


def test_photo_differing_from_the_archive_is_refused(tmp_path):
    readers, photos, labels, records, splits, _ = prepare_inputs(tmp_path)
    photos["hk0002"] = {**photos["hk0002"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="differs from the inspected archive"):
        cubit_data.prepare_cubit_corpus(
            readers, tmp_path / "out", photos, labels, records, splits, {}
        )
    assert not (tmp_path / "out").exists()


def test_oversized_photo_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cubit_data, "MAX_CUBIT_PHOTO_BYTES", 2)
    readers, photos, labels, records, splits, _ = prepare_inputs(tmp_path)
    with pytest.raises(ValueError, match="above the size limit"):
        cubit_data.prepare_cubit_corpus(
            readers, tmp_path / "out", photos, labels, records, splits, {}
        )


def test_an_empty_split_is_refused(tmp_path):
    readers, photos, labels, records, splits, _ = prepare_inputs(tmp_path)
    splits = {name: "train" for name in splits}
    with pytest.raises(ValueError, match="at least one photo"):
        cubit_data.prepare_cubit_corpus(
            readers, tmp_path / "out", photos, labels, records, splits, {}
        )


def test_unknown_split_name_is_refused(tmp_path):
    readers, photos, labels, records, splits, _ = prepare_inputs(tmp_path)
    splits = {**splits, "hk0003": "holdout"}
    with pytest.raises(ValueError, match="Unknown split name"):
        cubit_data.prepare_cubit_corpus(
            readers, tmp_path / "out", photos, labels, records, splits, {}
        )


def test_existing_destination_is_never_overwritten(tmp_path):
    readers, photos, labels, records, splits, _ = prepare_inputs(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        cubit_data.prepare_cubit_corpus(
            readers, tmp_path / "out", photos, labels, records, splits, {}
        )


def test_manifest_records_the_author_split_and_scene_group(tmp_path):
    readers, photos, labels, records, splits, _ = prepare_inputs(tmp_path)
    cubit_data.prepare_cubit_corpus(
        readers, tmp_path / "out", photos, labels, records, splits, {"hk0009": [{"line": 1}]}
    )
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    row = next(r for r in report["records"] if r["id"] == "hk0003")
    assert row["author_split"] == "test" and row["split"] == "test"
    assert row["scene_group"] == "hk0003"
    assert report["rejected_boxes"] == {"hk0009": [{"line": 1}]}
