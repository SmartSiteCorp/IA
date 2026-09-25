"""Vérifier l'accueil d'un corpus déjà préparé, sans toucher au parcours d'humidité."""

import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from smartsite_ia import box_data
from smartsite_ia.box_data import (
    config_classes,
    inspect_corpus,
    inspect_prepared_corpus,
    load_box_config,
    prepare_inputs,
    prepared_photo_size,
)

CLASSES = ["crack", "surface_loss", "moisture_trace"]


def photo_bytes(seed, size=(64, 48), frames=1):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    if frames == 2:
        # Un MPO à deux vues, comme les photos 8000x6000 du corpus.
        image.save(buffer, format="MPO", append_images=[image.copy()])
    else:
        image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def build_corpus(tmp_path, *, cross_group=False, frames=1):
    root = tmp_path / "corpus"
    records, documents = [], {}
    plan = {"train": ["a", "b"], "valid": ["c"], "test": ["d"]}
    for split, names in plan.items():
        (root / split).mkdir(parents=True)
        images, annotations = [], []
        for index, name in enumerate(names, 1):
            raw = photo_bytes(hash(name) % 1000, frames=frames if name == "a" else 1)
            (root / split / f"{name}.jpg").write_bytes(raw)
            group = "shared" if cross_group else name
            records.append(
                {
                    "id": name,
                    "image": f"{split}/{name}.jpg",
                    "split": split,
                    "scene_group": group,
                    "width": 64,
                    "height": 48,
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "boxes": 1,
                }
            )
            images.append({"id": index, "file_name": f"{name}.jpg", "width": 64, "height": 48})
            annotations.append(
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [1.0, 1.0, 10.0, 10.0],
                    "area": 100.0,
                    "iscrowd": 0,
                }
            )
        documents[split] = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES, 1)],
        }
        (root / split / "_annotations.coco.json").write_text(json.dumps(documents[split]))
    report = {
        "schema_version": 1,
        "approved_for_training": False,
        "class_names": CLASSES,
        "records": records,
    }
    (root / "report.json").write_text(json.dumps(report))
    return root


def make_config(root, tmp_path, **changes):
    files = ("report.json", "train/_annotations.coco.json", "valid/_annotations.coco.json")
    config = {
        "schema_version": 1,
        "purpose": "pilot_box_training",
        "corpus_kind": "prepared_coco",
        "class_names": CLASSES,
        "seed": 1,
        "epochs": 1,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "lr": 0.0001,
        "checkpoint_selection": "last_epoch",
        "corpus_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
        **changes,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return load_box_config(path)[0]


def test_prepared_corpus_is_inspected_with_its_own_classes(tmp_path):
    root = build_corpus(tmp_path)
    selection = inspect_corpus(root, make_config(root, tmp_path))
    assert selection["class_names"] == CLASSES
    assert selection["splits"]["train"]["images"] == 2
    assert selection["splits"]["valid"]["images"] == 1
    assert selection["test_used"] is False


def test_test_split_is_never_read(tmp_path):
    """Le lot de test du corpus reste ferme : l'inspection ne doit pas y toucher."""
    root = build_corpus(tmp_path)
    original = box_data.read_local
    seen = []

    def watched(base, path, limit):
        seen.append(str(path))
        return original(base, path, limit)

    box_data.read_local = watched
    try:
        inspect_corpus(root, make_config(root, tmp_path))
    finally:
        box_data.read_local = original
    assert not any(name.startswith("test/") for name in seen)


def test_scene_group_crossing_splits_is_refused(tmp_path):
    root = build_corpus(tmp_path, cross_group=True)
    with pytest.raises(ValueError, match="scene group crosses splits"):
        inspect_corpus(root, make_config(root, tmp_path))


def test_classes_disagreeing_with_the_corpus_are_refused(tmp_path):
    root = build_corpus(tmp_path)
    config = make_config(root, tmp_path, class_names=["crack", "surface_loss"])
    with pytest.raises(ValueError, match="classes disagree"):
        inspect_corpus(root, config)


def test_corpus_opened_for_training_is_refused(tmp_path):
    root = build_corpus(tmp_path)
    report = json.loads((root / "report.json").read_text())
    report["approved_for_training"] = True
    (root / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="classes disagree"):
        inspect_corpus(root, make_config(root, tmp_path))


def test_changed_photo_is_refused(tmp_path):
    root = build_corpus(tmp_path)
    config = make_config(root, tmp_path)
    (root / "train" / "a.jpg").write_bytes(photo_bytes(999))
    with pytest.raises(ValueError, match="Prepared photo changed"):
        inspect_prepared_corpus(root, config)


def test_photo_missing_from_the_manifest_is_refused(tmp_path):
    root = build_corpus(tmp_path)
    report = json.loads((root / "report.json").read_text())
    report["records"] = [r for r in report["records"] if r["id"] != "a"]
    (root / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="missing from the corpus manifest"):
        inspect_corpus(root, make_config(root, tmp_path))


def test_mpo_with_two_views_is_accepted(tmp_path):
    """Dix-neuf photos du corpus reel sont des MPO a deux vues."""
    assert prepared_photo_size(photo_bytes(1, frames=2)) == (64, 48)
    root = build_corpus(tmp_path, frames=2)
    assert inspect_corpus(root, make_config(root, tmp_path))["splits"]["train"]["images"] == 2


def test_animation_is_still_refused():
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (64, 48), "black"), Image.new("RGB", (64, 48), "white")]
    frames[0].save(buffer, format="MPO", append_images=frames[1:] * 2)
    with pytest.raises(ValueError, match="Unsupported prepared photo"):
        prepared_photo_size(buffer.getvalue())


def test_prepared_inputs_are_copied_without_the_test_split(tmp_path):
    root = build_corpus(tmp_path)
    selection = prepare_inputs(root, tmp_path / "out", make_config(root, tmp_path))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "selection.json",
        "train",
        "valid",
    ]
    assert (tmp_path / "out" / "train" / "a.jpg").read_bytes() == (
        root / "train" / "a.jpg"
    ).read_bytes()
    assert selection["splits"]["valid"]["annotations"] == 1


def test_moisture_configuration_keeps_its_classes_and_kind(tmp_path):
    """Une configuration sans les nouveaux champs garde le comportement d'origine."""
    config = {
        "schema_version": 1,
        "purpose": "pilot_box_training",
        "seed": 1,
        "epochs": 1,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "lr": 0.0001,
        "checkpoint_selection": "last_epoch",
        "corpus_sha256": {name: "0" * 64 for name in box_data.BOX_CORPUS_FILES},
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(config))
    loaded, _ = load_box_config(path)
    assert config_classes(loaded) == box_data.BOX_CLASSES
    assert loaded.get("corpus_kind", box_data.DEFAULT_CORPUS_KIND) == "reviewed_collection"


def test_unknown_corpus_kind_is_refused(tmp_path):
    root = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="Unknown box corpus kind"):
        make_config(root, tmp_path, corpus_kind="something_else")


@pytest.mark.parametrize(
    "names", [[], ["a"] * 11, ["crack", "crack"], [1], [""], ["x" * 41], "crack"]
)
def test_invalid_class_names_are_refused(tmp_path, names):
    root = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="Invalid box class names|bounded pilot"):
        make_config(root, tmp_path, class_names=names)


def test_changed_corpus_file_is_refused(tmp_path):
    root = build_corpus(tmp_path)
    config = make_config(root, tmp_path)
    report = json.loads((root / "report.json").read_text())
    report["records"][0]["boxes"] = 99
    (root / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Box corpus changed"):
        inspect_prepared_corpus(root, config)


def test_categories_in_a_different_order_are_refused(tmp_path):
    root = build_corpus(tmp_path)
    document = json.loads((root / "train" / "_annotations.coco.json").read_text())
    document["categories"] = [{"id": i, "name": n} for i, n in enumerate(reversed(CLASSES), 1)]
    (root / "train" / "_annotations.coco.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Unexpected categories in train"):
        inspect_corpus(root, make_config(root, tmp_path))


def test_dimensions_disagreeing_with_the_annotations_are_refused(tmp_path):
    root = build_corpus(tmp_path)
    document = json.loads((root / "train" / "_annotations.coco.json").read_text())
    document["images"][0]["width"] = 65
    (root / "train" / "_annotations.coco.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="dimensions changed"):
        inspect_corpus(root, make_config(root, tmp_path))


def test_the_same_photo_in_two_splits_is_refused(tmp_path):
    """Un fichier identique dans deux lots est un doublon franc, pas une ressemblance."""
    root = build_corpus(tmp_path)
    raw = (root / "train" / "a.jpg").read_bytes()
    (root / "valid" / "c.jpg").write_bytes(raw)
    report = json.loads((root / "report.json").read_text())
    for row in report["records"]:
        if row["id"] == "c":
            row["image_sha256"] = hashlib.sha256(raw).hexdigest()
    (root / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Identical photos cross splits"):
        inspect_corpus(root, make_config(root, tmp_path))


def test_empty_split_is_refused(tmp_path):
    root = build_corpus(tmp_path)
    document = json.loads((root / "valid" / "_annotations.coco.json").read_text())
    document["images"], document["annotations"] = [], []
    (root / "valid" / "_annotations.coco.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Empty split: valid"):
        inspect_corpus(root, make_config(root, tmp_path))


def test_batch_larger_than_the_training_split_is_refused(tmp_path):
    root = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="smaller than one batch"):
        inspect_corpus(root, make_config(root, tmp_path, batch_size=4))


def test_photo_changed_between_inspection_and_copy_is_refused(tmp_path, monkeypatch):
    """La copie revérifie : un fichier remplacé entre les deux étapes est refusé."""
    root = build_corpus(tmp_path)
    config = make_config(root, tmp_path)
    selection = inspect_prepared_corpus(root, config)
    monkeypatch.setattr(box_data, "inspect_prepared_corpus", lambda *a: selection)
    (root / "train" / "a.jpg").write_bytes(photo_bytes(777))
    with pytest.raises(ValueError, match="changed during preparation"):
        prepare_inputs(root, tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


def test_annotations_changed_between_inspection_and_copy_are_refused(tmp_path, monkeypatch):
    root = build_corpus(tmp_path)
    config = make_config(root, tmp_path)
    selection = inspect_prepared_corpus(root, config)
    monkeypatch.setattr(box_data, "inspect_prepared_corpus", lambda *a: selection)
    document = json.loads((root / "train" / "_annotations.coco.json").read_text())
    document["info"] = "modifié"
    (root / "train" / "_annotations.coco.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="annotations changed during preparation"):
        prepare_inputs(root, tmp_path / "out", config)


def test_large_photo_passes_the_post_training_check(tmp_path):
    """Non-regression : une photo de plus de 8 Mo faisait echouer le controle final.

    L'essai CUBIT du 24 septembre a tourne 28 minutes puis echoue a ce controle, alors
    que l'entrainement lui-meme avait reussi. Quinze photos de facade depassaient la
    limite commune, pensee pour des images de revue.
    """
    from smartsite_ia.box_data import corpus_photo_limit
    from smartsite_ia.learning import verify_training_data

    root = build_corpus(tmp_path)
    config = make_config(root, tmp_path)
    output = tmp_path / "run"
    prepare_inputs(root, output / "data", config)

    big = b"\xff\xd8" + b"\x00" * (9 * 1024 * 1024)
    (output / "data" / "train" / "a.jpg").write_bytes(big)
    selection = json.loads((output / "data" / "selection.json").read_text())
    for row in selection["splits"]["train"]["records"]:
        if row["id"] == "a":
            row["image_sha256"] = hashlib.sha256(big).hexdigest()
    (output / "data" / "selection.json").write_text(json.dumps(selection))

    report = {
        "selection_sha256": hashlib.sha256(
            (output / "data" / "selection.json").read_bytes()
        ).hexdigest(),
        "input_annotations_sha256": {
            split: hashlib.sha256(
                (output / "data" / split / "_annotations.coco.json").read_bytes()
            ).hexdigest()
            for split in ("train", "valid")
        },
    }
    # Avec la limite commune, le controle refuse la photo de neuf megaoctets.
    with pytest.raises(ValueError, match="exceeds its size limit"):
        verify_training_data(output, report)
    # Avec la limite du corpus prepare, il passe.
    verify_training_data(output, report, corpus_photo_limit(config))


def test_photo_limit_depends_on_the_corpus_kind(tmp_path):
    from smartsite_ia.box_data import MAX_PREPARED_PHOTO_BYTES, corpus_photo_limit
    from smartsite_ia.review import MAX_FILE_BYTES

    root = build_corpus(tmp_path)
    assert corpus_photo_limit(make_config(root, tmp_path)) == MAX_PREPARED_PHOTO_BYTES
    assert corpus_photo_limit({}) == MAX_FILE_BYTES
