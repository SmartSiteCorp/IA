import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import image_bytes
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from smartsite_ia import cli
from smartsite_ia.curation import read_document, require_text, staged_output
from smartsite_ia.importer import import_archive
from smartsite_ia.prepare import prepare_corpus


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def prepared_inputs(tmp_path, archive_factory, sample):
    # Deux classes et douze photos synthétiques, dont une paire strictement identique.
    other = {
        "category_id": 1,
        "segmentation": [[50, 50, 70, 50, 70, 70, 50, 70]],
        "bbox": [50, 50, 20, 20],
        "iscrowd": 0,
    }
    sample["document"]["annotations"].append(other)
    sample["yolo"] += (
        "1 " + " ".join(str(v / 640) for v in other["segmentation"][0]) + "\n"
    ).encode()
    mask = Image.new("RGB", (640, 640))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((10, 10, 30, 20), fill="red")
    draw.rectangle((50, 50, 70, 70), fill="blue")
    sample["mask"] = image_bytes(mask, "PNG")
    changes = {}
    for i in range(1, 13):
        values = np.random.default_rng(1 if i == 2 else i).integers(
            0, 256, (640, 640, 3), dtype=np.uint8
        )
        changes[f"Damage Segmentaion/Easy/Images/E ({i}).jpg"] = image_bytes(
            Image.fromarray(values), "JPEG"
        )
    archive, source = archive_factory(changes, copies=12)
    root = tmp_path / "source"
    import_archive(archive, root, source)
    policy = json.loads(Path("config/damsegment_review.json").read_text())
    policy["manifest_sha256"] = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    policy["quarantine_for_future_partitions"] = [
        {"id": "easy_0012", "reason": "Synthetic ambiguous case"}
    ]
    policy["must_stay_together"] = [
        {"members": ["easy_0001", "easy_0002"], "reason": "Synthetic identical pair"}
    ]
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    return root, policy_path, policy


def test_prepare_exports_loadable_coco_preserves_sources_and_is_reproducible(
    prepared_inputs, tmp_path
):
    root, policy, _ = prepared_inputs
    before = snapshot(root)
    left, right = tmp_path / "left", tmp_path / "right"
    report = prepare_corpus(root, left, policy)
    assert report == prepare_corpus(root, right, policy)
    assert snapshot(left) == snapshot(right)
    assert snapshot(root) == before
    assert report["retained_images"] == 11
    assert report["quarantined_images"] == 1
    assert report["approved_for_training"] is True
    assert report["usage"] == "experimental_training_only"
    assert report["approved_for_independent_evaluation"] is False
    records = {r["id"]: r for r in report["records"]}
    assert records["easy_0001"]["split"] == records["easy_0002"]["split"]
    assert records["easy_0012"]["split"] == "quarantine"
    all_ids = set()
    for split in ("train", "valid", "test"):
        coco = COCO(left / split / "_annotations.coco.json")
        assert coco.imgs and len(coco.cats) == 2
        assert not all_ids.intersection(coco.imgs)
        all_ids.update(coco.imgs)
        for entry in coco.imgs.values():
            filename = entry["file_name"]
            assert (left / split / filename).read_bytes() == (
                root / "images" / filename
            ).read_bytes()
        for ann in coco.anns.values():
            assert coco.annToMask(ann).sum() == ann["area"]
            assert ann["source_category_id"] + 1 == ann["category_id"]
    assert len(all_ids) == 11
    assert all(r["scene_group"] is None for r in records.values())


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "mapping",
        "categories",
        "quarantine",
        "duplicate",
        "reason",
        "groups",
        "members",
        "unknown",
    ],
)
def test_invalid_decisions_are_rejected(prepared_inputs, tmp_path, change):
    root, path, policy = prepared_inputs
    if change == "hash":
        policy["manifest_sha256"] = "0" * 64
    if change == "mapping":
        policy["semantic_mapping"]["status"] = "unconfirmed"
    if change == "categories":
        policy["semantic_mapping"]["categories"][0]["id"] = True
    if change == "quarantine":
        policy["quarantine_for_future_partitions"] = [None]
    if change == "duplicate":
        policy["quarantine_for_future_partitions"] *= 2
    if change == "reason":
        policy["quarantine_for_future_partitions"][0]["reason"] = ""
    if change == "groups":
        policy["must_stay_together"] = [None]
    if change == "members":
        policy["must_stay_together"][0]["members"] = None
    if change == "unknown":
        policy["must_stay_together"][0]["members"] = ["easy_0001", "easy_9999"]
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        prepare_corpus(root, tmp_path / "out", path)
    assert not (tmp_path / "out").exists()


def test_quarantine_propagates_to_connected_images(prepared_inputs, tmp_path):
    root, path, policy = prepared_inputs
    policy["quarantine_for_future_partitions"].append(
        {"id": "easy_0001", "reason": "Test quarantine propagation"}
    )
    path.write_text(json.dumps(policy))
    result = prepare_corpus(root, tmp_path / "out", path)
    assert result["quarantined_images"] == 3
    assert next(r for r in result["records"] if r["id"] == "easy_0002")["reason"].startswith(
        "Linked"
    )


def test_source_corruption_is_not_published(prepared_inputs, tmp_path):
    root, path, _ = prepared_inputs
    image = root / "images/easy_0001.jpg"
    image.write_bytes(image.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="manifest"):
        prepare_corpus(root, tmp_path / "out", path)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".smartsite-prepare-*"))


def test_prepare_cli(prepared_inputs, tmp_path, monkeypatch, capsys):
    root, path, _ = prepared_inputs
    monkeypatch.setattr(
        "sys.argv",
        [
            "smartsite-data",
            "prepare",
            str(root),
            "--policy",
            str(path),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["images"] == 12
    assert cli.main() == 1
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("document", [[], {}, {"schema_version": True}, {"schema_version": 2}])
def test_invalid_preparation_document(tmp_path, document):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        read_document(path)


def test_json_duplicates_and_missing_explanations_are_rejected(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_document(path)
    for value in (None, "", " " * 10, "a" * 4001):
        with pytest.raises(ValueError):
            require_text(value)


def test_publication_never_overwrites_or_writes_inside_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"), staged_output(root / "out", [root]):
        pass
    out = tmp_path / "out"
    out.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError), staged_output(out, [root]):
        pass
    out.unlink()
    with pytest.raises(FileExistsError), staged_output(out, [root]):
        out.mkdir()
        (out / "user-file").write_text("keep")
    assert (out / "user-file").read_text() == "keep"


def test_manifest_changed_during_export_is_not_published(prepared_inputs, tmp_path, monkeypatch):
    from smartsite_ia import prepare

    root, path, _ = prepared_inputs
    original = prepare.write_preparation_page

    def change(stage, report):
        original(stage, report)
        with (root / "manifest.json").open("ab") as stream:
            stream.write(b" ")

    monkeypatch.setattr(prepare, "write_preparation_page", change)
    with pytest.raises(ValueError, match="changed during"):
        prepare_corpus(root, tmp_path / "out", path)
    assert not (tmp_path / "out").exists()
