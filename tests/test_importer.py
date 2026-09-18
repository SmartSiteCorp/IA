import json
import subprocess
import sys

import pytest
from conftest import image_bytes
from PIL import Image
from pycocotools.coco import COCO

from smartsite_ia.importer import convert_sample, decode_image, import_archive


def test_import_preserves_data_and_produces_readable_coco(tmp_path, archive_factory, sample):
    archive, source = archive_factory()
    destination = tmp_path / "prepared"
    report = import_archive(archive, destination, source)
    assert report["images"] == 1 and report["annotations"] == 1
    assert report["approved_for_training"] is False
    assert report["training_blockers"]
    assert (destination / "images/easy_0001.jpg").read_bytes() == sample["image"]
    assert (destination / "masks/easy_0001.png").read_bytes() == sample["mask"]
    manifest = json.loads((destination / "manifest.json").read_text())["records"]
    assert manifest[0]["scene_group"] is None and manifest[0]["split"] is None
    coco = COCO(str(destination / "annotations.coco.json"))
    instance = coco.loadAnns(coco.getAnnIds())[0]
    assert instance["area"] == 200
    assert coco.annToMask(instance).sum() == 200
    assert coco.cats[1]["name"] == "source_class_0"


def test_reproducible_outputs(tmp_path, archive_factory):
    archive, source = archive_factory()
    left, right = tmp_path / "left", tmp_path / "right"
    assert import_archive(archive, left, source) == import_archive(archive, right, source)
    for item in left.rglob("*"):
        if item.is_file():
            assert item.read_bytes() == (right / item.relative_to(left)).read_bytes()


@pytest.mark.parametrize(
    "change",
    [
        {"Damage Segmentaion/Easy/Labels/Mask/E (1)_mask.png": None},
        {"Damage Segmentaion/Easy/Images/E (1).jpg": b"corrupt"},
        {"unexpected.txt": b"extra"},
        {"Damage Segmentaion/Easy/Labels/Pascal VOC/E (1).json": b"{"},
    ],
)
def test_failed_import_leaves_no_partial_output(tmp_path, archive_factory, change):
    archive, source = archive_factory(change)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        import_archive(archive, output, source)
    assert not output.exists()
    assert not list(tmp_path.glob(".smartsite-import-*"))


def test_existing_output_never_overwritten(tmp_path, archive_factory):
    archive, source = archive_factory()
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "user-file"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        import_archive(archive, output, source)
    assert sentinel.read_text() == "keep"


def test_duplicates_are_reported_not_split(tmp_path, archive_factory):
    archive, source = archive_factory(copies=2)
    report = import_archive(archive, tmp_path / "prepared", source)
    assert report["exact_pixel_duplicate_groups"] == [["easy_0001", "easy_0002"]]


@pytest.mark.parametrize("image", [Image.new("RGB", (641, 640)), Image.new("L", (640, 640))])
def test_invalid_dimensions_or_mode(image):
    with pytest.raises(ValueError, match="640x640"):
        decode_image(image_bytes(image, "PNG"), "PNG")


def test_unknown_mask_color_rejected(sample):
    mask = image_bytes(Image.new("RGB", (640, 640), "green"), "PNG")
    with pytest.raises(ValueError, match="palette"):
        convert_sample(
            sample["image"],
            mask,
            json.dumps(sample["document"]).encode(),
            sample["yolo"],
            "E (1).jpg",
        )


def test_cli_help_and_failure_code(tmp_path):
    help_result = subprocess.run(
        [sys.executable, "-m", "smartsite_ia.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0 and "download" in help_result.stdout
    failed = subprocess.run(
        [
            sys.executable,
            "-m",
            "smartsite_ia.cli",
            "import",
            str(tmp_path / "missing.zip"),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1 and "Preparation failed" in failed.stderr
    assert not (tmp_path / "out").exists()
