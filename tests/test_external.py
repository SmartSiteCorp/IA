import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from smartsite_ia import cli
from smartsite_ia.external import measure_pair, normalize_pair, open_photo, prepare_external


def jpeg(image, orientation=1):
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    image.save(stream, format="JPEG", exif=exif, quality=95)
    return stream.getvalue()


def mask_image(size):
    values = np.zeros((size[1], size[0]), dtype=np.uint8)
    values[:, size[0] // 3 : size[0] // 2] = 255
    values[:4, :] = 128
    return Image.fromarray(values)


@pytest.fixture
def external_inputs(tmp_path):
    root = tmp_path / "source"
    for folder in ("rgb", "BW"):
        (root / "extracted" / folder).mkdir(parents=True)
    policy = json.loads(Path("config/ccsd_preparation.json").read_text())
    policy["expected_images"] = 3
    policy["records"] = []
    for i in range(1, 4):
        sample_id = f"{i:03}"
        pixels = np.random.default_rng(1 if i < 3 else i).integers(
            0, 256, (64, 48, 3), dtype=np.uint8
        )
        photo = jpeg(Image.fromarray(pixels), 6)
        mask = jpeg(mask_image((64, 48)))
        record = {"id": sample_id, "exif_orientation": 6, "mask_exif_orientation": 1}
        for kind, folder, data in (("image", "rgb", photo), ("mask", "BW", mask)):
            relative = f"extracted/{folder}/{sample_id}.jpg"
            (root / relative).write_bytes(data)
            record[kind] = relative
            record[f"{kind}_sha256"] = hashlib.sha256(data).hexdigest()
        policy["records"].append(record)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return root, path, policy


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_external_export_preserves_native_pixels_binary_masks_groups_and_sources(
    external_inputs, tmp_path
):
    root, path, policy = external_inputs
    original = snapshot(root)
    first, second = tmp_path / "first", tmp_path / "second"
    report = prepare_external(root, first, path)
    assert report == prepare_external(root, second, path)
    assert snapshot(first) == snapshot(second)
    assert snapshot(root) == original
    assert report["split_counts"] == {"external_candidate": 2, "external_variant": 1}
    assert report["content_groups"] == 2
    assert report["approved_for_training"] is False
    assert report["approved_for_evaluation"] is False
    assert report["similar_pairs"][0]["binary_mask_iou"] == 1
    for source, entry in zip(policy["records"], report["records"], strict=True):
        with (
            Image.open(root / source["image"]) as image,
            Image.open(first / entry["image"]) as output,
        ):
            assert output.size == (64, 48)
            assert np.array_equal(np.asarray(output), np.asarray(ImageOps.exif_transpose(image)))
            assert output.getexif().get(274, 1) == 1
        with Image.open(first / entry["mask"]) as mask:
            assert set(np.unique(np.asarray(mask))) == {0, 255}
        assert entry["scene_group"] is None


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_orientation_and_threshold_applied_once(orientation):
    raw = jpeg(
        Image.fromarray(np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)), orientation
    )
    with Image.open(io.BytesIO(raw)) as image:
        expected = ImageOps.exif_transpose(image)
    mask_bytes = jpeg(mask_image(expected.size))
    image, mask, band, metrics = normalize_pair(raw, mask_bytes)
    assert np.array_equal(np.asarray(image), np.asarray(expected))
    with Image.open(io.BytesIO(mask_bytes)) as source:
        values = np.asarray(source.convert("L"))
    assert np.array_equal(np.asarray(mask) != 0, values >= 128)
    assert np.array_equal(np.asarray(band) != 0, (values >= 96) & (values < 160))
    assert (
        metrics["foreground_pixels_at_96"]
        >= metrics["foreground_pixels"]
        >= metrics["foreground_pixels_at_160"]
    )


@pytest.mark.parametrize("value", [0, 255])
def test_empty_or_full_mask_requires_explicit_review(value):
    with pytest.raises(ValueError, match="mask is empty"):
        normalize_pair(jpeg(Image.new("RGB", (32, 32))), jpeg(Image.new("L", (32, 32), value)))


def test_bad_dimensions_orientation_format_and_resource_limits():
    with pytest.raises(ValueError, match="dimensions differ"):
        normalize_pair(jpeg(Image.new("RGB", (32, 32))), jpeg(mask_image((16, 32))))
    with pytest.raises(ValueError, match="orientation"):
        open_photo(jpeg(Image.new("RGB", (32, 32)), 9))
    with pytest.raises(ValueError, match="dimensions"):
        open_photo(jpeg(Image.new("RGB", (4001, 4000))))
    with pytest.raises(OSError):
        open_photo(b"not a JPEG")


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "count",
        "empty",
        "record",
        "id",
        "duplicate",
        "path",
        "hash",
        "threshold",
        "provenance",
        "pair_policy",
        "pair_threshold",
    ],
)
def test_bad_external_inventory(external_inputs, tmp_path, change):
    root, path, policy = external_inputs
    if change == "source":
        policy["dataset"] = "unknown"
    if change == "count":
        policy["expected_images"] = True
    if change == "empty":
        policy["records"] = []
    if change == "record":
        policy["records"][0] = None
    if change == "id":
        policy["records"][0]["id"] = "../x"
    if change == "duplicate":
        policy["records"][1] = policy["records"][0]
    if change == "path":
        policy["records"][0]["image"] = "../private.jpg"
    if change == "hash":
        policy["records"][0]["image_sha256"] = "invalid"
    if change == "threshold":
        policy["mask_policy"]["threshold"] = True
    if change == "provenance":
        policy["provenance"] = None
    if change == "pair_policy":
        policy["pair_policy"] = None
    if change == "pair_threshold":
        policy["pair_policy"]["min_binary_mask_iou"] = 1.1
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        prepare_external(root, tmp_path / "out", path)
    assert not (tmp_path / "out").exists()


def test_changed_external_source_is_not_published(external_inputs, tmp_path):
    root, path, policy = external_inputs
    (root / policy["records"][0]["mask"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_external(root, tmp_path / "out", path)
    assert not (tmp_path / "out").exists()


def test_incorrect_orientation_inventory_is_rejected(external_inputs, tmp_path):
    root, path, policy = external_inputs
    policy["records"][0]["exif_orientation"] = 1
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="EXIF metadata"):
        prepare_external(root, tmp_path / "out", path)
    assert not (tmp_path / "out").exists()


def test_pair_rotation_and_label_disagreement_are_measured_at_native_size(tmp_path):
    for folder in ("images", "masks"):
        (tmp_path / folder).mkdir()
    image = Image.fromarray(np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3))
    mask = mask_image(image.size).point(lambda x: 255 if x >= 128 else 0)
    for folder, source in (("images", image), ("masks", mask)):
        source.save(tmp_path / folder / "001.png")
        source.transpose(Image.Transpose.ROTATE_270).save(tmp_path / folder / "002.png")
    pair = {"left": "001", "right": "002", "transform_right": "ROTATE_90"}
    assert measure_pair(tmp_path, pair)["review_required"] is False
    Image.new("L", (24, 32), 255).save(tmp_path / "masks/002.png")
    assert measure_pair(tmp_path, pair)["review_required"] is True
    Image.new("RGB", (3, 3)).save(tmp_path / "images/002.png")
    assert measure_pair(tmp_path, pair)["native_dimensions_match"] is False


def test_external_mask_conflict_quarantines_the_whole_group(external_inputs, tmp_path):
    root, path, policy = external_inputs
    entry = policy["records"][1]
    array = np.zeros((48, 64), dtype=np.uint8)
    array[20:30, :] = 255
    mask = jpeg(Image.fromarray(array))
    (root / entry["mask"]).write_bytes(mask)
    entry["mask_sha256"] = hashlib.sha256(mask).hexdigest()
    path.write_text(json.dumps(policy))
    result = prepare_external(root, tmp_path / "out", path)
    assert result["split_counts"] == {"quarantine": 2, "external_candidate": 1}
    assert result["similar_pairs"][0]["review_required"] is True


def test_external_cli(external_inputs, tmp_path, monkeypatch, capsys):
    root, path, _ = external_inputs
    monkeypatch.setattr(
        "sys.argv",
        [
            "smartsite-data",
            "prepare-external",
            str(root),
            "--policy",
            str(path),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["images"] == 3
    assert cli.main() == 1
    assert "already exists" in capsys.readouterr().err
