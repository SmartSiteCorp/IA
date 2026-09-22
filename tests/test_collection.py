"""Vérifier la collecte sans réseau et sans confondre annotation et prédiction."""

import copy
import io
import json

import pytest
from PIL import Image

from smartsite_ia.collection import boxes_for, load_collection, main, prepare_collection
from smartsite_ia.curation import digest


def photo_bytes(color, orientation=1):
    photo = Image.new("RGB", (64, 40), color)
    exif = photo.getexif()
    exif[274] = orientation
    raw = io.BytesIO()
    photo.save(raw, format="JPEG", exif=exif)
    return raw.getvalue()


@pytest.fixture
def collection(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    selection = config / "selection.json"
    records = []
    for i, color in enumerate(["red", "green", "blue"]):
        raw = photo_bytes(color, 6 if i == 0 else 1)
        sha = digest(raw)
        (cache / sha).write_bytes(raw)
        row = {
            "id": f"image_{i}",
            "source_id": "fixture",
            "source_split": ["train", "val", "unassigned"][i],
            "asset": {
                "url": f"https://upload.wikimedia.org/fixture{i}.jpg",
                "size": len(raw),
                "transfer_size": len(raw),
                "sha256": sha,
                "start": None,
                "total": len(raw),
                "encoding": "identity",
            },
            "credit": {
                "title": "Synthetic test photo",
                "author": "Test fixture",
                "license": "test",
                "source_url": "https://example.org/source",
                "license_url": "https://example.org/license",
            },
            "decision": "quarantine" if i == 1 else "review",
            "note": "<script>unsafe</script>",
        }
        if i < 2:
            label = "0 0.5 0.5 0.5 0.5\n1 0.2 0.2 0.1 0.1\n"
            row.update(source_label_text=label, source_label_sha256=digest(label.encode()))
        records.append(row)
    document = {
        "schema_version": 1,
        "limitations": "Synthetic fixtures, not training data.",
        "sources": {
            "fixture": {
                "coordinate_frame": "display_oriented",
                "class_names": ["Mold", "Chair"],
                "class_map": {"0": "mold_suspected"},
            }
        },
        "groups": [["image_0", "image_1"]],
        "records": records,
    }
    selection.write_text(json.dumps(document))
    return selection, cache, tmp_path / "output", document


def test_prepare_keeps_originals_coordinates_and_uncertainty(collection):
    selection, cache, output, document = collection
    before = {p.name: p.read_bytes() for p in cache.iterdir()}
    summary = prepare_collection(selection, cache, output)
    assert summary["images"] == 3
    assert summary["to_annotate"] == 1
    assert summary["quarantined"] == 1
    assert summary["orientation_corrected"] == 1
    assert summary["approved_for_training"] == 0
    report = json.loads((output / "report.json").read_text())
    first, second, third = report["records"]
    assert (first["width"], first["height"]) == (40, 64)
    assert first["boxes"][0]["xyxy_normalized"] == [0.25, 0.25, 0.75, 0.75]
    assert first["group"] == second["group"]
    assert report["source_split_conflicts"][0]["ids"] == ["image_0", "image_1"]
    assert third["annotation_status"] == "to_annotate"
    assert all(
        r["partition"] == "unassigned" and not r["approved_for_training"] for r in report["records"]
    )
    for row in document["records"]:
        assert (output / "originals" / (row["id"] + ".image")).read_bytes() == before[
            row["asset"]["sha256"]
        ]
    assert {p.name: p.read_bytes() for p in cache.iterdir()} == before
    assert (output / "source_labels/image_0.txt").read_text() == document["records"][0][
        "source_label_text"
    ]
    page = (output / "index.html").read_text()
    assert "<script>unsafe</script>" not in page
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in page
    assert "pas des détections" in page
    assert "Mise de côté" in page
    queue = json.loads((output / "review_queue.json").read_text())
    assert all(
        r["decision"] == "pending" and not r["exhaustive_annotation"] for r in queue["records"]
    )
    with pytest.raises(FileExistsError):
        prepare_collection(selection, cache, output)


def test_preparation_replays_identically(collection):
    selection, cache, output, _ = collection
    prepare_collection(selection, cache, output)
    other = output.with_name("replay")
    prepare_collection(selection, cache, other)
    for p in output.rglob("*"):
        if p.is_file():
            assert p.read_bytes() == (other / p.relative_to(output)).read_bytes()


@pytest.mark.parametrize(
    "bad",
    [
        "id",
        "duplicate",
        "test",
        "unknown_source",
        "credit",
        "link",
        "hash",
        "missing",
        "corrupt",
        "symlink",
        "group",
        "decision",
        "empty",
        "orientation",
    ],
)
def test_invalid_input_does_not_publish(collection, bad):
    selection, cache, output, document = collection
    row = document["records"][0]
    if bad == "id":
        row["id"] = "../escape"
    elif bad == "duplicate":
        document["records"].append(copy.deepcopy(row))
    elif bad == "test":
        row["source_split"] = "test"
    elif bad == "unknown_source":
        row["source_id"] = "missing"
    elif bad == "credit":
        del row["credit"]["author"]
    elif bad == "link":
        row["credit"]["source_url"] = "javascript:alert(1)"
    elif bad == "hash":
        (cache / row["asset"]["sha256"]).write_bytes(b"bad")
    elif bad == "missing":
        (cache / row["asset"]["sha256"]).unlink()
    elif bad == "symlink":
        target = cache / row["asset"]["sha256"]
        real = cache / "real"
        target.rename(real)
        target.symlink_to(real)
    elif bad == "group":
        document["groups"] = [["image_0", "missing"]]
    elif bad == "decision":
        row["decision"] = "approved"
    elif bad == "empty":
        document["records"] = []
    elif bad in {"corrupt", "orientation"}:
        raw = b"not an image" if bad == "corrupt" else photo_bytes("red", 0)
        row["asset"].update(
            size=len(raw), transfer_size=len(raw), total=len(raw), sha256=digest(raw)
        )
        (cache / digest(raw)).write_bytes(raw)
    selection.write_text(json.dumps(document))
    with pytest.raises((ValueError, OSError)):
        prepare_collection(selection, cache, output)
    assert not output.exists()
    assert not list(output.parent.glob(".smartsite-prepare-*"))


@pytest.mark.parametrize(
    "label",
    [
        "0 nan .5 .3 .3",
        "0 .5 .5 -1 .3",
        "0 .5 .5 0 .3",
        "0 1 .5 1 .3",
        "0 .5 .5 .3",
        "99 .5 .5 .3 .3",
        "-1 .5 .5 .3 .3",
        "0 inf .5 .3 .3",
        "0 .5 .5 .3 .3\n" * 201,
    ],
)
def test_invalid_boxes(label, collection):
    _, _, _, document = collection
    row = document["records"][0]
    row.update(source_label_text=label, source_label_sha256=digest(label.encode()))
    with pytest.raises(ValueError):
        boxes_for(row, document["sources"]["fixture"])


def test_annotation_hash_and_coordinate_frame_are_checked(collection):
    _, _, _, document = collection
    row = document["records"][0]
    source = document["sources"]["fixture"]
    row["source_label_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        boxes_for(row, source)
    row["source_label_sha256"] = digest(row["source_label_text"].encode())
    source["coordinate_frame"] = "unknown"
    with pytest.raises(ValueError, match="coordinate frame"):
        boxes_for(row, source)


def test_cli_reports_real_errors_and_requires_destination(collection, capsys):
    selection, cache, output, _ = collection
    with pytest.raises(SystemExit):
        main(["prepare", "--selection", str(selection), "--cache", str(cache)])
    args = [
        "prepare",
        "--selection",
        str(selection),
        "--cache",
        str(cache),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert main(args) == 1
    assert "FileExistsError" in capsys.readouterr().err
    assert main(["fetch", "--selection", str(selection), "--cache", str(cache)]) == 0
    assert load_collection(selection)[1] == digest(selection.read_bytes())


def test_mpo_primary_frame_is_opt_in_and_other_frame_is_not_used(collection):
    from smartsite_ia.prediction import decode_photo

    raw = io.BytesIO()
    first = Image.new("RGB", (64, 40), "red")
    second = Image.new("RGB", (32, 32), "blue")
    first.save(raw, format="MPO", save_all=True, append_images=[second])
    content = raw.getvalue()
    with pytest.raises(ValueError, match="animation"):
        decode_photo(content)
    photo = decode_photo(content, allow_primary_mpo=True)
    assert photo.size == (64, 40)
    assert photo.getpixel((0, 0))[0] > 240
    assert photo.getpixel((0, 0))[2] < 10
    with pytest.raises(ValueError, match="two-frame"):
        decode_photo(photo_bytes("red"), allow_primary_mpo=True)
    three = io.BytesIO()
    first.save(three, format="MPO", save_all=True, append_images=[second, second])
    with pytest.raises(ValueError, match="two-frame"):
        decode_photo(three.getvalue(), allow_primary_mpo=True)
    selection, cache, output, document = collection
    row = document["records"][0]
    row["image_policy"] = "mpo_primary_of_two"
    row["asset"].update(
        size=len(content), transfer_size=len(content), total=len(content), sha256=digest(content)
    )
    (cache / digest(content)).write_bytes(content)
    selection.write_text(json.dumps(document))
    assert prepare_collection(selection, cache, output)["images"] == 3
    assert (output / "originals/image_0.image").read_bytes() == content


def test_exact_duplicates_are_grouped_without_manual_links(collection):
    selection, cache, output, document = collection
    document["groups"] = []
    document["records"][1]["asset"] = copy.deepcopy(document["records"][0]["asset"])
    document["excluded_candidates"] = [{"id": "outside", "reason": "Invalid orientation"}]
    selection.write_text(json.dumps(document))
    prepare_collection(selection, cache, output)
    report = json.loads((output / "report.json").read_text())
    assert report["similar_pairs"][0]["kind"] == "exact_pixels"
    assert report["records"][0]["group"] == report["records"][1]["group"]
    assert report["summary"]["excluded_before_preparation"] == 1
    assert "Invalid orientation" in (output / "index.html").read_text()


def test_fetch_stops_on_failure_and_can_resume_cached_files(collection, monkeypatch):
    import smartsite_ia.collection as module

    selection, cache, _, document = collection
    for p in cache.iterdir():
        p.unlink()
    calls = []
    pauses = []

    def simulated(asset, cache):
        calls.append(asset.sha256)
        if len(calls) == 2:
            raise OSError("Remote source temporarily unavailable")
        (cache / asset.sha256).write_bytes(b"fixture download")

    monkeypatch.setattr(module, "fetch_asset", simulated)
    monkeypatch.setattr(module.time, "sleep", pauses.append)
    with pytest.raises(OSError):
        module.fetch_collection(selection, cache)
    assert len(calls) == 2
    assert len(list(cache.iterdir())) == 1
    assert pauses == [3]
