"""Rejouer la sélection exacte, y compris face à une archive invalide."""

import io
import json
from zipfile import ZipFile

import pytest
from PIL import Image

from smartsite_ia.curation import digest
from smartsite_ia.zero_shot_sample import prepare_sample


@pytest.fixture
def archive_files(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (48, 40), "gray").save(stream, format="JPEG")
    raw = stream.getvalue()
    # Le XML n'est pas interprété ici ; il est conservé comme preuve originale.
    annotation = b"<synthetic-test-fixture/>"
    entries = {
        "MBDD2025/JPEGImages/Hefei1.jpg": raw,
        "MBDD2025/Annotations/Hefei1.xml": annotation,
    }
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "id": "synthetic",
                    "url": "https://example.org",
                    "license": "test",
                    "scope": "Fixture synthétique",
                },
                "records": [
                    {
                        "id": "Hefei1",
                        "file": "images/Hefei1.jpg",
                        "sha256": digest(raw),
                        "source_annotation_sha256": digest(annotation),
                        "width": 48,
                        "height": 40,
                        "reference_boxes": [],
                        "review_note": "Fixture synthétique",
                    }
                ],
            }
        )
    )
    return tmp_path / "source.zip", selection, tmp_path / "output", entries


def write_archive(path, entries):
    with ZipFile(path, "w") as archive:
        for name, raw in entries.items():
            archive.writestr(name, raw)


def test_prepare_selected_files_without_changing_originals(archive_files):
    archive, selection, output, entries = archive_files
    entries["MBDD2025/JPEGImages/unselected.jpg"] = b"do not extract"
    write_archive(archive, entries)
    before = archive.read_bytes()
    prepare_sample(archive, selection, output)
    assert (output / "manifest.json").read_bytes() == selection.read_bytes()
    assert (output / "images/Hefei1.jpg").read_bytes() == entries["MBDD2025/JPEGImages/Hefei1.jpg"]
    assert sorted(p.name for p in (output / "images").iterdir()) == ["Hefei1.jpg"]
    assert archive.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_sample(archive, selection, output)


@pytest.mark.parametrize(
    "failure", ["path", "hash", "missing", "id", "destination", "empty", "linked"]
)
def test_bad_sources_never_publish_a_sample(archive_files, failure, tmp_path):
    archive, selection, output, entries = archive_files
    if failure == "path":
        entries["../escape.txt"] = b"danger"
    elif failure == "hash":
        entries["MBDD2025/Annotations/Hefei1.xml"] = b"changed"
    elif failure == "missing":
        del entries["MBDD2025/JPEGImages/Hefei1.jpg"]
    elif failure in {"id", "destination", "empty"}:
        data = json.loads(selection.read_text())
        if failure == "empty":
            data["records"] = []
        else:
            data["records"][0]["id" if failure == "id" else "file"] = "../escape"
        selection.write_text(json.dumps(data))
    write_archive(archive, entries)
    if failure == "linked":
        real = tmp_path / "real.zip"
        archive.rename(real)
        archive.symlink_to(real)
    with pytest.raises(ValueError):
        prepare_sample(archive, selection, output)
    assert not output.exists()
    assert not (tmp_path / "escape.txt").exists()
