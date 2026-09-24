"""Vérifier la sélection SDNET et ses refus, sur une archive fabriquée pour le test."""

import io
import json
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from smartsite_ia import sdnet_sample
from smartsite_ia.sdnet_sample import SdnetArchive


def patch_bytes(seed):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def build_archive(tmp_path, photos=("7001", "7002"), per_photo=4, duplicate=False):
    """Deux photos de mur, chacune avec des extraits sains et fissurés."""
    path = tmp_path / "SDNET2018.zip"
    seed = 0
    with ZipFile(path, "w") as archive:
        for photo in photos:
            for label in ("UW", "CW"):
                for index in range(1, per_photo + 1):
                    seed += 1
                    archive.writestr(f"W/{label}/{photo}-{index}.jpg", patch_bytes(seed))
        if duplicate:
            # Onze extraits de la vraie archive existent sous deux noms.
            archive.writestr(f"W/UW/{photos[0]}-1_2.jpg", patch_bytes(1))
    return path


def patch_ids(report):
    return [record["id"] for record in report["records"]]


def fake_source(path, patches, photos=2):
    return SdnetArchive(
        size=path.stat().st_size,
        sha256=sdnet_sample.hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_patches=patches,
        origin_photos=photos,
    )


def test_archive_is_checked_against_its_fingerprint(tmp_path):
    path = build_archive(tmp_path)
    source = fake_source(path, 16)
    assert sdnet_sample.verify_sdnet_archive(path, source) == source.sha256
    with pytest.raises(ValueError, match="SHA-256"):
        sdnet_sample.verify_sdnet_archive(path, SdnetArchive(size=source.size))


def test_archive_of_another_size_is_refused(tmp_path):
    path = build_archive(tmp_path)
    with pytest.raises(ValueError, match="differs from the inspected size"):
        sdnet_sample.verify_sdnet_archive(path, SdnetArchive(size=1))


def test_missing_archive_is_refused(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        sdnet_sample.verify_sdnet_archive(tmp_path / "absent.zip", SdnetArchive(size=1))


def test_selection_is_balanced_across_origin_photos(tmp_path):
    path = build_archive(tmp_path)
    report = sdnet_sample.prepare_sample(
        path, tmp_path / "out", per_photo=2, source=fake_source(path, 16)
    )
    assert report["counts"] == {"clear": 4, "cracked": 4}
    assert report["scene_groups"] == 2
    groups = [r["scene_group"] for r in report["records"] if r["author_state"] == "clear"]
    assert sorted(groups) == ["W-7001", "W-7001", "W-7002", "W-7002"]


def test_selection_is_reproducible_and_seed_dependent(tmp_path):
    path = build_archive(tmp_path, per_photo=8)
    source = fake_source(path, 32)
    first = sdnet_sample.prepare_sample(path, tmp_path / "a", per_photo=2, seed=42, source=source)
    same = sdnet_sample.prepare_sample(path, tmp_path / "b", per_photo=2, seed=42, source=source)
    other = sdnet_sample.prepare_sample(path, tmp_path / "c", per_photo=2, seed=7, source=source)
    assert patch_ids(first) == patch_ids(same)
    assert patch_ids(first) != patch_ids(other)


def test_repeated_patch_is_kept_once(tmp_path):
    path = build_archive(tmp_path, duplicate=True)
    report = sdnet_sample.prepare_sample(
        path, tmp_path / "out", per_photo=10, source=fake_source(path, 17)
    )
    assert report["selection"]["deduplicated_patches"] == 1
    fingerprints = [r["sha256"] for r in report["records"]]
    assert len(fingerprints) == len(set(fingerprints))


def test_unexpected_patch_count_is_refused(tmp_path):
    path = build_archive(tmp_path)
    with pytest.raises(ValueError, match="patches, found"):
        sdnet_sample.prepare_sample(path, tmp_path / "out", source=fake_source(path, 999))


def test_unexpected_origin_photo_count_is_refused(tmp_path):
    path = build_archive(tmp_path)
    with pytest.raises(ValueError, match="origin photos"):
        sdnet_sample.prepare_sample(path, tmp_path / "out", source=fake_source(path, 16, photos=99))


def test_entry_outside_the_expected_naming_is_refused(tmp_path):
    path = build_archive(tmp_path)
    with ZipFile(path, "a") as archive:
        archive.writestr("W/UW/notes.txt", b"x")
    with pytest.raises(ValueError, match="Unexpected SDNET entry"):
        sdnet_sample.prepare_sample(path, tmp_path / "out", source=fake_source(path, 17))


def test_unknown_surface_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown SDNET surface"):
        sdnet_sample.prepare_sample(tmp_path / "x.zip", tmp_path / "out", surface="Z")


@pytest.mark.parametrize("per_photo", [0, -1, 101])
def test_impossible_patch_counts_are_refused(per_photo):
    with pytest.raises(ValueError, match="between 1 and 100"):
        sdnet_sample.choose_patches({}, "W", "U", per_photo, 42)


def test_empty_selection_is_refused():
    with pytest.raises(ValueError, match="Empty or oversized"):
        sdnet_sample.choose_patches({}, "W", "U", 10, 42)


def test_sample_report_states_it_is_not_for_training(tmp_path):
    path = build_archive(tmp_path)
    report = sdnet_sample.prepare_sample(path, tmp_path / "out", source=fake_source(path, 16))
    assert report["approved_for_training"] is False
    assert any("cracks only" in limit for limit in report["limits"])
    assert report["provenance"]["license"] == "CC-BY-4.0"
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["records"] == report["records"]


def test_alert_measurement_compares_clear_and_cracked():
    report = {
        "records": [
            {"id": "a", "author_state": "clear", "scene_group": "W-1"},
            {"id": "b", "author_state": "clear", "scene_group": "W-1"},
            {"id": "c", "author_state": "cracked", "scene_group": "W-2"},
        ]
    }
    result = sdnet_sample.measure_alerts(report, {"a": 0, "b": 2, "c": 3})
    assert result["clear"]["alert_rate"] == 0.5
    assert result["clear"]["proposals_per_patch"] == 1.0
    assert result["cracked"]["alert_rate"] == 1.0
    assert result["clear"]["scene_groups"] == 1


def test_alert_measurement_refuses_a_missing_patch():
    report = {
        "records": [
            {"id": "a", "author_state": "clear", "scene_group": "W-1"},
            {"id": "c", "author_state": "cracked", "scene_group": "W-2"},
        ]
    }
    with pytest.raises(ValueError, match="Missing alert counts"):
        sdnet_sample.measure_alerts(report, {"a": 0})


def test_alert_measurement_refuses_a_missing_state():
    report = {"records": [{"id": "a", "author_state": "clear", "scene_group": "W-1"}]}
    with pytest.raises(ValueError, match="No patch to measure"):
        sdnet_sample.measure_alerts(report, {"a": 0})
