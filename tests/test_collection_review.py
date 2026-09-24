"""Sécuriser les corrections, les exclusions et le corpus pilote sans réseau/GPU."""

import copy
import io
import json

import pytest
from PIL import Image
from pycocotools.coco import COCO

from smartsite_ia.categories import CLASS_NAMES, DEFECT_FAMILIES, get_class_names
from smartsite_ia.collection import boxes_for
from smartsite_ia.collection_review import (
    apply_review,
    main,
    rectangle,
    source_audit,
    validate_review,
)
from smartsite_ia.curation import digest


@pytest.fixture
def review_case(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    selection = config / "selection.json"
    review = config / "review.json"
    source = {
        "coordinate_frame": "display_oriented",
        "class_names": ["Mold"],
        "class_map": {"0": "mold_suspected"},
    }
    originals, decisions = [], []
    for i, color in enumerate(["red", "blue", "green"]):
        photo = Image.new("RGB", (128, 128), color)
        buffer = io.BytesIO()
        photo.save(buffer, format="JPEG")
        raw = buffer.getvalue()
        sha = digest(raw)
        (cache / sha).write_bytes(raw)
        label = "0 .25 .25 .3 .3\n"
        original = {
            "id": f"photo_{i}",
            "source_id": "fixture",
            "source_split": "train",
            "decision": "review",
            "note": "Synthetic fixture, not real inspection data.",
            "source_label_text": label,
            "source_label_sha256": digest(label.encode()),
            "asset": {
                "url": f"https://upload.wikimedia.org/{i}.jpg",
                "sha256": sha,
                "size": len(raw),
                "transfer_size": len(raw),
                "start": None,
                "total": len(raw),
                "encoding": "identity",
            },
            "credit": {
                "title": "Synthetic",
                "author": "Fixture",
                "license": "Test",
                "source_url": "https://example.org/source",
                "license_url": "https://example.org/license",
            },
        }
        originals.append(original)
        decisions.append(
            {
                "id": original["id"],
                "decision": "excluded" if i == 2 else "candidate",
                "reviewed_scope": i != 2,
                "note": "<script>not executable</script>",
                "boxes": [
                    {
                        "class": "mold_suspected",
                        "xyxy_normalized": [0.1, 0.1, 0.4, 0.4],
                        "source_indices": [0],
                    },
                    {
                        "class": "peeling_paint",
                        "xyxy_normalized": [0.1, 0.5, 0.4, 0.6],
                        "source_indices": [],
                    },
                    {
                        "class": "moisture_trace",
                        "xyxy_normalized": [0.5, 0.1, 0.6, 0.4],
                        "source_indices": [],
                    },
                ]
                if i != 2
                else [],
                "removed_source_boxes": [] if i != 2 else [{"index": 0, "reason": "Ambiguous"}],
            }
        )
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {"fixture": source},
                "limitations": "Synthetic fixtures",
                "records": originals,
                "groups": [],
            }
        )
    )
    document = {
        "schema_version": 1,
        "selection_sha256": digest(selection.read_bytes()),
        "reviewer": "assistant_visual_review",
        "human_validation": "pending",
        "scope": ["mold_suspected", "moisture_trace", "peeling_paint"],
        "coordinate_frame": "display_oriented",
        "protocol": {
            "purpose": "pilot_training_only",
            "annotation_unit": "Visible regions",
            "coverage": "Synthetic cases",
            "limits": "No actual model or review",
            "class_rules": {
                c: "Synthetic rule" for c in ("mold_suspected", "moisture_trace", "peeling_paint")
            },
        },
        "group_partitions": {"photo_0": "train", "photo_1": "validation", "photo_2": "train"},
        "records": decisions,
        "negative_crops": [
            {
                "id": "negative",
                "parent_id": "photo_0",
                "reviewed_scope": True,
                "crop_xyxy_normalized": [0.65, 0.65, 1, 1],
                "note": "Reviewed crop",
            }
        ],
    }
    review.write_text(json.dumps(document))
    source_report = {
        "selection_sha256": document["selection_sha256"],
        "records": [{**r, "group": r["id"], "boxes": boxes_for(r, source)} for r in originals],
    }
    return selection, cache, review, tmp_path / "output", document, source_report


def test_review_exports_boxes_preserves_sources_and_replays(review_case):
    selection, cache, review, output, _, source = review_case
    before = {p.name: p.read_bytes() for p in cache.iterdir()}
    summary = apply_review(selection, cache, review, output)
    assert summary["candidate_photos"] == 2
    assert summary["excluded_photos"] == 1
    assert summary["source_box_outcomes"] == {"kept": 2, "removed": 1}
    coco = COCO(str(output / "train/_annotations.coco.json"))
    assert len(coco.imgs) == 2
    assert len(coco.anns) == 3
    assert coco.anns[1]["bbox"] == pytest.approx([12.8, 12.8, 38.4, 38.4])
    assert coco.anns[1]["area"] == pytest.approx(38.4**2)
    assert all("segmentation" not in a for a in coco.anns.values())
    assert not (output / "test").exists()
    assert not list((output / "train").glob("photo_2*"))
    assert not list((output / "validation").glob("negative*"))
    report = json.loads((output / "report.json").read_text())
    assert (output / "review.json").read_bytes() == review.read_bytes()
    assert report["review_sha256"] == digest((output / "review.json").read_bytes())
    assert all(r["human_validation"] == "pending" for r in report["records"])
    assert all(not r["approved_for_training"] for r in report["records"])
    assert report["final_test_created"] is False
    assert report["records"][-1]["group"] == "photo_0"
    assert report["records"][-1]["crop_xyxy_pixels"] == [83, 83, 128, 128]
    for r in source["records"]:
        assert (output / f"source/originals/{r['id']}.image").read_bytes() == before[
            r["asset"]["sha256"]
        ]
    page = (output / "index.html").read_text()
    assert "<script>not executable</script>" not in page
    assert "&lt;script&gt;not executable&lt;/script&gt;" in page
    assert "pas des détections" in page
    excluded_card = page.split('<article id="photo_2">')[1].split("</article>")[0]
    assert "Aucune des trois cibles" not in excluded_card
    assert "hors apprentissage" in excluded_card
    assert coco.cats[3]["supercategory"] == "surface_damage"
    replay = output.with_name("replay")
    apply_review(selection, cache, review, replay)
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == {
        p.relative_to(replay): p.read_bytes() for p in replay.rglob("*") if p.is_file()
    }
    assert before == {p.name: p.read_bytes() for p in cache.iterdir()}
    with pytest.raises(FileExistsError):
        apply_review(selection, cache, review, output)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [0, 0, 0, 1],
        [0, 0, 1.1, 1],
        [0, 0, float("nan"), 1],
        [0, False, 1, 1],
        [0, 1, 1, 0],
        [0, 0, float("inf"), 1],
    ],
)
def test_invalid_coordinates(value):
    with pytest.raises(ValueError):
        rectangle(value)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(selection_sha256="0" * 64),
        lambda d: d.update(typo=True),
        lambda d: d.update(human_validation="approved"),
        lambda d: d.update(scope=["mold_suspected"]),
        lambda d: d.update(coordinate_frame="raw"),
        lambda d: d["protocol"].update(purpose="production"),
        lambda d: d["protocol"].pop("limits"),
        lambda d: d["protocol"].update(class_rules={}),
        lambda d: d["records"].pop(),
        lambda d: d["records"].__setitem__(1, d["records"][0]),
        lambda d: d["records"][0].update(id="../escaped"),
        lambda d: d["records"][0].update(decision="approved"),
        lambda d: d["records"][0].update(reviewed_scope=False),
        lambda d: d["records"][0].update(boxes=[]),
        lambda d: d["records"][0].update(note=""),
        lambda d: d["records"][0].update(typo=True),
        lambda d: d["records"][0]["boxes"][0].update(**{"class": "surface_loss"}),
        lambda d: d["records"][0]["boxes"][0].update(source_indices=[False]),
        lambda d: d["records"][0]["boxes"][0].update(source_indices=[0, 0]),
        lambda d: d["records"][0]["boxes"][0].update(source_indices=[4]),
        lambda d: d["records"][0]["boxes"][0].update(source_indices=[]),
        lambda d: d["records"][0]["boxes"][0].update(typo=1),
        lambda d: d["records"][0]["boxes"].append(copy.deepcopy(d["records"][0]["boxes"][0])),
        lambda d: d["records"][0]["removed_source_boxes"].append({"index": 0, "reason": "Overlap"}),
        lambda d: d["records"][2]["removed_source_boxes"].append(
            {"index": 0, "reason": "Duplicate"}
        ),
        lambda d: d["records"][2]["removed_source_boxes"][0].update(index=-1),
        lambda d: d["records"][2]["removed_source_boxes"][0].update(typo=True),
        lambda d: d["group_partitions"].update(photo_0="test"),
        lambda d: d["group_partitions"].pop("photo_0"),
        lambda d: d["negative_crops"][0].update(parent_id="photo_1"),
        lambda d: d["negative_crops"][0].update(parent_id="photo_2"),
        lambda d: d["negative_crops"][0].update(parent_id="unknown"),
        lambda d: d["negative_crops"][0].update(reviewed_scope=False),
        lambda d: d["negative_crops"][0].update(id="../escaped"),
        lambda d: d["negative_crops"][0].update(id="photo_0"),
        lambda d: d["negative_crops"][0].update(typo=True),
        lambda d: d["negative_crops"][0].update(crop_xyxy_normalized=[0, 0, 0.5, 0.5]),
    ],
)
def test_bad_review_is_rejected(review_case, change):
    *_, document, source = review_case
    change(document)
    with pytest.raises(ValueError):
        validate_review(document, source)


def test_source_revision_and_split_are_audited(review_case):
    *_, document, source = review_case
    row = document["records"][0]
    row["boxes"][0]["class"] = "moisture_trace"
    assert source_audit(row, source["records"][0])[0]["outcome"] == "revised"
    row["boxes"][1]["source_indices"] = [0]
    audit = source_audit(row, source["records"][0])
    assert audit[0]["reviewed_indices"] == [0, 1]
    assert audit[0]["outcome"] == "revised"


def test_changed_grouping_and_failed_export_publish_nothing(review_case):
    selection, cache, review, output, document, _ = review_case
    source = json.loads(selection.read_text())
    source["groups"] = [["photo_0", "photo_1"]]
    selection.write_text(json.dumps(source))
    document["selection_sha256"] = digest(selection.read_bytes())
    review.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="whole source groups"):
        apply_review(selection, cache, review, output)
    assert not output.exists()
    assert not list(output.parent.glob(".smartsite-prepare-*"))


def test_incomplete_class_export_and_tiny_crop_fail_atomically(review_case):
    selection, cache, review, output, document, _ = review_case
    document["records"][1]["boxes"].pop()
    review.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="every target class"):
        apply_review(selection, cache, review, output)
    assert not output.exists()
    document["negative_crops"][0]["crop_xyxy_normalized"] = [0.9, 0.9, 1, 1]
    review.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="too small"):
        apply_review(selection, cache, review, output)
    assert not output.exists()


def test_pinned_source_tamper_and_cli_errors(review_case, capsys):
    selection, cache, review, output, document, _ = review_case
    args = [
        "--selection",
        str(selection),
        "--cache",
        str(cache),
        "--review",
        str(review),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert main(args) == 1
    assert "FileExistsError" in capsys.readouterr().err
    document["selection_sha256"] = "0" * 64
    review.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="SHA-256"):
        apply_review(selection, cache, review, output.with_name("other"))


def test_shared_alert_family_does_not_change_existing_model_contract():
    assert DEFECT_FAMILIES["surface_loss"] == DEFECT_FAMILIES["peeling_paint"]
    assert CLASS_NAMES == ("crack", "surface_loss")
    assert get_class_names({}) == CLASS_NAMES
    with pytest.raises(ValueError):
        get_class_names({"class_names": ["peeling_paint"]})


@pytest.mark.parametrize("split", ["train", "validation"])
def test_explicit_full_negative_is_exported_and_auditable(review_case, split):
    selection, cache, review, output, document, _ = review_case
    row = document["records"][2]
    row.update(decision="negative", reviewed_scope=True, boxes=[])
    document["group_partitions"]["photo_2"] = split
    review.write_text(json.dumps(document))
    summary = apply_review(selection, cache, review, output)
    assert summary["negative_photos"] == 1
    assert summary["excluded_photos"] == 0
    assert summary["partitions"][split]["negative_photos"] == 1
    coco = COCO(str(output / split / "_annotations.coco.json"))
    photo = next(i for i in coco.imgs.values() if i["file_name"] == "photo_2.jpg")
    assert coco.getAnnIds(imgIds=photo["id"]) == []
    report = json.loads((output / "report.json").read_text())
    exported = next(r for r in report["records"] if r["id"] == "photo_2")
    assert exported["decision"] == "negative"
    assert exported["eligible_for_pilot"] and not exported["approved_for_training"]
    assert exported["source_audit"][0]["outcome"] == "removed"
    assert "Photo entière revue sans cible visible" in (output / "index.html").read_text()


@pytest.mark.parametrize("invalid", ["unreviewed", "positive_boxes", "missing_audit"])
def test_full_negative_requires_scope_empty_boxes_and_source_audit(review_case, invalid):
    *_, document, source = review_case
    row = document["records"][2]
    row.update(decision="negative", reviewed_scope=True, boxes=[])
    if invalid == "unreviewed":
        row["reviewed_scope"] = False
    elif invalid == "positive_boxes":
        row["boxes"] = copy.deepcopy(document["records"][0]["boxes"])
        row["removed_source_boxes"] = []
    else:
        row["removed_source_boxes"] = []
    with pytest.raises(ValueError):
        validate_review(document, source)
