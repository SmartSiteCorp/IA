import hashlib
import json
from html.parser import HTMLParser

import pytest
from PIL import Image

from smartsite_ia import cli
from smartsite_ia.importer import import_archive
from smartsite_ia.review import compare_masks, load_samples, read_local, review_corpus


@pytest.fixture
def corpus(tmp_path, archive_factory):
    archive, source = archive_factory()
    root = tmp_path / "corpus"
    import_archive(archive, root, source)
    return root


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for name, value in attrs if name in ("src", "href"))


def test_review_is_reproducible_preserves_source_and_links_work(corpus, tmp_path):
    before = snapshot(corpus)
    left, right = tmp_path / "left", tmp_path / "right"
    report = review_corpus(corpus, left)
    assert report == review_corpus(corpus, right)
    assert snapshot(left) == snapshot(right)
    assert snapshot(corpus) == before
    assert report["images"] == 1 and report["annotations"] == 1
    assert report["approved_for_training"] is False
    assert report["records"][0]["scene_group"] is None
    assert report["records"][0]["review_status"] == "pending"
    assert (left / "samples/easy_0001/photo.jpg").read_bytes() == (
        corpus / "images/easy_0001.jpg"
    ).read_bytes()
    assert report["mask_totals"]["pillow_disagreement_pixels"] == 0
    assert report["mask_totals"]["coco_disagreement_pixels"] > 0
    assert report["mask_totals"]["disagreement_outside_1px_boundaries"] == 0
    page = (left / "index.html").read_text()
    assert "pas les résultats d'une IA" in page
    assert "<script" not in page
    parser = Links()
    parser.feed(page)
    for target in parser.targets:
        if not target.startswith(("#", "https:")):
            assert (left / target).is_file(), target


def test_identical_images_produce_pending_pair_and_group(tmp_path, archive_factory):
    archive, source = archive_factory(copies=2)
    root = tmp_path / "corpus"
    import_archive(archive, root, source)
    report = review_corpus(root, tmp_path / "review")
    assert report["candidate_groups"] == [["easy_0001", "easy_0002"]]
    assert report["similar_pairs"][0]["kind"] == "exact_pixels"
    decisions = json.loads((tmp_path / "review/review-decisions.template.json").read_text())
    assert decisions["pairs"][0]["decision"] == "pending"
    assert decisions["pairs"][0]["reviewer"] is None


def test_empty_annotations_are_flagged_not_assumed_healthy(tmp_path, archive_factory, sample):
    from conftest import image_bytes

    sample["document"]["annotations"] = []
    base = "Damage Segmentaion/Easy/Labels"
    archive, source = archive_factory(
        {
            f"{base}/Mask/E (1)_mask.png": image_bytes(Image.new("RGB", (640, 640)), "PNG"),
            f"{base}/Yolo/E (1).txt": b"",
        }
    )
    root = tmp_path / "corpus"
    import_archive(archive, root, source)
    report = review_corpus(root, tmp_path / "review")
    assert report["flag_counts"] == {"empty_annotations": 1}
    assert report["detailed_samples"] == ["easy_0001"]
    assert report["records"][0]["review_status"] == "pending"


@pytest.mark.parametrize(
    "relative", ["images/easy_0001.jpg", "masks/easy_0001.png", "source_annotations/easy_0001.json"]
)
def test_changed_source_leaves_no_partial_review(corpus, tmp_path, relative):
    path = corpus / relative
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "review"
    with pytest.raises(ValueError, match="differ from manifest"):
        review_corpus(corpus, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".smartsite-review-*"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "../escape"),
        ("image", "../escape.jpg"),
        ("difficulty", "Unknown"),
        ("scene_group", "guessed"),
        ("split", "train"),
        ("annotation_count", 99),
    ],
)
def test_invalid_or_edited_manifest_rejected(corpus, tmp_path, field, value):
    path = corpus / "manifest.json"
    data = json.loads(path.read_text())
    data["records"][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        review_corpus(corpus, tmp_path / "out")


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"schema_version": 2},
        {"schema_version": 1},
        {"schema_version": 1, "records": []},
        {"schema_version": 1, "records": [None]},
    ],
)
def test_bad_manifest_layout_rejected(corpus, document):
    (corpus / "manifest.json").write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_samples(corpus)


def test_duplicate_ids_rejected(corpus):
    path = corpus / "manifest.json"
    data = json.loads(path.read_text())
    data["records"] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        load_samples(corpus)


def test_output_cannot_overwrite_or_modify_corpus(corpus, tmp_path):
    existing = tmp_path / "out"
    existing.mkdir()
    (existing / "keep").write_text("user data")
    with pytest.raises(FileExistsError):
        review_corpus(corpus, existing)
    assert (existing / "keep").read_text() == "user data"
    with pytest.raises(ValueError, match="outside"):
        review_corpus(corpus, corpus / "review")
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        review_corpus(corpus, dangling)


def test_read_limits_and_symlinks(corpus, tmp_path):
    with pytest.raises(ValueError, match="size limit"):
        read_local(corpus, "manifest.json", 1)
    outside = tmp_path / "outside"
    outside.write_text("private")
    with pytest.raises(ValueError, match="escapes"):
        read_local(corpus, "../outside", 100)
    (corpus / "linked").symlink_to(corpus / "manifest.json")
    with pytest.raises(ValueError, match="Linked"):
        read_local(corpus, "linked", 10000)
    (corpus / "linked-dir").symlink_to(corpus / "images", target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        read_local(corpus, "linked-dir/easy_0001.jpg", 100000)


def test_large_interior_mask_disagreement_is_not_only_a_border_effect():
    source = Image.new("RGB", (640, 640))
    rendered = Image.new("RGB", source.size, "red")
    report = compare_masks(source, rendered, rendered)
    assert report["disagreement_outside_1px_boundaries"] == 640 * 640


def test_shifted_mask_triggers_iou_and_interior_flags(tmp_path, archive_factory):
    from conftest import image_bytes
    from PIL import ImageDraw

    mask = Image.new("RGB", (640, 640))
    ImageDraw.Draw(mask).rectangle((20, 10, 40, 20), fill="red")
    archive, source = archive_factory(
        {"Damage Segmentaion/Easy/Labels/Mask/E (1)_mask.png": image_bytes(mask, "PNG")}
    )
    corpus = tmp_path / "corpus"
    import_archive(archive, corpus, source)
    report = review_corpus(corpus, tmp_path / "review")
    assert report["flag_counts"]["mask_iou_below_review_threshold"] == 1
    assert report["flag_counts"]["mask_difference_away_from_boundary"] == 1


def test_overlapping_classes_are_not_hidden_by_last_rendered_color(
    tmp_path, archive_factory, sample
):
    from conftest import image_bytes
    from PIL import ImageDraw

    other = dict(sample["document"]["annotations"][0], category_id=1)
    sample["document"]["annotations"].append(other)
    sample["yolo"] += sample["yolo"].replace(b"0 ", b"1 ", 1)
    mask = Image.new("RGB", (640, 640))
    ImageDraw.Draw(mask).rectangle((10, 10, 30, 20), fill="blue")
    sample["mask"] = image_bytes(mask, "PNG")
    archive, source = archive_factory()
    corpus = tmp_path / "corpus"
    import_archive(archive, corpus, source)
    report = review_corpus(corpus, tmp_path / "review")
    record = report["records"][0]
    assert record["cross_class_overlap_pixels"] == 200
    assert record["overlap_fraction_of_smaller_class"] == 1.0
    assert report["flag_counts"]["source_classes_almost_fully_overlap"] == 1


def test_image_changed_during_review_is_not_published(corpus, tmp_path, monkeypatch):
    from smartsite_ia import review

    original = review.select_panels

    def change_file(records):
        image = corpus / "images/easy_0001.jpg"
        image.write_bytes(image.read_bytes() + b" ")
        return original(records)

    monkeypatch.setattr(review, "select_panels", change_file)
    with pytest.raises(ValueError, match="differ from manifest"):
        review_corpus(corpus, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_review_cli_runs_and_reports_errors(corpus, tmp_path, monkeypatch, capsys):
    output = tmp_path / "review"
    monkeypatch.setattr(
        "sys.argv", ["smartsite-data", "review", str(corpus), "--output", str(output)]
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["review_page"] == str(output / "index.html")
    assert cli.main() == 1
    assert "already exists" in capsys.readouterr().err
