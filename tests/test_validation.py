"""Des cas synthétiques pour vérifier les mesures et les fuites, sans moteur lourd."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from smartsite_ia import model_cli, validation, validation_metrics
from smartsite_ia.importer import write_json
from smartsite_ia.learning import file_hash
from smartsite_ia.model_assets import CLASS_NAMES


def rle(x=10):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:20, x : x + 20] = 1
    result = coco_mask.encode(np.asfortranarray(mask))
    result["counts"] = result["counts"].decode("ascii")
    return result


def document():
    return {
        "images": [{"id": 1, "file_name": "easy_0001.jpg", "width": 64, "height": 64}],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, 1)],
        "annotations": [
            dict(
                id=i,
                image_id=1,
                category_id=i,
                segmentation=rle(),
                bbox=[10, 8, 20, 12],
                area=240,
                iscrowd=0,
            )
            for i in (1, 2)
        ],
    }


def proposals():
    return [dict(image_id=1, category_id=i, segmentation=rle(), score=0.9) for i in (1, 2)]


def test_perfect_and_empty_predictions():
    gt = document()
    result = validation_metrics.coco_scores(gt, proposals())
    assert result["mask_map_50_95"] == pytest.approx(1.0)
    assert validation_metrics.coco_scores(gt, [])["mask_map_50_95"] == 0
    counts = validation_metrics.counts_for_image(gt["annotations"], [], 0.3)
    totals = validation_metrics.summarize_counts([counts])
    assert totals["crack"]["recall"] == 0 and totals["crack"]["precision"] is None


def test_duplicates_wrong_classes_and_low_scores():
    refs = document()["annotations"][:1]
    preds = proposals() + [proposals()[0], {**proposals()[0], "score": 0.2}]
    counts = validation_metrics.counts_for_image(refs, preds, 0.3)
    assert counts == {
        "crack": {"tp": 1, "fp": 1, "fn": 0},
        "surface_loss": {"tp": 0, "fp": 1, "fn": 0},
    }
    preds[0]["segmentation"] = rle(40)
    assert validation_metrics.counts_for_image(refs, preds[:1], 0.3)["crack"] == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }
    assert validation_metrics.finite_mean(np.array([-1])) is None


def test_error_diagnostics_distinguish_duplicates_and_partial_masks():
    refs = document()["annotations"][:1]
    preds = [
        proposals()[0],
        proposals()[0],
        {**proposals()[0], "segmentation": rle(22)},
        {**proposals()[0], "segmentation": rle(40)},
    ]
    result = validation_metrics.analyze_masks(refs, preds, 0.3)["crack"]
    assert result["counts"] == {"tp": 1, "fp": 3, "fn": 0}
    assert result["errors"] == {"duplicate": 1, "insufficient_overlap": 1, "no_overlap": 1}
    assert sum(result["errors"].values()) == result["counts"]["fp"]
    empty = validation_metrics.analyze_masks([], preds, 0.3)["crack"]
    assert empty["errors"]["no_overlap"] == 4


@pytest.fixture
def validation_inputs(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "valid").mkdir(parents=True)
    photo = corpus / "valid/easy_0001.jpg"
    Image.new("RGB", (64, 64), "gray").save(photo)
    record = dict(id="easy_0001", split="valid", difficulty="Easy", image_sha256=file_hash(photo))
    for name in ("manifest.json", "report.json"):
        write_json(corpus / name, dict(schema_version=1, records=[record]))
    write_json(corpus / "valid/_annotations.coco.json", document())
    report = dict(
        config={
            "corpus_sha256": {
                name: file_hash(corpus / name)
                for name in ("manifest.json", "report.json", "valid/_annotations.coco.json")
            }
        },
        checkpoint_sha256="a" * 64,
    )
    monkeypatch.setattr(validation, "verify_run", lambda run: (report, tmp_path / "weights"))
    monkeypatch.setattr(validation, "runtime", lambda device: {"device": device})
    mask = coco_mask.decode(rle()).astype(bool)
    det = SimpleNamespace(
        xyxy=np.array([[10, 8, 30, 20], [10, 8, 30, 20]]),
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([0, 1]),
        mask=np.stack([mask, mask]),
    )
    monkeypatch.setattr(
        validation, "load_engine", lambda *args: SimpleNamespace(predict=lambda *a, **kw: det)
    )
    return corpus, report


def test_complete_evaluation_comparison_without_test_access(
    validation_inputs, tmp_path, monkeypatch
):
    corpus, _ = validation_inputs
    original = validation.read_local

    def guarded(root, path, limit):
        assert not str(path).startswith("test/")
        return original(root, path, limit)

    monkeypatch.setattr(validation, "read_local", guarded)
    before, after = tmp_path / "before", tmp_path / "after"
    for out in (before, after):
        report = validation.evaluate_validation(tmp_path / "run", corpus, out, "cpu")
        assert report["images"] == 1 and report["test_used"] is False
        assert report["counts"]["crack"]["tp"] == 1
        assert report["coco"]["mask_map_50_95"] == pytest.approx(1.0)
        assert report["mask_errors"]["crack"]["no_overlap"] == 0
        assert report["inference"]["preprocessing"] == "public-v1"
    output = tmp_path / "comparison"
    validation.compare_validations(before, after, output)
    assert (output / "easy_0001/after.jpg").read_bytes() == (
        after / "easy_0001/prediction.jpg"
    ).read_bytes()
    assert "Avant" in (output / "index.html").read_text()
    assert "poids du modèle sont identiques" in (output / "index.html").read_text()
    with pytest.raises(FileExistsError):
        validation.compare_validations(before, after, output)


@pytest.mark.parametrize(
    "change",
    [
        "pin",
        "photo",
        "dimensions",
        "categories",
        "duplicate_image",
        "crowd",
        "annotation_image",
        "bad_id",
    ],
)
def test_validation_refuses_changed_inputs(validation_inputs, tmp_path, change):
    corpus, report = validation_inputs
    path = corpus / "valid/_annotations.coco.json"
    doc = json.loads(path.read_text())
    if change == "photo":
        (corpus / "valid/easy_0001.jpg").write_bytes(b"corrupt")
    elif change == "dimensions":
        doc["images"][0]["width"] = 65
    elif change == "categories":
        doc["categories"][0]["name"] = "person"
    elif change == "duplicate_image":
        doc["images"].append(doc["images"][0])
    elif change == "crowd":
        doc["annotations"][0]["iscrowd"] = 1
    elif change == "annotation_image":
        doc["annotations"][0]["image_id"] = 99
    elif change == "bad_id":
        doc["images"][0]["file_name"] = "../../outside"
    write_json(path, doc)
    if change != "pin":
        report["config"]["corpus_sha256"]["valid/_annotations.coco.json"] = file_hash(path)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        validation.evaluate_validation(tmp_path / "run", corpus, tmp_path / "bad", "cpu")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "change", ["protocol", "image_hash", "annotations", "split", "unsafe_id", "inference"]
)
def test_comparison_requires_same_data_protocol(validation_inputs, tmp_path, change):
    corpus, _ = validation_inputs
    report = validation.evaluate_validation(tmp_path / "run", corpus, tmp_path / "before", "cpu")
    second = copy.deepcopy(report)
    if change == "protocol":
        second["protocol"]["display_threshold"] = 0.8
    elif change == "image_hash":
        second["records"][0]["image_sha256"] = "b" * 64
    elif change == "annotations":
        second["annotations_sha256"] = "b" * 64
    elif change == "split":
        second["split"] = "test"
    elif change == "unsafe_id":
        second["records"][0]["id"] = "../../outside"
    elif change == "inference":
        second["inference"]["preprocessing"] = "unknown"
    (tmp_path / "after").mkdir()
    write_json(tmp_path / "after/report.json", second)
    with pytest.raises(ValueError):
        validation.compare_validations(tmp_path / "before", tmp_path / "after", tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("command", ["evaluate", "compare"])
def test_validation_cli(command, monkeypatch, capsys):
    args = (
        ["corpus", "--run", "run", "--device", "cpu"]
        if command == "evaluate"
        else ["before", "after"]
    )
    monkeypatch.setattr("sys.argv", ["smartsite-model", command, *args, "--output", "output"])
    monkeypatch.setattr(model_cli, "evaluate_validation", lambda *args: {"images": 1})
    monkeypatch.setattr(model_cli, "compare_validations", lambda *args: {})
    assert model_cli.main() == 0 and "report" in json.loads(capsys.readouterr().out)


def test_comparison_allows_changed_inference_but_keeps_measurement_rules(
    validation_inputs, tmp_path
):
    corpus, _ = validation_inputs
    before = tmp_path / "before"
    report = validation.evaluate_validation(tmp_path / "run", corpus, before, "cpu")
    after = tmp_path / "after"
    import shutil

    shutil.copytree(before, after)
    from smartsite_ia.inference import inference_metadata

    report["inference"] = inference_metadata("training-v1", 0.6)
    write_json(after / "report.json", report)
    validation.compare_validations(before, after, tmp_path / "comparison")
    page = (tmp_path / "comparison/index.html").read_text()
    assert "60%" in page and "aligné" in page
