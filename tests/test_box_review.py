"""Tester la revue sur des photos synthétiques, sans réseau ni moteur obligatoire."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_box_training import box_case as box_case
from test_box_training import fake_engine as fake_engine
from test_collection_review import review_case as review_case

from smartsite_ia import box_cli, box_review, box_training
from smartsite_ia.box_data import BOX_CLASSES
from smartsite_ia.box_metrics import match_boxes
from smartsite_ia.curation import digest
from smartsite_ia.importer import write_json


def detections(boxes=(), scores=(), labels=(), names=None):
    return SimpleNamespace(
        xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
        confidence=np.asarray(scores, dtype=float),
        class_id=np.asarray(labels, dtype=float),
        mask=None,
        data={} if names is None else {"class_name": np.asarray(names)},
    )


@pytest.fixture
def completed(fake_engine, monkeypatch):
    corpus, config, weights, run, _ = fake_engine
    parent = box_training.train_boxes(corpus, config, weights, run, "cpu")
    monkeypatch.setattr(box_review, "runtime", lambda device: parent["runtime"])
    selection = json.loads((run / "data/selection.json").read_text())
    row = selection["splits"]["valid"]["records"][0]
    calls = []

    class Engine:
        def predict(self, photo, threshold):
            calls.append((photo.size, threshold))
            boxes = [
                [
                    v * bound
                    for v, bound in zip(
                        box["xyxy_normalized"], (photo.width, photo.height) * 2, strict=True
                    )
                ]
                for box in row["boxes"]
            ]
            return detections(
                boxes, [0.8] * len(boxes), [BOX_CLASSES.index(b["class"]) for b in row["boxes"]]
            )

    monkeypatch.setattr(box_review, "load_box_engine", lambda *args: Engine())
    return corpus, config, run, calls


def test_complete_gallery_has_same_photos_correct_matches_and_no_training(completed, tmp_path):
    corpus, config, run, calls = completed
    before = {p: digest(p.read_bytes()) for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "gallery"
    report = box_review.review_boxes(run, corpus, config, output, "cpu")
    assert len(calls) == 1 and calls[0][1] == 0.3
    assert report["training_started"] is False and report["test_used"] is False
    assert report["human_validation"] == "pending" and not report["qualified_for_smartsite"]
    assert all(c["tp"] == 1 and c["fp"] == c["fn"] == 0 for c in report["summary"].values())
    row = report["records"][0]
    assert row["missed_reference_ids"] == [] and len(row["matches"]) == 3
    assert (output / row["id"] / "photo.jpg").read_bytes() == (
        run / "data/valid" / f"{row['id']}.jpg"
    ).read_bytes()
    page = (output / "index.html").read_text()
    assert "<script>" not in page and "Prédictions du modèle" in page
    assert "Retrouvées" in page and "seuil fixe" in page
    assert all(before[p] == digest(p.read_bytes()) for p in before)
    with pytest.raises(FileExistsError):
        box_review.review_boxes(run, corpus, config, output, "cpu")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "model",
        "checkpoint",
        "config_sha256",
        "checkpoint_sha256",
        "metrics_sha256",
        "last_epoch_metrics",
    ],
)
def test_changed_run_refused_before_model(completed, tmp_path, change):
    corpus, config, run, calls = completed
    report = json.loads((run / "run.json").read_text())
    report[change] = "changed"
    write_json(run / "run.json", report)
    with pytest.raises(ValueError):
        box_review.review_boxes(run, corpus, config, tmp_path / "out", "cpu")
    assert not calls and not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "change", ["photo", "selection", "metrics", "weights", "corpus", "versions"]
)
def test_corrupt_input_and_runtime_refused(completed, monkeypatch, tmp_path, change):
    corpus, config, run, calls = completed
    paths = {
        "photo": run / "data/valid/photo_1.jpg",
        "selection": run / "data/selection.json",
        "metrics": run / "checkpoints/metrics.csv",
        "weights": run / "checkpoints/checkpoint_best_total.pth",
        "corpus": corpus / "report.json",
    }
    if change == "versions":
        monkeypatch.setattr(box_review, "runtime", lambda device: {"versions": {}})
    elif change == "selection":
        doc = json.loads(paths[change].read_text())
        doc["human_validation"] = "approved"
        write_json(paths[change], doc)
    else:
        paths[change].write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        box_review.review_boxes(run, corpus, config, tmp_path / "out", "cpu")
    assert not calls


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failure_cleans_partial_gallery_and_preserves_run(
    completed, monkeypatch, tmp_path, failure
):
    corpus, config, run, _ = completed
    original = (run / "run.json").read_bytes()

    def stop(*args, **kwargs):
        raise failure("synthetic interruption")

    monkeypatch.setattr(box_review, "load_box_engine", stop)
    output = tmp_path / "out"
    with pytest.raises(failure):
        box_review.review_boxes(run, corpus, config, output, "cpu")
    assert not output.exists() and not list(tmp_path.glob(".smartsite-prepare-*"))
    assert (run / "run.json").read_bytes() == original


@pytest.mark.parametrize("threshold", [0, 1, True, float("nan"), float("inf")])
def test_invalid_threshold_fails_before_input_read(tmp_path, threshold):
    with pytest.raises(ValueError):
        box_review.review_boxes(tmp_path, tmp_path, tmp_path, tmp_path / "out", "cpu", threshold)


def test_duplicate_wrong_class_and_insufficient_overlap():
    refs = [
        {"id": 1, "class_name": BOX_CLASSES[0], "bbox_xyxy": [0, 0, 10, 10]},
        {"id": 2, "class_name": BOX_CLASSES[1], "bbox_xyxy": [40, 40, 50, 50]},
    ]
    predictions = [
        {"id": 1, "class_name": BOX_CLASSES[0], "bbox_xyxy": [0, 0, 10, 10], "score": 0.7},
        {"id": 2, "class_name": BOX_CLASSES[0], "bbox_xyxy": [0, 0, 10, 10], "score": 0.9},
        {"id": 3, "class_name": BOX_CLASSES[2], "bbox_xyxy": [40, 40, 50, 50], "score": 0.9},
        {"id": 4, "class_name": BOX_CLASSES[1], "bbox_xyxy": [40, 40, 60, 60], "score": 0.9},
    ]
    result = box_review.compare_boxes(refs, predictions)
    assert result["counts"][BOX_CLASSES[0]] == {"tp": 1, "fp": 1, "fn": 0}
    assert result["counts"][BOX_CLASSES[1]] == {"tp": 0, "fp": 1, "fn": 1}
    assert result["missed_reference_ids"] == [2]
    assert {m["reason"] for m in result["matches"]} == {
        "matched",
        "duplicate",
        "insufficient_overlap",
        "no_overlap",
    }
    assert next(m for m in result["matches"] if m["reason"] == "matched")["prediction_id"] == 2


def test_iou_boundary_score_ties_and_empty_cases():
    refs = [[0, 0, 10, 10]]
    predictions = [{"bbox_xyxy": [0, 0, 10, 20], "score": 0.5}] * 2
    result = match_boxes(refs, predictions, 0.5)
    assert result[0]["prediction_index"] == 0 and result[0]["matched_iou"] == 0.5
    assert result[1]["reason"] == "duplicate"
    assert box_review.compare_boxes([], [])["missed_reference_ids"] == []
    with pytest.raises(ValueError):
        match_boxes([], [], 0)


def test_coordinates_clipping_background_and_degenerate_are_traced():
    output = box_review.encode_boxes(
        detections(
            [[-2, 1, 66, 62], [0, 0, 10, 10], [64, 0, 80, 10]],
            [0.7, 0.9, 0.8],
            [0, 3, 1],
            [BOX_CLASSES[0], "__background__", BOX_CLASSES[1]],
        ),
        Image.new("RGB", (64, 64)),
    )
    assert output["predictions"][0]["bbox_xyxy"] == [0, 1, 64, 62]
    assert output["predictions"][0]["clipped"]
    assert output["predictions"][0]["raw_bbox_xyxy"] == [-2, 1, 66, 62]
    assert [r["reason"] for r in output["discarded"]] == ["background", "empty_after_clipping"]
    assert box_review.encode_boxes(detections(), Image.new("RGB", (64, 64)))["predictions"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: setattr(d, "confidence", None),
        lambda d: setattr(d, "mask", np.zeros((1, 64, 64))),
        lambda d: setattr(d, "class_id", np.array([4])),
        lambda d: setattr(d, "class_id", np.array([0.2])),
        lambda d: setattr(d, "confidence", np.array([float("nan")])),
        lambda d: setattr(d, "confidence", np.array([1.1])),
        lambda d: setattr(d, "xyxy", np.array([[10, 0, 0, 10]])),
        lambda d: setattr(d, "xyxy", np.array([[0, 0, float("inf"), 10]])),
        lambda d: setattr(d, "data", {"class_name": []}),
        lambda d: setattr(d, "data", {"class_name": [BOX_CLASSES[1]]}),
    ],
)
def test_invalid_engine_outputs_rejected(mutation):
    value = detections([[0, 0, 10, 10]], [0.7], [0])
    mutation(value)
    with pytest.raises(ValueError):
        box_review.encode_boxes(value, Image.new("RGB", (64, 64)))


def test_cli_review_and_escaped_empty_gallery(completed, monkeypatch, tmp_path):
    corpus, config, run, _ = completed
    monkeypatch.setattr(
        box_review,
        "load_box_engine",
        lambda *args: SimpleNamespace(predict=lambda *args, **kwargs: detections()),
    )
    output = tmp_path / "out"
    assert (
        box_cli.main(
            [
                "review",
                str(run),
                "--corpus",
                str(corpus),
                "--config",
                str(config),
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    doc = json.loads((output / "report.json").read_text())
    assert sum(r["fn"] for r in doc["summary"].values()) == 3
    assert "Aucune proposition au seuil choisi" in (output / "index.html").read_text()
    doc["records"][0]["annotation_note"] = '<script>alert("x")</script>'
    box_review.write_box_gallery(output, doc)
    assert "<script>" not in (output / "index.html").read_text()
    assert "&lt;script&gt;" in (output / "index.html").read_text()


def test_loader_disables_unsafe_checkpoint_loading(monkeypatch, tmp_path):
    calls = []

    class RFDETRNano:
        pass

    engine = RFDETRNano()

    def load(path, **options):
        calls.append((path, options))
        return engine

    monkeypatch.setattr(
        box_review.importlib,
        "import_module",
        lambda name: SimpleNamespace(RFDETR=SimpleNamespace(from_checkpoint=load)),
    )
    monkeypatch.setattr(box_review, "AlignedPredictor", lambda obj, **kwargs: (obj, kwargs))
    path = tmp_path / "model.pth"
    result = box_review.load_box_engine(path, "cpu")
    assert calls == [(str(path.resolve()), {"device": "cpu", "trust_checkpoint": False})]
    assert result == (engine, {"box_classes": BOX_CLASSES})
    engine = object()
    with pytest.raises(ValueError, match="Nano"):
        box_review.load_box_engine(path, "cpu")


def test_missing_runtime_and_output_inside_run_refused(completed):
    corpus, config, run, calls = completed
    with pytest.raises(ValueError, match="outside"):
        box_review.review_boxes(run, corpus, config, run / "gallery", "cpu")
    doc = json.loads((run / "run.json").read_text())
    doc.pop("runtime")
    write_json(run / "run.json", doc)
    with pytest.raises(ValueError, match="completed"):
        box_review.review_boxes(run, corpus, config, run / "gallery", "cpu")
    assert not calls


def test_oversized_engine_output_refused():
    with pytest.raises(ValueError, match="dimensions"):
        box_review.encode_boxes(
            detections([[0, 0, 10, 10]] * 501, [0.5] * 501, [0] * 501), Image.new("RGB", (64, 64))
        )


@pytest.mark.parametrize("change", ["photo", "checkpoint", "run", "score"])
def test_changed_inputs_during_prediction_cancel_publication(
    completed, tmp_path, monkeypatch, change
):
    corpus, config, run, _ = completed
    output = tmp_path / "out"

    def load(*args):
        if change == "photo":
            (run / "data/valid/photo_1.jpg").write_bytes(b"changed")

        def predict(*args, **kwargs):
            if change == "checkpoint":
                (run / "checkpoints/checkpoint_best_total.pth").write_bytes(b"changed")
            if change == "run":
                (run / "run.json").write_bytes(b"changed")
            return detections([[0, 0, 10, 10]], [0.01], [0]) if change == "score" else detections()

        return SimpleNamespace(predict=predict)

    monkeypatch.setattr(box_review, "load_box_engine", load)
    with pytest.raises(ValueError):
        box_review.review_boxes(run, corpus, config, output, "cpu")
    assert not output.exists()
