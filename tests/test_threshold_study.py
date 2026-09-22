"""Comparer des seuils sur des cas synthétiques, sans moteur ni accès au test réservé."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from pycocotools import mask as coco_mask
from test_validation import rle
from test_validation import validation_inputs as validation_inputs

from smartsite_ia import model_cli, threshold_study, validation
from smartsite_ia.curation import digest
from smartsite_ia.importer import write_json
from smartsite_ia.threshold_study import checked_thresholds, f1, study_thresholds, threshold_cases
from smartsite_ia.validation_metrics import analyze_masks


@pytest.fixture
def cached_validation(validation_inputs, tmp_path, monkeypatch):
    corpus, training = validation_inputs
    monkeypatch.setattr(threshold_study, "verify_run", lambda _: (training, tmp_path / "weights"))
    exact, elsewhere = (coco_mask.decode(rle(x)).astype(bool) for x in (10, 40))
    # Une vraie cible sous 0,30, un doublon plus faible et une alerte ailleurs.
    detections = SimpleNamespace(
        xyxy=np.array([[10, 8, 30, 20], [10, 8, 30, 20], [10, 8, 30, 20], [40, 8, 60, 20]]),
        confidence=np.array([0.9, 0.2, 0.1, 0.07]),
        class_id=np.array([0, 1, 1, 1]),
        mask=np.stack([exact, exact, exact, elsewhere]),
    )
    monkeypatch.setattr(
        validation, "load_engine", lambda *a: SimpleNamespace(predict=lambda *a, **k: detections)
    )
    saved = tmp_path / "evaluation"
    report = validation.evaluate_validation(tmp_path / "run", corpus, saved, "cpu")
    return corpus, saved, report


def test_complete_cached_study_is_reproducible_and_does_not_run_model(
    cached_validation, tmp_path, monkeypatch
):
    corpus, saved, original = cached_validation
    monkeypatch.setattr(validation, "load_engine", lambda *a: pytest.fail("Unexpected inference"))
    read = threshold_study.read_local

    def guarded(root, path, limit):
        assert not path.startswith("test/")
        return read(root, path, limit)

    monkeypatch.setattr(threshold_study, "read_local", guarded)
    reports = []
    for folder in (tmp_path / "first", tmp_path / "second"):
        reports.append(study_thresholds(tmp_path / "run", corpus, saved, folder))
    assert reports[0] == reports[1]
    report = reports[0]
    assert report["selected_threshold"] is None and report["test_used"] is False
    assert report["best_observed_f1_thresholds"] == [0.15, 0.2]
    baseline = next(s for s in report["summaries"] if s["threshold"] == 0.3)
    assert baseline["counts"] == original["counts"]
    loose = report["summaries"][0]
    assert {k: loose["counts"]["surface_loss"][k] for k in ("tp", "fp", "fn")} == {
        "tp": 1,
        "fp": 2,
        "fn": 0,
    }
    assert loose["errors"] == {"duplicate": 1, "insufficient_overlap": 0, "no_overlap": 1}
    assert all(s["counts"]["crack"] == baseline["counts"]["crack"] for s in report["summaries"])
    assert [p["reason"] for p in report["records"][0]["variants"][0]["added"]] == [
        "matched",
        "duplicate",
        "no_overlap",
    ]
    assert report["source_hashes"]["predictions/easy_0001"] == digest(
        (saved / "easy_0001/predictions.json").read_bytes()
    )
    assert (tmp_path / "first/easy_0001/photo.jpg").read_bytes() == (
        corpus / "valid/easy_0001.jpg"
    ).read_bytes()
    for path in (tmp_path / "first").rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes()
                == (tmp_path / "second" / path.relative_to(tmp_path / "first")).read_bytes()
            )
    assert "Aucun seuil de production" in (tmp_path / "first/index.html").read_text()
    with pytest.raises(FileExistsError):
        study_thresholds(tmp_path / "run", corpus, saved, tmp_path / "first")
    with pytest.raises(ValueError, match="outside"):
        study_thresholds(tmp_path / "run", corpus, saved, saved / "nested")


@pytest.mark.parametrize(
    "change",
    [
        "test",
        "classes",
        "checkpoint",
        "inference",
        "protocol",
        "records",
        "hash",
        "photo",
        "dimensions",
        "counts",
        "errors",
        "aggregate",
    ],
)
def test_invalid_cache_fails_without_partial_output(cached_validation, tmp_path, change):
    corpus, saved, report = cached_validation
    if change == "test":
        report["split"] = "test"
    elif change == "classes":
        report["class_names"] = ["crack"]
    elif change == "checkpoint":
        report["checkpoint_sha256"] = "b" * 64
    elif change == "inference":
        report["inference"]["preprocessing"] = "unknown"
    elif change == "protocol":
        report["protocol"]["score_floor"] = 0.4
    elif change == "records":
        report["records"][0]["id"] = "../../outside"
    elif change == "hash":
        report["records"][0]["image_sha256"] = "b" * 64
    elif change == "photo":
        (corpus / "valid/easy_0001.jpg").write_bytes(b"bad")
    elif change == "dimensions":
        report["records"][0]["image_id"] = 99
    elif change == "counts":
        report["records"][0]["counts"]["crack"]["tp"] = 9
    elif change == "errors":
        report["records"][0]["mask_errors"]["crack"]["no_overlap"] = 9
    else:
        report["counts"]["crack"]["tp"] = 9
    write_json(saved / "report.json", report)
    with pytest.raises(ValueError):
        study_thresholds(tmp_path / "run", corpus, saved, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".smartsite-prepare-*"))


@pytest.mark.parametrize(
    "grid",
    [
        (0.3,),
        (0.1, 0.2),
        (0.3, 0.3),
        (0.0001, 0.3),
        (0.3, 1),
        (False, 0.3),
        (float("nan"), 0.3),
        (0.1,) * 21,
    ],
)
def test_grid_rejects_unmeasurable_or_ambiguous_thresholds(grid):
    with pytest.raises(ValueError):
        checked_thresholds(grid)


def test_matching_at_score_boundary_and_removals():
    refs = [{"id": 41, "segmentation": rle()}]
    proposals = [
        {"id": 3, "score": 0.3, "mask_rle": rle()},
        {"id": 2, "score": 0.2, "mask_rle": rle()},
    ]
    cases = threshold_cases(refs, proposals, (0.2, 0.3, 0.4))
    assert cases[0]["added"][0]["reason"] == "duplicate"
    assert cases[1]["counts"] == {"tp": 1, "fp": 0, "fn": 0}
    assert cases[2]["removed"][0]["reference_id"] == 41
    assert cases[2]["counts"] == {"tp": 0, "fp": 0, "fn": 1}
    for case in cases:
        expected = analyze_masks(
            [{**refs[0], "category_id": 1}],
            [
                {"category_id": 1, "segmentation": p["mask_rle"], "score": p["score"]}
                for p in proposals
            ],
            case["threshold"],
        )["crack"]
        assert case["counts"] == expected["counts"]
    assert f1({"tp": 0, "fp": 0, "fn": 0}) is None
    assert threshold_cases([], [], (0.3,))[0]["counts"] == {"tp": 0, "fp": 0, "fn": 0}
    assert threshold_cases([], proposals, (0.3,))[0]["counts"]["fp"] == 1


def test_study_cli_forwards_grid_and_class(monkeypatch, capsys):
    def run(*args):
        assert args[-2:] == ("crack", (0.2, 0.3))
        return {"images": 1, "best_observed_f1_thresholds": [0.2]}

    monkeypatch.setattr(model_cli, "study_thresholds", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "smartsite-model",
            "study-thresholds",
            "corpus",
            "--run",
            "run",
            "--evaluation",
            "evaluation",
            "--output",
            "out",
            "--class",
            "crack",
            "--thresholds",
            "0.2",
            "0.3",
        ],
    )
    assert model_cli.main() == 0
    assert json.loads(capsys.readouterr().out)["selected_threshold"] is None
