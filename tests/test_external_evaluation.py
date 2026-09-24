"""Vérifier la lecture du corpus externe et ses refus, sans charger le moteur."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from smartsite_ia import external_evaluation
from smartsite_ia.curation import digest
from smartsite_ia.importer import write_json


def build_corpus(tmp_path, *, records=None, **report_changes):
    """Un corpus externe minimal : une photo, son masque et le rapport de préparation."""
    root = tmp_path / "ccsd"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)

    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8), mode="RGB")
    photo.save(root / "images" / "001.png")
    mask = np.zeros((64, 48), dtype=np.uint8)
    mask[10:20, 5:15] = 255
    Image.fromarray(mask, mode="L").save(root / "masks" / "001.png")

    record = {
        "id": "001",
        "split": "external_candidate",
        "content_group": "g1",
        "image": "images/001.png",
        "mask": "masks/001.png",
        "width": 48,
        "height": 64,
        "foreground_pixels": 100,
        "image_sha256": digest((root / "images" / "001.png").read_bytes()),
        "mask_sha256": digest((root / "masks" / "001.png").read_bytes()),
    }
    report = {
        "schema_version": 1,
        "dataset": "concrete_crack_segmentation_v1",
        "approved_for_training": False,
        "records": [record] if records is None else records,
        **report_changes,
    }
    write_json(root / "report.json", report)
    return root, record


def test_selection_keeps_only_the_requested_split(tmp_path):
    root, record = build_corpus(
        tmp_path, records=[record_of("a", "quarantine"), {**base_record(), "id": "001"}]
    )
    records, report_hash = external_evaluation.load_external_records(root, "external_candidate")
    assert [r["id"] for r in records] == ["001"]
    assert len(report_hash) == 64


def base_record():
    return {"id": "001", "split": "external_candidate", "content_group": "g1"}


def record_of(name, split):
    return {"id": name, "split": split, "content_group": "g0"}


def test_another_dataset_is_refused(tmp_path):
    root, _ = build_corpus(tmp_path, dataset="damsegment_v1")
    with pytest.raises(ValueError, match="CCSD"):
        external_evaluation.load_external_records(root, "external_candidate")


def test_corpus_opened_for_training_is_refused(tmp_path):
    """Réserve d'évaluation : elle ne doit jamais devenir un corpus d'apprentissage."""
    root, _ = build_corpus(tmp_path, approved_for_training=True)
    with pytest.raises(ValueError, match="outside training"):
        external_evaluation.load_external_records(root, "external_candidate")


def test_empty_selection_is_refused(tmp_path):
    root, _ = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="selection size"):
        external_evaluation.load_external_records(root, "absent_split")


def test_reference_mask_is_read_as_a_crack_zone(tmp_path):
    root, record = build_corpus(tmp_path)
    encoded = external_evaluation.read_reference_mask(root, record)
    assert int(coco_mask.area(encoded)) == 100


def test_changed_mask_is_refused(tmp_path):
    root, record = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="mask changed"):
        external_evaluation.read_reference_mask(root, {**record, "mask_sha256": "0" * 64})


def test_mask_disagreeing_with_its_record_is_refused(tmp_path):
    root, record = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="foreground disagrees"):
        external_evaluation.read_reference_mask(root, {**record, "foreground_pixels": 99})


def test_grey_mask_is_refused(tmp_path):
    """Un masque non binarisé fausserait la zone de référence sans prévenir."""
    root, record = build_corpus(tmp_path)
    grey = np.zeros((64, 48), dtype=np.uint8)
    grey[10:20, 5:15] = 128
    Image.fromarray(grey, mode="L").save(root / "masks" / "001.png")
    changed = {**record, "mask_sha256": digest((root / "masks" / "001.png").read_bytes())}
    with pytest.raises(ValueError, match="not binary"):
        external_evaluation.read_reference_mask(root, changed)


def test_mask_of_another_size_is_refused(tmp_path):
    root, record = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="Unsupported external mask"):
        external_evaluation.read_reference_mask(root, {**record, "width": 40})


def test_photo_is_checked_against_its_fingerprint(tmp_path):
    root, record = build_corpus(tmp_path)
    assert external_evaluation.read_photo(root, record).size == (48, 64)
    with pytest.raises(ValueError, match="photo changed"):
        external_evaluation.read_photo(root, {**record, "image_sha256": "0" * 64})


def test_photo_dimensions_must_match_the_record(tmp_path):
    root, record = build_corpus(tmp_path)
    with pytest.raises(ValueError, match="dimensions disagree"):
        external_evaluation.read_photo(root, {**record, "height": 65})


def test_protocol_states_what_the_measurement_does_not_prove():
    protocol = external_evaluation.external_protocol({"preprocessing": "public-v1"}, "split")
    assert protocol["threshold_calibrated_on_this_corpus"] is False
    assert protocol["class"] == "crack"
    assert any("qualification" in limit for limit in protocol["limits"])


def test_reported_protocol_is_independent():
    first = external_evaluation.external_protocol({}, "split")
    first["limits"].append("modifié")
    assert len(external_evaluation.external_protocol({}, "split")["limits"]) == 4


@pytest.fixture
def external_run(tmp_path, monkeypatch):
    """Un moteur simulé : on vérifie l'assemblage du rapport, pas les poids réels."""
    root, record = build_corpus(tmp_path)
    report = {
        "model": "rf-detr-seg-medium",
        "class_names": ["crack", "surface_loss"],
        "checkpoint": "checkpoints/checkpoint_best_total.pth",
        "checkpoint_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        external_evaluation, "verify_run", lambda run: (report, tmp_path / "weights")
    )
    monkeypatch.setattr(external_evaluation, "runtime", lambda device: {"device": device})
    monkeypatch.setattr(external_evaluation, "process_peak_rss", lambda: 1)

    mask = np.zeros((64, 48), dtype=bool)
    mask[10:20, 5:10] = True  # La moitié de la zone de référence, et rien à côté.
    detection = SimpleNamespace(
        xyxy=np.array([[5, 10, 10, 20], [5, 10, 10, 20]]),
        confidence=np.array([0.9, 0.9]),
        class_id=np.array([0, 1]),
        mask=np.stack([mask, mask]),
    )
    monkeypatch.setattr(
        external_evaluation,
        "load_engine",
        lambda *args: SimpleNamespace(predict=lambda *a, **kw: detection),
    )
    return root, record


def test_complete_external_measurement(external_run, tmp_path):
    root, _ = external_run
    result = external_evaluation.evaluate_external(tmp_path / "run", root, tmp_path / "out", "cpu")
    assert result["images"] == 1
    assert result["content_groups"] == 1
    assert result["checkpoint_sha256"] == "a" * 64
    coverage = result["coverage"]["crack"]
    # La proposition couvre la moitié de la référence, et tombe entièrement dessus.
    assert coverage["zone_recall"] == pytest.approx(0.5)
    assert coverage["zone_precision"] == pytest.approx(1.0)
    assert coverage["photos_found"] == 1
    assert json.loads((tmp_path / "out" / "report.json").read_text())["images"] == 1


def test_only_the_crack_class_is_measured(external_run, tmp_path):
    """La source n'annote pas les pertes de matière : les compter serait un faux résultat."""
    root, _ = external_run
    result = external_evaluation.evaluate_external(tmp_path / "run", root, tmp_path / "out", "cpu")
    assert set(result["coverage"]) == {"crack"}
    assert result["records"][0]["proposals"] == 1


def test_existing_output_is_never_overwritten(external_run, tmp_path):
    root, _ = external_run
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        external_evaluation.evaluate_external(tmp_path / "run", root, tmp_path / "out", "cpu")


def build_patch_sample(tmp_path, states=("clear", "cracked")):
    """Un échantillon d'extraits comme le prépare la commande SDNET."""
    root = tmp_path / "patches"
    records = []
    rng = np.random.default_rng(1)
    for state in states:
        (root / state).mkdir(parents=True)
        for index in range(2):
            patch_id = f"{state}-{index}"
            image = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB")
            image.save(root / state / f"{patch_id}.jpg", quality=92)
            records.append(
                {
                    "id": patch_id,
                    "path": f"{state}/{patch_id}.jpg",
                    "sha256": digest((root / state / f"{patch_id}.jpg").read_bytes()),
                    "author_state": state,
                    "scene_group": f"W-{index}",
                }
            )
    write_json(
        root / "report.json",
        {
            "schema_version": 1,
            "dataset": "sdnet2018",
            "approved_for_training": False,
            "limits": ["Author labels cover cracks only"],
            "selection": {"surface": "wall"},
            "records": records,
        },
    )
    return root


@pytest.fixture
def patch_engine(tmp_path, monkeypatch):
    """Le moteur alerte sur tout : on vérifie le comptage, pas la qualité du modèle."""
    report = {"model": "rf-detr-seg-medium", "checkpoint_sha256": "b" * 64}
    monkeypatch.setattr(
        external_evaluation, "verify_run", lambda run: (report, tmp_path / "weights")
    )
    monkeypatch.setattr(external_evaluation, "runtime", lambda device: {"device": device})
    monkeypatch.setattr(external_evaluation, "process_peak_rss", lambda: 1)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    detection = SimpleNamespace(
        xyxy=np.array([[10, 10, 20, 20]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        mask=np.stack([mask]),
    )
    monkeypatch.setattr(
        external_evaluation,
        "load_engine",
        lambda *args: SimpleNamespace(predict=lambda *a, **kw: detection),
    )


def test_patch_alerts_count_both_states(patch_engine, tmp_path):
    sample = build_patch_sample(tmp_path)
    result = external_evaluation.evaluate_patch_alerts(
        tmp_path / "run", sample, tmp_path / "out", "cpu"
    )
    assert result["patches"] == 4
    assert result["alerts"]["clear"]["alert_rate"] == 1.0
    assert result["alerts"]["cracked"]["alert_rate"] == 1.0
    assert result["protocol"]["reference"].startswith("author image labels")
    assert result["sample_limits"] == ["Author labels cover cracks only"]


def test_patch_changed_since_preparation_is_refused(patch_engine, tmp_path):
    sample = build_patch_sample(tmp_path)
    document = json.loads((sample / "report.json").read_text())
    document["records"][0]["sha256"] = "0" * 64
    write_json(sample / "report.json", document)
    with pytest.raises(ValueError, match="Patch changed"):
        external_evaluation.evaluate_patch_alerts(tmp_path / "run", sample, tmp_path / "out", "cpu")


def test_patch_sample_opened_for_training_is_refused(patch_engine, tmp_path):
    sample = build_patch_sample(tmp_path)
    document = json.loads((sample / "report.json").read_text())
    document["approved_for_training"] = True
    write_json(sample / "report.json", document)
    with pytest.raises(ValueError, match="outside training"):
        external_evaluation.evaluate_patch_alerts(tmp_path / "run", sample, tmp_path / "out", "cpu")


def test_patch_sample_without_a_state_is_refused(patch_engine, tmp_path):
    """Sans extraits fissurés, un taux nul sur le sain ne voudrait rien dire."""
    sample = build_patch_sample(tmp_path, states=("clear",))
    with pytest.raises(ValueError, match="No patch to measure"):
        external_evaluation.evaluate_patch_alerts(tmp_path / "run", sample, tmp_path / "out", "cpu")


def test_empty_patch_sample_is_refused(patch_engine, tmp_path):
    sample = build_patch_sample(tmp_path)
    document = json.loads((sample / "report.json").read_text())
    document["records"] = []
    write_json(sample / "report.json", document)
    with pytest.raises(ValueError, match="Empty or oversized patch sample"):
        external_evaluation.evaluate_patch_alerts(tmp_path / "run", sample, tmp_path / "out", "cpu")


def window_engine(monkeypatch, filled=True):
    """Un moteur simulé qui répond à la taille de la fenêtre qu'il reçoit."""

    def predict(view, **_):
        width, height = view.size
        if not filled:
            return SimpleNamespace(
                xyxy=np.zeros((0, 4)),
                confidence=np.zeros((0,)),
                class_id=np.zeros((0,), dtype=int),
                mask=np.zeros((0, height, width), dtype=bool),
            )
        mask = np.zeros((height, width), dtype=bool)
        mask[: height // 2, : width // 2] = True
        return SimpleNamespace(
            xyxy=np.array([[0.0, 0.0, width / 2, height / 2]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            mask=np.stack([mask]),
        )

    monkeypatch.setattr(
        external_evaluation, "load_engine", lambda *args: SimpleNamespace(predict=predict)
    )


def test_windowed_measurement_covers_the_same_reference(external_run, tmp_path, monkeypatch):
    """Même corpus et mêmes seuils que la mesure directe : seule la découpe change."""
    root, _ = external_run
    window_engine(monkeypatch)
    result = external_evaluation.evaluate_windowed(
        tmp_path / "run", root, tmp_path / "out", "cpu", window=32, overlap=8
    )
    assert result["images"] == 1
    assert result["protocol"]["method"] == "windowed"
    assert result["protocol"]["window"] == 32
    assert result["windows_per_photo"] > 1
    assert result["records"][0]["window_proposals"] >= result["records"][0]["merged_proposals"]
    assert result["coverage"]["crack"]["reference_pixels"] == 100


def test_windowed_photo_without_any_proposal_reports_no_zone(external_run, tmp_path, monkeypatch):
    """Aucune proposition ne doit pas produire une zone vide comptée comme signalement."""
    root, _ = external_run
    window_engine(monkeypatch, filled=False)
    result = external_evaluation.evaluate_windowed(
        tmp_path / "run", root, tmp_path / "out", "cpu", window=32, overlap=8
    )
    coverage = result["coverage"]["crack"]
    assert coverage["photos_signalled"] == 0
    assert coverage["zone_recall"] == 0.0
    assert coverage["zone_precision"] is None


def test_windowed_ignores_proposals_below_the_threshold(external_run, tmp_path, monkeypatch):
    """Le seuil est celui de la mesure directe : une proposition faible ne compte pas."""
    root, _ = external_run

    def predict(view, **_):
        width, height = view.size
        mask = np.zeros((height, width), dtype=bool)
        mask[: height // 2, : width // 2] = True
        return SimpleNamespace(
            xyxy=np.array([[0.0, 0.0, width / 2, height / 2]]),
            confidence=np.array([0.05]),
            class_id=np.array([0]),
            mask=np.stack([mask]),
        )

    monkeypatch.setattr(
        external_evaluation, "load_engine", lambda *args: SimpleNamespace(predict=predict)
    )
    result = external_evaluation.evaluate_windowed(
        tmp_path / "run", root, tmp_path / "out", "cpu", window=32, overlap=8
    )
    assert result["records"][0]["window_proposals"] == 0
    assert result["coverage"]["crack"]["photos_signalled"] == 0
