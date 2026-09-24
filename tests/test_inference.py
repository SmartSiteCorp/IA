"""Vérifier les profils sans installer le moteur dans les tests courants."""

from types import SimpleNamespace

import pytest

from smartsite_ia import inference, prediction
from smartsite_ia.categories import CLASS_NAMES


def test_legacy_profile_keeps_the_public_engine():
    engine = object()
    assert inference.configure_inference(engine, "public-v1") is engine
    assert inference.inference_metadata("public-v1")["mask_probability_threshold"] == 0.5


def test_unknown_profile_refused_before_loading_weights(tmp_path):
    with pytest.raises(ValueError, match="preprocessing"):
        prediction.load_engine(tmp_path / "absent", "cpu", "typo")


def test_profile_metadata_is_independent():
    first = inference.inference_metadata("training-v1")
    first["preprocessing"] = "modified"
    assert inference.inference_metadata("training-v1")["preprocessing"] == "training-v1"


@pytest.mark.parametrize("threshold", [0, 1, -1, float("nan"), float("inf"), None, True, "0.5"])
def test_invalid_mask_threshold_refused(threshold):
    with pytest.raises(ValueError, match="Mask threshold"):
        inference.inference_metadata("training-v1", threshold)


def test_legacy_profile_cannot_silently_ignore_mask_threshold():
    with pytest.raises(ValueError, match="requires"):
        inference.inference_metadata("public-v1", 0.7)


def test_aligned_profile_pins_engine_version(monkeypatch):
    monkeypatch.setattr(inference, "version", lambda name: "unknown")
    with pytest.raises(ValueError, match="version"):
        inference.configure_inference(object(), "training-v1")


@pytest.mark.parametrize("change", ["optimized", "cleared", "classes", "head", "slots"])
def test_incompatible_engine_refused_before_inference(monkeypatch, change):
    monkeypatch.setattr(inference, "version", lambda name: inference.MODEL_VERSION)
    engine = SimpleNamespace(
        model=SimpleNamespace(model=object(), args=SimpleNamespace(num_classes=2)),
        model_config=SimpleNamespace(segmentation_head=True),
        class_names=list(CLASS_NAMES),
        _is_optimized_for_inference=False,
    )
    if change == "optimized":
        engine._is_optimized_for_inference = True
    elif change == "cleared":
        engine.model.model = None
    elif change == "classes":
        engine.class_names = ["person"]
    elif change == "slots":
        engine.model.args.num_classes = 80
    else:
        engine.model_config.segmentation_head = False
    with pytest.raises(ValueError, match="segmenter|class names"):
        inference.configure_inference(engine, "training-v1")


def test_box_adapter_refuses_wrong_labels_and_mask_threshold(monkeypatch):
    monkeypatch.setattr(inference, "version", lambda name: inference.MODEL_VERSION)
    engine = SimpleNamespace(model=object(), model_config=object(), class_names=["mold_suspected"])
    with pytest.raises(ValueError, match="class names"):
        inference.AlignedPredictor(engine, box_classes=("moisture_trace",))
    with pytest.raises(ValueError, match="no mask threshold"):
        inference.AlignedPredictor(engine, mask_threshold=0.7, box_classes=("mold_suspected",))
