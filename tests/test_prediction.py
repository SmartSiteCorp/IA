"""Vérifier les coordonnées, masques, photos et erreurs sans télécharger un réseau."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import image_bytes
from PIL import Image
from pycocotools import mask as coco_mask

from smartsite_ia import prediction
from smartsite_ia.model_assets import CLASS_NAMES, MODEL_NAME


def detections(count=1, size=(64, 48)):
    masks = np.zeros((count, size[1], size[0]), dtype=bool)
    masks[:, 8:20, 10:30] = True
    return SimpleNamespace(
        xyxy=np.tile([10.0, 8.0, 30.0, 20.0], (count, 1)),
        confidence=np.full(count, 0.8),
        class_id=np.zeros(count, dtype=int),
        mask=masks,
    )


def test_mask_roundtrip_original_coordinates_and_original_untouched():
    photo = Image.new("RGB", (64, 48), "gray")
    source = photo.tobytes()
    records, annotated = prediction.describe_predictions(detections(), photo)
    assert photo.tobytes() == source and annotated.tobytes() != source
    record = records[0]
    assert record["class_name"] == "crack" and record["bbox_xyxy"] == [10, 8, 30, 20]
    assert record["mask_pixels"] == 240
    np.testing.assert_array_equal(coco_mask.decode(record["mask_rle"]), detections().mask[0])


def test_no_detection_is_a_valid_result():
    photo = Image.new("RGB", (64, 48))
    records, rendered = prediction.describe_predictions(detections(0), photo)
    assert records == [] and rendered.tobytes() == photo.tobytes()


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_id", np.array([2])),
        ("class_id", np.array([0.5])),
        ("confidence", np.array([float("nan")])),
        ("confidence", np.array([1.1])),
        ("xyxy", np.array([[30, 8, 10, 20]])),
        ("xyxy", np.array([[-1, 0, 20, 20]])),
        ("xyxy", np.array([[1, 0, 80, 20]])),
        ("xyxy", np.array([[0, 0, np.inf, 20]])),
        ("mask", np.zeros((1, 48, 64), dtype=float)),
        ("mask", np.zeros((1, 64, 48), dtype=bool)),
        ("mask", None),
        ("confidence", None),
        ("class_id", None),
    ],
)
def test_invalid_predictions_refused(field, value):
    values = detections()
    setattr(values, field, value)
    with pytest.raises(ValueError):
        prediction.describe_predictions(values, Image.new("RGB", (64, 48)))


def test_exif_corrected_before_model_and_stripped():
    image = Image.new("RGB", (64, 48))
    exif = Image.Exif()
    exif[274] = 6
    import io

    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    result = prediction.decode_photo(stream.getvalue())
    assert result.size == (48, 64) and not result.getexif()


@pytest.mark.parametrize(
    "mode,size,format",
    [
        ("RGB", (12, 64), "PNG"),
        ("RGBA", (64, 64), "PNG"),
        ("RGB", (64, 64), "GIF"),
        ("RGB", (4001, 4000), "JPEG"),
    ],
)
def test_unsupported_photos_refused(mode, size, format):
    with pytest.raises((ValueError, OSError)):
        prediction.decode_photo(image_bytes(Image.new(mode, size), format))


def test_invalid_exif_refused():
    import io

    image = Image.new("RGB", (64, 48))
    exif = Image.Exif()
    exif[274] = 9
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    with pytest.raises(ValueError, match="orientation"):
        prediction.decode_photo(stream.getvalue())


def test_predict_engine_checks_classes_and_safe_loading(monkeypatch, tmp_path):
    captured = {}

    class Engine:
        class_names = list(CLASS_NAMES)

        def predict(self, image, **kwargs):
            return detections()

        @classmethod
        def from_checkpoint(cls, *args, **kwargs):
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(
        prediction.importlib, "import_module", lambda name: SimpleNamespace(RFDETR=Engine)
    )
    prediction.predict_engine(tmp_path / "weights", Image.new("RGB", (64, 48)), "cpu", 0.3)
    assert captured["trust_checkpoint"] is False
    Engine.class_names = ["person"]
    with pytest.raises(ValueError, match="class names"):
        prediction.predict_engine(tmp_path / "weights", Image.new("RGB", (64, 48)), "cpu", 0.3)


def test_prediction_export_is_honest_safe_and_atomic(tmp_path, monkeypatch):
    photo = tmp_path / "<script>photo.jpg"
    Image.new("RGB", (64, 48)).save(photo)
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(
        prediction,
        "verify_run",
        lambda root: (
            {"model": MODEL_NAME, "checkpoint_sha256": "a" * 64, "purpose": "smoke"},
            run / "weights",
        ),
    )
    monkeypatch.setattr(prediction, "runtime", lambda device: {"device": device})
    monkeypatch.setattr(prediction, "predict_engine", lambda *args: detections())
    out = tmp_path / "out"
    result = prediction.predict_photo(run, photo, out, "cpu")
    assert not result["qualified_for_smartsite"] and not result["threshold_calibrated"]
    assert json.loads((out / "prediction.json").read_text()) == result
    page = (out / "index.html").read_text()
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert "pas les annotations" in page and "pas que la surface est conforme" in page
    with pytest.raises(FileExistsError):
        prediction.predict_photo(run, photo, out, "cpu")

    def fail(*args):
        raise RuntimeError("engine failed")

    monkeypatch.setattr(prediction, "predict_engine", fail)
    with pytest.raises(RuntimeError):
        prediction.predict_photo(run, photo, tmp_path / "failed", "cpu")
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize("threshold", [0, 1, float("nan"), float("inf"), -0.2])
def test_invalid_threshold_refused_before_loading_model(tmp_path, threshold):
    with pytest.raises(ValueError, match="threshold"):
        prediction.predict_photo(tmp_path, tmp_path / "photo", tmp_path / "out", "cpu", threshold)
