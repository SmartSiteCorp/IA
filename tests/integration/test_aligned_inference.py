"""Contrôler le vrai traitement RF-DETR sur CPU, avec l'extra training installé."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from rfdetr.datasets.coco import make_coco_transforms_square_div_64
from rfdetr.models.postprocess import PostProcess

from smartsite_ia.inference import AlignedPredictor
from smartsite_ia.prediction import encode_predictions


class Network(torch.nn.Module):
    def forward(self, tensor):
        assert not torch.is_grad_enabled() and not self.training
        self.input = tensor
        return {
            "pred_logits": torch.tensor([[[2.0, -9.0, -9.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
            "pred_masks": torch.tensor([[[[-0.8, 1.2], [0.3, -1.4]]]]),
        }


@pytest.fixture
def predictor():
    context = SimpleNamespace(
        model=Network(),
        postprocess=PostProcess(num_select=3),
        args=SimpleNamespace(num_classes=2),
        resolution=432,
        device=torch.device("cpu"),
    )
    return AlignedPredictor(
        SimpleNamespace(
            model=context,
            model_config=SimpleNamespace(
                segmentation_head=True,
                patch_size=12,
                num_windows=2,
            ),
            class_names=["crack", "surface_loss"],
            _is_optimized_for_inference=False,
        )
    )


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (432, 432)])
def test_real_transform_native_mask_and_box_coordinates(predictor, size):
    # Le damier révèle un changement de filtre que masquerait une image unie.
    pixels = np.indices((size[1], size[0])).sum(axis=0) % 2 * 255
    photo = Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2).astype("uint8"))
    original = photo.tobytes()
    result = predictor.predict(photo, threshold=0.3)
    expected, _ = make_coco_transforms_square_div_64("val", 432)(photo, None)
    torch.testing.assert_close(predictor.network.input[0], expected, rtol=0, atol=0)
    assert photo.tobytes() == original
    assert result.mask.shape == (1, size[1], size[0])
    np.testing.assert_allclose(
        result.xyxy, [[size[0] * 0.25, size[1] * 0.25, size[0] * 0.75, size[1] * 0.75]]
    )
    # La frontière suit les logits interpolés, pas un masque binaire agrandi.
    logits = torch.tensor([[[[-0.8, 1.2], [0.3, -1.4]]]])
    reference = (
        torch.nn.functional.interpolate(
            logits, size=(size[1], size[0]), mode="bilinear", align_corners=False
        )
        > 0
    )
    np.testing.assert_array_equal(result.mask, reference[:, 0].numpy())
    assert len(encode_predictions(result, photo)) == 1


def test_empty_output_has_native_shape(predictor):
    photo = Image.new("RGB", (96, 48))
    result = predictor.predict(photo, threshold=0.99)
    assert result.mask.shape == (0, 48, 96)
    assert encode_predictions(result, photo) == []


def test_stricter_mask_threshold_keeps_object_scores_and_native_geometry(predictor):
    photo = Image.new("RGB", (96, 48))
    original = predictor.predict(photo, threshold=0.3)
    adjusted = AlignedPredictor(
        SimpleNamespace(
            model=predictor.context,
            model_config=SimpleNamespace(
                segmentation_head=True,
                patch_size=12,
                num_windows=2,
            ),
            class_names=["crack", "surface_loss"],
            _is_optimized_for_inference=False,
        ),
        mask_threshold=0.7,
    )
    stricter = adjusted.predict(photo, threshold=0.3)
    np.testing.assert_array_equal(original.xyxy, stricter.xyxy)
    np.testing.assert_array_equal(original.confidence, stricter.confidence)
    assert stricter.mask.sum() < original.mask.sum()
    assert not (stricter.mask & ~original.mask).any()


@pytest.mark.parametrize("mode,threshold", [("L", 0.3), ("RGB", float("nan")), ("RGB", 0)])
def test_bad_input_refused(predictor, mode, threshold):
    with pytest.raises(ValueError):
        predictor.predict(Image.new(mode, (64, 64)), threshold)
