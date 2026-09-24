"""Vérifier le vrai lecteur RF-DETR sur CPU, avec l'extra training installé."""

import torch
from PIL import Image
from rfdetr.config import RFDETRNanoConfig
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64

from smartsite_ia.box_data import BOX_CLASSES
from smartsite_ia.collection_review import box_coco_document
from smartsite_ia.importer import write_json


def test_real_box_loader_maps_classes_keeps_negatives_and_has_no_masks(tmp_path):
    rows = []
    for index in range(2):
        Image.new("RGB", (128, 64), "gray").save(tmp_path / f"image_{index}.jpg")
        rows.append(
            {
                "id": f"image_{index}",
                "width": 128,
                "height": 64,
                "group": "synthetic",
                "credit": {"title": "Synthetic software test"},
                "boxes": [
                    {"class": name, "xyxy_normalized": [0.25, 0.25, 0.75, 0.75]}
                    for name in BOX_CLASSES
                ]
                if index == 0
                else [],
            }
        )
    annotations = tmp_path / "_annotations.coco.json"
    write_json(annotations, box_coco_document(rows))
    dataset = CocoDetection(
        tmp_path,
        annotations,
        make_coco_transforms_square_div_64("val", 384),
        include_masks=False,
        remap_category_ids=True,
    )
    assert dataset.cat2label == {1: 0, 2: 1, 3: 2}
    image, positive = dataset[0]
    _, negative = dataset[1]
    assert image.shape == (3, 384, 384)
    assert positive["labels"].tolist() == [0, 1, 2]
    torch.testing.assert_close(positive["boxes"], torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 3))
    assert negative["labels"].numel() == 0 and negative["boxes"].shape == (0, 4)
    assert "masks" not in positive and "masks" not in negative
    config = RFDETRNanoConfig(device="cpu")
    assert config.resolution == 384 and not config.segmentation_head


def test_aligned_boxes_use_validation_pixels_original_coordinates_and_score_filter():
    from types import SimpleNamespace

    import numpy as np
    from rfdetr.models.postprocess import PostProcess

    from smartsite_ia.box_review import encode_boxes
    from smartsite_ia.inference import AlignedPredictor

    class Network(torch.nn.Module):
        def forward(self, tensor):
            assert not torch.is_grad_enabled() and not self.training
            self.input = tensor
            return {
                "pred_logits": torch.tensor([[[2.0, -9.0, -9.0, -9.0]]]),
                "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
            }

    network = Network()
    engine = SimpleNamespace(
        model=SimpleNamespace(
            model=network,
            postprocess=PostProcess(num_select=4),
            args=SimpleNamespace(num_classes=3),
            resolution=384,
            device=torch.device("cpu"),
        ),
        model_config=SimpleNamespace(segmentation_head=False, patch_size=16, num_windows=2),
        class_names=list(BOX_CLASSES),
        _is_optimized_for_inference=False,
    )
    predictor = AlignedPredictor(engine, box_classes=BOX_CLASSES)
    for width, height in ((96, 64), (64, 96), (96, 96)):
        pixels = np.indices((height, width)).sum(axis=0) % 2 * 255
        photo = Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2).astype("uint8"))
        result = predictor.predict(photo, 0.3)
        expected, _ = make_coco_transforms_square_div_64(
            "val", 384, patch_size=16, num_windows=2, scale_jitter=False
        )(photo, None)
        torch.testing.assert_close(network.input[0], expected, rtol=0, atol=0)
        assert result.mask is None and result.class_id.tolist() == [0]
        np.testing.assert_allclose(
            result.xyxy, [[width * 0.25, height * 0.25, width * 0.75, height * 0.75]]
        )
        assert len(encode_boxes(result, photo)["predictions"]) == 1
        empty = predictor.predict(photo, 0.99)
        assert empty.mask is None and empty.xyxy.shape == (0, 4)
