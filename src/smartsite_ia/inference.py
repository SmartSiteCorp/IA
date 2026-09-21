"""Garder le traitement des photos explicite, sans modifier la bibliothèque du moteur."""

import importlib
import math
from importlib.metadata import version
from typing import Any

import numpy as np
from PIL import Image

from smartsite_ia.categories import get_class_names
from smartsite_ia.model_assets import MODEL_VERSION

PROFILES = ("public-v1", "training-v1")
DEFAULT_PROFILE = "public-v1"


def inference_metadata(profile: str, mask_threshold: float = 0.5) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown preprocessing profile: {profile}")
    if (
        type(mask_threshold) not in (int, float)
        or not math.isfinite(mask_threshold)
        or not 0 < mask_threshold < 1
    ):
        raise ValueError("Mask threshold must be finite and between 0 and 1")
    if profile == "public-v1" and mask_threshold != 0.5:
        raise ValueError("Custom mask threshold requires training-v1 preprocessing")
    return {
        "preprocessing": profile,
        "resize": (
            "RGB tensor bilinear, antialias=False"
            if profile == "public-v1"
            else "RF-DETR torchvision validation transforms, PIL bilinear antialias"
        ),
        "mask_probability_threshold": mask_threshold,
        "mask_coordinates": "Original oriented photo; resize logits before binarization",
    }


def configure_inference(engine: Any, profile: str, mask_threshold: float = 0.5) -> Any:
    """Le mode historique reste disponible pour reproduire les anciennes mesures."""
    inference_metadata(profile, mask_threshold)
    return engine if profile == "public-v1" else AlignedPredictor(engine, mask_threshold)


class AlignedPredictor:
    """Réutiliser les transformations et le décodage RF-DETR, avec les pixels d'origine."""

    def __init__(self, engine: Any, mask_threshold: float = 0.5) -> None:
        inference_metadata("training-v1", mask_threshold)
        self.mask_logit = math.log(mask_threshold / (1 - mask_threshold))
        if version("rfdetr") != MODEL_VERSION:
            raise ValueError("Aligned inference requires the pinned RF-DETR version")
        self.context = engine.model
        config = engine.model_config
        self.class_names = get_class_names({"class_names": list(engine.class_names)})
        if (
            self.context.model is None
            or engine._is_optimized_for_inference
            or self.context.args.num_classes != len(self.class_names)
            or not config.segmentation_head
        ):
            raise ValueError("Aligned inference expects the unoptimized SmartSite segmenter")
        self.torch = importlib.import_module("torch")
        self.detections = importlib.import_module("supervision").Detections
        transforms = importlib.import_module("rfdetr.datasets.coco")
        # C'est le même traitement déterministe que la validation de l'entraînement.
        # Aucune annotation n'entre ici : le modèle reçoit uniquement la photo.
        self.transform = transforms.make_coco_transforms_square_div_64(
            "val",
            self.context.resolution,
            patch_size=config.patch_size,
            num_windows=config.num_windows,
            scale_jitter=False,
        )
        self.network = self.context.model.to(self.context.device).eval()

    def predict(self, photo: Image.Image, threshold: float) -> Any:
        if photo.mode != "RGB" or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Expected RGB photo and finite threshold between 0 and 1")
        with self.torch.inference_mode():
            tensor, _ = self.transform(photo, None)
            raw = self.network(tensor.unsqueeze(0).to(self.context.device))
            if self.mask_logit:
                # Le moteur découpe les masques au logit zéro. Décaler les logits
                # change ce seuil sans toucher aux poids ni au score du défaut.
                raw = {**raw, "pred_masks": raw["pred_masks"] - self.mask_logit}
            sizes = self.torch.tensor([[photo.height, photo.width]], device=self.context.device)
            # On redimensionne les logits directement vers la photo originale.
            # Agrandir un masque déjà binaire abîmerait les contours fins.
            result = self.context.postprocess(raw, sizes, score_threshold=threshold)[0]
            labels = result["labels"].cpu().numpy()
            return self.detections(
                xyxy=result["boxes"].float().cpu().numpy(),
                confidence=result["scores"].float().cpu().numpy(),
                class_id=labels,
                mask=result["masks"].squeeze(1).cpu().numpy(),
                data={
                    "class_name": np.array(
                        [
                            self.class_names[label]
                            if 0 <= label < len(self.class_names)
                            else "__background__"
                            if label == len(self.class_names)
                            else ""
                            for label in labels
                        ],
                        dtype=object,
                    )
                },
            )
