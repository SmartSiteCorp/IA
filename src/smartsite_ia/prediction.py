"""Montrer de vraies prédictions et leurs masques, sans les confondre avec un corrigé."""

import html
import importlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as coco_mask

from smartsite_ia.categories import class_legend, get_class_names
from smartsite_ia.curation import digest, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.inference import DEFAULT_PROFILE, configure_inference, inference_metadata
from smartsite_ia.learning import runtime, verify_run
from smartsite_ia.model_assets import CLASS_NAMES
from smartsite_ia.review import MAX_FILE_BYTES, read_local


def decode_photo(raw: bytes, *, allow_primary_mpo: bool = False) -> Image.Image:
    """Les coordonnées du résultat concernent la photo une fois remise à l'endroit."""
    with Image.open(io.BytesIO(raw), formats=["JPEG", "PNG"]) as image:
        if image.width < 32 or image.height < 32 or image.width * image.height > 16_000_000:
            raise ValueError("Prediction expects a photo between 32 pixels and 16 megapixels")
        frames = getattr(image, "n_frames", 1)
        # L'inférence ordinaire continue à refuser les fichiers contenant plusieurs images
        primary_mpo = allow_primary_mpo and image.format == "MPO" and frames == 2
        if allow_primary_mpo and not primary_mpo:
            raise ValueError("Expected the reviewed two-frame MPO format")
        if image.mode not in ("RGB", "L") or (frames != 1 and not primary_mpo):
            raise ValueError("Unsupported photo mode or animation")
        orientation = image.getexif().get(274, 1)
        if type(orientation) is not int or orientation not in range(1, 9):
            raise ValueError("Invalid photo orientation")
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        oriented.info.clear()
        return oriented


def load_engine(
    checkpoint: Path,
    device: str,
    preprocessing: str = DEFAULT_PROFILE,
    mask_threshold: float = 0.5,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> Any:
    """Charger une seule fois les poids quand on analyse plusieurs photos."""
    inference_metadata(preprocessing, mask_threshold)
    engine = importlib.import_module("rfdetr").RFDETR.from_checkpoint(
        str(checkpoint.resolve()), device=device, trust_checkpoint=False
    )
    if list(engine.class_names) != list(class_names):
        raise ValueError("Checkpoint class names do not match SmartSite")
    return configure_inference(engine, preprocessing, mask_threshold)


def predict_engine(
    checkpoint: Path,
    photo: Image.Image,
    device: str,
    threshold: float,
    preprocessing: str = DEFAULT_PROFILE,
    mask_threshold: float = 0.5,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> Any:
    return load_engine(checkpoint, device, preprocessing, mask_threshold, class_names).predict(
        photo, threshold=threshold
    )


def encode_predictions(
    detections: Any,
    photo: Image.Image,
    *,
    allow_empty_boxes: bool = False,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> list[dict[str, Any]]:
    """Vérifier les sorties du moteur avant d'enregistrer leurs coordonnées et masques."""
    boxes, scores, classes = detections.xyxy, detections.confidence, detections.class_id
    masks = detections.mask
    count = len(boxes)
    if (
        count > 200
        or scores is None
        or classes is None
        or masks is None
        or boxes.shape != (count, 4)
        or scores.shape != (count,)
        or classes.shape != (count,)
        or masks.shape != (count, photo.height, photo.width)
    ):
        raise ValueError("Unexpected prediction dimensions or missing masks")
    records = []
    for index in range(count):
        label, score, box = classes[index], float(scores[index]), boxes[index].astype(float)
        # RF-DETR expose parfois sa classe « aucun objet » à très faible score.
        # On la retire seulement si le moteur la nomme explicitement ainsi.
        names = getattr(detections, "data", {}).get("class_name")
        if label == len(class_names) and names is not None and names[index] == "__background__":
            continue
        if (
            int(label) != label
            or not 0 <= label < len(class_names)
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or not np.isfinite(box).all()
            or np.any(box < 0)
            or box[0] > box[2]
            or box[1] > box[3]
            or (not allow_empty_boxes and (box[0] == box[2] or box[1] == box[3]))
            or box[2] > photo.width
            or box[3] > photo.height
        ):
            raise ValueError("Invalid prediction class, score or box")
        mask = masks[index]
        if mask.dtype != np.bool_:
            raise ValueError("Prediction masks must be binary")
        label = int(label)
        encoded = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
        encoded["counts"] = encoded["counts"].decode("ascii")
        records.append(
            {
                "id": index + 1,
                "class_id": label,
                "class_name": class_names[label],
                "score": score,
                "bbox_xyxy": box.tolist(),
                "mask_rle": encoded,
                "mask_pixels": int(mask.sum()),
            }
        )
    return records


def render_predictions(
    photo: Image.Image,
    records: list[dict[str, Any]],
    *,
    show_boxes: bool = True,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> Image.Image:
    """Dessiner seulement les résultats qu'on veut montrer, dans les pixels d'origine."""
    annotated = np.asarray(photo).copy()
    # La couleur depend du défaut, pas de sa position dans la tête du modèle
    palette = {"crack": (255, 70, 50), "surface_loss": (30, 140, 255)}
    colors = [palette[name] for name in class_names]
    for record in records:
        mask = coco_mask.decode(record["mask_rle"]).astype(bool)
        rgb = np.array(colors[record["class_id"]])
        annotated[mask] = (annotated[mask] * 0.5 + rgb * 0.5).astype(np.uint8)
    result = Image.fromarray(annotated)
    # Sur les photos très chargées, les étiquettes cacheraient les petites fissures.
    if not show_boxes:
        return result
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=14)
    for record in records:
        box = record["bbox_xyxy"]
        color = colors[record["class_id"]]
        draw.rectangle(box, outline=color, width=2)
        # La petite police embarquée ne couvre pas tous les accents ; le HTML les garde.
        label = "Fissure" if class_names[record["class_id"]] == "crack" else "Perte de matiere"
        text = f"{label} {record['score']:.0%}"
        point = (box[0], max(0, box[1] - 17))
        draw.rectangle(draw.textbbox(point, text, font=font), fill="white")
        draw.text(point, text, fill=color, font=font)
    return result


def describe_predictions(
    detections: Any, photo: Image.Image, class_names: tuple[str, ...] = CLASS_NAMES
) -> tuple[list[dict[str, Any]], Image.Image]:
    records = encode_predictions(detections, photo, class_names=class_names)
    return records, render_predictions(photo, records, class_names=class_names)


def predict_photo(
    run: Path,
    image_path: Path,
    output: Path,
    device: str,
    threshold: float = 0.3,
    preprocessing: str = DEFAULT_PROFILE,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    settings = inference_metadata(preprocessing, mask_threshold)
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Prediction threshold must be between 0 and 1")
    report, checkpoint = verify_run(run)
    names = get_class_names(report.get("config", {}))
    environment = runtime(device)
    raw = read_local(image_path.parent, image_path.name, MAX_FILE_BYTES)
    photo = decode_photo(raw)
    with staged_output(output, [run, image_path]) as stage:
        detections = predict_engine(
            checkpoint, photo, device, threshold, preprocessing, mask_threshold, names
        )
        records, annotated = describe_predictions(detections, photo, names)
        result = {
            "schema_version": 1,
            "model": report["model"],
            "class_names": list(names),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "source_image_sha256": digest(raw),
            "width": photo.width,
            "height": photo.height,
            "coordinate_space": "EXIF-oriented image, pixels",
            "runtime": environment,
            "inference": settings,
            "threshold": threshold,
            "threshold_calibrated": False,
            "qualified_for_smartsite": False,
            "training_purpose": report["purpose"],
            "predictions": records,
        }
        photo.save(stage / "photo.png")
        annotated.save(stage / "prediction.png")
        write_json(stage / "prediction.json", result)
        # Rapport léger ...
        title = html.escape(image_path.name)
        (stage / "index.html").write_text(
            f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy"
content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'">
<title>SmartSite — première prédiction</title><style>
body{{font:17px system-ui;max-width:1150px;margin:36px auto;padding:0 24px;color:#20323c}}
.notice{{padding:18px;background:#fff4d3}}.photos{{display:flex;gap:20px;flex-wrap:wrap}}
figure{{flex:1;min-width:260px;margin:0}}img{{width:100%;height:auto}}
figcaption{{font-weight:600;margin:10px 0}}
a{{color:#245480}}</style><h1>Première prédiction SmartSite</h1>
<p class="notice">Modèle expérimental : ce résultat ne valide pas un chantier.
Les couleurs montrent des prédictions, pas les annotations du dataset.
Aucun défaut détecté ne signifie pas que la surface est conforme.</p>
<p>Photo : {title} · {len(records)} proposition(s) · seuil d'affichage : {threshold:.0%}.
Ce seuil n'est pas encore calibré ; les scores ne mesurent pas la gravité.</p>
<div class="photos"><figure><figcaption>Photo orientée</figcaption>
<a href="photo.png"><img src="photo.png" alt="Photo originale orientée"></a></figure>
<figure><figcaption>Prédictions du modèle</figcaption><a href="prediction.png">
<img src="prediction.png" alt="Masques et boîtes prédits"></a></figure></div>
<p>Catégories recherchées — {class_legend(names)}.</p>
<p><a href="prediction.json">Résultat détaillé et masques</a></p></html>""",
            encoding="utf-8",
        )
    return result
