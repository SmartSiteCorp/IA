"""Mesurer un modèle sur une autre source que celle qui l'a entraîné.

Le modèle de référence a appris sur un seul barrage en béton. Cette mesure demande
s'il retrouve les fissures de bâtiments d'une source indépendante, jamais utilisée
pour l'apprentissage ni pour choisir un seuil. Les références de cette source sont
des masques sémantiques : on ne peut donc pas apparier un défaut pour un défaut,
seulement mesurer la zone couverte et le fait qu'une photo soit signalée.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from smartsite_ia.curation import digest, read_document, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.inference import DEFAULT_PROFILE, inference_metadata
from smartsite_ia.learning import process_peak_rss, runtime, verify_run
from smartsite_ia.prediction import decode_photo, encode_predictions, load_engine
from smartsite_ia.review import MAX_FILE_BYTES, read_local
from smartsite_ia.sdnet_sample import measure_alerts
from smartsite_ia.validation import DISPLAY_THRESHOLD, SCORE_FLOOR
from smartsite_ia.validation_metrics import coverage_for_image, summarize_coverage
from smartsite_ia.windowed_inference import (
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW,
    empty_zone,
    encode_zone,
    merge_proposals,
    place_mask,
    shift_box,
    window_bounds,
    windows,
)

# Cette source n'annote que les fissures : mesurer une autre classe n'aurait pas de sens.
EXTERNAL_CLASS = "crack"
MAX_EXTERNAL_IMAGES = 500

# Ces photos sont des PNG sans perte de presque huit mégapixels : elles dépassent la
# limite commune de huit mégaoctets. On garde une limite propre à cette lecture, et
# le nombre de pixels reste borné par le décodage commun.
MAX_EXTERNAL_PHOTO_BYTES = 24 * 1024 * 1024

# Borne le lot d'extraits mesuré en une fois.
MAX_PATCHES = 5000


def load_external_records(corpus: Path, split: str) -> tuple[list[dict[str, Any]], str]:
    """Reprendre la sélection déjà décidée, sans la refaire au moment de mesurer."""
    report, report_hash = read_document(corpus / "report.json")
    if report.get("dataset") != "concrete_crack_segmentation_v1":
        raise ValueError("External evaluation expects the prepared CCSD corpus")
    if report.get("approved_for_training") is not False:
        raise ValueError("This corpus must stay outside training")
    records = [record for record in report["records"] if record.get("split") == split]
    if not 1 <= len(records) <= MAX_EXTERNAL_IMAGES:
        raise ValueError("Unexpected external selection size")
    return sorted(records, key=lambda record: record["id"]), report_hash


def read_reference_mask(corpus: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Relire le masque de référence en vérifiant son empreinte et ses dimensions."""
    raw = read_local(corpus, record["mask"], MAX_FILE_BYTES)
    if digest(raw) != record["mask_sha256"]:
        raise ValueError(f"External mask changed: {record['id']}")
    with Image.open(corpus / record["mask"], formats=["PNG"]) as mask:
        if mask.mode != "L" or mask.size != (record["width"], record["height"]):
            raise ValueError(f"Unsupported external mask: {record['id']}")
        pixels = np.asarray(mask, dtype=np.uint8)
    # La préparation a déjà binarisé à 128 ; on refuse toute valeur intermédiaire.
    if set(np.unique(pixels).tolist()) - {0, 255}:
        raise ValueError(f"External mask is not binary: {record['id']}")
    foreground = int((pixels == 255).sum())
    if foreground != record["foreground_pixels"]:
        raise ValueError(f"External mask foreground disagrees: {record['id']}")
    encoded: dict[str, Any] = coco_mask.encode(np.asfortranarray(pixels == 255))
    return encoded


def read_photo(corpus: Path, record: dict[str, Any]) -> Image.Image:
    raw = read_local(corpus, record["image"], MAX_EXTERNAL_PHOTO_BYTES)
    if digest(raw) != record["image_sha256"]:
        raise ValueError(f"External photo changed: {record['id']}")
    photo = decode_photo(raw)
    if photo.size != (record["width"], record["height"]):
        raise ValueError(f"External photo dimensions disagree: {record['id']}")
    return photo


def evaluate_external(
    run: Path,
    corpus: Path,
    output: Path,
    device: str,
    split: str = "external_candidate",
    preprocessing: str = DEFAULT_PROFILE,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """Un chargement du modèle, une photo à la fois, et aucune écriture avant la fin."""
    settings = inference_metadata(preprocessing, mask_threshold)
    report, checkpoint = verify_run(run)
    records, corpus_hash = load_external_records(corpus, split)
    environment = runtime(device)
    started = time.monotonic()
    rows: list[dict[str, dict[str, Any]]] = []
    details: list[dict[str, Any]] = []
    with staged_output(output, [run, corpus]) as stage:
        engine = load_engine(checkpoint, device, preprocessing, mask_threshold)
        for record in records:
            photo = read_photo(corpus, record)
            reference = read_reference_mask(corpus, record)
            before = time.monotonic()
            proposals = encode_predictions(
                engine.predict(photo, threshold=SCORE_FLOOR), photo, allow_empty_boxes=True
            )
            seconds = time.monotonic() - before
            predicted = [
                {
                    "category_id": 1,
                    "score": proposal["score"],
                    "segmentation": proposal["mask_rle"],
                }
                for proposal in proposals
                if proposal["class_name"] == EXTERNAL_CLASS
            ]
            row = coverage_for_image(
                [{"category_id": 1, "segmentation": reference}],
                predicted,
                DISPLAY_THRESHOLD,
                (EXTERNAL_CLASS,),
            )
            rows.append(row)
            details.append(
                {
                    "id": record["id"],
                    "content_group": record.get("content_group"),
                    "proposals": len(predicted),
                    "retained_proposals": sum(p["score"] >= DISPLAY_THRESHOLD for p in predicted),
                    "inference_seconds": seconds,
                    **row[EXTERNAL_CLASS],
                }
            )
        summary = summarize_coverage(rows, (EXTERNAL_CLASS,))
        result = {
            "schema_version": 1,
            "protocol": external_protocol(settings, split),
            "model": report["model"],
            "model_class_names": report["class_names"],
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "corpus_report_sha256": corpus_hash,
            "images": len(records),
            "content_groups": len({record.get("content_group") for record in records}),
            "coverage": summary,
            "records": details,
            "runtime": environment,
            "elapsed_seconds": time.monotonic() - started,
            "process_peak_rss_bytes": process_peak_rss(),
        }
        write_json(stage / "report.json", result)
    return result


def external_protocol(settings: dict[str, Any], split: str) -> dict[str, Any]:
    """Écrire noir sur blanc ce que cette mesure prouve et ce qu'elle ne prouve pas."""
    return {
        "version": 1,
        "score_floor": SCORE_FLOOR,
        "display_threshold": DISPLAY_THRESHOLD,
        "threshold_calibrated_on_this_corpus": False,
        "selection": split,
        "class": EXTERNAL_CLASS,
        "reference": "semantic crack masks; no instance matching possible",
        **settings,
        "limits": [
            "Source externe jamais utilisée pour l'apprentissage ni le choix des seuils",
            "Un masque de référence incomplet compte une vraie fissure comme fausse alerte",
            "Les groupes de contenu sont des groupes de ressemblance, pas des bâtiments connus",
            "Mesure exploratoire de généralisation, pas une qualification SmartSite",
        ],
    }


def evaluate_patch_alerts(
    run: Path,
    sample: Path,
    output: Path,
    device: str,
    preprocessing: str = DEFAULT_PROFILE,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compter les alertes de fissure sur des extraits étiquetés sains ou fissurés.

    Ces extraits n'ont pas de contour de référence : on ne mesure donc pas une zone,
    seulement la fréquence à laquelle le modèle propose une fissure. Le lot fissuré
    sert de contrôle : sans lui, un taux nul sur le sain ne voudrait rien dire.
    """
    settings = inference_metadata(preprocessing, mask_threshold)
    report, checkpoint = verify_run(run)
    document, sample_hash = read_document(sample / "report.json")
    if document.get("dataset") != "sdnet2018" or document.get("approved_for_training") is not False:
        raise ValueError("Patch alerts expect the prepared SDNET sample, kept outside training")
    records = document["records"]
    if not 1 <= len(records) <= MAX_PATCHES:
        raise ValueError("Empty or oversized patch sample")
    environment = runtime(device)
    started = time.monotonic()
    alerts: dict[str, int] = {}
    with staged_output(output, [run, sample]) as stage:
        engine = load_engine(checkpoint, device, preprocessing, mask_threshold)
        for record in records:
            raw = read_local(sample, record["path"], MAX_FILE_BYTES)
            if digest(raw) != record["sha256"]:
                raise ValueError(f"Patch changed since preparation: {record['id']}")
            photo = decode_photo(raw)
            proposals = encode_predictions(
                engine.predict(photo, threshold=SCORE_FLOOR), photo, allow_empty_boxes=True
            )
            alerts[record["id"]] = sum(
                proposal["class_name"] == EXTERNAL_CLASS and proposal["score"] >= DISPLAY_THRESHOLD
                for proposal in proposals
            )
        result = {
            "schema_version": 1,
            "protocol": {
                **external_protocol(settings, document["selection"]["surface"]),
                "reference": "author image labels; no outline, so no zone measured",
            },
            "model": report["model"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "sample_report_sha256": sample_hash,
            "sample_limits": document["limits"],
            "patches": len(records),
            "alerts": measure_alerts(document, alerts),
            "alerts_per_patch": alerts,
            "runtime": environment,
            "elapsed_seconds": time.monotonic() - started,
            "process_peak_rss_bytes": process_peak_rss(),
        }
        write_json(stage / "report.json", result)
    return result


def evaluate_windowed(
    run: Path,
    corpus: Path,
    output: Path,
    device: str,
    split: str = "external_candidate",
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
    preprocessing: str = DEFAULT_PROFILE,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    """Mesurer la même chose que `evaluate_external`, mais en découpant chaque photo.

    On garde exactement le même corpus, les mêmes poids et le même seuil : seule la
    façon de présenter la photo au moteur change, sinon la comparaison ne dirait rien.
    """
    settings = inference_metadata(preprocessing, mask_threshold)
    report, checkpoint = verify_run(run)
    records, corpus_hash = load_external_records(corpus, split)
    environment = runtime(device)
    started = time.monotonic()
    rows: list[dict[str, dict[str, Any]]] = []
    details: list[dict[str, Any]] = []
    with staged_output(output, [run, corpus]) as stage:
        engine = load_engine(checkpoint, device, preprocessing, mask_threshold)
        for record in records:
            photo = read_photo(corpus, record)
            reference = read_reference_mask(corpus, record)
            bounds = window_bounds(photo.width, photo.height, window, overlap)
            zone = empty_zone(photo)
            proposals: list[dict[str, Any]] = []
            before = time.monotonic()
            for box, view in zip(bounds, windows(photo, bounds), strict=True):
                found = encode_predictions(
                    engine.predict(view, threshold=SCORE_FLOOR), view, allow_empty_boxes=True
                )
                for proposal in found:
                    if (
                        proposal["class_name"] != EXTERNAL_CLASS
                        or proposal["score"] < DISPLAY_THRESHOLD
                    ):
                        continue
                    place_mask(proposal["mask_rle"], box, zone)
                    proposals.append(
                        {
                            "class_name": proposal["class_name"],
                            "score": proposal["score"],
                            "bbox_xyxy": shift_box(proposal["bbox_xyxy"], box),
                        }
                    )
            seconds = time.monotonic() - before
            merged = merge_proposals(proposals)
            # La zone accumulée remplace ici les masques : elle décrit la même surface.
            predicted = (
                [{"category_id": 1, "score": 1.0, "segmentation": encode_zone(zone)}]
                if proposals
                else []
            )
            row = coverage_for_image(
                [{"category_id": 1, "segmentation": reference}],
                predicted,
                DISPLAY_THRESHOLD,
                (EXTERNAL_CLASS,),
            )
            rows.append(row)
            details.append(
                {
                    "id": record["id"],
                    "content_group": record.get("content_group"),
                    "windows": len(bounds),
                    "window_proposals": len(proposals),
                    "merged_proposals": len(merged),
                    "inference_seconds": seconds,
                    **row[EXTERNAL_CLASS],
                }
            )
        summary = summarize_coverage(rows, (EXTERNAL_CLASS,))
        result = {
            "schema_version": 1,
            "protocol": {
                **external_protocol(settings, split),
                "method": "windowed",
                "window": window,
                "overlap": overlap,
                "merge_rule": "same class, overlap of the smaller box >= 0.3, boxes only",
            },
            "model": report["model"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "corpus_report_sha256": corpus_hash,
            "images": len(records),
            "content_groups": len({record.get("content_group") for record in records}),
            "coverage": summary,
            "proposals_per_photo": sum(d["merged_proposals"] for d in details) / len(details),
            "windows_per_photo": sum(d["windows"] for d in details) / len(details),
            "seconds_per_photo": sum(d["inference_seconds"] for d in details) / len(details),
            "records": details,
            "runtime": environment,
            "elapsed_seconds": time.monotonic() - started,
            "process_peak_rss_bytes": process_peak_rss(),
        }
        write_json(stage / "report.json", result)
    return result
