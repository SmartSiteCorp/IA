"""Essayer un modèle déjà entraîné, avec des photos et des réglages figés."""

import argparse
import importlib
import json
import re
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

from smartsite_ia.annotations import number
from smartsite_ia.curation import digest, read_document, require_text, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.prediction import decode_photo
from smartsite_ia.review import MAX_FILE_BYTES, read_local
from smartsite_ia.source import Source, verify_archive
from smartsite_ia.validation_metrics import summarize_counts
from smartsite_ia.zero_shot_html import write_probe_html

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MODEL_FILES = {
    "README.md",
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
}
TARGET = "moisture_trace"


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    """La consigne et les seuils restent ceux du protocole, même si le résultat déçoit."""
    config, checksum = read_document(path)
    if config.get("model_id") != MODEL_ID or not re.fullmatch(
        r"[a-f0-9]{40}", str(config.get("revision", ""))
    ):
        raise ValueError("Expected the pinned Grounding DINO Tiny model and revision")
    prompt = require_text(config.get("prompt"))
    if len(prompt) > 200 or not prompt.endswith("."):
        raise ValueError("Expected a short prompt ending with a period")
    for key in ("box_threshold", "text_threshold", "match_iou"):
        if not 0 < number(config.get(key)) < 1:
            raise ValueError("Thresholds must be finite and between zero and one")
    if type(config.get("max_images")) is not int or not 1 <= config["max_images"] <= 64:
        raise ValueError("A probe must be bounded to at most 64 images")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {"torch", "transformers", "safetensors"}:
        raise ValueError("Expected pinned runtime versions")
    for value in runtime.values():
        require_text(value)
    files = config.get("files")
    if not isinstance(files, list) or len(files) != len(MODEL_FILES):
        raise ValueError("Expected the complete safe model file list")
    names = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"name", "size", "sha256"}:
            raise ValueError("Invalid model file specification")
        if (
            not isinstance(item["name"], str)
            or item["name"] not in MODEL_FILES
            or type(item["size"]) is not int
            or not 0 < item["size"] <= 800_000_000
            or not re.fullmatch(r"[a-f0-9]{64}", str(item["sha256"]))
        ):
            raise ValueError("Invalid model filename, size or SHA256")
        names.append(item["name"])
    if set(names) != MODEL_FILES:
        raise ValueError("Duplicate or missing model files")
    return config, checksum


def checked_box(raw: object, width: int, height: int) -> list[float]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("Expected an XYXY box")
    box = [number(value) for value in raw]
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("Box is empty or outside the oriented image")
    return box


def load_sample(path: Path, limit: int) -> tuple[dict[str, Any], str]:
    """Vérifier les photos avant de charger le moteur ou de commencer le calcul."""
    manifest, checksum = read_document(path)
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Missing sample provenance")
    for key in ("id", "url", "license", "scope"):
        require_text(source.get(key))
    rows = manifest.get("records")
    if not isinstance(rows, list) or not 1 <= len(rows) <= limit:
        raise ValueError("Sample is empty or exceeds the protocol limit")
    ids, hashes = set(), set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", row["id"])
        ):
            raise ValueError("Invalid sample identifier")
        require_text(row.get("file"))
        raw = read_local(path.parent, row["file"], MAX_FILE_BYTES)
        if digest(raw) != row.get("sha256"):
            raise ValueError("Image differs from sample manifest")
        photo = decode_photo(raw)
        if any(type(row.get(key)) is not int for key in ("width", "height")) or photo.size != (
            row["width"],
            row["height"],
        ):
            raise ValueError("Image dimensions differ from manifest")
        boxes = row.get("reference_boxes")
        if not isinstance(boxes, list) or len(boxes) > 200:
            raise ValueError("Invalid reference box list")
        for box in boxes:
            checked_box(box, photo.width, photo.height)
        require_text(row.get("review_note"))
        if row["id"] in ids or row["sha256"] in hashes:
            raise ValueError("Duplicate sample identifier or image content")
        ids.add(row["id"])
        hashes.add(row["sha256"])
    return manifest, checksum


def check_model(root: Path, protocol: dict[str, Any]) -> None:
    """Les poids sont en safetensors ; aucun pickle ou code distant n'est chargé."""
    for item in protocol["files"]:
        verify_archive(root / item["name"], Source("", item["size"], item["sha256"], 0))


class GroundingEngine:
    """Adaptateur limité au moteur publié, chargé une fois pour tout le petit lot."""

    def __init__(self, root: Path, protocol: dict[str, Any]) -> None:
        for package, expected in protocol["runtime"].items():
            if version(package) != expected:
                raise ValueError(f"Probe expects {package}=={expected}")
        self.torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        self.protocol = protocol
        self.previous_threads = self.torch.get_num_threads()
        # Le premier essai est sur CPU, pas de compatibilité MPS supposée...
        self.torch.set_num_threads(4)
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                root, local_files_only=True, trust_remote_code=False
            )
            self.model = (
                transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
                    root, local_files_only=True, trust_remote_code=False, use_safetensors=True
                )
                .to("cpu")
                .eval()
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.torch.set_num_threads(self.previous_threads)

    def predict(self, photo: Image.Image) -> list[dict[str, Any]]:
        with self.torch.inference_mode():
            inputs = self.processor(
                images=photo, text=self.protocol["prompt"], return_tensors="pt"
            ).to("cpu")
            outputs = self.model(**inputs)
            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.protocol["box_threshold"],
                text_threshold=self.protocol["text_threshold"],
                target_sizes=[(photo.height, photo.width)],
            )[0]
        records = []
        for box, score, label in zip(
            result["boxes"].tolist(),
            result["scores"].tolist(),
            result["text_labels"],
            strict=True,
        ):

            raw = [number(value) for value in box]
            clipped = [
                max(0.0, min(value, bound))
                for value, bound in zip(
                    raw, (photo.width, photo.height, photo.width, photo.height), strict=True
                )
            ]
            records.append(
                {"bbox_xyxy": clipped, "raw_bbox_xyxy": raw, "score": score, "text_label": label}
            )
        return records


def check_predictions(records: object, photo: Image.Image) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) > 900:
        raise ValueError("Invalid or excessive prediction count")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid prediction record")
        checked_box(record.get("bbox_xyxy"), photo.width, photo.height)
        if not 0 <= number(record.get("score")) <= 1:
            raise ValueError("Invalid prediction score")
        if not isinstance(record.get("text_label"), str) or len(record["text_label"]) > 200:
            raise ValueError("Invalid prediction label")
    return records


def box_counts(
    references: list[list[float]], predictions: list[dict[str, Any]], overlap: float
) -> dict[str, int]:
    """Compter l'accord avec les boîtes auteur, sans l'appeler une vérité terrain."""
    ordered = sorted(predictions, key=lambda item: item["score"], reverse=True)

    def xywh(box: list[float]) -> list[float]:
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    ious = (
        coco_mask.iou(
            [xywh(p["bbox_xyxy"]) for p in ordered],
            [xywh(box) for box in references],
            [0] * len(references),
        )
        if references and ordered
        else np.zeros((len(ordered), len(references)))
    )
    used: set[int] = set()
    for row in ious:
        candidates = [i for i in range(len(references)) if i not in used and row[i] >= overlap]
        if candidates:
            used.add(max(candidates, key=lambda i: row[i]))
    return {"tp": len(used), "fp": len(predictions) - len(used), "fn": len(references) - len(used)}


def draw_boxes(photo: Image.Image, boxes: list[list[float]], color: str) -> Image.Image:
    result = photo.copy()
    draw = ImageDraw.Draw(result)
    for box in boxes:
        draw.rectangle(box, outline=color, width=max(2, photo.width // 320))
    return result


def run_probe(sample: Path, model: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(protocol_path)
    manifest, manifest_sha = load_sample(sample, protocol["max_images"])
    check_model(model, protocol)
    start = time.perf_counter()
    with staged_output(output, [sample.parent, model, protocol_path]) as stage:
        engine = GroundingEngine(model, protocol)
        rows = []
        try:
            for index, row in enumerate(manifest["records"], 1):
                raw = read_local(sample.parent, row["file"], MAX_FILE_BYTES)
                if digest(raw) != row["sha256"]:
                    raise ValueError("Image changed during probe")
                photo = decode_photo(raw)
                begun = time.perf_counter()
                predictions = check_predictions(engine.predict(photo), photo)
                counts = box_counts(row["reference_boxes"], predictions, protocol["match_iou"])
                result = {
                    **row,
                    "predictions": predictions,
                    "counts": counts,
                    "inference_seconds": time.perf_counter() - begun,
                }
                folder = stage / row["id"]
                folder.mkdir()
                photo.save(folder / "photo.jpg", quality=92)
                draw_boxes(photo, row["reference_boxes"], "#00a884").save(
                    folder / "reference.jpg", quality=92
                )
                draw_boxes(photo, [p["bbox_xyxy"] for p in predictions], "#ed7425").save(
                    folder / "prediction.jpg", quality=92
                )
                write_json(folder / "result.json", result)
                rows.append(result)
                print(
                    f"Photo {index}/{len(manifest['records'])} : {row['id']} — "
                    f"{len(predictions)} proposition(s)",
                    flush=True,
                )
        finally:
            engine.close()
        summary = summarize_counts([{TARGET: row["counts"]} for row in rows], (TARGET,))[TARGET]
        report = {
            "schema_version": 1,
            "status": "completed",
            "training_started": False,
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "sample_sha256": manifest_sha,
            "source": manifest["source"],
            "device": "cpu",
            "threads": 4,
            "elapsed_seconds": time.perf_counter() - start,
            "summary": summary,
            "records": rows,
            "limitations": "Exploration sur annotations auteur ; aucune qualification chantier. "
            "Absence de label ≠ surface saine. Photos proches, bâtiments non identifiés. "
            "Aucune preuve d'indépendance vis-à-vis du préentraînement. "
            "Une trace n'établit ni humidité actuelle ni fuite active.",
        }
        write_json(stage / "report.json", report)
        write_probe_html(stage, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded, offline moisture probe; no training"
    )
    parser.add_argument("sample", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_probe(args.sample, args.model, args.protocol, args.output)
    except (ValueError, OSError, ImportError, PackageNotFoundError) as error:
        print(f"Probe failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "training_started": False,
                "report": str(args.output / "index.html"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
