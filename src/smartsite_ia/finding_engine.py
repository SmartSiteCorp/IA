"""Relier le détecteur Nano existant au dossier local, sans entraînement ni réseau."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from smartsite_ia import finding_validation as check
from smartsite_ia.box_review import encode_boxes, load_box_engine
from smartsite_ia.categories import DEFECT_FAMILIES
from smartsite_ia.curation import digest, read_document
from smartsite_ia.finding import (
    ENABLED_CLASSES,
    analysis_identity,
    build_analysis,
    require_model,
    validate_request,
)
from smartsite_ia.finding_store import create_case, read_case, unlinked
from smartsite_ia.learning import file_hash, runtime
from smartsite_ia.model_assets import BOX_MODEL_NAME
from smartsite_ia.photo_quality import assess_photo
from smartsite_ia.prediction import decode_photo
from smartsite_ia.review import MAX_FILE_BYTES, read_local


def verified_box_run(root: Path) -> tuple[dict[str, Any], Path, str]:
    """Lire seulement le manifeste et les poids ; le service futur n'a pas besoin du corpus."""
    unlinked(root)
    report, run_sha = read_document(root / "run.json")
    names = report.get("class_names")
    if (
        report.get("status") != "completed"
        or report.get("model") != BOX_MODEL_NAME
        or report.get("checkpoint") != "checkpoints/checkpoint_best_total.pth"
        or report.get("checkpoint_selection") != "last_epoch"
        or not isinstance(names, list)
        or any(not isinstance(n, str) or n not in DEFECT_FAMILIES for n in names)
        or len(set(names)) != len(names)
        or not set(ENABLED_CLASSES).issubset(names)
        or not isinstance(report.get("config"), dict)
        or report["config"].get("class_names") != names
        or not isinstance(report.get("runtime"), dict)
        or not isinstance(report["runtime"].get("versions"), dict)
    ):
        raise ValueError("Expected a completed Nano box run with the enabled finding classes")
    checkpoint = root / report["checkpoint"]
    unlinked(checkpoint)
    if file_hash(checkpoint) != check.sha256(report.get("checkpoint_sha256")):
        raise ValueError("Checkpoint changed after training")
    return report, checkpoint, run_sha


def analyse_photo(
    image_path: Path,
    *,
    run: Path,
    output: Path,
    request: Mapping[str, Any],
    device: str,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Produire un dossier ou retrouver sa version existante, retours humains inclus."""
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Unsupported device")
    request = validate_request(request)
    # Un dossier ne doit jamais être créé dans l'arbre des poids ou sur la photo d'entrée.
    if output.resolve().is_relative_to(run.resolve()) or output.resolve() == image_path.resolve():
        raise ValueError("Finding output must be separate from model and input photo")
    unlinked(image_path)
    unlinked(output)
    raw = read_local(image_path.parent, image_path.name, MAX_FILE_BYTES)
    photo = decode_photo(raw)
    quality = assess_photo(photo)
    parent, checkpoint, run_sha = verified_box_run(run)
    model = {
        "run": run.name,
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "score_threshold": threshold,
    }
    require_model(model)
    engine_trace = {
        "run_sha256": run_sha,
        "class_names": parent["class_names"],
        "preprocessing": "training-v1",
        "runtime_versions": parent["runtime"]["versions"],
    }
    expected_id = analysis_identity(digest(raw), request, model, engine_trace, quality)
    if output.exists():
        existing = read_case(output, request["context"]["site_id"])
        if existing["analysis"]["id"] != expected_id:
            raise check.ConflictError("Output already belongs to another analysis")
        return existing
    predictions = []
    if quality["verdict"] != "unusable":
        environment = runtime(device)
        if environment["versions"] != parent["runtime"]["versions"]:
            raise ValueError("Inference runtime differs from the trained model")
        engine = load_box_engine(checkpoint, device, tuple(parent["class_names"]))
        encoded = encode_boxes(
            engine.predict(photo, threshold=threshold),
            photo,
            class_names=tuple(parent["class_names"]),
        )
        predictions = [
            {k: p[k] for k in ("class_name", "score", "bbox_xyxy")} for p in encoded["predictions"]
        ]
        del engine
    if (
        file_hash(checkpoint) != parent["checkpoint_sha256"]
        or file_hash(run / "run.json") != run_sha
    ):
        raise ValueError("Model changed during inference")
    analysis = build_analysis(
        photo_sha256=digest(raw),
        request=request,
        model=model,
        engine=engine_trace,
        quality=quality,
        predictions=predictions,
        analysed_at=check.now(),
    )
    return create_case(output, analysis, raw)
