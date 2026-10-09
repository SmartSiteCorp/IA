"""Construire un constat traçable, sans confondre score, gravité et conformité.

Le constat garde la proposition originale. Les retours humains vivent dans un
journal séparé, traité par `finding_events`. Aucun import ne lance un modèle.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from smartsite_ia import finding_validation as check
from smartsite_ia.categories import CLASS_NAMES, DEFECT_FAMILIES, FAMILY_LABELS
from smartsite_ia.photo_quality import (
    GRID,
    VERDICTS,
    check_thresholds,
    choose_verdict,
    list_reasons,
)

SCHEMA_VERSION = 2
INITIAL_STATE = "proposed"
ENABLED_CLASSES = CLASS_NAMES
MAX_FINDINGS = 200
MAX_PREDICTIONS = 500
OUTCOMES = (
    "defects_found",
    "no_target_defect_in_readable_area",
    "partial_analysis",
    "not_assessable",
)


def finding_identifier(photo_sha256: str, class_name: str, box: Sequence[float]) -> str:
    """Identité géométrique locale ; le constat ajoute chantier et modèle à cette base.

    Le pixel arrondi absorbe seulement de très petits écarts numériques. Ce n'est
    ni un rapprochement entre photos, ni une identification du défaut dans le temps.
    """
    check.sha256(photo_sha256)
    defect_class(class_name)
    if len(box) != 4:
        raise ValueError("Detection box is malformed")
    return check.identity([photo_sha256, class_name, [round(check.number(n)) for n in box]])


def defect_class(value: object) -> str:
    if not isinstance(value, str) or value not in DEFECT_FAMILIES:
        raise ValueError("Unknown defect class")
    return value


def require_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Les identifiants locaux ne prouvent pas les droits d'accès d'un utilisateur."""
    if not isinstance(context, Mapping) or not context.get("site_id"):
        raise ValueError("A finding needs the site it belongs to")
    check.fields(context, ["site_id"], ["zone_id", "task_id"])
    try:
        site = check.text(context["site_id"], "site_id")
    except ValueError as exc:
        raise ValueError("A finding needs the site it belongs to") from exc
    return {
        "site_id": site,
        **{k: check.optional_text(context.get(k), k) for k in ("zone_id", "task_id")},
    }


def require_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Tracer les poids et le seuil, sans transformer ce score en probabilité validée."""
    try:
        check.fields(model, ["run", "checkpoint_sha256", "score_threshold"])
        run = check.text(model["run"], "model run")
        sha = check.sha256(model["checkpoint_sha256"])
        threshold = check.number(model["score_threshold"])
        if not 0 < threshold < 1:
            raise ValueError("Invalid threshold")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "A finding must carry a valid model run, fingerprint and threshold"
        ) from exc
    return {
        "run": run,
        "checkpoint_sha256": sha,
        "score_threshold": threshold,
        "qualified_for_smartsite": False,
    }


def build_finding(
    detection: Mapping[str, Any],
    *,
    photo_sha256: str,
    photo_size: tuple[int, int],
    context: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Garder la prédiction et ses coordonnées dans la photo remise à l'endroit."""
    check.fields(detection, ["class_name", "score", "bbox_xyxy"])
    class_name = defect_class(detection["class_name"])
    try:
        score = check.number(detection["score"])
        if not 0 <= score <= 1:
            raise ValueError("Invalid score")
    except ValueError as exc:
        raise ValueError("Detection score must sit between 0 and 1") from exc
    width, height = check.size(photo_size)
    coords = check.box(detection["bbox_xyxy"], (width, height))
    check.sha256(photo_sha256)
    checked_context, source = require_context(context), require_model(model)
    local_id = finding_identifier(photo_sha256, class_name, coords)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": check.identity(
            [local_id, checked_context["site_id"], checked_context["zone_id"], source]
        ),
        "origin": "model",
        "state": INITIAL_STATE,
        "defect": {
            "class": class_name,
            "family": DEFECT_FAMILIES[class_name],
            "label": FAMILY_LABELS[class_name],
            "material": None,
        },
        "location": {
            "bbox_xyxy": coords,
            "bbox_normalized": [
                coords[0] / width,
                coords[1] / height,
                coords[2] / width,
                coords[3] / height,
            ],
            "photo_width": width,
            "photo_height": height,
            "coordinate_space": "EXIF-oriented image, pixels",
            "point_3d": None,
        },
        "evidence": {"photo_sha256": photo_sha256},
        "context": checked_context,
        "source": source,
        "confidence": {
            "model_score": score,
            "is_not_a_severity": True,
            "is_calibrated_probability": False,
        },
        "severity": None,
        "measurement": None,
    }


def analyse_outcome(quality: Mapping[str, Any], findings: Sequence[Mapping[str, Any]]) -> str:
    """La qualité reste jointe même lorsqu'une photo partielle porte un signalement."""
    verdict = quality.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError("Unknown photo quality verdict")
    if verdict == "unusable":
        return "not_assessable"
    if findings:
        return "defects_found"
    return "partial_analysis" if verdict == "partial" else "no_target_defect_in_readable_area"


def validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Décrire ce qui est demandé sans inventer une date ou une capture manquante."""
    check.fields(request, ["context"], ["capture_id", "captured_at", "media_id"])
    return {
        "context": require_context(request["context"]),
        "capture_id": check.optional_text(request.get("capture_id"), "capture_id"),
        "media_id": check.optional_text(request.get("media_id"), "media_id"),
        "captured_at": check.timestamp(request["captured_at"])
        if request.get("captured_at") is not None
        else None,
    }


def analysis_identity(
    photo_sha256: str,
    request: Mapping[str, Any],
    model: Mapping[str, Any],
    engine: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> str:
    """Une même demande sur les mêmes entrées ne crée pas une seconde analyse."""
    return check.identity(
        [
            SCHEMA_VERSION,
            check.sha256(photo_sha256),
            validate_request(request),
            require_model(model),
            engine,
            quality,
        ]
    )


def validate_quality(quality: Mapping[str, Any]) -> tuple[int, int]:
    """La forme du bilan est contrôlée ici ; le stockage le recalcule sur la photo."""
    check.fields(
        quality,
        [
            "schema_version",
            "photo",
            "verdict",
            "reasons",
            "measures",
            "lost_regions",
            "thresholds",
            "limits",
        ],
    )
    if type(quality["schema_version"]) is not int or quality["schema_version"] != 1:
        raise ValueError("Unsupported photo quality schema")
    check.fields(quality["photo"], ["width", "height"])
    dimensions = check.size((quality["photo"]["width"], quality["photo"]["height"]))
    limits = check_thresholds(quality["thresholds"])
    measure_names = [
        "mean_luminance",
        "dark_fraction",
        "bright_fraction",
        "readable_fraction",
        "counting_scale",
        "sharpness",
    ]
    check.fields(quality["measures"], measure_names)
    measures = {k: check.number(quality["measures"][k]) for k in measure_names}
    if (
        not 0 <= measures["mean_luminance"] <= 255
        or any(not 0 <= measures[k] <= 1 for k in measure_names[1:4])
        or measures["counting_scale"] < 1
        or measures["sharpness"] < 0
    ):
        raise ValueError("Invalid photo quality measures")
    if quality["verdict"] != choose_verdict(measures, limits) or quality["reasons"] != list_reasons(
        measures, limits
    ):
        raise ValueError("Photo quality verdict or reasons differ from its measures")
    check.text(quality["limits"], "photo quality limits", 1000)
    regions = quality["lost_regions"]
    if not isinstance(regions, list) or len(regions) > GRID * GRID:
        raise ValueError("Invalid photo quality regions")
    for region in regions:
        check.fields(region, ["x", "y", "width", "height", "reason"])
        x, y, w, h = [check.number(region[k]) for k in ("x", "y", "width", "height")]
        check.box([x, y, x + w, y + h], dimensions)
        if region["reason"] not in ("too_dark", "too_bright"):
            raise ValueError("Unknown lost-region reason")
    return dimensions


def unreadable_fraction(coords: list[float], quality: Mapping[str, Any]) -> float:
    """Les cases perdues de la grille sont disjointes : leurs intersections s'additionnent."""
    x, y, u, v = coords
    lost = sum(
        max(0, min(u, r["x"] + r["width"]) - max(x, r["x"]))
        * max(0, min(v, r["y"] + r["height"]) - max(y, r["y"]))
        for r in quality["lost_regions"]
    )
    return float(min(1.0, lost / ((u - x) * (v - y))))


def build_analysis(
    *,
    photo_sha256: str,
    request: Mapping[str, Any],
    model: Mapping[str, Any],
    engine: Mapping[str, Any],
    quality: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    analysed_at: str,
) -> dict[str, Any]:
    """Limiter les constats automatiques aux deux classes retenues, en traçant le reste."""
    request = validate_request(request)
    model = {k: v for k, v in require_model(model).items() if k != "qualified_for_smartsite"}
    check.fields(engine, ["run_sha256", "class_names", "preprocessing", "runtime_versions"])
    check.sha256(engine["run_sha256"])
    names = engine["class_names"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) or n not in DEFECT_FAMILIES for n in names)
        or len(set(names)) != len(names)
        or not set(ENABLED_CLASSES).issubset(names)
    ):
        raise ValueError("Unexpected engine classes for the finding scope")
    if engine["preprocessing"] != "training-v1" or not isinstance(engine["runtime_versions"], dict):
        raise ValueError("Unexpected engine provenance")
    for name, version in engine["runtime_versions"].items():
        check.text(name, "runtime name")
        check.text(version, "runtime version")
    dimensions = validate_quality(quality)
    at = check.timestamp(analysed_at)
    if request["captured_at"] is not None and request["captured_at"] > at:
        raise ValueError("Capture cannot be later than analysis")
    if len(predictions) > MAX_PREDICTIONS or (quality["verdict"] == "unusable" and predictions):
        raise ValueError("Too many predictions or prediction on an unusable photo")
    analysis_id = analysis_identity(photo_sha256, request, model, engine, quality)
    findings: list[dict[str, Any]] = []
    withheld, normalized = [], []
    seen: dict[str, tuple[int, int]] = {}
    for detection in predictions:
        finding = build_finding(
            detection,
            photo_sha256=photo_sha256,
            photo_size=dimensions,
            context=request["context"],
            model=model,
        )
        if finding["defect"]["class"] not in names:
            raise ValueError("Prediction class is absent from model provenance")
        finding["id"] = check.identity([analysis_id, finding["id"]])
        normalized.append({k: detection[k] for k in ("class_name", "score", "bbox_xyxy")})
        fraction = unreadable_fraction(finding["location"]["bbox_xyxy"], quality)
        finding["observability"] = {
            "unreadable_box_fraction": fraction,
            "status": "partial" if fraction > 0 or quality["verdict"] == "partial" else "usable",
        }
        reason = None
        prediction_index = len(normalized) - 1
        if finding["defect"]["class"] not in ENABLED_CLASSES:
            reason = "class_not_enabled"
        elif finding["confidence"]["model_score"] <= model["score_threshold"]:
            reason = "below_or_equal_threshold"
        elif fraction >= 1:
            reason = "unreadable_region"
        elif finding["id"] in seen:
            slot, previous_index = seen[finding["id"]]
            if finding["confidence"]["model_score"] > findings[slot]["confidence"]["model_score"]:
                findings[slot] = finding
                seen[finding["id"]] = (slot, prediction_index)
                prediction_index = previous_index
            reason = "duplicate_geometry"
        if reason:
            withheld.append({"prediction_index": prediction_index, "reason": reason})
        else:
            seen[finding["id"]] = (len(findings), prediction_index)
            findings.append(finding)
    if len(findings) > MAX_FINDINGS:
        raise ValueError("Too many findings; analysis must not silently truncate them")
    return {
        "schema_version": SCHEMA_VERSION,
        "id": analysis_id,
        "analysed_at": at,
        "photo_sha256": photo_sha256,
        "request": request,
        "model": model,
        "engine": dict(engine),
        "quality": dict(quality),
        "enabled_classes": list(ENABLED_CLASSES),
        "predictions": normalized,
        "withheld": withheld,
        "findings": findings,
        "outcome": analyse_outcome(quality, findings),
        "inference_executed": quality["verdict"] != "unusable",
        "qualified_for_smartsite": False,
        "conformity": "not_assessed",
    }


def validate_analysis(value: Mapping[str, Any]) -> None:
    """Rejouer la construction permet de vérifier aussi les champs dérivés du JSON."""
    expected = build_analysis(
        **{
            k: value[k]
            for k in (
                "photo_sha256",
                "request",
                "model",
                "engine",
                "quality",
                "predictions",
                "analysed_at",
            )
        }
    )
    if check.canonical(expected) != check.canonical(value):
        raise ValueError("Analysis differs from its immutable inputs")
