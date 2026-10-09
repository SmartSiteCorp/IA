"""Transformer une détection en constat exploitable par le chantier.

Le moteur rend des boîtes et des scores. Un chef de chantier a besoin d'autre chose :
savoir quel type d'anomalie se trouve où, sur quelle photo, avec quel modèle, et ce
qui reste à vérifier. Ce module fait cette traduction, sans rien inventer.

Deux règles guident tout le fichier. Le score du réseau n'est pas une gravité : il dit
la confiance du modèle, pas le danger, et la gravité reste vide jusqu'à ce qu'une
personne la décide. Et l'absence de détection n'est pas une conformité : une photo
trop sombre ou partiellement lisible doit le dire.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from smartsite_ia.categories import DEFECT_FAMILIES, FAMILY_LABELS
from smartsite_ia.curation import digest

SCHEMA_VERSION = 1

# L'état de départ d'un constat. La suite du cycle de vie viendra avec la vérification.
INITIAL_STATE = "proposed"

# Les quatre réponses possibles d'une analyse, distinguées par le cadrage de septembre.
OUTCOMES = (
    "defects_found",
    "no_target_defect_in_readable_area",
    "partial_analysis",
    "not_assessable",
)

MAX_FINDINGS = 200


def finding_identifier(photo_sha256: str, class_name: str, box: Sequence[float]) -> str:
    """Donner au constat un identifiant stable, reproductible depuis ses propres faits.

    Deux analyses de la même photo par le même modèle doivent produire le même
    identifiant, sinon relancer une analyse créerait des doublons dans le suivi. Les
    coordonnées sont arrondies au pixel : un écart infime de calcul ne doit pas donner
    un nouveau constat.
    """
    rounded = ",".join(f"{round(float(value))}" for value in box)
    return digest(f"{photo_sha256}:{class_name}:{rounded}".encode())[:32]


def build_finding(
    detection: Mapping[str, Any],
    *,
    photo_sha256: str,
    photo_size: tuple[int, int],
    context: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    """Construire un constat à partir d'une détection et de son contexte de chantier."""
    class_name = detection["class_name"]
    if class_name not in DEFECT_FAMILIES:
        raise ValueError(f"Unknown defect class: {class_name}")
    score = detection["score"]
    if type(score) not in (int, float) or not 0 <= score <= 1:
        raise ValueError("Detection score must sit between 0 and 1")
    box = [float(value) for value in detection["bbox_xyxy"]]
    width, height = photo_size
    if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
        raise ValueError("Detection box is malformed")
    if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
        raise ValueError("Detection box falls outside the photo")
    return {
        "schema_version": SCHEMA_VERSION,
        "id": finding_identifier(photo_sha256, class_name, box),
        "state": INITIAL_STATE,
        "defect": {
            "class": class_name,
            "family": DEFECT_FAMILIES[class_name],
            "label": FAMILY_LABELS[class_name],
            # Le matériau se renseigne au moment de la vérification : le modèle ne le voit pas.
            "material": None,
        },
        "location": {
            "bbox_xyxy": box,
            "bbox_normalized": [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
            "photo_width": width,
            "photo_height": height,
            # Sans association fiable entre image, profondeur et pose, on ne place
            # pas le constat dans l'espace. Inventer un point 3D serait pire que rien.
            "point_3d": None,
        },
        "evidence": {"photo_sha256": photo_sha256},
        "context": require_context(context),
        "source": require_model(model),
        "confidence": {
            "model_score": float(score),
            # Un score élevé dit que le modèle est sûr de reconnaître la forme,
            # pas que le défaut est grave. Les deux champs restent séparés.
            "is_not_a_severity": True,
        },
        "severity": None,
        "measurement": None,
    }


def require_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Un constat sans chantier ne peut être ni affiché ni suivi."""
    site = context.get("site_id")
    if not isinstance(site, str) or not site.strip():
        raise ValueError("A finding needs the site it belongs to")
    for key in ("zone_id", "task_id"):
        value = context.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Invalid {key} in the finding context")
    return {
        "site_id": site,
        "zone_id": context.get("zone_id"),
        "task_id": context.get("task_id"),
    }


def require_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Tracer quel modèle a produit le constat, pour pouvoir le rejouer plus tard."""
    run = model.get("run")
    checkpoint = model.get("checkpoint_sha256")
    threshold = model.get("score_threshold")
    if not isinstance(run, str) or not run.strip():
        raise ValueError("A finding must name the run that produced it")
    if not isinstance(checkpoint, str) or len(checkpoint) != 64:
        raise ValueError("A finding must carry the checkpoint fingerprint")
    # `isinstance` plutôt que `type(...) in` : le typage strict peut alors conclure
    # que la valeur n'est plus absente. Le cas booléen est écarté à part.
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not 0 < threshold < 1
    ):
        raise ValueError("A finding must record the score threshold used")
    return {
        "run": run,
        "checkpoint_sha256": checkpoint,
        "score_threshold": float(threshold),
        "qualified_for_smartsite": False,
    }


def analyse_outcome(quality: Mapping[str, Any], findings: Sequence[Mapping[str, Any]]) -> str:
    """Dire ce que l'analyse a réellement établi, sans transformer un silence en conformité."""
    verdict = quality["verdict"]
    if verdict == "unusable":
        return "not_assessable"
    if findings:
        return "defects_found"
    return "partial_analysis" if verdict == "partial" else "no_target_defect_in_readable_area"
