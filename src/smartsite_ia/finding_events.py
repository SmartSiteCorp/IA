"""Rejouer les décisions humaines sans modifier les prédictions originales.

Une confirmation décrit le défaut. La reprise des travaux possède son propre état.
Les noms d'auteurs sont déclarés localement ; l'authentification viendra du backend.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from smartsite_ia import finding_validation as check
from smartsite_ia.categories import DEFECT_FAMILIES, FAMILY_LABELS
from smartsite_ia.finding import MAX_FINDINGS, SCHEMA_VERSION, defect_class

MAX_EVENTS = 1000
DECISIONS = ("confirmed", "rejected", "indeterminate")
CORRECTION_STATES = ("planned", "in_progress", "resolved", "not_required")


def event_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Vérifier une demande de retour avant de l'ajouter à l'historique."""
    check.fields(value, ["request_id", "kind", "finding_id", "actor", "payload"], ["occurred_at"])
    kind = value["kind"]
    if kind not in ("manual", "review", "correction", "note"):
        raise ValueError("Unknown finding event kind")
    target = None if kind == "manual" else check.identifier(value["finding_id"])
    if kind == "manual" and value["finding_id"] is not None:
        raise ValueError("A manual report receives its own identifier")
    payload = value["payload"]
    if kind == "manual":
        check.fields(payload, ["class_name", "bbox_xyxy", "note"])
        defect_class(payload["class_name"])
        if not isinstance(payload["bbox_xyxy"], (list, tuple)) or len(payload["bbox_xyxy"]) != 4:
            raise ValueError("Manual report needs four coordinates")
        payload = {**payload, "bbox_xyxy": [check.number(v) for v in payload["bbox_xyxy"]]}
    elif kind == "review":
        check.fields(payload, ["decision", "note"], ["class_name", "task_id", "severity"])
        if payload["decision"] not in DECISIONS:
            raise ValueError("Unknown human decision")
        category = (
            defect_class(payload["class_name"]) if payload.get("class_name") is not None else None
        )
        payload = {
            **payload,
            "class_name": category,
            "task_id": check.optional_text(payload.get("task_id"), "task_id"),
            "severity": check.optional_text(payload.get("severity"), "human severity", 500),
        }
        if payload["decision"] != "confirmed" and any(
            payload[k] is not None for k in ("class_name", "task_id", "severity")
        ):
            raise ValueError("Classification, task and severity require a confirmed finding")
    elif kind == "correction":
        check.fields(payload, ["status", "note", "evidence_sha256"])
        if payload["status"] not in CORRECTION_STATES:
            raise ValueError("Unknown correction state")
        evidence = payload["evidence_sha256"]
        if not isinstance(evidence, list) or len(evidence) > 4:
            raise ValueError("Correction evidence must be a bounded list without duplicates")
        for sha in evidence:
            check.sha256(sha)
        if len(set(evidence)) != len(evidence):
            raise ValueError("Duplicate correction evidence")
        if payload["status"] == "resolved" and not evidence:
            raise ValueError("Resolving a correction requires photographic evidence")
    else:
        check.fields(payload, ["note"])
    payload = {**payload, "note": check.text(payload["note"], "human note", 4000)}
    return {
        "request_id": check.text(value["request_id"], "request_id", 128),
        "kind": kind,
        "finding_id": target,
        "actor": check.text(value["actor"], "actor"),
        "occurred_at": check.timestamp(value["occurred_at"])
        if value.get("occurred_at") is not None
        else None,
        "payload": payload,
    }


def empty_current(finding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "finding": deepcopy(finding),
        "verification": {"state": "proposed"},
        "correction": {"status": "not_assessed"},
        "notes": [],
    }


def manual_finding(analysis: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    """Un ajout humain n'acquiert ni score de réseau ni fausse origine automatique."""
    payload = event["request"]["payload"]
    dimensions = (analysis["quality"]["photo"]["width"], analysis["quality"]["photo"]["height"])
    coords = check.box(payload["bbox_xyxy"], dimensions)
    category = payload["class_name"]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": check.identity([analysis["id"], event["request"]["request_id"], "manual"]),
        "origin": "human",
        "state": "proposed",
        "defect": {
            "class": category,
            "family": DEFECT_FAMILIES[category],
            "label": FAMILY_LABELS[category],
            "material": None,
        },
        "location": {
            "bbox_xyxy": coords,
            "bbox_normalized": [v / dimensions[i % 2] for i, v in enumerate(coords)],
            "photo_width": dimensions[0],
            "photo_height": dimensions[1],
            "coordinate_space": "EXIF-oriented image, pixels",
            "point_3d": None,
        },
        "evidence": {"photo_sha256": analysis["photo_sha256"]},
        "context": deepcopy(analysis["request"]["context"]),
        "source": {"actor": event["request"]["actor"], "recorded_at": event["recorded_at"]},
        "confidence": {
            "model_score": None,
            "is_not_a_severity": True,
            "is_calibrated_probability": False,
        },
        "severity": None,
        "measurement": None,
    }


def replay(analysis: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """L'état courant se reconstruit depuis le journal ; aucune décision n'est écrasée."""
    if len(events) > MAX_EVENTS:
        raise ValueError("Finding history exceeds its event limit")
    current = {f["id"]: empty_current(f) for f in analysis["findings"]}
    seen: set[str] = set()
    last_time = analysis["analysed_at"]
    for revision, event in enumerate(events, 1):
        check.fields(event, ["revision", "recorded_at", "request"])
        if type(event["revision"]) is not int or event["revision"] != revision:
            raise ValueError("Broken finding history revision sequence")
        at = check.timestamp(event["recorded_at"])
        if at < last_time or at != event["recorded_at"]:
            raise ValueError("Finding history timestamps are inconsistent")
        request = event_request(event["request"])
        if (
            check.canonical(request) != check.canonical(event["request"])
            or request["request_id"] in seen
        ):
            raise ValueError("Noncanonical or duplicate finding event")
        if request["occurred_at"] is not None and request["occurred_at"] > at:
            raise ValueError("Human observation cannot be later than its recording")
        last_time = at
        seen.add(request["request_id"])
        kind, payload = request["kind"], request["payload"]
        if kind == "manual":
            if len(current) >= MAX_FINDINGS:
                raise ValueError("Too many findings")
            finding = manual_finding(analysis, event)
            current[finding["id"]] = empty_current(finding)
            current[finding["id"]]["notes"].append(deepcopy(event))
            continue
        target = request["finding_id"]
        if target not in current:
            raise ValueError("Finding is absent from this analysis")
        view = current[target]
        if kind == "review":
            old = view["verification"]
            category = payload["class_name"] or view["finding"]["defect"]["class"]
            if old["state"] != payload["decision"] or old.get("effective_class") != category:
                # Une nouvelle conclusion ne confirme pas une ancienne réparation.
                view["correction"] = {"status": "not_assessed"}
            view["verification"] = {
                **payload,
                "state": payload["decision"],
                "effective_class": category,
                "actor": request["actor"],
                "recorded_at": at,
                "occurred_at": request["occurred_at"],
            }
        elif kind == "correction":
            if view["verification"]["state"] != "confirmed":
                raise ValueError("A correction requires a confirmed finding")
            view["correction"] = {
                **payload,
                "actor": request["actor"],
                "recorded_at": at,
                "occurred_at": request["occurred_at"],
            }
        else:
            view["notes"].append(deepcopy(event))
    return {
        "analysis_id": analysis["id"],
        "revision": len(events),
        "findings": list(current.values()),
        "automatic_outcome": analysis["outcome"],
        "human_reports": sum(v["finding"]["origin"] == "human" for v in current.values()),
        "notification": {"status": "not_connected", "sent": False},
        "conformity": "not_assessed",
    }


def append_event(
    analysis: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    *,
    expected_revision: int,
    recorded_at: str,
) -> list[dict[str, Any]]:
    """Une relance identique est sans effet ; une édition concurrente est refusée."""
    if type(expected_revision) is not int or expected_revision < 0:
        raise check.ConflictError("Expected revision must be a nonnegative integer")
    normalized = event_request(request)
    for event in events:
        if event["request"]["request_id"] == normalized["request_id"]:
            if check.canonical(event["request"]) != check.canonical(normalized):
                raise check.ConflictError("Request identifier was already used for another event")
            return [deepcopy(dict(e)) for e in events]
    if expected_revision != len(events):
        raise check.ConflictError("Finding history changed; reload its current revision")
    updated = [
        *[deepcopy(dict(e)) for e in events],
        {
            "revision": len(events) + 1,
            "recorded_at": check.timestamp(recorded_at),
            "request": normalized,
        },
    ]
    replay(analysis, updated)
    return updated
