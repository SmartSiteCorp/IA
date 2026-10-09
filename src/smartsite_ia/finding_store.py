"""Conserver un dossier local atomique avec ses photos et son historique.

Ce stockage est un outil hors ligne. Le contrôle du chantier évite une erreur de
ciblage, mais ne remplace pas l'authentification et les droits du futur serveur.
"""

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from smartsite_ia import finding_validation as check
from smartsite_ia.annotations import unique_object
from smartsite_ia.curation import digest, staged_output
from smartsite_ia.finding import SCHEMA_VERSION, validate_analysis
from smartsite_ia.finding_events import append_event, replay
from smartsite_ia.photo_quality import assess_photo
from smartsite_ia.prediction import decode_photo
from smartsite_ia.review import MAX_FILE_BYTES, MAX_JSON_BYTES, read_local


class IntegrityError(ValueError):
    """Le dossier est incomplet, incohérent ou ses preuves ont changé."""


class BusyError(RuntimeError):
    """Une autre opération écrit déjà ce dossier."""


def unlinked(path: Path) -> None:
    """Aucune redirection par lien dans les chemins de ce stockage local."""
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Linked paths are not accepted in a finding dossier")


@contextmanager
def case_lock(root: Path) -> Iterator[None]:
    """Le verrou est extérieur au dossier pour protéger aussi sa création atomique."""
    unlinked(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / ("." + root.name + ".finding.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BusyError("Another operation is writing this finding dossier") from exc
        yield


def atomic_record(path: Path, value: Mapping[str, Any]) -> None:
    """Une interruption avant le remplacement laisse la version précédente lisible."""
    raw = check.canonical(value) + b"\n"
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Finding dossier exceeds the JSON size limit")
    atomic_bytes(path, raw)


def atomic_bytes(path: Path, raw: bytes) -> None:
    """Publier un fichier entier ; une preuve interrompue reste hors du journal."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".finding-", delete=False) as out:
            temporary = Path(out.name)
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def evidence_hashes(events: list[dict[str, Any]]) -> set[str]:
    return {
        sha
        for e in events
        if e["request"]["kind"] == "correction"
        for sha in e["request"]["payload"]["evidence_sha256"]
    }


def read_case(root: Path, site_id: str) -> dict[str, Any]:
    """Vérifier schéma, calculs, chantier et preuves avant tout usage du dossier."""
    unlinked(root)
    try:
        raw = read_local(root, "record.json", MAX_JSON_BYTES)
        value = json.loads(raw, object_pairs_hook=unique_object)
        check.fields(value, ["schema_version", "analysis", "events"])
        if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported finding dossier schema")
        if not isinstance(value["events"], list):
            raise ValueError("Expected an event list")
        analysis = value["analysis"]
        validate_analysis(analysis)
        if analysis["request"]["context"]["site_id"] != check.text(site_id, "site_id"):
            raise check.ConflictError("This finding dossier belongs to another site")
        source = read_local(root, "source.bin", MAX_FILE_BYTES)
        if digest(source) != analysis["photo_sha256"]:
            raise ValueError("Source photo changed")
        photo = decode_photo(source)
        if assess_photo(photo, thresholds=analysis["quality"]["thresholds"]) != analysis["quality"]:
            raise ValueError("Photo quality or coordinates differ from the evidence")
        replay(analysis, value["events"])
        for sha in evidence_hashes(value["events"]):
            proof = read_local(root, f"evidence/{sha}.bin", MAX_FILE_BYTES)
            if digest(proof) != sha:
                raise ValueError("Correction evidence changed")
            decode_photo(proof)
    except check.ConflictError:
        raise
    except (KeyError, TypeError, IndexError, ValueError, OSError, RecursionError) as exc:
        raise IntegrityError(
            "Finding dossier is invalid, incomplete or its evidence changed"
        ) from exc
    return dict(value)


def create_case(root: Path, analysis: Mapping[str, Any], source: bytes) -> dict[str, Any]:
    """Réutiliser une demande identique, sans perdre les retours déjà enregistrés."""
    validate_analysis(analysis)
    if len(source) > MAX_FILE_BYTES or digest(source) != analysis["photo_sha256"]:
        raise ValueError("Source photo differs from the analysis")
    photo = decode_photo(source)
    if assess_photo(photo, thresholds=analysis["quality"]["thresholds"]) != analysis["quality"]:
        raise ValueError("Photo quality differs from the analysis")
    with case_lock(root):
        if root.exists():
            old = read_case(root, analysis["request"]["context"]["site_id"])
            if old["analysis"]["id"] != analysis["id"]:
                raise check.ConflictError("Output already belongs to another analysis")
            return old
        record = {"schema_version": SCHEMA_VERSION, "analysis": dict(analysis), "events": []}
        with staged_output(root, []) as stage:
            (stage / "source.bin").write_bytes(source)
            atomic_record(stage / "record.json", record)
        return record


def record_event(
    root: Path,
    site_id: str,
    request: Mapping[str, Any],
    *,
    expected_revision: int,
    evidence: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Écrire d'abord les nouvelles preuves, puis publier le journal en une fois."""
    attachments = dict(evidence or {})
    if len(attachments) > 4:
        raise ValueError("Too many evidence photos")
    for sha, raw in attachments.items():
        check.sha256(sha)
        if len(raw) > MAX_FILE_BYTES or digest(raw) != sha:
            raise ValueError("Invalid evidence photo fingerprint or size")
        decode_photo(raw)
    with case_lock(root):
        current = read_case(root, site_id)
        events = append_event(
            current["analysis"],
            current["events"],
            request,
            expected_revision=expected_revision,
            recorded_at=check.now(),
        )
        needed = evidence_hashes(events)
        if not set(attachments).issubset(needed):
            raise ValueError("Unreferenced evidence photo")
        for sha in needed:
            if sha not in attachments:
                raw = read_local(root, f"evidence/{sha}.bin", MAX_FILE_BYTES)
                if digest(raw) != sha:
                    raise IntegrityError("Missing or changed correction evidence")
        if events == current["events"]:
            return current
        if attachments:
            folder = root / "evidence"
            existing = list(folder.iterdir()) if folder.exists() else []
            new_count = sum(not (folder / f"{sha}.bin").exists() for sha in attachments)
            if len(existing) + new_count > 64:
                raise ValueError("Dossier exceeds its limit of 64 evidence photos")
        if attachments:
            unlinked(root / "evidence")
            (root / "evidence").mkdir(exist_ok=True)
        for sha, raw in attachments.items():
            path = root / "evidence" / f"{sha}.bin"
            unlinked(path)
            if path.exists():
                if read_local(root, f"evidence/{sha}.bin", MAX_FILE_BYTES) != raw:
                    raise IntegrityError("Existing evidence changed")
            else:
                # En cas d’arrêt, garder un fichier orphelin plutôt que perdre une preuve publiée.
                atomic_bytes(path, raw)
        updated = {**current, "events": events}
        atomic_record(root / "record.json", updated)
        return updated
