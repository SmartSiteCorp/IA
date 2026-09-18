"""Partager les contrôles et le regroupement utilisés pour préparer les corpus."""

import hashlib
import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from smartsite_ia.annotations import unique_object
from smartsite_ia.review import MAX_JSON_BYTES, read_local


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_document(path: Path) -> tuple[dict[str, Any], str]:
    """On garde l'empreinte du document exact qui a piloté la préparation."""
    raw = read_local(path.parent, path.name, MAX_JSON_BYTES)
    data = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        raise ValueError("Expected a versioned JSON object")
    if data["schema_version"] != 1:
        raise ValueError("Unsupported preparation schema")
    return data, digest(raw)


def require_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError("Expected a nonempty, bounded explanation")
    return value


@contextmanager
def staged_output(destination: Path, sources: list[Path]) -> Iterator[Path]:
    """On publie seulement un résultat complet, sans toucher aux entrées."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Preparation destination already exists")
    if any(destination.resolve().is_relative_to(root.resolve()) for root in sources):
        raise ValueError("Preparation output must be outside all source directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".smartsite-prepare-", dir=destination.parent) as work:
        stage = Path(work) / "corpus"
        stage.mkdir()
        yield stage
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Preparation destination appeared during preparation")
        stage.rename(destination)


def content_groups(ids: set[str], links: list[list[str]]) -> dict[str, str]:
    """Relier aussi les paires transitives ; ce ne sont pas des identifiants de scène."""
    parents = {sample_id: sample_id for sample_id in ids}

    def root(sample_id: str) -> str:
        while parents[sample_id] != sample_id:
            parents[sample_id] = parents[parents[sample_id]]
            sample_id = parents[sample_id]
        return sample_id

    for members in links:
        if (
            not isinstance(members, list)
            or len(members) < 2
            or any(not isinstance(member, str) or member not in ids for member in members)
            or len(set(members)) != len(members)
        ):
            raise ValueError("Invalid group members or unknown sample ID")
        for member in members[1:]:
            left, right = sorted((root(members[0]), root(member)))
            parents[right] = left
    return {sample_id: root(sample_id) for sample_id in sorted(ids)}
