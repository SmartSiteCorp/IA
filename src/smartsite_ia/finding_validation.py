"""Valider les petites valeurs du contrat de constat, sans modèle ni accès disque."""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any


class ConflictError(ValueError):
    """La demande contredit une version ou une opération déjà enregistrée."""


def text(value: object, label: str, limit: int = 160) -> str:
    """Refuser les textes vides, trop longs ou contenant des caractères de contrôle."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > limit
        or any(ord(c) < 32 and c not in "\n\t" for c in value)
    ):
        raise ValueError(f"Invalid {label}")
    return value


def optional_text(value: object, label: str, limit: int = 160) -> str | None:
    return None if value is None else text(value, label, limit)


def sha256(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid SHA-256 fingerprint")
    return value


def identifier(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{32}", value) is None:
        raise ValueError("Invalid finding identifier")
    return value


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite number, without boolean or string conversion")
    return float(value)


def size(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(n) is not int or n < 1 for n in value)
        or value[0] * value[1] > 16_000_000
    ):
        raise ValueError("Invalid photo dimensions")
    return value[0], value[1]


def box(value: object, dimensions: tuple[int, int]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Detection box is malformed")
    coords = [number(n) for n in value]
    if coords[0] >= coords[2] or coords[1] >= coords[3]:
        raise ValueError("Detection box is malformed or empty")
    if min(coords) < 0 or coords[2] > dimensions[0] or coords[3] > dimensions[1]:
        raise ValueError("Detection box falls outside the photo")
    return coords


def timestamp(value: object) -> str:
    """Conserver un instant explicite en UTC ; une date sans fuseau est ambiguë."""
    raw = text(value, "timestamp", 40)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp needs a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def now() -> str:
    return timestamp(datetime.now(UTC).isoformat())


def canonical(value: Any) -> bytes:
    """L'ordre des clés ne change pas une identité ; NaN n'est jamais du JSON valide."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()[:32]


def fields(
    value: object, required: Sequence[str], optional: Sequence[str] = ()
) -> Mapping[str, Any]:
    """Les fautes de champ doivent provoquer une erreur, pas disparaître silencieusement."""
    if (
        not isinstance(value, Mapping)
        or set(required) - value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise ValueError("Missing or unknown contract fields")
    return value
