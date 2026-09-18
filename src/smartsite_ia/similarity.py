"""Repérer des images proches, sans les déclarer identiques ni retrouver leur scène."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

TRANSFORMS = (None, *Image.Transpose)
MAX_IMAGES = 5000
MAX_PAIRS = 10000


@dataclass(frozen=True)
class Fingerprint:
    sample_id: str
    pixel_sha256: str
    hashes: tuple[int, ...]
    thumbnails: tuple[bytes, ...]


@dataclass(frozen=True)
class SimilarPair:
    left: str
    right: str
    kind: str
    transform_right: str
    hamming_distance: int
    mean_absolute_error: float


def difference_hash(image: Image.Image) -> int:
    gray = image.convert("L")
    horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    vertical = np.asarray(gray.resize((8, 9), Image.Resampling.LANCZOS))
    bits = np.concatenate(
        (
            (horizontal[:, 1:] > horizontal[:, :-1]).ravel(),
            (vertical[1:, :] > vertical[:-1, :]).ravel(),
        )
    )
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def fingerprint(sample_id: str, image: Image.Image, pixel_sha256: str) -> Fingerprint:
    hashes, thumbnails = [], []
    for transform in TRANSFORMS:
        view = image if transform is None else image.transpose(transform)
        hashes.append(difference_hash(view))
        thumbnails.append(view.resize((32, 32), Image.Resampling.LANCZOS).convert("RGB").tobytes())
    return Fingerprint(sample_id, pixel_sha256, tuple(hashes), tuple(thumbnails))


def find_similar_pairs(
    fingerprints: list[Fingerprint], max_distance: int = 8, max_error: float = 20.0
) -> list[SimilarPair]:
    """Les seuils servent à trier les cas à revoir, pas à supprimer des photos."""
    if type(max_distance) is not int or not 0 <= max_distance <= 128:
        raise ValueError("Hash distance must be an integer between 0 and 128")
    if not np.isfinite(max_error) or not 0 <= max_error <= 255:
        raise ValueError("Pixel error threshold must be between 0 and 255")
    if len(fingerprints) > MAX_IMAGES:
        raise ValueError("Too many images for the bounded pairwise comparison")
    ordered = sorted(fingerprints, key=lambda f: f.sample_id)
    if len({f.sample_id for f in ordered}) != len(ordered):
        raise ValueError("Duplicate sample IDs in similarity search")
    result = []
    for i, left in enumerate(ordered):
        reference = np.frombuffer(left.thumbnails[0], dtype=np.uint8).astype(np.int16)
        for right in ordered[i + 1 :]:
            exact = left.pixel_sha256 == right.pixel_sha256
            matches = []
            for index, hashed in enumerate(right.hashes):
                distance = (left.hashes[0] ^ hashed).bit_count()
                if distance > max_distance and not exact:
                    continue
                target = np.frombuffer(right.thumbnails[index], dtype=np.uint8).astype(np.int16)
                error = float(np.mean(np.abs(reference - target)))
                if error <= max_error or exact:
                    matches.append((error, distance, index))
            if matches:
                error, distance, index = min(matches)
                transform = TRANSFORMS[index]
                result.append(
                    SimilarPair(
                        left.sample_id,
                        right.sample_id,
                        "exact_pixels" if exact else "candidate",
                        "IDENTITY" if transform is None else transform.name,
                        distance,
                        error,
                    )
                )
                if len(result) > MAX_PAIRS:
                    raise ValueError("Too many candidate pairs; narrow thresholds before retrying")
    return result


def candidate_groups(pairs: list[SimilarPair]) -> list[list[str]]:
    """Relier les candidats pour la revue ; ces groupes ne prouvent pas une origine commune."""
    neighbors: dict[str, set[str]] = {}
    for pair in pairs:
        neighbors.setdefault(pair.left, set()).add(pair.right)
        neighbors.setdefault(pair.right, set()).add(pair.left)
    remaining = set(neighbors)
    groups = []
    while remaining:
        pending = [min(remaining)]
        group = set()
        while pending:
            current = pending.pop()
            if current not in group:
                group.add(current)
                pending.extend(neighbors[current] - group)
        remaining -= group
        groups.append(sorted(group))
    return groups
