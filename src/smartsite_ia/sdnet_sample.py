"""Préparer un échantillon d'extraits SDNET2018 pour mesurer les alertes.

Cette source découpe 230 photos de béton en extraits de 256 pixels, étiquetés
fissuré ou sain par les auteurs. Elle sert ici à deux choses : des surfaces saines,
qui manquaient pour mesurer les fausses alertes, et un contrôle positif dans les
mêmes conditions. Le téléchargement n'est pas automatisable : l'éditeur refuse les
outils, la personne récupère l'archive et le module la vérifie par son empreinte.
"""

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from smartsite_ia.archive import checked_members, read_member
from smartsite_ia.curation import digest, staged_output
from smartsite_ia.importer import write_json

# `<surface>/<label>/<photo>-<extrait>[_<variante>].jpg`, par exemple `W/UW/7069-100.jpg`.
PATCH_NAME = re.compile(r"(?P<surface>[DPW])/(?P<label>[CU][DPW])/(?P<photo>\d+)-(\d+)(_\d+)?\.jpg")

SURFACES = {"D": "bridge_deck", "W": "wall", "P": "pavement"}
# Les auteurs n'étiquettent que la fissure : un extrait sain peut porter un autre défaut.
CLEAR_PREFIX, CRACKED_PREFIX = "U", "C"

MAX_ARCHIVE_MEMBERS = 70_000
MAX_ARCHIVE_EXPANDED_BYTES = 1024 * 1024 * 1024
MAX_SAMPLE_PATCHES = 5000


@dataclass(frozen=True)
class SdnetArchive:
    """L'archive interne publiée, épinglée par sa taille et son empreinte."""

    name: str = "SDNET2018.zip"
    size: int = 528284810
    sha256: str = "617afc074b53bfb11121b9bbd10512784e1b0b21529f542c61f2c422ebf77331"
    expected_patches: int = 56092
    origin_photos: int = 230


SDNET2018 = SdnetArchive()

PROVENANCE = {
    "dataset": "SDNET2018",
    "doi": "10.15142/T3TD19",
    "url": "https://digitalcommons.usu.edu/all_datasets/48/",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "authors": ["Marc Maguire", "Sattar Dorafshan", "Robert J. Thomas"],
    "license_evidence": "Publication page links CC BY 4.0; archive ReadMe names the authors",
    "camera": "Nikon COOLPIX L830",
    "site": "Utah State University campus, Logan, Utah, USA",
    "download": "manual; the publisher answers HTTP 403 to tool user agents",
}


def verify_sdnet_archive(path: Path, source: SdnetArchive = SDNET2018) -> str:
    """Refuser une archive absente, liée ou différente de celle qui a été inspectée."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size != source.size:
        raise ValueError("SDNET archive is missing, linked or differs from the inspected size")
    with path.open("rb") as stream:
        found = hashlib.file_digest(stream, "sha256").hexdigest()
    if found != source.sha256:
        raise ValueError("SDNET archive SHA-256 does not match the inspected file")
    return found


def index_patches(members: dict[str, Any]) -> dict[tuple[str, str, str], list[str]]:
    """Regrouper les extraits par surface, étiquette et photo d'origine."""
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for name in members:
        match = PATCH_NAME.fullmatch(name)
        if match is None:
            raise ValueError(f"Unexpected SDNET entry: {name}")
        key = (match["surface"], match["label"][0], match["photo"])
        grouped[key].append(name)
    return {key: sorted(names) for key, names in grouped.items()}


def choose_patches(
    grouped: dict[tuple[str, str, str], list[str]],
    surface: str,
    label: str,
    per_photo: int,
    seed: int,
) -> list[str]:
    """Tirer le même nombre d'extraits par photo, pour ne pas surreprésenter une scène.

    Le tirage suit l'ordre trié des photos et des noms : à graine égale, la sélection
    est reproductible, et elle ne dépend pas de l'ordre de lecture de l'archive.
    """
    if not 1 <= per_photo <= 100:
        raise ValueError("Patches per origin photo must sit between 1 and 100")
    chosen: list[str] = []
    for key in sorted(key for key in grouped if key[0] == surface and key[1] == label):
        names = grouped[key]
        generator = random.Random(f"{seed}:{'/'.join(key)}")
        chosen.extend(sorted(generator.sample(names, min(per_photo, len(names)))))
    if not 1 <= len(chosen) <= MAX_SAMPLE_PATCHES:
        raise ValueError("Empty or oversized SDNET selection")
    return chosen


def prepare_sample(
    archive_path: Path,
    destination: Path,
    surface: str = "W",
    per_photo: int = 10,
    seed: int = 42,
    source: SdnetArchive = SDNET2018,
) -> dict[str, Any]:
    """Écrire les extraits choisis, leurs empreintes et leurs groupes de scène."""
    if surface not in SURFACES:
        raise ValueError("Unknown SDNET surface")
    archive_hash = verify_sdnet_archive(archive_path, source)
    with ZipFile(archive_path) as archive:
        members = checked_members(
            archive,
            max_members=MAX_ARCHIVE_MEMBERS,
            max_expanded_bytes=MAX_ARCHIVE_EXPANDED_BYTES,
        )
        if len(members) != source.expected_patches:
            raise ValueError(f"Expected {source.expected_patches} patches, found {len(members)}")
        grouped = index_patches(members)
        photos = {key[2] for key in grouped}
        if len(photos) != source.origin_photos:
            raise ValueError(f"Expected {source.origin_photos} origin photos, found {len(photos)}")

        with staged_output(destination, [archive_path]) as stage:
            records: list[dict[str, Any]] = []
            seen: dict[str, str] = {}
            duplicates = 0
            for label, state in ((CLEAR_PREFIX, "clear"), (CRACKED_PREFIX, "cracked")):
                folder = stage / state
                folder.mkdir(parents=True)
                for name in choose_patches(grouped, surface, label, per_photo, seed):
                    match = PATCH_NAME.fullmatch(name)
                    assert match is not None
                    raw = read_member(archive, members[name])
                    fingerprint = digest(raw)
                    # Onze extraits sont publiés deux fois sous des noms différents.
                    if fingerprint in seen:
                        duplicates += 1
                        continue
                    seen[fingerprint] = name
                    patch_id = Path(name).stem
                    (folder / f"{patch_id}.jpg").write_bytes(raw)
                    records.append(
                        {
                            "id": patch_id,
                            "path": f"{state}/{patch_id}.jpg",
                            "source_member": name,
                            "sha256": fingerprint,
                            "author_label": match["label"],
                            "author_state": state,
                            "surface": SURFACES[surface],
                            "scene_group": f"{match['surface']}-{match['photo']}",
                        }
                    )
            report = {
                "schema_version": 1,
                "dataset": "sdnet2018",
                "provenance": PROVENANCE,
                "archive_sha256": archive_hash,
                "selection": {
                    "surface": SURFACES[surface],
                    "patches_per_origin_photo": per_photo,
                    "seed": seed,
                    "deduplicated_patches": duplicates,
                },
                "limits": [
                    "Author labels cover cracks only; a clear patch may show scaling or a hole",
                    "256 px patches; the training corpus used larger photos",
                    "Patches of one origin photo stay in the same scene group",
                    "Walls come from a single building on one campus, with one camera",
                ],
                "approved_for_training": False,
                "counts": {
                    state: sum(r["author_state"] == state for r in records)
                    for state in ("clear", "cracked")
                },
                "scene_groups": len({r["scene_group"] for r in records}),
                "records": records,
            }
            write_json(stage / "report.json", report)
    return report


def measure_alerts(report: dict[str, Any], alerts: dict[str, int]) -> dict[str, Any]:
    """Comparer la fréquence des alertes sur le sain et sur le fissuré.

    Sans le contrôle positif, un taux nul sur les surfaces saines resterait
    ininterprétable : il pourrait seulement dire que le modèle ne voit rien ici.
    """
    result: dict[str, Any] = {}
    for state in ("clear", "cracked"):
        rows = [r for r in report["records"] if r["author_state"] == state]
        if not rows:
            raise ValueError(f"No patch to measure for state: {state}")
        missing = [r["id"] for r in rows if r["id"] not in alerts]
        if missing:
            raise ValueError(f"Missing alert counts for {len(missing)} patches")
        counts = [alerts[r["id"]] for r in rows]
        alerted = sum(1 for count in counts if count > 0)
        result[state] = {
            "patches": len(rows),
            "scene_groups": len({r["scene_group"] for r in rows}),
            "patches_with_alert": alerted,
            "alert_rate": alerted / len(rows),
            "proposals_per_patch": sum(counts) / len(rows),
        }
    return result
