"""Séparer les groupes de contenu de façon stable, sans inventer leur scène d'origine."""

import hashlib
from collections import defaultdict
from typing import Any

SPLITS = ("train", "valid", "test")


def assign_splits(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Répartir environ 80/10/10, en équilibrant difficulté et présence des classes."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 2**32 - 1")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["content_group"]].append(record)
    strata: dict[str, list[str]] = defaultdict(list)
    for group, members in groups.items():
        # Une paire peut traverser Easy/Medium/Hard : elle reste pourtant indivisible.
        difficulty = "+".join(sorted({r["difficulty"] for r in members}))
        categories = sorted({c for r in members for c in r["source_classes"]})
        strata[f"{difficulty}:{categories}"].append(group)
    result = {}
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
        )
        # Les petites strates vont au train. Ailleurs, on réserve au moins un groupe
        # à chaque évaluation ; les comptes exacts sont publiés dans le rapport.
        held_out = max(1, round(len(ordered) * 0.1)) if len(ordered) >= 3 else 0
        for index, group in enumerate(ordered):
            split = "valid" if index < held_out else "test" if index < 2 * held_out else "train"
            result.update({r["id"]: split for r in groups[group]})
    if set(result.values()) != set(SPLITS):
        raise ValueError("Not enough independent content groups for three nonempty splits")
    all_classes = {c for r in records for c in r["source_classes"]}
    for split in SPLITS:
        present = {c for r in records if result[r["id"]] == split for c in r["source_classes"]}
        if present != all_classes:
            raise ValueError(f"A source class is absent from {split}; revise the grouping policy")
    return result
