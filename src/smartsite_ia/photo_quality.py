"""Dire ce qu'une photo permet vraiment d'analyser, avant de parler de défauts.

Une photo trop sombre, brûlée ou floue ne prouve pas qu'un mur est sain. Ce module
mesure ce qui est observable et rend un verdict explicite, avec ses raisons et les
zones mises de côté. Il ne charge aucun modèle et ne décide d'aucun défaut.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

# Grille fixe : le nombre de zones ne dépend pas de la taille de la photo, donc le
# coût et la taille du résultat restent bornés, même sur une image de 16 mégapixels.
GRID = 16

# Fenêtres prélevées en pixels d'origine pour juger la netteté. On ne redimensionne
# pas la photo entière : réduire une photo floue la ferait paraître nette.
SHARPNESS_WINDOWS = 4
SHARPNESS_WINDOW_SIZE = 256

# Au-delà, on moyenne la photo par blocs entiers avant de compter les pixels.
# Une zone sombre reste sombre ; seuls les pixels isolés disparaissent, ce qui est
# voulu : on cherche des zones inexploitables, pas des points.
EXPOSURE_PIXEL_BUDGET = 4_000_000

# Seuils de départ, volontairement prudents. Ils ne sont pas qualifiés sur des photos
# de chantier : il faudra les régler sur de vraies captures, téléphone et drone séparés.
DEFAULT_THRESHOLDS: Mapping[str, float] = {
    "dark_level": 16.0,
    "bright_level": 245.0,
    "block_lost_fraction": 0.80,
    "partial_fraction": 0.95,
    "unusable_fraction": 0.40,
    "low_detail_floor": 12.0,
}

VERDICTS = ("usable", "partial", "unusable")


def assess_photo(
    image: Image.Image,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Mesurer une photo déjà décodée et remise à l'endroit.

    Le décodage, l'orientation et les limites de taille restent à l'appelant
    (`prediction.decode_photo`), pour garder une seule définition de ces règles.
    """
    limits = check_thresholds(thresholds)
    if image.width < 1 or image.height < 1:
        raise ValueError("Photo quality expects a non-empty image")
    gray = image.convert("L")
    lost, measures = read_exposure(gray, limits)
    measures["sharpness"] = read_sharpness(gray)
    reasons = list_reasons(measures, limits)
    return {
        "schema_version": 1,
        "photo": {"width": image.width, "height": image.height},
        "verdict": choose_verdict(measures, limits),
        "reasons": reasons,
        "measures": measures,
        "lost_regions": lost,
        "thresholds": dict(limits),
        "limits": (
            "Mesures d'image seulement. Une zone exploitable n'est pas une zone conforme, "
            "et ces seuils ne sont pas encore réglés sur des photos de chantier."
        ),
    }


def check_thresholds(thresholds: Mapping[str, float] | None) -> Mapping[str, float]:
    """Refuser tout de suite un réglage incohérent plutôt que de rendre un verdict faux."""
    limits = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    if set(limits) != set(DEFAULT_THRESHOLDS):
        raise ValueError("Unknown photo quality threshold")
    if any(type(value) not in (int, float) or not np.isfinite(value) for value in limits.values()):
        raise ValueError("Photo quality thresholds must be finite numbers")
    if not 0 <= limits["dark_level"] < limits["bright_level"] <= 255:
        raise ValueError("Dark level must stay below bright level inside 0-255")
    if not 0 < limits["block_lost_fraction"] <= 1:
        raise ValueError("Block loss fraction must sit in ]0, 1]")
    if not 0 <= limits["unusable_fraction"] < limits["partial_fraction"] <= 1:
        raise ValueError("Unusable fraction must stay below partial fraction inside 0-1")
    if limits["low_detail_floor"] < 0:
        raise ValueError("Low detail floor cannot be negative")
    return limits


def read_exposure(
    gray: Image.Image,
    limits: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Repérer les zones perdues dans le noir ou le blanc, et les rendre en rectangles."""
    reduced, factor = shrink_for_counting(gray)
    pixels = np.asarray(reduced, dtype=np.uint8)
    dark = pixels <= limits["dark_level"]
    bright = pixels >= limits["bright_level"]
    rows = block_bounds(pixels.shape[0])
    columns = block_bounds(pixels.shape[1])
    lost: list[dict[str, Any]] = []
    lost_area = 0
    for top, bottom in rows:
        for left, right in columns:
            area = (bottom - top) * (right - left)
            dark_share = float(dark[top:bottom, left:right].sum()) / area
            bright_share = float(bright[top:bottom, left:right].sum()) / area
            if dark_share + bright_share < limits["block_lost_fraction"]:
                continue
            lost_area += area
            # Les coordonnées repartent dans les pixels de la photo d'origine. La
            # réduction arrondit vers le haut, donc on reste borné par ses bords.
            x, y = left * factor, top * factor
            lost.append(
                {
                    "x": x,
                    "y": y,
                    "width": min(right * factor, gray.width) - x,
                    "height": min(bottom * factor, gray.height) - y,
                    "reason": "too_dark" if dark_share >= bright_share else "too_bright",
                }
            )
    total = float(pixels.size)
    return lost, {
        "mean_luminance": float(pixels.mean()),
        "dark_fraction": float(dark.sum()) / total,
        "bright_fraction": float(bright.sum()) / total,
        "readable_fraction": 1.0 - lost_area / total,
        "counting_scale": float(factor),
    }


def shrink_for_counting(gray: Image.Image) -> tuple[Image.Image, int]:
    """Moyenner par blocs entiers quand la photo est grande, pour borner le calcul."""
    pixels = gray.width * gray.height
    if pixels <= EXPOSURE_PIXEL_BUDGET:
        return gray, 1
    factor = int(np.ceil(np.sqrt(pixels / EXPOSURE_PIXEL_BUDGET)))
    # `reduce` garde des blocs entiers ; les bords incomplets sont moyennés de même.
    return gray.reduce(factor), factor


def block_bounds(size: int) -> list[tuple[int, int]]:
    """Découper un côté en cases de la grille, sans case vide sur les petites photos."""
    edges = [round(size * index / GRID) for index in range(GRID + 1)]
    return [(edges[i], edges[i + 1]) for i in range(GRID) if edges[i + 1] > edges[i]]


def read_sharpness(gray: Image.Image) -> float:
    """Mesurer le détail présent dans quelques fenêtres prises en pixels d'origine.

    On garde la valeur la plus forte : sur une photo nette, au moins une fenêtre
    contient du détail. Attention, un mur lisse bien net donne aussi peu de détail :
    vérifié sur le corpus, une photo de moisissures sur plâtre blanc tombe à 5,9 et
    un aplat de peinture à 2,2. Cette mesure ne prouve donc jamais un flou à elle
    seule, et elle ne sert pas à écarter une photo.
    """
    best = 0.0
    for top, left, height, width in sample_windows(gray.width, gray.height):
        window = np.asarray(gray.crop((left, top, left + width, top + height)), dtype=np.float32)
        if window.shape[0] < 3 or window.shape[1] < 3:
            continue
        middle = window[1:-1, 1:-1]
        # Laplacien à quatre voisins : il réagit aux contours, pas à la luminosité.
        edges = (
            4.0 * middle
            - window[:-2, 1:-1]
            - window[2:, 1:-1]
            - window[1:-1, :-2]
            - window[1:-1, 2:]
        )
        best = max(best, float(edges.var()))
    return best


def sample_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Répartir les fenêtres de mesure sur la photo, sans sortir de ses bords."""
    side = min(SHARPNESS_WINDOW_SIZE, width, height)
    positions: list[tuple[int, int, int, int]] = []
    for row in range(SHARPNESS_WINDOWS):
        for column in range(SHARPNESS_WINDOWS):
            top = round((height - side) * row / max(SHARPNESS_WINDOWS - 1, 1))
            left = round((width - side) * column / max(SHARPNESS_WINDOWS - 1, 1))
            if (top, left, side, side) not in positions:
                positions.append((top, left, side, side))
    return positions


def list_reasons(measures: Mapping[str, float], limits: Mapping[str, float]) -> list[str]:
    """Expliquer le verdict en clair : une alerte sans raison n'est pas exploitable."""
    reasons: list[str] = []
    if measures["readable_fraction"] < limits["partial_fraction"]:
        reasons.append("zones perdues dans le noir ou le blanc")
    if measures["sharpness"] < limits["low_detail_floor"]:
        reasons.append("peu de détail mesuré : photo floue ou surface lisse, à regarder")
    return reasons


def choose_verdict(measures: Mapping[str, float], limits: Mapping[str, float]) -> str:
    """Trois états seulement, pour ne pas confondre « rien vu » et « rien à voir ».

    Seules les zones illisibles écartent une photo. Le manque de détail descend au
    plus à « partiel » : on ne sait pas distinguer un flou d'une surface lisse, donc
    on le signale à la personne qui vérifie au lieu de jeter la photo.
    """
    if measures["readable_fraction"] < limits["unusable_fraction"]:
        return "unusable"
    if (
        measures["readable_fraction"] < limits["partial_fraction"]
        or measures["sharpness"] < limits["low_detail_floor"]
    ):
        return "partial"
    return "usable"
