"""Vérifier ce que le contrôle qualité affirme, et surtout ce qu'il n'affirme pas."""

import numpy as np
import pytest
from PIL import Image

from smartsite_ia import photo_quality
from smartsite_ia.photo_quality import DEFAULT_THRESHOLDS, assess_photo


def noisy_photo(width: int = 320, height: int = 320, seed: int = 0) -> Image.Image:
    """Une image texturée : bien exposée et riche en détail, comme un mur photographié."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(88, 168, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_textured_photo_is_fully_usable():
    report = assess_photo(noisy_photo())
    assert report["verdict"] == "usable"
    assert report["reasons"] == []
    assert report["lost_regions"] == []
    assert report["measures"]["readable_fraction"] == 1.0
    assert report["photo"] == {"width": 320, "height": 320}


def test_flat_sharp_surface_is_reported_not_rejected():
    """Non-régression : un aplat net (mur lisse) avait été déclaré inexploitable."""
    flat = Image.new("RGB", (320, 320), (200, 198, 196))
    report = assess_photo(flat)
    assert report["verdict"] == "partial"
    assert report["reasons"] == ["peu de détail mesuré : photo floue ou surface lisse, à regarder"]
    assert report["lost_regions"] == []


def test_half_dark_photo_locates_the_lost_zone():
    pixels = np.asarray(noisy_photo()).copy()
    pixels[:160] = 0
    report = assess_photo(Image.fromarray(pixels, mode="RGB"))
    assert report["verdict"] == "partial"
    assert report["reasons"] == ["zones perdues dans le noir ou le blanc"]
    assert {region["reason"] for region in report["lost_regions"]} == {"too_dark"}
    assert all(region["y"] + region["height"] <= 160 for region in report["lost_regions"])
    assert report["measures"]["readable_fraction"] == pytest.approx(0.5)


def test_burnt_photo_reports_bright_zones():
    pixels = np.asarray(noisy_photo()).copy()
    pixels[:, 160:] = 255
    report = assess_photo(Image.fromarray(pixels, mode="RGB"))
    assert {region["reason"] for region in report["lost_regions"]} == {"too_bright"}
    assert report["measures"]["bright_fraction"] == pytest.approx(0.5)


def test_mostly_unreadable_photo_is_refused():
    pixels = np.asarray(noisy_photo()).copy()
    pixels[:288] = 0
    report = assess_photo(Image.fromarray(pixels, mode="RGB"))
    assert report["verdict"] == "unusable"
    # Quatorze rangées de cases sur seize sont perdues ; la case à cheval reste lisible.
    assert report["measures"]["readable_fraction"] == pytest.approx(0.125)
    assert report["measures"]["dark_fraction"] == pytest.approx(0.9)


def test_large_photo_keeps_original_coordinates(monkeypatch):
    """La réduction sert à borner le calcul, pas à décaler les zones signalées."""
    monkeypatch.setattr(photo_quality, "EXPOSURE_PIXEL_BUDGET", 10_000)
    pixels = np.asarray(noisy_photo(451, 301)).copy()
    pixels[:150] = 0
    report = assess_photo(Image.fromarray(pixels, mode="RGB"))
    assert report["measures"]["counting_scale"] > 1
    assert report["lost_regions"]
    for region in report["lost_regions"]:
        assert region["x"] >= 0 and region["x"] + region["width"] <= 451
        assert region["y"] >= 0 and region["y"] + region["height"] <= 301


def test_small_photo_produces_no_empty_block():
    report = assess_photo(noisy_photo(9, 7))
    assert report["measures"]["readable_fraction"] == 1.0
    assert report["verdict"] in photo_quality.VERDICTS


def test_reported_thresholds_cannot_be_modified_through_the_result():
    first = assess_photo(noisy_photo())
    first["thresholds"]["dark_level"] = 200.0
    assert assess_photo(noisy_photo())["thresholds"] == dict(DEFAULT_THRESHOLDS)


def test_caller_thresholds_are_applied():
    flat = Image.new("RGB", (320, 320), (200, 198, 196))
    assert assess_photo(flat, thresholds={"low_detail_floor": 0.0})["verdict"] == "usable"


@pytest.mark.parametrize(
    "thresholds",
    [
        {"unknown": 1.0},
        {"dark_level": 250.0},
        {"bright_level": 10.0},
        {"dark_level": -1.0},
        {"block_lost_fraction": 0.0},
        {"block_lost_fraction": 1.5},
        {"partial_fraction": 0.2},
        {"unusable_fraction": -0.1},
        {"partial_fraction": 1.5},
        {"low_detail_floor": -1.0},
        {"dark_level": float("nan")},
        {"low_detail_floor": float("inf")},
        {"dark_level": "16"},
        {"low_detail_floor": None},
    ],
)
def test_inconsistent_thresholds_are_refused(thresholds):
    with pytest.raises(ValueError):
        assess_photo(noisy_photo(), thresholds=thresholds)


def test_empty_photo_is_refused():
    with pytest.raises(ValueError, match="non-empty"):
        assess_photo(Image.new("RGB", (0, 0)))


def test_tiny_photo_reports_no_detail_instead_of_failing():
    """Une vignette de deux pixels n'a pas de voisinage mesurable : pas d'erreur."""
    report = assess_photo(noisy_photo(2, 2))
    assert report["measures"]["sharpness"] == 0.0
    assert report["verdict"] == "partial"
