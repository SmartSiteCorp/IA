import hashlib
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from smartsite_ia import similarity
from smartsite_ia.similarity import (
    SimilarPair,
    candidate_groups,
    difference_hash,
    find_similar_pairs,
    fingerprint,
)


def patterned_image(seed=7):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).resize((128, 128))


def describe(name, image):
    return fingerprint(name, image, hashlib.sha256(image.tobytes()).hexdigest())


def test_exact_pixels_and_stable_order():
    image = patterned_image()
    a, b = describe("a", image), describe("b", image)
    result = find_similar_pairs([b, a])
    assert len(result) == 1
    assert result[0] == SimilarPair("a", "b", "exact_pixels", "IDENTITY", 0, 0.0)
    assert result == find_similar_pairs([a, b])


@pytest.mark.parametrize("transform", list(Image.Transpose))
def test_rotated_or_flipped_image_remains_a_candidate(transform):
    image = patterned_image()
    transformed = image.transpose(transform)
    pairs = find_similar_pairs([describe("a", image), describe("b", transformed)])
    assert len(pairs) == 1 and pairs[0].kind == "candidate"
    assert pairs[0].hamming_distance == 0
    assert pairs[0].mean_absolute_error == 0


def test_minor_brightness_change_found_but_unrelated_image_not_found():
    image = patterned_image()
    adjusted = ImageEnhance.Brightness(image).enhance(1.02)
    pairs = find_similar_pairs(
        [describe("a", image), describe("b", adjusted), describe("c", patterned_image(123))]
    )
    assert [(p.left, p.right) for p in pairs] == [("a", "b")]


def test_uniform_image_hash_collision_requires_color_agreement():
    black, white = Image.new("RGB", (128, 128)), Image.new("RGB", (128, 128), "white")
    assert difference_hash(black) == difference_hash(white)
    assert not find_similar_pairs([describe("a", black), describe("b", white)])


@pytest.mark.parametrize(
    "distance,error",
    [(-1, 20), (129, 20), (True, 20), (8, -1), (8, 256), (8, float("nan")), (8, float("inf"))],
)
def test_invalid_thresholds_rejected(distance, error):
    with pytest.raises(ValueError):
        find_similar_pairs([], distance, error)


def test_duplicate_ids_and_resource_limits(monkeypatch):
    f = describe("a", patterned_image())
    with pytest.raises(ValueError, match="Duplicate"):
        find_similar_pairs([f, f])
    monkeypatch.setattr(similarity, "MAX_IMAGES", 1)
    with pytest.raises(ValueError, match="Too many images"):
        find_similar_pairs([f, replace(f, sample_id="b")])
    monkeypatch.setattr(similarity, "MAX_IMAGES", 5)
    monkeypatch.setattr(similarity, "MAX_PAIRS", 0)
    with pytest.raises(ValueError, match="Too many candidate"):
        find_similar_pairs([f, replace(f, sample_id="b")])


def test_candidate_groups_are_transitive_and_do_not_include_singletons():
    pairs = [
        SimilarPair(a, b, "candidate", "IDENTITY", 1, 1.0)
        for a, b in [("c", "a"), ("b", "c"), ("x", "y")]
    ]
    assert candidate_groups(pairs) == [["a", "b", "c"], ["x", "y"]]
    assert candidate_groups([]) == []


def bulk_hash(bits):
    """Construire une empreinte de 128 bits à partir d'une liste de positions à un."""
    value = 0
    for position in bits:
        value |= 1 << position
    return value


def test_bulk_search_finds_the_same_close_pairs_as_the_detailed_one():
    hashes = {"a": bulk_hash([0, 5]), "b": bulk_hash([0, 5, 9]), "c": bulk_hash(range(64))}
    pairs = similarity.close_pairs(hashes, max_distance=2)
    assert pairs == [("a", "b", 1)]


def test_bulk_search_counts_the_exact_bit_distance():
    hashes = {"a": 0, "b": bulk_hash([1, 2, 3])}
    assert similarity.close_pairs(hashes, max_distance=8) == [("a", "b", 3)]
    assert similarity.close_pairs(hashes, max_distance=2) == []


def test_identical_images_have_no_distance():
    value = bulk_hash([3, 40, 127])
    assert similarity.close_pairs({"a": value, "b": value}, max_distance=0) == [("a", "b", 0)]


def test_each_pair_is_reported_once_and_ordered():
    value = bulk_hash([7])
    pairs = similarity.close_pairs({"c": value, "a": value, "b": value}, max_distance=0)
    assert pairs == [("a", "b", 0), ("a", "c", 0), ("b", "c", 0)]


def test_bulk_search_crosses_its_internal_blocks(monkeypatch):
    """Les blocs sont un détail de calcul : ils ne doivent pas couper une paire."""
    monkeypatch.setattr(similarity, "BULK_BLOCK", 2)
    value = bulk_hash([11])
    hashes = {name: value for name in ("a", "b", "c", "d", "e")}
    assert len(similarity.close_pairs(hashes, max_distance=0)) == 10


def test_empty_bulk_search_returns_nothing():
    assert similarity.close_pairs({}, max_distance=4) == []


@pytest.mark.parametrize("distance", [-1, 129, 1.5, "4", None])
def test_invalid_bulk_distance_is_refused(distance):
    with pytest.raises(ValueError, match="Hash distance"):
        similarity.close_pairs({"a": 1}, distance)


def test_oversized_bulk_search_is_refused(monkeypatch):
    monkeypatch.setattr(similarity, "MAX_BULK_IMAGES", 2)
    with pytest.raises(ValueError, match="Too many images"):
        similarity.close_pairs({"a": 1, "b": 2, "c": 3})


def test_groups_link_photos_through_a_common_neighbour():
    pairs = [("a", "b", 1), ("b", "c", 2), ("x", "y", 0)]
    assert similarity.groups_from_pairs(pairs) == [["a", "b", "c"], ["x", "y"]]


def test_groups_without_pairs_are_empty():
    assert similarity.groups_from_pairs([]) == []


def test_bulk_hashes_match_the_real_image_hash():
    """L'empreinte groupée doit être celle du module, pas une variante."""
    first = Image.new("RGB", (64, 64), "black")
    second = first.copy()
    second.paste((255, 255, 255), (0, 0, 32, 64))
    hashes = {
        "uniform": similarity.difference_hash(first),
        "split": similarity.difference_hash(second),
    }
    assert similarity.close_pairs(hashes, max_distance=0) == []
    assert similarity.close_pairs(hashes, max_distance=128)[0][2] > 0
