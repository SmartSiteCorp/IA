"""Vérifier le découpage, le recollage et la fusion, sans charger le moteur."""

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from smartsite_ia import windowed_inference as wi


def box_rle(height, width, top, left, box_height, box_width, as_text=False):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top : top + box_height, left : left + box_width] = 1
    encoded = coco_mask.encode(np.asfortranarray(mask))
    if as_text:
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def test_grid_covers_the_whole_photo():
    bounds = wi.window_bounds(1000, 700, window=400, overlap=100)
    assert all(right - left == 400 and bottom - top == 400 for left, top, right, bottom in bounds)
    assert max(right for _, _, right, _ in bounds) == 1000
    assert max(bottom for *_, bottom in bounds) == 700
    assert min(left for left, *_ in bounds) == 0


def test_last_window_sits_on_the_edge_instead_of_being_cropped():
    """Une fenêtre rognée n'aurait plus la taille attendue par le moteur."""
    assert wi.offsets(1000, 400, 300) == [0, 300, 600]
    assert wi.offsets(1001, 400, 300) == [0, 300, 600, 601]


def test_photo_smaller_than_a_window_gives_one_window():
    assert wi.window_bounds(100, 80, window=400, overlap=100) == [(0, 0, 100, 80)]


@pytest.mark.parametrize(("window", "overlap"), [(16, 0), (400, 400), (400, 500), (400, -1)])
def test_impossible_window_settings_are_refused(window, overlap):
    with pytest.raises(ValueError, match="Window must be"):
        wi.window_bounds(1000, 1000, window, overlap)


def test_oversized_grid_is_refused():
    with pytest.raises(ValueError, match="Empty or oversized window grid"):
        wi.window_bounds(20000, 20000, window=64, overlap=0)


def test_windows_are_cut_without_touching_the_photo():
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8), mode="RGB")
    before = np.asarray(photo).copy()
    bounds = wi.window_bounds(300, 200, window=100, overlap=0)
    views = list(wi.windows(photo, bounds))
    assert len(views) == len(bounds)
    assert all(view.size == (100, 100) for view in views)
    assert np.array_equal(np.asarray(photo), before)


def test_window_mask_lands_at_the_right_place():
    zone = np.zeros((200, 300), dtype=bool)
    # Un carré de 10 px au coin d'une fenêtre qui commence à (100, 50).
    wi.place_mask(box_rle(100, 100, 0, 0, 10, 10), (100, 50, 200, 150), zone)
    assert zone[50:60, 100:110].all()
    assert zone.sum() == 100


def test_masks_encoded_as_text_are_accepted():
    zone = np.zeros((100, 100), dtype=bool)
    wi.place_mask(box_rle(50, 50, 0, 0, 5, 5, as_text=True), (0, 0, 50, 50), zone)
    assert zone.sum() == 25


def test_mask_of_another_size_than_its_window_is_refused():
    zone = np.zeros((200, 300), dtype=bool)
    with pytest.raises(ValueError, match="does not match its window"):
        wi.place_mask(box_rle(50, 50, 0, 0, 5, 5), (0, 0, 100, 100), zone)


def test_box_returns_to_the_original_coordinates():
    assert wi.shift_box([10.0, 20.0, 30.0, 40.0], (100, 200, 740, 840)) == [
        110.0,
        220.0,
        130.0,
        240.0,
    ]


def test_same_defect_seen_by_two_windows_is_merged_once():
    """Le cas que le recouvrement des fenêtres doit produire : un défaut vu deux fois."""
    whole = {"class_name": "crack", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 100.0, 20.0]}
    piece = {"class_name": "crack", "score": 0.5, "bbox_xyxy": [80.0, 0.0, 120.0, 20.0]}
    merged = wi.merge_proposals([whole, piece])
    assert len(merged) == 1
    assert merged[0]["score"] == 0.9
    assert merged[0]["merged_from"] == 2
    # La boîte gardée couvre le défaut complet, pas seulement la meilleure vue.
    assert merged[0]["bbox_xyxy"] == [0.0, 0.0, 120.0, 20.0]


def test_distant_defects_stay_separate():
    first = {"class_name": "crack", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}
    second = {"class_name": "crack", "score": 0.8, "bbox_xyxy": [500.0, 500.0, 510.0, 510.0]}
    assert len(wi.merge_proposals([first, second])) == 2


def test_two_classes_at_the_same_place_are_not_merged():
    first = {"class_name": "crack", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}
    second = {"class_name": "surface_loss", "score": 0.8, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}
    assert len(wi.merge_proposals([first, second])) == 2


def test_merge_keeps_the_best_score_whatever_the_input_order():
    low = {"class_name": "crack", "score": 0.4, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}
    high = {"class_name": "crack", "score": 0.95, "bbox_xyxy": [1.0, 1.0, 11.0, 11.0]}
    for order in ([low, high], [high, low]):
        merged = wi.merge_proposals(order)
        assert len(merged) == 1 and merged[0]["score"] == 0.95


def test_empty_proposals_merge_to_nothing():
    assert wi.merge_proposals([]) == []


@pytest.mark.parametrize("overlap", [0, -0.5, 1.5])
def test_impossible_merge_overlap_is_refused(overlap):
    with pytest.raises(ValueError, match="Merge overlap"):
        wi.merge_proposals([], overlap)


def test_flat_box_never_divides_by_zero():
    flat = {"class_name": "crack", "score": 0.9, "bbox_xyxy": [5.0, 5.0, 5.0, 20.0]}
    other = {"class_name": "crack", "score": 0.8, "bbox_xyxy": [0.0, 0.0, 10.0, 30.0]}
    assert len(wi.merge_proposals([flat, other])) == 2


def test_accumulated_zone_is_encoded_like_other_masks():
    photo = Image.new("RGB", (60, 40))
    zone = wi.empty_zone(photo)
    assert zone.shape == (40, 60)
    zone[10:20, 5:15] = True
    assert int(coco_mask.area(wi.encode_zone(zone))) == 100
