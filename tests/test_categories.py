"""Les IDs peuvent changer, le sens du défaut et ses contours doivent rester identiques."""

import copy

import pytest

from smartsite_ia.categories import CLASS_NAMES, get_class_names, project_categories


@pytest.mark.parametrize(
    "names", [[], None, "crack", [0], ["mold"], ["crack", "crack"], ["surface_loss", "crack"]]
)
def test_invalid_class_selection_is_refused(names):
    with pytest.raises(ValueError, match="class names"):
        get_class_names({"class_names": names})


@pytest.mark.parametrize("names", [("crack",), ("surface_loss",), CLASS_NAMES])
def test_projection_keeps_images_geometry_provenance_and_original(names):
    source = {
        "info": {"description": "Synthetic software test"},
        "licenses": [{"id": 1}],
        "categories": [
            {"id": i, "name": name, "source_category_id": i - 1}
            for i, name in enumerate(CLASS_NAMES, 1)
        ],
        "images": [{"id": 1}, {"id": 2}, {"id": 3}],
        "annotations": [
            {
                "id": i,
                "category_id": i,
                "source_category_id": i - 1,
                "image_id": i,
                "segmentation": [[1, 1, 4, 1, 4, 4]],
                "bbox": [1, 1, 3, 3],
                "area": 4.5,
            }
            for i in (1, 2)
        ],
    }
    original = copy.deepcopy(source)
    result = project_categories(source, names)
    assert source == original
    assert result["images"] == original["images"]
    assert result["info"] == original["info"] and result["licenses"] == original["licenses"]
    assert [(c["id"], c["name"]) for c in result["categories"]] == list(enumerate(names, 1))
    for annotation, name in zip(result["annotations"], names, strict=True):
        source_id = CLASS_NAMES.index(name)
        expected = original["annotations"][source_id]
        assert annotation == {**expected, "category_id": names.index(name) + 1}
        assert annotation["source_category_id"] == source_id


@pytest.mark.parametrize("category", [0, 3, True, "1"])
def test_projection_does_not_hide_unknown_categories(category):
    source = {
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, 1)],
        "annotations": [{"category_id": category}],
        "images": [],
    }
    with pytest.raises(ValueError, match="annotation category"):
        project_categories(source, ("crack",))


def test_legacy_configuration_still_selects_both_categories():
    assert get_class_names({}) == CLASS_NAMES
