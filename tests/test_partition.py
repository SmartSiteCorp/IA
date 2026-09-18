import pytest

from smartsite_ia.curation import content_groups
from smartsite_ia.partition import assign_splits


def records(count=30):
    return [
        {"id": str(i), "content_group": str(i), "difficulty": "Easy", "source_classes": [0, 1]}
        for i in range(count)
    ]


def test_groups_are_transitive_stable_and_include_singletons():
    ids = {"a", "b", "c", "d"}
    links = [["c", "b"], ["a", "b"]]
    expected = {"a": "a", "b": "a", "c": "a", "d": "d"}
    assert content_groups(ids, links) == expected
    assert content_groups(ids, list(reversed(links))) == expected


@pytest.mark.parametrize("links", [[["a"]], [["a", "z"]], [["a", "a"]], [None], [[1, "a"]]])
def test_invalid_links_are_rejected(links):
    with pytest.raises(ValueError):
        content_groups({"a", "b"}, links)


def test_split_preserves_groups_even_across_difficulties_and_input_order():
    data = records()
    data[1]["content_group"] = "0"
    data[1]["difficulty"] = "Hard"
    result = assign_splits(data, 42)
    assert result["0"] == result["1"]
    assert set(result.values()) == {"train", "valid", "test"}
    assert result == assign_splits(list(reversed(data)), 42)
    assert result != assign_splits(data, 43)


@pytest.mark.parametrize("seed", [True, -1, 2**32, "42"])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="Seed"):
        assign_splits(records(), seed)


def test_too_few_groups_or_missing_class_does_not_publish_a_fake_test():
    with pytest.raises(ValueError, match="Not enough"):
        assign_splits(records(2), 1)
    data = records()
    data[-1]["source_classes"] = [0, 1, 2]
    with pytest.raises(ValueError, match="class is absent"):
        assign_splits(data, 1)
