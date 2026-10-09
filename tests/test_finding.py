"""Vérifier ce qu'un constat affirme, et surtout ce qu'il refuse d'affirmer."""

import pytest

from smartsite_ia import finding
from smartsite_ia.finding import analyse_outcome, build_finding, finding_identifier

PHOTO = "a" * 64
CONTEXT = {"site_id": "chantier-12", "zone_id": "salle-de-bain", "task_id": "T-45"}
MODEL = {"run": "cubit_boxes_v3", "checkpoint_sha256": "b" * 64, "score_threshold": 0.3}


def detection(**changes):
    return {"class_name": "crack", "score": 0.82, "bbox_xyxy": [10.0, 20.0, 60.0, 90.0], **changes}


def make(**changes):
    return build_finding(
        detection(**changes.pop("detection", {})),
        photo_sha256=changes.pop("photo_sha256", PHOTO),
        photo_size=changes.pop("photo_size", (200, 100)),
        context=changes.pop("context", CONTEXT),
        model=changes.pop("model", MODEL),
    )


def test_finding_carries_the_defect_and_its_place():
    result = make()
    assert result["defect"]["class"] == "crack"
    assert result["defect"]["family"] == "crack"
    assert result["defect"]["label"] == "Fissures"
    assert result["location"]["bbox_xyxy"] == [10.0, 20.0, 60.0, 90.0]
    assert result["location"]["bbox_normalized"] == [0.05, 0.2, 0.3, 0.9]
    assert result["context"] == CONTEXT
    assert result["state"] == "proposed"


def test_score_is_never_presented_as_a_severity():
    """Un score dit la confiance du modele, pas le danger : les deux restent separes."""
    result = make()
    assert result["confidence"]["model_score"] == 0.82
    assert result["severity"] is None


def test_unavailable_facts_stay_empty_instead_of_being_invented():
    result = make()
    assert result["location"]["point_3d"] is None
    assert result["measurement"] is None
    assert result["defect"]["material"] is None
    assert result["source"]["qualified_for_smartsite"] is False


def test_the_model_that_produced_the_finding_is_traced():
    result = make()
    assert result["source"]["run"] == "cubit_boxes_v3"
    assert result["source"]["checkpoint_sha256"] == "b" * 64
    assert result["source"]["score_threshold"] == 0.3
    assert result["evidence"]["photo_sha256"] == PHOTO


def test_the_same_detection_always_gives_the_same_identifier():
    """Relancer une analyse ne doit pas creer un doublon dans le suivi."""
    assert make()["id"] == make()["id"]
    assert len(make()["id"]) == 32


def test_a_tiny_coordinate_drift_keeps_the_same_identifier():
    first = finding_identifier(PHOTO, "crack", [10.0, 20.0, 60.0, 90.0])
    second = finding_identifier(PHOTO, "crack", [10.0004, 19.9996, 60.0, 90.0])
    assert first == second


def test_another_photo_class_or_place_gives_another_identifier():
    base = finding_identifier(PHOTO, "crack", [10.0, 20.0, 60.0, 90.0])
    assert base != finding_identifier("c" * 64, "crack", [10.0, 20.0, 60.0, 90.0])
    assert base != finding_identifier(PHOTO, "surface_loss", [10.0, 20.0, 60.0, 90.0])
    assert base != finding_identifier(PHOTO, "crack", [11.0, 20.0, 60.0, 90.0])


def test_unknown_defect_class_is_refused():
    with pytest.raises(ValueError, match="Unknown defect class"):
        make(detection={"class_name": "inondation"})


@pytest.mark.parametrize("score", [-0.1, 1.5, "0.5", None])
def test_impossible_score_is_refused(score):
    with pytest.raises(ValueError, match="score must sit"):
        make(detection={"score": score})


@pytest.mark.parametrize(
    "box",
    [[60.0, 20.0, 10.0, 90.0], [10.0, 90.0, 60.0, 20.0], [10.0, 20.0, 60.0]],
)
def test_malformed_box_is_refused(box):
    with pytest.raises(ValueError, match="malformed"):
        make(detection={"bbox_xyxy": box})


@pytest.mark.parametrize(
    "box", [[-1.0, 20.0, 60.0, 90.0], [10.0, 20.0, 260.0, 90.0], [10.0, 20.0, 60.0, 190.0]]
)
def test_box_outside_the_photo_is_refused(box):
    with pytest.raises(ValueError, match="outside the photo"):
        make(detection={"bbox_xyxy": box})


def test_a_finding_without_a_site_is_refused():
    with pytest.raises(ValueError, match="needs the site"):
        make(context={"zone_id": "z"})
    with pytest.raises(ValueError, match="needs the site"):
        make(context={"site_id": "   "})


def test_zone_and_task_are_optional_but_checked():
    result = make(context={"site_id": "chantier-12"})
    assert result["context"]["zone_id"] is None and result["context"]["task_id"] is None
    with pytest.raises(ValueError, match="Invalid zone_id"):
        make(context={"site_id": "s", "zone_id": ""})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run", ""),
        ("run", None),
        ("checkpoint_sha256", "court"),
        ("score_threshold", 0),
        ("score_threshold", 1),
        ("score_threshold", None),
        ("score_threshold", True),
    ],
)
def test_incomplete_model_trace_is_refused(field, value):
    with pytest.raises(ValueError, match="finding must"):
        make(model={**MODEL, field: value})


def quality(verdict):
    return {"verdict": verdict}


def test_outcome_says_what_the_analysis_established():
    assert analyse_outcome(quality("usable"), [{"id": "x"}]) == "defects_found"
    assert analyse_outcome(quality("usable"), []) == "no_target_defect_in_readable_area"
    assert analyse_outcome(quality("partial"), []) == "partial_analysis"
    assert analyse_outcome(quality("unusable"), []) == "not_assessable"


def test_an_unreadable_photo_never_reports_a_clean_surface():
    """Une photo trop sombre ne prouve rien : elle ne doit pas ressembler a un mur sain."""
    assert analyse_outcome(quality("unusable"), [{"id": "x"}]) == "not_assessable"
    assert analyse_outcome(quality("unusable"), []) != "no_target_defect_in_readable_area"


def test_a_partial_photo_with_a_defect_still_reports_the_defect():
    assert analyse_outcome(quality("partial"), [{"id": "x"}]) == "defects_found"


def test_every_outcome_is_declared():
    produced = {
        analyse_outcome(quality(v), f)
        for v in ("usable", "partial", "unusable")
        for f in ([], [{"id": "x"}])
    }
    assert produced == set(finding.OUTCOMES)
