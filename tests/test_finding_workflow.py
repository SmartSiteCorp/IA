"""Recette locale du constat : preuves, retours, erreurs et relances sans réseau."""

import io
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from smartsite_ia import finding, finding_cli, finding_engine, finding_events, finding_store
from smartsite_ia import finding_validation as check
from smartsite_ia.curation import digest
from smartsite_ia.finding_html import export_case
from smartsite_ia.photo_quality import assess_photo
from smartsite_ia.prediction import decode_photo

AT = "2026-10-09T10:00:00.000000Z"
LATER = "2026-10-09T10:01:00.000000Z"
MODEL = {"run": "synthetic-test", "checkpoint_sha256": "b" * 64, "score_threshold": 0.3}
ENGINE = {
    "run_sha256": "c" * 64,
    "class_names": ["crack", "surface_loss", "moisture_trace"],
    "preprocessing": "training-v1",
    "runtime_versions": {"synthetic": "1"},
}
REQUEST = {
    "context": {"site_id": "synthetic-site", "zone_id": None, "task_id": None},
    "capture_id": None,
    "captured_at": None,
    "media_id": None,
}
DETECTION = {"class_name": "crack", "score": 0.8, "bbox_xyxy": [10.0, 20.0, 60.0, 90.0]}


def photo_bytes(seed=42):
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(40, 220, (128, 128, 3), dtype=np.uint8))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def analysis(raw=None, **changes):
    raw = raw or photo_bytes()
    return finding.build_analysis(
        **{
            "photo_sha256": digest(raw),
            "request": REQUEST,
            "model": MODEL,
            "engine": ENGINE,
            "quality": assess_photo(decode_photo(raw)),
            "predictions": [DETECTION],
            "analysed_at": AT,
            **changes,
        }
    )


@pytest.fixture
def case(tmp_path):
    raw = photo_bytes()
    a = analysis(raw)
    root = tmp_path / "case"
    finding_store.create_case(root, a, raw)
    return root, a, raw


def event(a, kind="review", **changes):
    payload = {
        "decision": "confirmed",
        "note": "Contrôle synthétique, sans validation chantier réelle.",
    }
    return {
        "request_id": "review-1",
        "kind": kind,
        "finding_id": a["findings"][0]["id"],
        "actor": "synthetic-reviewer",
        "payload": payload,
        **changes,
    }


def confirmed(case):
    root, a, _ = case
    finding_store.record_event(root, "synthetic-site", event(a), expected_revision=0)
    return root, a


def test_automatic_scope_and_threshold_are_explicit():
    a = analysis(
        predictions=[
            DETECTION,
            {**DETECTION, "class_name": "moisture_trace"},
            {**DETECTION, "score": 0.3},
            {**DETECTION, "score": 0.2},
            DETECTION,
        ]
    )
    assert len(a["findings"]) == 1
    assert [r["reason"] for r in a["withheld"]] == [
        "class_not_enabled",
        "below_or_equal_threshold",
        "below_or_equal_threshold",
        "duplicate_geometry",
    ]
    assert a["conformity"] == "not_assessed" and not a["qualified_for_smartsite"]
    assert a["request"]["captured_at"] is None
    finding.validate_analysis(a)


@pytest.mark.parametrize("field", ["request", "model", "engine"])
def test_analysis_identity_includes_capture_site_and_model(field):
    base = analysis()
    value = deepcopy(base[field])
    if field == "request":
        value["capture_id"] = "another-capture"
    elif field == "model":
        value["checkpoint_sha256"] = "f" * 64
    else:
        value["runtime_versions"]["synthetic"] = "2"
    assert analysis(**{field: value})["id"] != base["id"]


def test_unknown_fields_and_forged_derived_values_are_rejected():
    a = analysis()
    a["outcome"] = "no_target_defect_in_readable_area"
    with pytest.raises(ValueError):
        finding.validate_analysis(a)
    with pytest.raises(ValueError):
        analysis(request={**REQUEST, "url": "https://invalid/"})
    with pytest.raises(ValueError):
        analysis(predictions=[{**DETECTION, "extra": True}])


@pytest.mark.parametrize("time", ["not-a-date", "2026-10-09T12:00:00", "2027-01-01T00:00:00Z"])
def test_bad_or_future_capture_time_is_not_invented(time):
    with pytest.raises(ValueError):
        analysis(request={**REQUEST, "captured_at": time})


def test_capture_time_is_normalized_without_losing_its_instant():
    assert (
        analysis(request={**REQUEST, "captured_at": "2026-10-09T11:30:00+02:00"})["request"][
            "captured_at"
        ]
        == "2026-10-09T09:30:00.000000Z"
    )


def test_excess_predictions_and_findings_are_refused_not_truncated():
    with pytest.raises(ValueError):
        analysis(predictions=[DETECTION] * 501)
    boxes = [
        {**DETECTION, "bbox_xyxy": [i % 100, i // 100, (i % 100) + 1, (i // 100) + 1]}
        for i in range(201)
    ]
    with pytest.raises(ValueError, match="Too many findings"):
        analysis(predictions=boxes)


def test_unusable_photo_never_runs_a_fake_positive_analysis():
    out = io.BytesIO()
    Image.new("RGB", (128, 128), "black").save(out, format="PNG")
    raw = out.getvalue()
    assert analysis(raw, predictions=[])["outcome"] == "not_assessable"
    with pytest.raises(ValueError):
        analysis(raw)


def test_repeated_creation_preserves_human_decisions(case):
    root, a, raw = case
    confirmed(case)
    result = finding_store.create_case(root, analysis(raw, analysed_at=LATER), raw)
    assert len(result["events"]) == 1
    assert result["analysis"]["analysed_at"] == AT
    assert result["analysis"] == a


def test_another_analysis_cannot_replace_the_existing_dossier(case):
    root, _, raw = case
    with pytest.raises(check.ConflictError):
        finding_store.create_case(
            root, analysis(raw, request={**REQUEST, "capture_id": "next"}), raw
        )
    with pytest.raises(check.ConflictError):
        finding_store.read_case(root, "another-site")


def test_review_changes_only_the_human_view_and_keeps_original_prediction(case):
    root, a, _ = case
    request = event(
        a,
        payload={
            "decision": "confirmed",
            "note": "Erreur de catégorie constatée.",
            "class_name": "surface_loss",
            "severity": "À reprendre selon le contrôleur",
            "task_id": "T-123",
        },
    )
    result = finding_store.record_event(root, "synthetic-site", request, expected_revision=0)
    view = finding_events.replay(a, result["events"])["findings"][0]
    assert view["finding"]["defect"]["class"] == "crack"
    assert view["verification"]["effective_class"] == "surface_loss"
    assert view["verification"]["task_id"] == "T-123"
    assert result["analysis"] == a
    assert view["correction"]["status"] == "not_assessed"
    assert finding_events.replay(a, result["events"])["notification"]["sent"] is False


def test_retry_and_conflicting_edit_are_distinguished(case):
    root, a, _ = case
    r = event(a)
    first = finding_store.record_event(root, "synthetic-site", r, expected_revision=0)
    assert finding_store.record_event(root, "synthetic-site", r, expected_revision=0) == first
    with pytest.raises(check.ConflictError):
        finding_store.record_event(
            root, "synthetic-site", {**r, "actor": "different"}, expected_revision=1
        )
    with pytest.raises(check.ConflictError):
        finding_store.record_event(
            root, "synthetic-site", {**r, "request_id": "second"}, expected_revision=0
        )


def test_unavailable_target_and_cross_site_request_do_not_write(case):
    root, a, _ = case
    before = (root / "record.json").read_bytes()
    with pytest.raises(ValueError):
        finding_store.record_event(
            root, "synthetic-site", event(a, finding_id="f" * 32), expected_revision=0
        )
    with pytest.raises(check.ConflictError):
        finding_store.record_event(root, "other", event(a), expected_revision=0)
    assert (root / "record.json").read_bytes() == before


def test_repair_needs_confirmation_and_evidence(case):
    root, a, _ = case
    request = event(
        a,
        "correction",
        payload={"status": "resolved", "note": "Réparation test.", "evidence_sha256": []},
    )
    with pytest.raises(ValueError, match="evidence"):
        finding_store.record_event(root, "synthetic-site", request, expected_revision=0)
    raw = photo_bytes(43)
    sha = digest(raw)
    request["payload"]["evidence_sha256"] = [sha]
    with pytest.raises(ValueError, match="confirmed"):
        finding_store.record_event(
            root, "synthetic-site", request, expected_revision=0, evidence={sha: raw}
        )
    assert not (root / "evidence").exists()


def test_repair_and_reassessment_keep_all_evidence_and_history(case):
    root, a = confirmed(case)
    raw = photo_bytes(43)
    sha = digest(raw)
    request = event(
        a,
        "correction",
        request_id="repair-1",
        payload={"status": "resolved", "note": "Réparation synthétique.", "evidence_sha256": [sha]},
    )
    r = finding_store.record_event(
        root, "synthetic-site", request, expected_revision=1, evidence={sha: raw}
    )
    assert (
        finding_events.replay(a, r["events"])["findings"][0]["correction"]["status"] == "resolved"
    )
    changed = event(
        a,
        request_id="recheck",
        payload={"decision": "indeterminate", "note": "Nouvelle vérification nécessaire."},
    )
    r = finding_store.record_event(root, "synthetic-site", changed, expected_revision=2)
    assert (
        finding_events.replay(a, r["events"])["findings"][0]["correction"]["status"]
        == "not_assessed"
    )
    assert (root / "evidence" / f"{sha}.bin").read_bytes() == raw
    assert len(r["events"]) == 3


def test_manual_reports_and_worker_notes_have_no_fake_model_score(case):
    root, a, _ = case
    manual = event(
        a,
        "manual",
        finding_id=None,
        request_id="manual-1",
        payload={
            "class_name": "moisture_trace",
            "bbox_xyxy": [1, 2, 50, 60],
            "note": "Signalement manuel synthétique.",
        },
    )
    r = finding_store.record_event(root, "synthetic-site", manual, expected_revision=0)
    view = finding_events.replay(a, r["events"])
    added = view["findings"][1]["finding"]
    assert view["human_reports"] == 1 and added["confidence"]["model_score"] is None
    assert added["origin"] == "human" and added["state"] == "proposed"
    note = event(
        a,
        "note",
        request_id="note-1",
        finding_id=added["id"],
        payload={"note": "Accès momentanément impossible."},
    )
    r = finding_store.record_event(root, "synthetic-site", note, expected_revision=1)
    assert len(finding_events.replay(a, r["events"])["findings"][1]["notes"]) == 2
    assert r["analysis"]["outcome"] == "defects_found"


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": "automatic", "note": "Test"},
        {"decision": "rejected", "note": "Test", "task_id": "x"},
        {"decision": "indeterminate", "note": "Test", "severity": "high"},
        {"decision": "confirmed", "note": ""},
        {"decision": "confirmed", "note": "x" * 4001},
        {"decision": "confirmed", "note": "ok", "unknown": True},
    ],
)
def test_invalid_human_requests_are_refused(case, payload):
    root, a, _ = case
    with pytest.raises(ValueError):
        finding_store.record_event(
            root, "synthetic-site", event(a, payload=payload), expected_revision=0
        )


def test_history_order_future_times_and_event_limit():
    a = analysis()
    r = event(a)
    one = finding_events.append_event(a, [], r, expected_revision=0, recorded_at=LATER)
    for change in ({"revision": 2}, {"recorded_at": AT.replace("10:00", "09:00")}):
        with pytest.raises(ValueError):
            finding_events.replay(a, [{**one[0], **change}])
    with pytest.raises(ValueError):
        finding_events.replay(a, one * 1001)
    with pytest.raises(ValueError):
        finding_events.append_event(
            a,
            [],
            {**r, "occurred_at": "2099-01-01T00:00:00Z"},
            expected_revision=0,
            recorded_at=LATER,
        )


@pytest.mark.parametrize(
    "corruption", ["photo", "derived", "unknown", "schema", "missing", "event", "json"]
)
def test_corrupt_bundle_is_never_reported_as_clean(case, corruption):
    root, a, _ = case
    if corruption == "photo":
        (root / "source.bin").write_bytes(photo_bytes(43))
    elif corruption == "missing":
        (root / "source.bin").unlink()
    elif corruption == "json":
        (root / "record.json").write_text('{"schema_version":2,"schema_version":2}')
    else:
        record = json.loads((root / "record.json").read_text())
        if corruption == "derived":
            record["analysis"]["findings"][0]["location"]["point_3d"] = [1, 2, 3]
        elif corruption == "unknown":
            record["extra"] = True
        elif corruption == "schema":
            record["schema_version"] = 1
        else:
            record["events"] = [{"revision": 1}]
        (root / "record.json").write_text(json.dumps(record))
    with pytest.raises(finding_store.IntegrityError):
        finding_store.read_case(root, "synthetic-site")


def test_symlinks_in_evidence_or_dossier_are_refused(case, tmp_path):
    root, _, raw = case
    original = root / "source.bin"
    original.unlink()
    elsewhere = tmp_path / "other.png"
    elsewhere.write_bytes(raw)
    original.symlink_to(elsewhere)
    with pytest.raises(finding_store.IntegrityError):
        finding_store.read_case(root, "synthetic-site")
    link = tmp_path / "linked"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        finding_store.read_case(link, "synthetic-site")


def test_write_lock_and_interruption_preserve_prior_state(case, monkeypatch):
    root, a, _ = case
    before = (root / "record.json").read_bytes()
    with finding_store.case_lock(root), pytest.raises(finding_store.BusyError):
        finding_store.record_event(root, "synthetic-site", event(a), expected_revision=0)

    def fail_replace(*args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(finding_store.os, "replace", fail_replace)
    with pytest.raises(OSError):
        finding_store.record_event(root, "synthetic-site", event(a), expected_revision=0)
    assert (root / "record.json").read_bytes() == before
    assert not list(root.glob(".finding-*"))
    assert finding_store.read_case(root, "synthetic-site")["events"] == []


def test_interruption_after_commit_does_not_delete_published_evidence(case, monkeypatch):
    root, a = confirmed(case)
    raw = photo_bytes(43)
    sha = digest(raw)
    original = finding_store.atomic_record

    def interrupted(path, value):
        original(path, value)
        raise KeyboardInterrupt()

    monkeypatch.setattr(finding_store, "atomic_record", interrupted)
    request = event(
        a,
        "correction",
        request_id="repair",
        payload={"status": "resolved", "note": "Exemple.", "evidence_sha256": [sha]},
    )
    with pytest.raises(KeyboardInterrupt):
        finding_store.record_event(
            root, "synthetic-site", request, expected_revision=1, evidence={sha: raw}
        )
    assert len(finding_store.read_case(root, "synthetic-site")["events"]) == 2


def test_export_is_portable_and_escapes_user_text(case, tmp_path):
    root, a, _ = case
    finding_store.record_event(
        root,
        "synthetic-site",
        event(a, payload={"decision": "rejected", "note": '<script>alert("test")</script>'}),
        expected_revision=0,
    )
    out = tmp_path / "export"
    export_case(root, out, "synthetic-site")
    assert finding_store.read_case(root, "synthetic-site") == finding_store.read_case(
        out, "synthetic-site"
    )
    page = (out / "index.html").read_text()
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "Rejeté par une personne" in page and "Aucune notification envoyée" in page
    assert (out / "photo.png").is_file() and (out / "annotated.jpg").is_file()
    with pytest.raises(FileExistsError):
        export_case(root, out, "synthetic-site")
    with pytest.raises(ValueError):
        export_case(root, root / "nested", "synthetic-site")


def test_cli_record_show_export_and_invalid_requests(case, tmp_path, capsys):
    root, a, _ = case
    request = tmp_path / "review.json"
    request.write_text(json.dumps(event(a)))
    assert (
        finding_cli.main(
            [
                "record",
                str(root),
                "--site",
                "synthetic-site",
                "--request",
                str(request),
                "--revision",
                "0",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["revision"] == 1
    assert finding_cli.main(["show", str(root), "--site", "synthetic-site"]) == 0
    assert (
        json.loads(capsys.readouterr().out)["findings"][0]["verification"]["state"] == "confirmed"
    )
    assert (
        finding_cli.main(
            ["export", str(root), "--site", "synthetic-site", "--output", str(tmp_path / "html")]
        )
        == 0
    )
    capsys.readouterr()
    assert finding_cli.main(["show", str(root), "--site", "other"]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "conflict"
    request.write_text('{"secret": "DO-NOT-LOG"}')
    assert (
        finding_cli.main(
            [
                "record",
                str(root),
                "--site",
                "synthetic-site",
                "--request",
                str(request),
                "--revision",
                "1",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert "DO-NOT-LOG" not in output.err
    assert json.loads(output.err)["error"]["code"] == "invalid_input"


def test_cli_rejects_nonobject_payload_with_evidence(case, tmp_path, capsys):
    root, a, raw = case
    request = tmp_path / "invalid.json"
    request.write_text(json.dumps(event(a, "correction", payload=[])))
    proof = tmp_path / "proof.png"
    proof.write_bytes(raw)
    assert (
        finding_cli.main(
            [
                "record",
                str(root),
                "--site",
                "synthetic-site",
                "--request",
                str(request),
                "--revision",
                "0",
                "--evidence",
                str(proof),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_input"
    assert finding_store.read_case(root, "synthetic-site")["events"] == []


@pytest.mark.parametrize("revision", [False, -1])
def test_identical_retry_still_requires_a_valid_revision(case, revision):
    root, a = confirmed(case)
    with pytest.raises(check.ConflictError):
        finding_store.record_event(root, "synthetic-site", event(a), expected_revision=revision)


@pytest.fixture
def engine_case(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/checkpoint_best_total.pth"
    checkpoint.write_bytes(b"synthetic weights only")
    raw = photo_bytes()
    photo = tmp_path / "photo.png"
    photo.write_bytes(raw)
    parent = {
        "schema_version": 1,
        "status": "completed",
        "model": "RFDETRNano",
        "checkpoint": "checkpoints/checkpoint_best_total.pth",
        "checkpoint_sha256": digest(checkpoint.read_bytes()),
        "checkpoint_selection": "last_epoch",
        "class_names": ENGINE["class_names"],
        "config": {"class_names": ENGINE["class_names"]},
        "runtime": {"versions": ENGINE["runtime_versions"]},
    }
    (run / "run.json").write_text(json.dumps(parent))
    calls = []

    class Engine:
        def predict(self, image, threshold):
            calls.append((image.size, threshold))
            return SimpleNamespace(
                xyxy=np.array([[10.0, 20.0, 60.0, 90.0], [70, 70, 100, 100]]),
                confidence=np.array([0.8, 0.9]),
                class_id=np.array([0, 2]),
                mask=None,
                data={"class_name": np.array(["crack", "moisture_trace"])},
            )

    monkeypatch.setattr(finding_engine, "load_box_engine", lambda *args: Engine())
    monkeypatch.setattr(
        finding_engine, "runtime", lambda device: {"versions": ENGINE["runtime_versions"]}
    )
    return run, photo, tmp_path / "dossier", calls


def test_native_bridge_maps_cubit_classes_and_reuses_the_dossier(engine_case):
    run, photo, out, calls = engine_case
    one = finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
    assert [f["defect"]["class"] for f in one["analysis"]["findings"]] == ["crack"]
    assert one["analysis"]["withheld"] == [{"prediction_index": 1, "reason": "class_not_enabled"}]
    assert len(calls) == 1
    assert (
        finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
        == one
    )
    assert len(calls) == 1
    with pytest.raises(check.ConflictError):
        finding_engine.analyse_photo(
            photo, run=run, output=out, request={**REQUEST, "capture_id": "different"}, device="cpu"
        )
    assert len(calls) == 1


def test_unusable_photo_skips_gpu_and_model_loading(engine_case, monkeypatch):
    run, photo, out, calls = engine_case
    Image.new("RGB", (128, 128), "black").save(photo)

    def forbid(*args):
        raise AssertionError("No engine or GPU for an unusable photo")

    monkeypatch.setattr(finding_engine, "runtime", forbid)
    monkeypatch.setattr(finding_engine, "load_box_engine", forbid)
    result = finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
    assert result["analysis"]["outcome"] == "not_assessable" and calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "training"),
        ("model", "another"),
        ("class_names", ["crack"]),
        ("checkpoint", "../../elsewhere"),
        ("checkpoint_sha256", "z" * 64),
        ("config", {}),
        ("runtime", {}),
        ("checkpoint_selection", "best_validation"),
    ],
)
def test_bad_model_provenance_prevents_loading(engine_case, field, value):
    run, photo, out, calls = engine_case
    p = run / "run.json"
    r = json.loads(p.read_text())
    r[field] = value
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
    assert calls == [] and not out.exists()


def test_engine_failure_and_version_mismatch_leave_no_result(engine_case, monkeypatch):
    run, photo, out, calls = engine_case
    monkeypatch.setattr(finding_engine, "runtime", lambda device: {"versions": {"synthetic": "2"}})
    with pytest.raises(ValueError):
        finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
    assert not out.exists() and not calls
    monkeypatch.setattr(
        finding_engine, "runtime", lambda device: {"versions": ENGINE["runtime_versions"]}
    )

    def fail(*args):
        raise RuntimeError("Synthetic engine failure")

    monkeypatch.setattr(finding_engine, "load_box_engine", fail)
    with pytest.raises(RuntimeError):
        finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
    assert not out.exists()


def test_changed_weights_and_paths_are_rejected(engine_case):
    run, photo, out, calls = engine_case
    with pytest.raises(ValueError):
        finding_engine.analyse_photo(
            photo, run=run, output=run / "nested", request=REQUEST, device="cpu"
        )
    with pytest.raises(ValueError):
        finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="invalid")
    (run / "checkpoints/checkpoint_best_total.pth").write_bytes(b"changed")
    with pytest.raises(ValueError):
        finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
    assert calls == []


def test_model_changed_during_inference_is_not_published(engine_case, monkeypatch):
    run, photo, out, _ = engine_case

    class Changing:
        def predict(self, image, threshold):
            (run / "checkpoints/checkpoint_best_total.pth").write_bytes(b"changed")
            return SimpleNamespace(
                xyxy=np.empty((0, 4)),
                confidence=np.array([]),
                class_id=np.array([]),
                mask=None,
                data={},
            )

    monkeypatch.setattr(finding_engine, "load_box_engine", lambda *args: Changing())
    with pytest.raises(ValueError, match="changed during"):
        finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
    assert not out.exists()


def test_exif_orientation_is_used_for_the_finding_and_export(engine_case, tmp_path):
    run, photo, out, calls = engine_case
    image = Image.open(io.BytesIO(photo_bytes())).resize((160, 128))
    exif = Image.Exif()
    exif[274] = 6
    image.save(photo, format="JPEG", exif=exif)
    result = finding_engine.analyse_photo(photo, run=run, output=out, request=REQUEST, device="cpu")
    assert calls == [((128, 160), 0.3)]
    assert result["analysis"]["quality"]["photo"] == {"width": 128, "height": 160}
    export_case(out, tmp_path / "html", "synthetic-site")
    with Image.open(tmp_path / "html/photo.png") as saved:
        assert saved.size == (128, 160) and not saved.getexif()


def test_cli_can_analyse_without_calling_any_external_service(engine_case, capsys):
    run, photo, out, calls = engine_case
    assert (
        finding_cli.main(
            [
                "analyse",
                str(photo),
                "--run",
                str(run),
                "--output",
                str(out),
                "--site",
                "synthetic-site",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["findings"] == 1 and not result["notification_sent"] and len(calls) == 1


def test_cli_records_real_local_evidence_without_fake_repair_metadata(case, tmp_path, capsys):
    root, a = confirmed(case)
    proof = tmp_path / "proof.png"
    proof.write_bytes(photo_bytes(43))
    request = tmp_path / "repair.json"
    request.write_text(
        json.dumps(
            event(
                a,
                "correction",
                request_id="repair",
                payload={
                    "status": "resolved",
                    "note": "Preuve synthétique.",
                    "evidence_sha256": [],
                },
            )
        )
    )
    args = [
        "record",
        str(root),
        "--site",
        "synthetic-site",
        "--request",
        str(request),
        "--revision",
        "1",
        "--evidence",
        str(proof),
    ]
    assert finding_cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["findings"][0]["correction"]["evidence_sha256"] == [digest(proof.read_bytes())]
    assert finding_cli.main(args) == 0 and json.loads(capsys.readouterr().out)["revision"] == 2
    export_case(root, tmp_path / "portable", "synthetic-site")
    assert list((tmp_path / "portable/evidence").glob("*.png"))


@pytest.mark.parametrize(
    "exception,code",
    [
        (KeyboardInterrupt(), "interrupted"),
        (finding_store.BusyError(), "busy"),
        (OSError("PRIVATE_PATH"), "io_error"),
        (RuntimeError("PRIVATE_KEY"), "engine_error"),
    ],
)
def test_cli_errors_have_stable_codes_and_do_not_leak_values(
    case, monkeypatch, capsys, exception, code
):
    root, _, _ = case

    def fail(*args):
        raise exception

    monkeypatch.setattr(finding_cli, "read_case", fail)
    assert finding_cli.main(["show", str(root), "--site", "synthetic-site"]) in (1, 130)
    output = capsys.readouterr()
    assert json.loads(output.err)["error"]["code"] == code
    assert "PRIVATE" not in output.err


def test_nonfinite_values_unicode_controls_and_nested_corruption_are_rejected(case):
    for value in (True, float("nan"), float("inf"), "1"):
        with pytest.raises(ValueError):
            check.number(value)
    with pytest.raises(ValueError):
        check.text("a\x00b", "name")
    with pytest.raises(ValueError):
        check.text(" leading", "name")
    with pytest.raises(ValueError):
        check.canonical({"value": float("nan")})
    root, _, _ = case
    r = json.loads((root / "record.json").read_text())
    r["analysis"]["quality"]["photo"] = []
    (root / "record.json").write_text(json.dumps(r))
    with pytest.raises(finding_store.IntegrityError):
        finding_store.read_case(root, "synthetic-site")


def test_missing_changed_and_unreferenced_evidence_are_rejected(case):
    root, a = confirmed(case)
    raw = photo_bytes(43)
    sha = digest(raw)
    r = event(
        a,
        "correction",
        request_id="repair",
        payload={"status": "resolved", "note": "Test.", "evidence_sha256": [sha]},
    )
    with pytest.raises(ValueError):
        finding_store.record_event(root, "synthetic-site", r, expected_revision=1)
    with pytest.raises(ValueError):
        finding_store.record_event(
            root,
            "synthetic-site",
            event(a, request_id="unused"),
            expected_revision=1,
            evidence={sha: raw},
        )
    with pytest.raises(ValueError):
        finding_store.record_event(
            root, "synthetic-site", r, expected_revision=1, evidence={sha: b"wrong bytes"}
        )
    finding_store.record_event(root, "synthetic-site", r, expected_revision=1, evidence={sha: raw})
    (root / "evidence" / f"{sha}.bin").write_bytes(photo_bytes(44))
    with pytest.raises(finding_store.IntegrityError):
        finding_store.read_case(root, "synthetic-site")


def test_an_export_without_findings_remains_explicit(tmp_path):
    raw = photo_bytes()
    a = analysis(raw, predictions=[])
    root = tmp_path / "source"
    finding_store.create_case(root, a, raw)
    export_case(root, tmp_path / "export", "synthetic-site")
    assert "Aucune cible détectée" in (tmp_path / "export/index.html").read_text()
    assert "Aucun constat" in (tmp_path / "export/index.html").read_text()


def test_a_prediction_fully_in_an_unreadable_region_is_withheld():
    pixels = np.asarray(decode_photo(photo_bytes())).copy()
    pixels[:, :64] = 0
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    raw = stream.getvalue()
    a = analysis(
        raw,
        predictions=[
            {**DETECTION, "bbox_xyxy": [1, 10, 50, 80]},
            {**DETECTION, "bbox_xyxy": [60, 10, 80, 80]},
        ],
    )
    assert a["quality"]["verdict"] == "partial"
    assert len(a["findings"]) == 1
    assert a["withheld"][0]["reason"] == "unreadable_region"
    assert a["findings"][0]["observability"]["unreadable_box_fraction"] == pytest.approx(0.2)
    assert a["findings"][0]["observability"]["status"] == "partial"


def test_duplicate_geometry_keeps_the_strongest_score_without_losing_outputs():
    a = analysis(predictions=[{**DETECTION, "score": 0.5}, {**DETECTION, "score": 0.9}])
    assert len(a["predictions"]) == 2
    assert a["findings"][0]["confidence"]["model_score"] == 0.9
    assert a["withheld"] == [{"prediction_index": 0, "reason": "duplicate_geometry"}]
