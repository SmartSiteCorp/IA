"""Les fixtures vérifient le logiciel ; elles ne mesurent pas la qualité du modèle."""

import copy
import json

import pytest
from PIL import Image

from smartsite_ia import zero_shot as probe
from smartsite_ia.curation import digest


@pytest.fixture
def files(tmp_path):
    images = tmp_path / "sample"
    images.mkdir()
    photo = images / "photo.png"
    Image.new("RGB", (48, 40), "gray").save(photo)
    record = {
        "id": "case_1",
        "file": "photo.png",
        "sha256": digest(photo.read_bytes()),
        "width": 48,
        "height": 40,
        "reference_boxes": [[5, 5, 20, 25]],
        "review_note": "<script>Annotation synthétique pour les tests</script>",
    }
    manifest = {
        "schema_version": 1,
        "source": {
            "id": "synthetic",
            "url": "https://example.org",
            "license": "test",
            "scope": "Fixture synthétique, aucune mesure IA",
        },
        "records": [record],
    }
    sample = images / "manifest.json"
    sample.write_text(json.dumps(manifest))
    model = tmp_path / "model"
    model.mkdir()
    specifications = []
    for name in sorted(probe.MODEL_FILES):
        raw = f"synthetic {name}".encode()
        (model / name).write_bytes(raw)
        specifications.append({"name": name, "size": len(raw), "sha256": digest(raw)})
    config = {
        "schema_version": 1,
        "model_id": probe.MODEL_ID,
        "revision": "a" * 40,
        "runtime": {name: "test" for name in ("torch", "transformers", "safetensors")},
        "files": specifications,
        "prompt": "water stain.",
        "box_threshold": 0.25,
        "text_threshold": 0.25,
        "match_iou": 0.5,
        "max_images": 32,
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(config))
    return sample, model, protocol, tmp_path / "output"


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_bounded_manifest_and_model_integrity(files):
    sample, model, protocol, _ = files
    config, checksum = probe.load_protocol(protocol)
    assert len(checksum) == 64
    manifest, _ = probe.load_sample(sample, config["max_images"])
    assert manifest["records"][0]["width"] == 48
    probe.check_model(model, config)
    (model / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size"):
        probe.check_model(model, config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "some/untrusted-model"),
        ("revision", "main"),
        ("prompt", ""),
        ("prompt", "a" * 201 + "."),
        ("prompt", "missing period"),
        ("box_threshold", float("nan")),
        ("text_threshold", True),
        ("match_iou", 1),
        ("max_images", 0),
        ("max_images", 65),
        ("max_images", True),
        ("runtime", {}),
        ("files", []),
    ],
)
def test_invalid_protocol_is_rejected(files, field, value):
    protocol = files[2]
    edit(protocol, lambda data: data.update({field: value}))
    with pytest.raises(ValueError):
        probe.load_protocol(protocol)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["files"][0].update(name="../escape"),
        lambda data: data["files"][0].update(name=[]),
        lambda data: data["files"][0].update(sha256="bad"),
        lambda data: data["files"][0].update(size=True),
        lambda data: data["files"].__setitem__(0, copy.deepcopy(data["files"][1])),
    ],
)
def test_model_file_specification_is_checked(files, mutate):
    edit(files[2], mutate)
    with pytest.raises(ValueError):
        probe.load_protocol(files[2])


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "../escape"),
        ("id", 123),
        ("file", "../outside.png"),
        ("sha256", "wrong"),
        ("width", 49),
        ("height", True),
        ("reference_boxes", [[0, 0, 50, 39]]),
        ("reference_boxes", [[0, 0, 0, 30]]),
        ("reference_boxes", [[0, 0, float("nan"), 30]]),
        ("reference_boxes", None),
        ("review_note", ""),
    ],
)
def test_invalid_photo_record_is_rejected(files, field, value):
    sample = files[0]
    edit(sample, lambda data: data["records"][0].update({field: value}))
    with pytest.raises(ValueError):
        probe.load_sample(sample, 32)


def test_duplicate_and_empty_samples_rejected(files):
    sample = files[0]
    edit(sample, lambda data: data["records"].append(copy.deepcopy(data["records"][0])))
    with pytest.raises(ValueError, match="Duplicate"):
        probe.load_sample(sample, 32)
    with pytest.raises(ValueError, match="limit"):
        probe.load_sample(sample, 1)
    edit(sample, lambda data: data.update(records=[]))
    with pytest.raises(ValueError, match="empty"):
        probe.load_sample(sample, 32)


def test_linked_photo_rejected(files, tmp_path):
    photo = files[0].parent / "photo.png"
    outside = tmp_path / "outside.png"
    photo.rename(outside)
    photo.symlink_to(outside)
    with pytest.raises(ValueError):
        probe.load_sample(files[0], 32)


def prediction(box, score=0.9):
    return {"bbox_xyxy": box, "score": score, "text_label": "water stain"}


def test_box_matching_does_not_count_duplicates_twice():
    references = [[0, 0, 10, 10], [30, 30, 40, 40]]
    predictions = [
        prediction([0, 0, 10, 10]),
        prediction([0, 0, 10, 10], 0.8),
        prediction([20, 20, 25, 25], 0.7),
    ]
    assert probe.box_counts(references, predictions, 0.5) == {"tp": 1, "fp": 2, "fn": 1}
    assert probe.box_counts(references, [], 0.5) == {"tp": 0, "fp": 0, "fn": 2}
    assert probe.box_counts([], predictions, 0.5) == {"tp": 0, "fp": 3, "fn": 0}
    assert probe.box_counts([], [], 0.5) == {"tp": 0, "fp": 0, "fn": 0}


def test_box_overlap_uses_iou_not_reference_coverage():
    # Une boîte couvrant tout le mur ne doit pas valider chaque petite tache.
    assert probe.box_counts([[0, 0, 10, 10]], [prediction([0, 0, 100, 100])], 0.5) == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }


@pytest.mark.parametrize(
    "records",
    [
        None,
        [prediction([0, 0, 20, 20])] * 901,
        [prediction([0, 0, 20, 20], float("nan"))],
        [prediction([0, 0, 20, 20], 1.1)],
        [prediction([-1, 0, 20, 20])],
        [prediction([0, 0, 70, 20])],
        [{}],
    ],
)
def test_engine_outputs_are_validated(records):
    with pytest.raises(ValueError):
        probe.check_predictions(records, Image.new("RGB", (48, 40)))


def test_complete_probe_is_offline_and_escapes_notes(files, monkeypatch):
    engines = []

    class Engine:
        def __init__(self, *args):
            self.closed = False
            engines.append(self)

        def predict(self, photo):
            assert photo.size == (48, 40)
            return [prediction([5, 5, 20, 25])]

        def close(self):
            self.closed = True

    monkeypatch.setattr(probe, "GroundingEngine", Engine)
    report = probe.run_probe(*files)
    assert report["training_started"] is False
    assert report["summary"]["tp"] == 1
    assert report["summary"]["recall"] == 1.0
    assert engines[0].closed
    output = files[3]
    page = (output / "index.html").read_text()
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert (output / "case_1" / "prediction.jpg").is_file()
    assert json.loads((output / "report.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        probe.run_probe(*files)
    assert len(engines) == 1


def test_failed_prediction_leaves_no_completed_report(files, monkeypatch):
    closed = []

    class Engine:
        def __init__(self, *args):
            pass

        def predict(self, photo):
            raise RuntimeError("Synthetic model failure")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(probe, "GroundingEngine", Engine)
    with pytest.raises(RuntimeError, match="Synthetic"):
        probe.run_probe(*files)
    assert closed == [True]
    assert not files[3].exists()
    assert files[0].is_file()


def test_cli_reports_invalid_inputs(files, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            str(files[0]),
            "--model",
            str(files[1]),
            "--protocol",
            str(files[2]),
            "--output",
            str(files[3]),
        ],
    )
    monkeypatch.setattr(probe, "run_probe", lambda *args: (_ for _ in ()).throw(ValueError("bad")))
    assert probe.main() == 1
    assert "Probe failed: bad" in capsys.readouterr().err


def test_runtime_mismatch_is_rejected_before_model_import(files, monkeypatch):
    config, _ = probe.load_protocol(files[2])
    monkeypatch.setattr(probe, "version", lambda name: "wrong")
    with pytest.raises(ValueError, match="Probe expects"):
        probe.GroundingEngine(files[1], config)


@pytest.mark.parametrize("loading_fails", [False, True])
def test_adapter_uses_local_safe_weights_and_restores_threads(files, monkeypatch, loading_fails):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from numpy import array

    config, _ = probe.load_protocol(files[2])
    threads = []
    calls = []
    torch = SimpleNamespace(
        get_num_threads=lambda: 8,
        set_num_threads=threads.append,
        inference_mode=nullcontext,
    )

    class Inputs(dict):
        input_ids = "synthetic_tokens"

        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        def __call__(self, **kwargs):
            assert kwargs["text"] == "water stain."
            assert kwargs["images"].size == (48, 40)
            return Inputs()

        def post_process_grounded_object_detection(self, outputs, tokens, **kwargs):
            assert tokens == "synthetic_tokens"
            assert kwargs == {"threshold": 0.25, "text_threshold": 0.25, "target_sizes": [(40, 48)]}
            return [
                {
                    "boxes": array([[-1, 2, 49, 39]]),
                    "scores": array([0.7]),
                    "text_labels": ["water stain"],
                }
            ]

    class Model:
        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return "synthetic_outputs"

    def loader(kind, root, **kwargs):
        calls.append(kwargs)
        assert root == files[1]
        if kind == "model" and loading_fails:
            raise OSError("Incomplete model")
        return Model() if kind == "model" else Processor()

    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(
            from_pretrained=lambda *a, **kw: loader("processor", *a, **kw)
        ),
        AutoModelForZeroShotObjectDetection=SimpleNamespace(
            from_pretrained=lambda *a, **kw: loader("model", *a, **kw)
        ),
    )
    monkeypatch.setattr(probe, "version", lambda name: "test")
    monkeypatch.setattr(
        probe.importlib, "import_module", {"torch": torch, "transformers": transformers}.get
    )
    if loading_fails:
        with pytest.raises(OSError, match="Incomplete"):
            probe.GroundingEngine(files[1], config)
    else:
        engine = probe.GroundingEngine(files[1], config)
        predicted = engine.predict(Image.new("RGB", (48, 40)))
        assert predicted[0]["bbox_xyxy"] == [0, 2, 48, 39]
        assert predicted[0]["raw_bbox_xyxy"] == [-1, 2, 49, 39]
        assert predicted[0]["score"] == 0.7
        engine.close()
    assert threads == [4, 8]
    assert calls == [
        {"local_files_only": True, "trust_remote_code": False},
        {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True},
    ]
