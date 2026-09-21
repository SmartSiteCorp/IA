"""Les petits jeux ci-dessous testent le logiciel, pas la qualité de l'IA."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from smartsite_ia import learning, model_cli, training_data
from smartsite_ia.importer import write_json
from smartsite_ia.model_assets import CLASS_NAMES, MODEL_VERSION, PRETRAINED


@pytest.fixture
def training_inputs(tmp_path):
    root = tmp_path / "corpus"
    records = []
    for split, count, offset in (("train", 8, 0), ("valid", 4, 20), ("test", 2, 30)):
        (root / split).mkdir(parents=True)
        images, annotations = [], []
        for i in range(count):
            sample_id = f"easy_{offset + i:04}"
            path = root / split / f"{sample_id}.jpg"
            Image.new("RGB", (64, 64), (i * 10, 100, 130)).save(path)
            records.append(
                {
                    "id": sample_id,
                    "split": split,
                    "difficulty": "Easy",
                    "content_group": f"easy_{offset + i // 2 * 2:04}",
                    "source_classes": [0, 1],
                    "image": f"{split}/{sample_id}.jpg",
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            images.append({"id": i, "file_name": path.name, "width": 64, "height": 64})
            for label in (1, 2):
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": i,
                        "category_id": label,
                        "segmentation": [[1, 1, 20, 1, 20, 20, 1, 20]],
                        "bbox": [1, 1, 19, 19],
                        "area": 361,
                        "iscrowd": 0,
                    }
                )
        write_json(
            root / split / "_annotations.coco.json",
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES, 1)],
            },
        )
    write_json(root / "manifest.json", {"schema_version": 1, "records": records})
    write_json(
        root / "report.json",
        {
            "schema_version": 1,
            "dataset": "damsegment_v1",
            "approved_for_training": True,
            "usage": "experimental_training_only",
            "records": records,
        },
    )
    config = {
        "schema_version": 1,
        "purpose": "smoke",
        "seed": 42,
        "epochs": 1,
        "train_images": 4,
        "valid_images": 2,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "lr": 0.0001,
        "corpus_sha256": {
            name: learning.file_hash(root / name) for name in training_data.CORPUS_FILES
        },
    }
    path = tmp_path / "config.json"
    write_json(path, config)
    return root, path, config


def repin(root, config):
    config["corpus_sha256"] = {
        name: learning.file_hash(root / name) for name in training_data.CORPUS_FILES
    }


def test_training_selection_is_reproducible_grouped_and_never_reads_test(
    training_inputs, tmp_path, monkeypatch
):
    root, path, config = training_inputs
    original = training_data.read_local

    def guarded(folder, relative, maximum):
        assert not str(relative).startswith("test/")
        return original(folder, relative, maximum)

    monkeypatch.setattr(training_data, "read_local", guarded)
    config, _ = training_data.load_training_config(path)
    first = training_data.prepare_training_inputs(root, tmp_path / "one", config)
    second = training_data.prepare_training_inputs(root, tmp_path / "two", config)
    assert first == second
    assert not first["test_used"] and not (tmp_path / "one/test").exists()
    assert len(first["splits"]["train"]["records"]) == 4
    groups = {}
    for split, details in first["splits"].items():
        for record in details["records"]:
            groups.setdefault(record["content_group"], []).append(split)
            assert (tmp_path / "one" / record["image"]).read_bytes() == (
                root / record["image"]
            ).read_bytes()
    assert all(len(v) == 2 and len(set(v)) == 1 for v in groups.values())


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", True),
        ("seed", -1),
        ("epochs", 0),
        ("epochs", 1001),
        ("lr", float("nan")),
        ("lr", 0),
        ("lr", True),
        ("purpose", "production"),
        ("run_test", True),
        ("corpus_sha256", {}),
        ("corpus_sha256", {"bad": "bad"}),
        ("class_names", None),
        ("class_names", ["person"]),
        ("class_names", ["crack", "crack"]),
        ("checkpoint_selection", "test_score"),
        ("checkpoint_selection", None),
        ("checkpoint_selection", True),
    ],
)
def test_training_config_rejects_invalid_options(training_inputs, key, value):
    _, path, config = training_inputs
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        training_data.load_training_config(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "unapproved",
        "changed_manifest",
        "group_leak",
        "duplicate_id",
        "unsafe_id",
        "invalid_split",
        "wrong_image_path",
        "wrong_categories",
        "missing_coco_image",
        "missing_class",
    ],
)
def test_training_inputs_refuse_inconsistent_corpus(training_inputs, tmp_path, mutation):
    root, _, config = training_inputs
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    coco = json.loads((root / "train/_annotations.coco.json").read_text())
    records = manifest["records"]
    if mutation == "unapproved":
        report["approved_for_training"] = False
    elif mutation == "group_leak":
        records[-1]["content_group"] = records[0]["content_group"]
    elif mutation == "duplicate_id":
        records[1]["id"] = records[0]["id"]
    elif mutation == "unsafe_id":
        records[0]["id"] = "../oops"
    elif mutation == "invalid_split":
        records[0]["split"] = "other"
    elif mutation == "wrong_image_path":
        for record in records:
            record["image"] = "../outside.jpg"
    elif mutation == "wrong_categories":
        coco["categories"][0]["name"] = "person"
    elif mutation == "missing_coco_image":
        coco["images"] = []
    elif mutation == "missing_class":
        coco["annotations"] = [a for a in coco["annotations"] if a["category_id"] == 1]
    if mutation != "changed_manifest":
        report["records"] = records
    else:
        records[0]["reason"] = "changed"
    write_json(root / "manifest.json", manifest)
    write_json(root / "report.json", report)
    write_json(root / "train/_annotations.coco.json", coco)
    repin(root, config)
    with pytest.raises(ValueError):
        training_data.prepare_training_inputs(root, tmp_path / "bad", config)
    assert not (tmp_path / "bad").exists()


def test_changed_pinned_file_or_photo_refused(training_inputs, tmp_path):
    root, _, config = training_inputs
    manifest = root / "manifest.json"
    raw = manifest.read_bytes()
    manifest.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="corpus changed"):
        training_data.prepare_training_inputs(root, tmp_path / "bad", config)
    manifest.write_bytes(raw)
    for photo in (root / "train").glob("*.jpg"):
        photo.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="photo changed"):
        training_data.prepare_training_inputs(root, tmp_path / "bad", config)


def test_group_budget_never_splits_pair(training_inputs):
    root, _, _ = training_inputs
    records = json.loads((root / "manifest.json").read_text())["records"][:8]
    assert len(training_data.select_groups(records, 3, 10)) == 2
    for r in records:
        r["source_classes"] = [0]
    with pytest.raises(ValueError, match="both classes"):
        training_data.select_groups(records, 4, 10)


@pytest.fixture
def fake_engine(monkeypatch):
    calls = []
    monkeypatch.setattr(learning, "runtime", lambda device: {"device": device})
    monkeypatch.setattr(learning, "verify_archive", lambda *args: PRETRAINED.sha256)
    original_hash = learning.file_hash
    monkeypatch.setattr(
        learning,
        "file_hash",
        lambda path, maximum=2_000_000_000: (
            PRETRAINED.sha256 if path.name == "weights" else original_hash(path, maximum)
        ),
    )

    def fit(data, output, weights, config, device):
        assert not (data / "test").exists()
        calls.append((config, device))
        output.mkdir()
        (output / "checkpoint_best_total.pth").write_bytes(b"synthetic checkpoint, no real model")
        (output / "metrics.csv").write_text("epoch,train/loss,val/segm_mAP_50_95\n0,12,0.01\n")

    monkeypatch.setattr(learning, "fit_engine", fit)
    return calls


def test_run_records_success_and_verifies_checkpoint(training_inputs, tmp_path, fake_engine):
    root, config, _ = training_inputs
    out = tmp_path / "run"
    report = learning.train_model(root, out, config, tmp_path / "weights", "cpu")
    assert report["status"] == "completed" and not report["test_used"]
    assert report["checkpoint_selection"] == "best_validation"
    assert report["selected"]["train"]["images"] == 4
    assert len(fake_engine) == 1
    assert learning.verify_run(out)[0] == report
    (out / report["checkpoint"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="checkpoint changed"):
        learning.verify_run(out)
    with pytest.raises(FileExistsError):
        learning.train_model(root, out, config, tmp_path / "weights", "cpu")


def test_last_epoch_selection_is_validated_and_recorded(training_inputs, tmp_path, fake_engine):
    root, config_path, config = training_inputs
    config["checkpoint_selection"] = "last_epoch"
    write_json(config_path, config)
    report = learning.train_model(root, tmp_path / "run", config_path, tmp_path / "weights", "cpu")
    assert report["checkpoint_selection"] == "last_epoch"
    assert fake_engine[0][0]["checkpoint_selection"] == "last_epoch"


@pytest.mark.parametrize(
    "error,status", [(RuntimeError("failed"), "failed"), (KeyboardInterrupt(), "interrupted")]
)
def test_failed_runs_keep_state_without_success(
    training_inputs, tmp_path, fake_engine, monkeypatch, error, status
):
    root, config, _ = training_inputs

    def fail(*args):
        raise error

    monkeypatch.setattr(learning, "fit_engine", fail)
    with pytest.raises(type(error)):
        learning.train_model(root, tmp_path / "run", config, tmp_path / "weights", "cpu")
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == status
    with pytest.raises(ValueError, match="completed"):
        learning.verify_run(tmp_path / "run")


def test_run_refuses_output_in_corpus(training_inputs, fake_engine):
    root, config, _ = training_inputs
    with pytest.raises(ValueError, match="separate"):
        learning.train_model(root, root / "run", config, root / "weights", "cpu")


@pytest.mark.parametrize("content", ["train/loss,val/segm_mAP_50_95\nNaN,1\n", "epoch\n1\n"])
def test_invalid_metrics_are_not_success(tmp_path, content):
    path = tmp_path / "metrics.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        learning.summarize_metrics(path)


def test_file_hash_refuses_symlink_and_size(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"123")
    link = tmp_path / "link"
    link.symlink_to(path)
    for p, maximum in [(link, 10), (path, 2)]:
        with pytest.raises(ValueError):
            learning.file_hash(p, maximum)


def test_run_refuses_linked_checkpoint_directory(training_inputs, tmp_path, fake_engine):
    root, config, _ = training_inputs
    out = tmp_path / "run"
    learning.train_model(root, out, config, tmp_path / "weights", "cpu")
    (out / "checkpoints").rename(tmp_path / "outside")
    (out / "checkpoints").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        learning.verify_run(out)


def test_runtime_checks_device_version_and_disables_remote_access(monkeypatch):
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(learning.importlib, "import_module", lambda name: torch)
    monkeypatch.setattr(
        learning, "version", lambda name: MODEL_VERSION if name == "rfdetr" else "test"
    )
    assert learning.runtime("cpu")["available"]["cpu"]
    for device in ["cuda", "mps", "unknown"]:
        with pytest.raises(ValueError, match="unavailable"):
            learning.runtime(device)
    monkeypatch.setattr(learning, "version", lambda name: "wrong")
    with pytest.raises(ValueError, match="version"):
        learning.runtime("cpu")


@pytest.mark.parametrize(
    "selection,skipped", [(None, 0), ("best_validation", 0), ("last_epoch", 2)]
)
def test_engine_adapter_passes_explicit_safe_options(tmp_path, monkeypatch, selection, skipped):
    captured = {}

    class Engine:
        def __init__(self, **kwargs):
            captured["model"] = kwargs

        def train(self, **kwargs):
            captured["train"] = kwargs

    modules = {
        "rfdetr": SimpleNamespace(RFDETRSegMedium=Engine),
        "pytorch_lightning": SimpleNamespace(seed_everything=lambda *a, **kw: None),
    }
    monkeypatch.setattr(learning.importlib, "import_module", lambda name: modules[name])
    config = {
        "seed": 42,
        "epochs": 3,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "lr": 0.0001,
        "purpose": "smoke",
    }
    if selection is not None:
        config["checkpoint_selection"] = selection
    learning.fit_engine(tmp_path / "data", tmp_path / "run", tmp_path / "weights", config, "cpu")
    assert captured["model"]["amp"] is False
    assert all(
        captured["train"][key] is False
        for key in ["run_test", "wandb", "mlflow", "clearml", "tensorboard"]
    )
    assert captured["train"]["notes"]["classes"] == list(CLASS_NAMES)
    assert captured["train"]["skip_best_epochs"] == skipped


@pytest.mark.parametrize("command", ["weights", "doctor", "train", "predict"])
def test_model_cli_dispatch(command, monkeypatch, capsys):
    arguments = {
        "weights": ["--output", "weights"],
        "doctor": ["--device", "cpu"],
        "train": [
            "corpus",
            "--config",
            "cfg",
            "--weights",
            "weights",
            "--output",
            "run",
            "--device",
            "cpu",
        ],
        "predict": ["photo.jpg", "--run", "run", "--output", "prediction", "--device", "cpu"],
    }
    monkeypatch.setattr("sys.argv", ["smartsite-model", command, *arguments[command]])
    monkeypatch.setattr(model_cli, "download_archive", lambda *a: Path("weights"))
    monkeypatch.setattr(model_cli, "runtime", lambda *a: {"device": "cpu"})
    monkeypatch.setattr(model_cli, "train_model", lambda *a: {"status": "completed"})
    monkeypatch.setattr(model_cli, "predict_photo", lambda *a: {"predictions": []})
    assert model_cli.main() == 0
    assert json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("error", [ImportError("training"), ValueError("invalid")])
def test_model_cli_explains_failure(monkeypatch, capsys, error):
    def fail(*args):
        raise error

    monkeypatch.setattr("sys.argv", ["smartsite-model", "doctor", "--device", "cpu"])
    monkeypatch.setattr(model_cli, "runtime", fail)
    assert model_cli.main() == 1
    assert capsys.readouterr().err


def test_resume_keeps_data_and_restores_only_verified_checkpoint(
    training_inputs, tmp_path, fake_engine, monkeypatch
):
    root, cfg_path, _ = training_inputs
    out = tmp_path / "run"
    original = learning.fit_engine

    def interrupted(data, output, weights, config, device):
        output.mkdir()
        (output / "last.ckpt").write_bytes(b"synthetic full checkpoint")
        raise KeyboardInterrupt()

    monkeypatch.setattr(learning, "fit_engine", interrupted)
    with pytest.raises(KeyboardInterrupt):
        learning.train_model(root, out, cfg_path, tmp_path / "weights", "cpu")
    selection = (out / "data/selection.json").read_bytes()

    def resumed(data, output, weights, config, device):
        assert config["_resume"] == str((output / "last.ckpt").resolve())
        assert learning.os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"
        (output / "checkpoint_best_total.pth").write_bytes(b"new weights")
        (output / "metrics.csv").write_text("epoch,train/loss,val/segm_mAP_50_95\n0,12,0.1\n")

    monkeypatch.setattr(learning, "fit_engine", resumed)
    result = learning.train_model(root, out, cfg_path, tmp_path / "weights", "cpu", resume=True)
    assert result["status"] == "completed"
    assert result["attempts"][0]["status"] == "interrupted"
    assert (out / "data/selection.json").read_bytes() == selection
    assert result["process_peak_rss_bytes"] > 0
    with pytest.raises(ValueError, match="interrupted"):
        learning.train_model(root, out, cfg_path, tmp_path / "weights", "cpu", resume=True)
    monkeypatch.setattr(learning, "fit_engine", original)


@pytest.mark.parametrize(
    "changed", ["checkpoint", "selection", "annotation", "photo", "device", "config"]
)
def test_resume_refuses_changed_inputs(
    training_inputs, tmp_path, fake_engine, monkeypatch, changed
):
    root, cfg_path, config = training_inputs
    out = tmp_path / "run"

    def interrupted(data, output, *args):
        output.mkdir()
        (output / "last.ckpt").write_bytes(b"synthetic full checkpoint")
        raise KeyboardInterrupt()

    monkeypatch.setattr(learning, "fit_engine", interrupted)
    with pytest.raises(KeyboardInterrupt):
        learning.train_model(root, out, cfg_path, tmp_path / "weights", "cpu")
    targets = {
        "checkpoint": out / "checkpoints/last.ckpt",
        "selection": out / "data/selection.json",
        "annotation": out / "data/train/_annotations.coco.json",
        "photo": next((out / "data/train").glob("*.jpg")),
    }
    if changed in targets:
        targets[changed].write_bytes(b"changed")
    elif changed == "config":
        config["epochs"] = 2
        write_json(cfg_path, config)
    with pytest.raises(ValueError):
        learning.train_model(
            root,
            out,
            cfg_path,
            tmp_path / "weights",
            "cuda" if changed == "device" else "cpu",
            resume=True,
        )


def test_early_engine_return_cannot_be_reported_as_success(training_inputs, tmp_path, fake_engine):
    root, path, config = training_inputs
    config["epochs"] = 2
    write_json(path, config)
    with pytest.raises(RuntimeError, match="final epoch"):
        learning.train_model(root, tmp_path / "run", path, tmp_path / "weights", "cpu")
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == "failed"


def test_safe_loading_environment_is_restored(training_inputs, tmp_path, fake_engine, monkeypatch):
    root, path, _ = training_inputs
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")
    learning.train_model(root, tmp_path / "run", path, tmp_path / "weights", "cpu")
    assert learning.os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"


@pytest.mark.parametrize("link", ["data", "data/train", "checkpoints"])
def test_resume_refuses_linked_folders(training_inputs, tmp_path, fake_engine, monkeypatch, link):
    root, path, cfg = training_inputs
    out = tmp_path / "run"

    def interrupted(data, output, *args):
        output.mkdir()
        (output / "last.ckpt").write_bytes(b"synthetic checkpoint")
        raise KeyboardInterrupt()

    monkeypatch.setattr(learning, "fit_engine", interrupted)
    with pytest.raises(KeyboardInterrupt):
        learning.train_model(root, out, path, tmp_path / "weights", "cpu")
    target = out / link
    target.rename(tmp_path / "outside")
    target.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError):
        learning.train_model(root, out, path, tmp_path / "weights", "cpu", resume=True)


def test_last_train_log_does_not_impersonate_a_completed_validation(
    training_inputs, tmp_path, fake_engine, monkeypatch
):
    root, path, config = training_inputs
    config["epochs"] = 2
    write_json(path, config)

    def partial_fit(data, output, *args):
        output.mkdir()
        (output / "checkpoint_best_total.pth").write_bytes(b"earlier best checkpoint")
        (output / "metrics.csv").write_text("epoch,train/loss,val/segm_mAP_50_95\n0,12,.1\n1,10,\n")

    monkeypatch.setattr(learning, "fit_engine", partial_fit)
    with pytest.raises(RuntimeError, match="final epoch"):
        learning.train_model(root, tmp_path / "run", path, tmp_path / "weights", "cpu")
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == "failed"


@pytest.fixture
def completed_parent(training_inputs, tmp_path, fake_engine):
    root, path, config = training_inputs
    parent = tmp_path / "parent"
    report = learning.train_model(root, parent, path, tmp_path / "weights", "cpu")
    report["runtime"]["versions"] = {"rfdetr": MODEL_VERSION}
    write_json(parent / "run.json", report)
    config["initial_checkpoint_sha256"] = report["checkpoint_sha256"]
    write_json(path, config)
    return parent, report


def test_fine_tuning_keeps_parent_and_records_cold_optimizer(
    training_inputs, tmp_path, completed_parent, monkeypatch
):
    root, path, _ = training_inputs
    parent, parent_report = completed_parent
    original = {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()}

    def fit(data, output, weights, config, device):
        assert weights == parent / parent_report["checkpoint"]
        assert "_resume" not in config
        output.mkdir()
        (output / "checkpoint_best_total.pth").write_bytes(b"new learned weights")
        (output / "metrics.csv").write_text("epoch,train/loss,val/segm_mAP_50_95\n0,10,.2\n")

    monkeypatch.setattr(learning, "fit_engine", fit)
    report = learning.train_model(root, tmp_path / "child", path, None, "cpu", from_run=parent)
    assert report["initialization"] == {
        "mode": "fine_tune",
        "checkpoint_sha256": parent_report["checkpoint_sha256"],
        "parent_run": str(parent.resolve()),
        "parent_report_sha256": learning.file_hash(parent / "run.json"),
        "optimizer_restored": False,
        "scheduler_restored": False,
        "epoch_numbering": "Restarted at zero in this new run",
    }
    assert all(p.read_bytes() == raw for p, raw in original.items())
    assert learning.verify_run(tmp_path / "child")[0] == report


@pytest.mark.parametrize("change", ["hash", "corpus", "version", "classes", "status", "weights"])
def test_fine_tuning_refuses_changed_parent_before_output(
    training_inputs, tmp_path, completed_parent, change
):
    root, path, _ = training_inputs
    parent, report = completed_parent
    if change == "hash":
        report["checkpoint_sha256"] = "a" * 64
    elif change == "corpus":
        report["config"]["corpus_sha256"]["manifest.json"] = "a" * 64
    elif change == "version":
        report["runtime"]["versions"]["rfdetr"] = "unknown"
    elif change == "classes":
        report["class_names"] = ["other"]
    elif change == "status":
        report["status"] = "interrupted"
    else:
        (parent / report["checkpoint"]).write_bytes(b"changed")
    write_json(parent / "run.json", report)
    with pytest.raises(ValueError):
        learning.train_model(root, tmp_path / "child", path, None, "cpu", from_run=parent)
    assert not (tmp_path / "child").exists()


@pytest.mark.parametrize("pin", [None, "", "x" * 64, True])
def test_fine_tuning_requires_pinned_hash(training_inputs, tmp_path, completed_parent, pin):
    root, path, config = training_inputs
    parent, _ = completed_parent
    config["initial_checkpoint_sha256"] = pin
    write_json(path, config)
    with pytest.raises(ValueError, match="initial checkpoint"):
        learning.train_model(root, tmp_path / "child", path, None, "cpu", from_run=parent)


def test_fine_tuning_refuses_parent_output_and_ambiguous_origin(
    training_inputs, tmp_path, completed_parent
):
    root, path, config = training_inputs
    parent, _ = completed_parent
    for out, weights, source in [
        (parent / "child", None, parent),
        (tmp_path / "child", tmp_path / "weights", parent),
        (tmp_path / "child", None, None),
        (tmp_path / "child", tmp_path / "weights", None),
    ]:
        with pytest.raises(ValueError):
            learning.train_model(root, out, path, weights, "cpu", from_run=source)
    config.pop("initial_checkpoint_sha256")
    write_json(path, config)
    with pytest.raises(ValueError, match="differs"):
        learning.train_model(root, tmp_path / "child", path, None, "cpu", from_run=parent)


@pytest.mark.parametrize("names", [list(CLASS_NAMES), ["crack"], ["surface_loss"]])
def test_fine_tuning_adapter_keeps_learned_classes(tmp_path, monkeypatch, names):
    captured = {}

    class Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def train(self, **kwargs):
            assert kwargs["resume"] is None

    modules = {
        "rfdetr": SimpleNamespace(RFDETRSegMedium=Engine),
        "pytorch_lightning": SimpleNamespace(seed_everything=lambda *a, **kw: None),
    }
    monkeypatch.setattr(learning.importlib, "import_module", lambda name: modules[name])
    config = {
        "seed": 42,
        "epochs": 1,
        "batch_size": 1,
        "grad_accum_steps": 2,
        "lr": 0.0001,
        "purpose": "experiment",
        "initial_checkpoint_sha256": "a" * 64,
        "class_names": names,
    }
    learning.fit_engine(tmp_path / "data", tmp_path / "out", tmp_path / "parent.pth", config, "cpu")
    assert captured["num_classes"] == len(names)
    assert captured["pretrain_weights"] == str((tmp_path / "parent.pth").resolve())


def test_fine_tuning_cli_forwards_parent_and_resume(monkeypatch):
    def train(*args, **kwargs):
        assert args[3] is None
        assert kwargs == {"from_run": Path("parent"), "resume": True}
        return {"status": "completed"}

    monkeypatch.setattr(model_cli, "train_model", train)
    monkeypatch.setattr(
        "sys.argv",
        [
            "smartsite-model",
            "train",
            "corpus",
            "--config",
            "cfg",
            "--from-run",
            "parent",
            "--output",
            "child",
            "--device",
            "cpu",
            "--resume",
        ],
    )
    assert model_cli.main() == 0


def test_interrupted_fine_tuning_restores_its_own_optimizer_checkpoint(
    training_inputs, tmp_path, completed_parent, monkeypatch
):
    root, path, _ = training_inputs
    parent, _ = completed_parent
    out = tmp_path / "child"

    def interrupt(data, output, *args):
        output.mkdir()
        (output / "last.ckpt").write_bytes(b"child optimizer checkpoint")
        raise KeyboardInterrupt()

    monkeypatch.setattr(learning, "fit_engine", interrupt)
    with pytest.raises(KeyboardInterrupt):
        learning.train_model(root, out, path, None, "cpu", from_run=parent)

    def resumed(data, output, weights, config, device):
        assert config["_resume"] == str((out / "checkpoints/last.ckpt").resolve())
        assert Path(config["_resume"]).read_bytes() == b"child optimizer checkpoint"
        (output / "checkpoint_best_total.pth").write_bytes(b"child learned weights")
        (output / "metrics.csv").write_text("epoch,train/loss,val/segm_mAP_50_95\n0,9,.3\n")

    monkeypatch.setattr(learning, "fit_engine", resumed)
    result = learning.train_model(root, out, path, None, "cpu", from_run=parent, resume=True)
    assert result["status"] == "completed"
    assert result["attempts"][0]["status"] == "interrupted"
    assert result["initialization"]["optimizer_restored"] is False
    assert result["attempts"][0]["resume_checkpoint_sha256"]


def test_parent_changes_during_fine_tuning_are_reported_as_failure(
    training_inputs, tmp_path, completed_parent, monkeypatch
):
    root, path, _ = training_inputs
    parent, _ = completed_parent

    def fit(data, output, weights, config, device):
        output.mkdir()
        weights.write_bytes(b"parent modified externally")
        (output / "checkpoint_best_total.pth").write_bytes(b"child weights")

    monkeypatch.setattr(learning, "fit_engine", fit)
    with pytest.raises(ValueError, match="Initial checkpoint changed"):
        learning.train_model(root, tmp_path / "child", path, None, "cpu", from_run=parent)
    assert json.loads((tmp_path / "child/run.json").read_text())["status"] == "failed"


def test_resume_refuses_different_parent_report(
    training_inputs, tmp_path, completed_parent, monkeypatch
):
    root, path, _ = training_inputs
    parent, parent_report = completed_parent
    out = tmp_path / "child"

    def interrupt(data, output, *args):
        output.mkdir()
        (output / "last.ckpt").write_bytes(b"child optimizer checkpoint")
        raise KeyboardInterrupt()

    monkeypatch.setattr(learning, "fit_engine", interrupt)
    with pytest.raises(KeyboardInterrupt):
        learning.train_model(root, out, path, None, "cpu", from_run=parent)
    parent_report["purpose"] = "changed"
    write_json(parent / "run.json", parent_report)
    with pytest.raises(ValueError, match="initialization changed"):
        learning.train_model(root, out, path, None, "cpu", from_run=parent, resume=True)


@pytest.fixture
def mixed_training_inputs(training_inputs):
    """Deux photos sans fissure : une autre anomalie, puis aucune annotation."""
    root, path, config = training_inputs
    manifest = json.loads((root / "manifest.json").read_text())
    for split in ("train", "valid"):
        target = root / split / "_annotations.coco.json"
        doc = json.loads(target.read_text())
        doc["annotations"] = [
            a
            for a in doc["annotations"]
            if a["image_id"] != 1 and not (a["image_id"] == 0 and a["category_id"] == 1)
        ]
        write_json(target, doc)
        records = [r for r in manifest["records"] if r["split"] == split]
        records[0]["source_classes"], records[1]["source_classes"] = [1], []
    report = json.loads((root / "report.json").read_text())
    report["records"] = manifest["records"]
    write_json(root / "manifest.json", manifest)
    write_json(root / "report.json", report)
    config.update(train_images=8, valid_images=4)
    repin(root, config)
    write_json(path, config)
    return root, path, config


def test_specialist_keeps_negative_photos_groups_originals_and_identical_selection(
    mixed_training_inputs, tmp_path, monkeypatch
):
    root, _, config = mixed_training_inputs
    original = {name: (root / name).read_bytes() for name in training_data.CORPUS_FILES}
    read = training_data.read_local

    def guarded(folder, relative, limit):
        assert not str(relative).startswith("test/")
        return read(folder, relative, limit)

    monkeypatch.setattr(training_data, "read_local", guarded)
    baseline = training_data.prepare_training_inputs(root, tmp_path / "both", config)
    config = {**config, "class_names": ["crack"]}
    first = training_data.prepare_training_inputs(root, tmp_path / "crack", config)
    second = training_data.prepare_training_inputs(root, tmp_path / "repeat", config)
    assert first == second and first["class_names"] == ["crack"]
    assert first["configuration"] == config
    assert first["source_corpus_sha256"] == config["corpus_sha256"]
    for split in ("train", "valid"):
        details = first["splits"][split]
        assert details["records"] == baseline["splits"][split]["records"]
        assert details["images_without_target_annotations"] == 2
        assert details["annotations"] == details["images"] - 2
        assert details["excluded_annotations"] == details["images"] - 1
        target = f"{split}/_annotations.coco.json"
        doc = json.loads((tmp_path / "crack" / target).read_text())
        source = json.loads(original[target])
        assert doc["images"] == source["images"]
        assert doc["annotations"] == [a for a in source["annotations"] if a["category_id"] == 1]
        assert (tmp_path / "crack" / target).read_bytes() == (
            tmp_path / "repeat" / target
        ).read_bytes()
        for record in details["records"]:
            assert (tmp_path / "crack" / record["image"]).read_bytes() == (
                root / record["image"]
            ).read_bytes()
    assert all((root / name).read_bytes() == raw for name, raw in original.items())
    assert not (tmp_path / "crack/test").exists()
    page = (tmp_path / "crack/index.html").read_text()
    assert "pas les détections" in page and "garantie sans fissure" in page
    assert page.count("<figure>") == 4
    assert "test/easy_" not in page


@pytest.mark.parametrize("names", [["crack"], ["surface_loss"], list(CLASS_NAMES)])
def test_class_configuration_reaches_the_run_and_its_inputs(
    training_inputs, tmp_path, fake_engine, names
):
    root, path, config = training_inputs
    config["class_names"] = names
    write_json(path, config)
    assert training_data.load_training_config(path)[0]["class_names"] == names
    out = tmp_path / "run"
    report = learning.train_model(root, out, path, tmp_path / "weights", "cpu")
    assert report["class_names"] == names
    assert fake_engine[0][0]["class_names"] == names
    assert learning.verify_run(out)[0] == report
    doc = json.loads((out / "data/train/_annotations.coco.json").read_text())
    assert [c["name"] for c in doc["categories"]] == names
    assert {a["category_id"] for a in doc["annotations"]} == set(range(1, len(names) + 1))
    report["class_names"] = ["surface_loss"] if names == ["crack"] else ["crack"]
    write_json(out / "run.json", report)
    with pytest.raises(ValueError, match="completed"):
        learning.verify_run(out)


def test_fine_tuning_cannot_silently_change_a_parents_class_mapping(
    training_inputs, tmp_path, completed_parent
):
    root, path, config = training_inputs
    parent, _ = completed_parent
    config["class_names"] = ["crack"]
    write_json(path, config)
    with pytest.raises(ValueError, match="classes"):
        learning.train_model(root, tmp_path / "child", path, None, "cpu", from_run=parent)
    assert not (tmp_path / "child").exists()


def test_prepare_cli_runs_without_loading_the_model(training_inputs, tmp_path, monkeypatch, capsys):
    root, path, config = training_inputs
    config["class_names"] = ["crack"]
    write_json(path, config)
    out = tmp_path / "prepared"

    def forbidden(*args):
        raise AssertionError("No model or training during data preparation")

    monkeypatch.setattr(model_cli, "runtime", forbidden)
    monkeypatch.setattr(model_cli, "train_model", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        ["smartsite-model", "prepare", str(root), "--config", str(path), "--output", str(out)],
    )
    assert model_cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["class_names"] == ["crack"]
    assert result["splits"]["train"]["images"] == 4
    assert (out / "selection.json").exists()
    assert not (out / "checkpoints").exists()
