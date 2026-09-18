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
    assert report["selected"]["train"]["images"] == 4
    assert len(fake_engine) == 1
    assert learning.verify_run(out)[0] == report
    (out / report["checkpoint"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="checkpoint changed"):
        learning.verify_run(out)
    with pytest.raises(FileExistsError):
        learning.train_model(root, out, config, tmp_path / "weights", "cpu")


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


def test_engine_adapter_passes_explicit_safe_options(tmp_path, monkeypatch):
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
        "epochs": 1,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "lr": 0.0001,
        "purpose": "smoke",
    }
    learning.fit_engine(tmp_path / "data", tmp_path / "run", tmp_path / "weights", config, "cpu")
    assert captured["model"]["amp"] is False
    assert all(
        captured["train"][key] is False
        for key in ["run_test", "wandb", "mlflow", "clearml", "tensorboard"]
    )
    assert captured["train"]["notes"]["classes"] == list(CLASS_NAMES)


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
