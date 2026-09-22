"""Vérifier la continuation et les arrêts sans entraîner le détecteur dans les tests courants."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_training import training_inputs as training_inputs

from smartsite_ia import continuation as c
from smartsite_ia import learning, model_cli, threshold_study, validation
from smartsite_ia.importer import write_json
from smartsite_ia.model_assets import CLASS_NAMES, MODEL_NAME
from smartsite_ia.training_data import prepare_training_inputs


@pytest.fixture
def parent_run(training_inputs, tmp_path, monkeypatch):
    root, _, config = training_inputs
    config["checkpoint_selection"] = "last_epoch"
    parent = tmp_path / "parent"
    prepare_training_inputs(root, parent / "data", config)
    checkpoints = parent / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "checkpoint_best_total.pth").write_bytes(b"parent weights")
    (checkpoints / "last.ckpt").write_bytes(b"parent full state")
    versions = {"rfdetr": "test", "torch": "test"}
    environment = {"device": "cpu", "versions": versions}
    for module in (learning, c):
        monkeypatch.setattr(module, "runtime", lambda device: environment)
    engine = {
        "train_config": {
            **config,
            "lr_scheduler": "step",
            "lr_scheduler_kwargs": {},
            "warmup_epochs": 0.0,
            "use_ema": False,
            "early_stopping": False,
            "drop_path": 0.0,
            "lr_drop": 100,
            "lr_scheduler_interval": "step",
            "optimizer": "adamw",
            "optimizer_kwargs": {},
            "augmentation_backend": "torchvision",
            "multi_scale": False,
            "expanded_scales": False,
            "scale_jitter": False,
        }
    }
    write_json(checkpoints / "training_config.json", engine)
    report = {
        "schema_version": 1,
        "status": "completed",
        "model": MODEL_NAME,
        "config": config,
        "class_names": list(CLASS_NAMES),
        "runtime": environment,
        "checkpoint": "checkpoints/checkpoint_best_total.pth",
        "checkpoint_sha256": c.file_hash(checkpoints / "checkpoint_best_total.pth"),
        "selection_sha256": c.file_hash(parent / "data/selection.json"),
        "input_annotations_sha256": {
            s: c.file_hash(parent / f"data/{s}/_annotations.coco.json") for s in ("train", "valid")
        },
    }
    write_json(parent / "run.json", report)
    state = {
        "epoch": 0,
        "global_step": 4,
        "state_dict": {"model.x": 42},
        "loops": {"fit_loop": {"position": 1}},
        "optimizer_states": [
            {"state": {0: {"step": 4, "exp_avg": 7}}, "param_groups": [{"params": [0]}]}
        ],
        "lr_schedulers": [{"last_epoch": 4, "base_lrs": [0.0001]}],
        "callbacks": {"ModelCheckpoint{test}": {"dirpath": str(checkpoints)}},
    }

    def load(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return copy.deepcopy(state if Path(path).name == "last.ckpt" else {"model": {"x": 42}})

    fake = SimpleNamespace(
        load=load, equal=lambda a, b: a == b, save=lambda s, f: f.write(json.dumps(s).encode())
    )
    monkeypatch.setattr(c.importlib, "import_module", lambda name: fake)
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "schema_version": 1,
            "parent_report_sha256": c.file_hash(parent / "run.json"),
            "parent_state_sha256": c.file_hash(checkpoints / "last.ckpt"),
            "target_epochs": 3,
        },
    )
    return root, parent, plan, state


def next_config(parent):
    previous = json.loads((parent / "run.json").read_text())["config"]
    return {
        **previous,
        "epochs": previous["epochs"] + 1,
        "continuation_checkpoint_sha256": c.file_hash(parent / "checkpoints/last.ckpt"),
    }


def test_preflight_does_not_create_output_or_train(parent_run, tmp_path, monkeypatch):
    root, parent, plan, _ = parent_run
    monkeypatch.setattr(c, "train_model", lambda *a, **kw: pytest.fail("No training"))
    monkeypatch.setattr(
        validation, "evaluate_validation", lambda *a, **kw: pytest.fail("No inference")
    )
    result = c.extend_training(root, parent, tmp_path / "out", plan, "cpu", check=True)
    assert result["status"] == "ready" and not result["training_started"]
    assert result["protocol"]["first_epoch"] == 2 and result["protocol"]["target_epochs"] == 3
    assert not (tmp_path / "out").exists()


def test_state_copy_preserves_learning_and_removes_parent_paths(parent_run, tmp_path):
    _, parent, _, state = parent_run
    original = (parent / "checkpoints/last.ckpt").read_bytes()
    destination = tmp_path / "state.ckpt"
    c.copy_continuation_state(parent, next_config(parent), destination)
    copied = json.loads(destination.read_text())
    assert copied["callbacks"] == {}
    assert copied["state_dict"] == state["state_dict"]
    assert copied["lr_schedulers"] == state["lr_schedulers"]
    assert copied["global_step"] == 4 and copied["loops"] == state["loops"]
    assert copied["optimizer_states"][0]["state"]["0"] == state["optimizer_states"][0]["state"][0]
    assert (parent / "checkpoints/last.ckpt").read_bytes() == original
    with pytest.raises(FileExistsError):
        c.copy_continuation_state(parent, next_config(parent), destination)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", 1),
        ("global_step", 0),
        ("optimizer_states", []),
        ("lr_schedulers", []),
        ("lr_schedulers", [{"last_epoch": 0}]),
        ("loops", {}),
        ("state_dict", {"model.x": 43}),
        ("callbacks", {"EMA": {}}),
    ],
)
def test_bad_full_state_refused(parent_run, field, value):
    _, parent, _, state = parent_run
    state[field] = value
    with pytest.raises(ValueError):
        c.read_full_state(parent, c.file_hash(parent / "checkpoints/last.ckpt"))


def test_changed_hash_refused_before_loading(parent_run):
    _, parent, _, _ = parent_run
    with pytest.raises(ValueError, match="changed"):
        c.read_full_state(parent, "0" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 3),
        ("lr", 0.001),
        ("class_names", ["crack"]),
        ("checkpoint_selection", "best_validation"),
        ("initial_checkpoint_sha256", "a" * 64),
    ],
)
def test_changed_learning_setup_refused(parent_run, tmp_path, field, value):
    _, parent, _, _ = parent_run
    config = next_config(parent)
    config[field] = value
    with pytest.raises(ValueError):
        c.continuation_origin(parent, tmp_path / "child", config, "cpu")


def test_changed_device_and_engine_settings_refused(parent_run, tmp_path):
    _, parent, _, _ = parent_run
    with pytest.raises(ValueError, match="runtime"):
        c.continuation_origin(parent, tmp_path / "child", next_config(parent), "mps")
    p = parent / "checkpoints/training_config.json"
    doc = json.loads(p.read_text())
    doc["train_config"]["lr_scheduler"] = "cosine"
    write_json(p, doc)
    with pytest.raises(ValueError, match="engine settings"):
        c.continuation_origin(parent, tmp_path / "child", next_config(parent), "cpu")


def test_parent_inputs_and_output_protected(parent_run, tmp_path):
    root, parent, plan, _ = parent_run
    with pytest.raises(ValueError, match="separate"):
        c.continuation_origin(parent, parent / "child", next_config(parent), "cpu")
    with pytest.raises(ValueError, match="outside"):
        c.extend_training(root, parent, root / "child", plan, "cpu", check=True)
    photo = next((parent / "data/train").glob("*.jpg"))
    photo.write_bytes(b"changed")
    with pytest.raises(ValueError, match="photo changed"):
        c.continuation_origin(parent, tmp_path / "child", next_config(parent), "cpu")


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_epochs", True),
        ("target_epochs", 1),
        ("target_epochs", 11),
        ("parent_state_sha256", "bad"),
        ("parent_report_sha256", "f" * 64),
        ("schema_version", True),
        ("unexpected", 1),
    ],
)
def test_plan_rejects_invalid_or_unpinned_budget(parent_run, tmp_path, field, value):
    root, parent, plan, _ = parent_run
    doc = json.loads(plan.read_text())
    doc[field] = value
    write_json(plan, doc)
    with pytest.raises(ValueError):
        c.extend_training(root, parent, tmp_path / "out", plan, "cpu", check=True)
    assert not (tmp_path / "out").exists()


def test_corpus_change_refused(parent_run, tmp_path):
    root, parent, plan, _ = parent_run
    (root / "manifest.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corpus"):
        c.extend_training(root, parent, tmp_path / "out", plan, "cpu", check=True)


def test_concurrent_extension_refused(tmp_path):
    with (
        c.extension_lock(tmp_path),
        pytest.raises(ValueError, match="already running"),
        c.extension_lock(tmp_path),
    ):
        pass
    c.extension_lock(tmp_path)


def test_linked_lock_refused(tmp_path):
    (tmp_path / ".extension.lock").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="linked"), c.extension_lock(tmp_path):
        pass


def test_training_uses_full_state_not_only_weights(parent_run, tmp_path, monkeypatch):
    root, parent, _, _ = parent_run
    config = tmp_path / "next.json"
    write_json(config, next_config(parent))
    out = tmp_path / "child"

    def fit(data, output, weights, settings, device):
        restored = json.loads(Path(settings["_resume"]).read_text())
        assert restored["global_step"] == 4 and restored["callbacks"] == {}
        assert restored["optimizer_states"] and restored["lr_schedulers"]
        output.mkdir()
        (output / "checkpoint_best_total.pth").write_bytes(b"new weights")
        (output / "last.ckpt").write_bytes(b"new full state")
        (output / "metrics.csv").write_text("epoch,train/loss,val/segm_mAP_50_95\n1,2,.2\n")

    monkeypatch.setattr(learning, "fit_engine", fit)
    result = learning.train_model(root, out, config, None, "cpu", continue_run=parent)
    assert result["status"] == "completed"
    assert result["initialization"]["mode"] == "continuation"
    assert result["initialization"]["start_epoch"] == 1
    assert result["resume_checkpoint_sha256"] == c.file_hash(out / "checkpoints/last.ckpt")
    assert result["initial_state_sha256"] == c.file_hash(out / "initial_state.ckpt")
    with pytest.raises(ValueError, match="cannot be mixed"):
        learning.train_model(
            root, tmp_path / "bad", config, Path("weights"), "cpu", continue_run=parent
        )
    with pytest.raises(ValueError, match="requires its full"):
        learning.train_model(root, tmp_path / "bad", config, None, "cpu")


def test_interrupted_continuation_uses_child_state(parent_run, tmp_path, monkeypatch):
    root, parent, _, _ = parent_run
    config = tmp_path / "next.json"
    write_json(config, next_config(parent))
    out = tmp_path / "child"

    def stop(data, output, *args):
        output.mkdir()
        (output / "last.ckpt").write_bytes(b"child state")
        raise KeyboardInterrupt()

    monkeypatch.setattr(learning, "fit_engine", stop)
    with pytest.raises(KeyboardInterrupt):
        learning.train_model(root, out, config, None, "cpu", continue_run=parent)

    def finish(data, output, weights, settings, device):
        assert Path(settings["_resume"]).read_bytes() == b"child state"
        (output / "checkpoint_best_total.pth").write_bytes(b"new weights")
        (output / "metrics.csv").write_text("epoch,train/loss,val/segm_mAP_50_95\n1,2,.2\n")

    monkeypatch.setattr(learning, "fit_engine", finish)
    result = learning.train_model(root, out, config, None, "cpu", continue_run=parent, resume=True)
    assert result["status"] == "completed" and len(result["attempts"]) == 1


@pytest.fixture
def orchestration(parent_run, monkeypatch):
    root, parent, plan, _ = parent_run
    calls = []

    def train(root, output, path, weights, device, **kwargs):
        cfg = json.loads(path.read_text())
        epoch = cfg["epochs"]
        calls.append(("train", epoch, kwargs.get("resume", False)))
        output.mkdir(exist_ok=True)
        (output / "checkpoints").mkdir(exist_ok=True)
        (output / "checkpoints/checkpoint_best_total.pth").write_bytes(f"epoch {epoch}".encode())
        (output / "checkpoints/last.ckpt").write_bytes(f"full {epoch}".encode())
        report = {
            "schema_version": 1,
            "status": "completed",
            "model": MODEL_NAME,
            "config": cfg,
            "class_names": list(CLASS_NAMES),
            "checkpoint": "checkpoints/checkpoint_best_total.pth",
            "checkpoint_sha256": c.file_hash(output / "checkpoints/checkpoint_best_total.pth"),
            "initialization": {
                "parent_report_sha256": c.file_hash(kwargs["continue_run"] / "run.json"),
                "full_checkpoint_sha256": cfg["continuation_checkpoint_sha256"],
            },
        }
        write_json(output / "run.json", report)
        return report

    def evaluate(run, root, target, device):
        epoch = json.loads((run / "run.json").read_text())["config"]["epochs"]
        calls.append(("evaluate", epoch))
        target.mkdir()
        result = {
            "schema_version": 1,
            "counts": {n: {"tp": epoch, "fp": 2, "fn": 3} for n in CLASS_NAMES},
            "coco": {},
        }
        write_json(target / "report.json", result)
        (target / "index.html").write_text("synthetic validation")
        return result

    monkeypatch.setattr(c, "train_model", train)
    monkeypatch.setattr(validation, "evaluate_validation", evaluate)
    monkeypatch.setattr(
        threshold_study,
        "load_study_inputs",
        lambda run, root, target, name: (
            json.loads((target / "report.json").read_text()),
            None,
            None,
            None,
        ),
    )
    return root, parent, plan, calls


def test_epochs_are_evaluated_in_order_and_finished_work_is_not_repeated(orchestration, tmp_path):
    root, parent, plan, calls = orchestration
    out = tmp_path / "out"
    result = c.extend_training(root, parent, out, plan, "cpu")
    assert result["status"] == "completed"
    assert calls == [
        ("evaluate", 1),
        ("train", 2, False),
        ("evaluate", 2),
        ("train", 3, False),
        ("evaluate", 3),
    ]
    assert [r["epoch"] for r in result["epochs"]] == [1, 2, 3]
    assert "validation_003/index.html" in (out / "index.html").read_text()
    calls.clear()
    assert c.extend_training(root, parent, out, plan, "cpu", resume=True) == result
    assert calls == []
    with pytest.raises(FileExistsError):
        c.extend_training(root, parent, out, plan, "cpu")


def test_failed_evaluation_is_retried_without_retraining(orchestration, tmp_path, monkeypatch):
    root, parent, plan, calls = orchestration
    out = tmp_path / "out"
    evaluate = validation.evaluate_validation

    def stop(run, root, target, device):
        if target.name == "validation_002":
            raise KeyboardInterrupt()
        return evaluate(run, root, target, device)

    monkeypatch.setattr(validation, "evaluate_validation", stop)
    with pytest.raises(KeyboardInterrupt):
        c.extend_training(root, parent, out, plan, "cpu")
    assert json.loads((out / "extension.json").read_text())["status"] == "interrupted"
    monkeypatch.setattr(validation, "evaluate_validation", evaluate)
    c.extend_training(root, parent, out, plan, "cpu", resume=True)
    assert calls.count(("train", 2, False)) == 1 and calls.count(("evaluate", 2)) == 1


@pytest.mark.parametrize("mutation", ["report", "config", "protocol", "missing_evaluation"])
def test_resume_refuses_changed_recorded_artifacts(orchestration, tmp_path, mutation):
    root, parent, plan, _ = orchestration
    out = tmp_path / "out"
    c.extend_training(root, parent, out, plan, "cpu")
    if mutation == "report":
        p = out / "validation_002/report.json"
        p.write_text(p.read_text() + " ")
    elif mutation == "config":
        p = out / "epoch_002.json"
        d = json.loads(p.read_text())
        d["lr"] = 0.2
        write_json(p, d)
    elif mutation == "protocol":
        p = out / "extension.json"
        d = json.loads(p.read_text())
        d["protocol"]["device"] = "mps"
        write_json(p, d)
    else:
        (out / "validation_002").rename(out / "removed_evaluation")
    with pytest.raises(ValueError):
        c.extend_training(root, parent, out, plan, "cpu", resume=True)


def test_extend_cli_forwards_check_and_resume(monkeypatch):
    def extend(*args, **kwargs):
        assert args == (Path("corpus"), Path("parent"), Path("out"), Path("plan"), "mps")
        assert kwargs == {"check": True, "resume": False}
        return {"status": "ready"}

    monkeypatch.setattr(model_cli, "extend_training", extend)
    monkeypatch.setattr(
        "sys.argv",
        [
            "smartsite-model",
            "extend",
            "corpus",
            "--parent",
            "parent",
            "--plan",
            "plan",
            "--output",
            "out",
            "--device",
            "mps",
            "--check",
        ],
    )
    assert model_cli.main() == 0


def test_partial_epoch_without_checkpoint_restarts_only_that_epoch(
    orchestration, tmp_path, monkeypatch
):
    root, parent, plan, calls = orchestration
    out = tmp_path / "out"
    train = c.train_model

    def stopped(root, child, path, weights, device, **kwargs):
        if child.name == "epoch_002":
            child.mkdir()
            (child / "notes.txt").write_text("partial attempt preserved")
            write_json(
                child / "run.json",
                {
                    "schema_version": 1,
                    "status": "interrupted",
                    "config": json.loads(path.read_text()),
                },
            )
            raise KeyboardInterrupt()
        return train(root, child, path, weights, device, **kwargs)

    monkeypatch.setattr(c, "train_model", stopped)
    with pytest.raises(KeyboardInterrupt):
        c.extend_training(root, parent, out, plan, "cpu")
    monkeypatch.setattr(c, "train_model", train)
    result = c.extend_training(root, parent, out, plan, "cpu", resume=True)
    assert result["status"] == "completed"
    assert (out / "epoch_002_attempt_001/notes.txt").read_text() == "partial attempt preserved"
    assert len(result["restarted_attempts"]) == 1
    assert calls.count(("evaluate", 1)) == 1
    assert calls.count(("train", 2, False)) == 1


def test_recorded_resumable_epoch_uses_resume_option(orchestration, tmp_path, monkeypatch):
    root, parent, plan, _ = orchestration
    out = tmp_path / "out"
    train = c.train_model

    def stopped(root, child, path, weights, device, **kwargs):
        child.mkdir()
        write_json(
            child / "run.json",
            {
                "schema_version": 1,
                "status": "interrupted",
                "config": json.loads(path.read_text()),
                "resume_checkpoint_sha256": "a" * 64,
            },
        )
        raise KeyboardInterrupt()

    monkeypatch.setattr(c, "train_model", stopped)
    with pytest.raises(KeyboardInterrupt):
        c.extend_training(root, parent, out, plan, "cpu")
    calls = []

    def resumed(*args, **kwargs):
        calls.append(kwargs.get("resume", False))
        return train(*args, **kwargs)

    monkeypatch.setattr(c, "train_model", resumed)
    c.extend_training(root, parent, out, plan, "cpu", resume=True)
    assert calls == [True, False]
