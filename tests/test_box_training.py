"""Contrôler les contrats du pilote par boîtes ; les images de test sont synthétiques."""

import copy
import fcntl
import json
import os

import pytest
from test_collection_review import review_case as review_case

from smartsite_ia import box_cli, box_data, box_training, learning
from smartsite_ia.collection_review import apply_review
from smartsite_ia.curation import digest
from smartsite_ia.importer import write_json
from smartsite_ia.source import Source


def pin(root, config):
    config["corpus_sha256"] = {
        name: digest((root / name).read_bytes()) for name in box_data.BOX_CORPUS_FILES
    }


@pytest.fixture
def box_case(review_case, tmp_path):
    selection, cache, review, root, *_ = review_case
    apply_review(selection, cache, review, root)
    config = {
        "schema_version": 1,
        "purpose": "pilot_box_training",
        "seed": 42,
        "epochs": 5,
        "batch_size": 2,
        "grad_accum_steps": 2,
        "lr": 0.0001,
        "checkpoint_selection": "last_epoch",
    }
    pin(root, config)
    path = tmp_path / "config" / "training.json"
    write_json(path, config)
    return root, path, config


def test_preparation_preserves_rectangles_crops_groups_and_no_test(box_case, tmp_path, monkeypatch):
    root, path, config = box_case
    actual = box_data.read_local

    def guarded(folder, relative, maximum):
        assert not relative.startswith("test/")
        assert "photo_2.jpg" not in relative
        return actual(folder, relative, maximum)

    monkeypatch.setattr(box_data, "read_local", guarded)
    assert box_data.load_box_config(path)[0] == config
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        result = box_data.prepare_box_inputs(root, output, config)
        assert result["splits"]["train"]["images"] == 2
        assert result["splits"]["train"]["negative_crops"] == 1
        assert result["splits"]["valid"]["images"] == 1
        assert not result["test_used"] and not (output / "test").exists()
        assert (output / "train/_annotations.coco.json").read_bytes() == (
            root / "train/_annotations.coco.json"
        ).read_bytes()
        assert (output / "valid/_annotations.coco.json").read_bytes() == (
            root / "validation/_annotations.coco.json"
        ).read_bytes()
        assert (output / "train/negative.jpg").read_bytes() == (
            root / "train/negative.jpg"
        ).read_bytes()

    def hashes(folder):
        return {
            p.relative_to(folder): digest(p.read_bytes()) for p in folder.rglob("*") if p.is_file()
        }

    assert hashes(outputs[0]) == hashes(outputs[1])
    with pytest.raises(FileExistsError):
        box_data.prepare_box_inputs(root, outputs[0], config)
    with pytest.raises(ValueError, match="outside"):
        box_data.prepare_box_inputs(root, root / "nested", config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 0),
        ("epochs", 11),
        ("epochs", True),
        ("batch_size", 0),
        ("batch_size", 5),
        ("grad_accum_steps", 9),
        ("seed", -1),
        ("lr", float("nan")),
        ("lr", float("inf")),
        ("lr", True),
        ("lr", 0.1),
        ("purpose", "production"),
        ("checkpoint_selection", "best"),
        ("corpus_sha256", {}),
        ("corpus_sha256", {"report.json": "bad"}),
        ("unknown", True),
        ("schema_version", 2),
    ],
)
def test_invalid_config_rejected(box_case, key, value):
    _, path, config = box_case
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        box_data.load_box_config(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(purpose="production"),
        lambda d: d.update(human_validation="approved"),
        lambda d: d.update(final_test_created=True),
        lambda d: d.update(review_sha256="0" * 64),
        lambda d: d.update(records=None),
        lambda d: d["records"].append(d["records"][0]),
        lambda d: d["records"].pop(),
        lambda d: d["records"][0].update(id="../escape"),
        lambda d: d["records"][0].update(partition="test"),
        lambda d: d["records"][2].update(partition="train", decision="candidate"),
        lambda d: d["records"][0].update(group="photo_1"),
        lambda d: d["records"][0].update(boxes=[]),
        lambda d: d["records"][0].update(width=False),
        lambda d: d["records"][0].update(image_sha256="bad"),
        lambda d: d["records"][-1].update(parent_id="photo_1"),
        lambda d: d["records"][-1].update(width=9000),
    ],
)
def test_changed_review_cannot_be_smuggled_into_training(box_case, tmp_path, change):
    root, _, config = box_case
    path = root / "report.json"
    report = json.loads(path.read_text())
    change(report)
    write_json(path, report)
    pin(root, config)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        box_data.prepare_box_inputs(root, output, config)
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value", [("bbox", [1, 2, 3, 4]), ("category_id", 99), ("segmentation", [])]
)
def test_changed_coco_geometry_or_classes_rejected(box_case, field, value):
    root, _, config = box_case
    path = root / "train/_annotations.coco.json"
    document = json.loads(path.read_text())
    document["annotations"][0][field] = value
    write_json(path, document)
    pin(root, config)
    with pytest.raises(ValueError, match="COCO"):
        box_data.inspect_box_corpus(root, config)


def test_hash_corruption_links_and_duplicate_pixels_rejected(box_case):
    root, _, config = box_case
    photo = root / "train/photo_0.jpg"
    raw = photo.read_bytes()
    photo.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="photo changed"):
        box_data.inspect_box_corpus(root, config)
    photo.unlink()
    photo.symlink_to(root / "validation/photo_1.jpg")
    with pytest.raises(ValueError, match="Linked"):
        box_data.inspect_box_corpus(root, config)
    photo.unlink()
    photo.write_bytes(raw)
    val = root / "validation/photo_1.jpg"
    val.write_bytes(raw)
    report = json.loads((root / "report.json").read_text())
    report["records"][1]["image_sha256"] = digest(raw)
    write_json(root / "report.json", report)
    pin(root, config)
    with pytest.raises(ValueError, match="pixels cross"):
        box_data.inspect_box_corpus(root, config)


def test_unpinned_document_and_malformed_source_refused(box_case):
    root, _, config = box_case
    path = root / "source/report.json"
    source = json.loads(path.read_text())
    source["records"][0]["group"] = []
    write_json(path, source)
    with pytest.raises(ValueError, match="corpus changed"):
        box_data.inspect_box_corpus(root, config)
    pin(root, config)
    with pytest.raises(ValueError, match="Malformed"):
        box_data.inspect_box_corpus(root, config)


@pytest.fixture
def fake_engine(box_case, tmp_path, monkeypatch):
    root, path, config = box_case
    weights = tmp_path / "weights" / "nano.pth"
    weights.parent.mkdir()
    weights.write_bytes(b"official fixture")
    source = Source(
        "https://storage.googleapis.com/test",
        weights.stat().st_size,
        digest(weights.read_bytes()),
        0,
    )
    monkeypatch.setattr(box_training, "BOX_PRETRAINED", source)
    monkeypatch.setattr(
        box_training, "runtime", lambda device: {"device": device, "versions": {"engine": "test"}}
    )
    calls = []

    def fit(data, output, weights, params, device, **options):
        calls.append(copy.deepcopy(params))
        assert options == {"model_name": "RFDETRNano", "class_names": box_data.BOX_CLASSES}
        assert os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"
        output.mkdir(exist_ok=True)
        (output / "checkpoint_best_total.pth").write_bytes(b"trained fixture")
        (output / "last.ckpt").write_bytes(b"full state fixture")
        (output / "metrics.csv").write_text("epoch,train/loss,val/mAP_50_95\n4,2.5,0\n")

    monkeypatch.setattr(box_training, "fit_engine", fit)
    return root, path, weights, tmp_path / "run", calls


def test_training_and_check_have_distinct_effects(fake_engine):
    root, path, weights, output, calls = fake_engine
    checked = box_training.check_box_training(root, path, weights, "cpu")
    assert checked["status"] == "inputs_checked" and not checked["training_started"]
    assert not calls and not output.exists()
    old = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
    report = box_training.train_boxes(root, path, weights, output, "cpu")
    assert report["status"] == "completed" and report["last_epoch_metrics"]["val/mAP_50_95"] == 0
    assert not report["qualified_for_smartsite"] and report["checkpoint_selection"] == "last_epoch"
    assert os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD") == old
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        box_training.train_boxes(root, path, weights, output, "cpu")
    with pytest.raises(ValueError, match="recorded interruption"):
        box_training.train_boxes(root, path, weights, output, "cpu", resume=True)


def test_interrupt_resume_and_no_concurrent_writer(fake_engine, monkeypatch):
    root, path, weights, output, calls = fake_engine
    actual = box_training.fit_engine

    def stop(data, dest, *args, **kwargs):
        dest.mkdir()
        (dest / "last.ckpt").write_bytes(b"partial full state")
        raise KeyboardInterrupt

    monkeypatch.setattr(box_training, "fit_engine", stop)
    with pytest.raises(KeyboardInterrupt):
        box_training.train_boxes(root, path, weights, output, "cpu")
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == "interrupted" and report["resume_checkpoint_sha256"]
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            box_training.train_boxes(root, path, weights, output, "cpu", resume=True)
    monkeypatch.setattr(box_training, "fit_engine", actual)
    report = box_training.train_boxes(root, path, weights, output, "cpu", resume=True)
    assert report["status"] == "completed"
    assert calls[0]["_resume"] == str(output / "checkpoints/last.ckpt")
    assert report["attempts"][0]["status"] == "interrupted"


@pytest.mark.parametrize("mutation", ["weights", "data", "last", "device", "config", "versions"])
def test_changed_resume_inputs_rejected(fake_engine, monkeypatch, mutation):
    root, path, weights, output, calls = fake_engine
    box_training.train_boxes(root, path, weights, output, "cpu")
    report = json.loads((output / "run.json").read_text())
    report["status"] = "interrupted"
    if mutation == "weights":
        weights.write_bytes(b"bad")
    elif mutation == "data":
        (output / "data/train/photo_0.jpg").write_bytes(b"bad")
    elif mutation == "last":
        (output / "checkpoints/last.ckpt").write_bytes(b"bad")
    elif mutation == "device":
        report["runtime"]["device"] = "mps"
    elif mutation == "versions":
        report["runtime"]["versions"] = {}
    else:
        report["config_sha256"] = "0" * 64
    write_json(output / "run.json", report)
    with pytest.raises(ValueError):
        box_training.train_boxes(root, path, weights, output, "cpu", resume=True)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["short", "missing", "nan", "same", "exception"])
def test_incomplete_run_is_never_reported_success(fake_engine, monkeypatch, mode):
    root, path, weights, output, calls = fake_engine
    actual = box_training.fit_engine

    def broken(*args, **kwargs):
        actual(*args, **kwargs)
        checkpoint = output / "checkpoints/checkpoint_best_total.pth"
        metrics = output / "checkpoints/metrics.csv"
        if mode == "short":
            metrics.write_text("epoch,train/loss,val/mAP_50_95\n0,1,0\n")
        elif mode == "missing":
            checkpoint.unlink()
        elif mode == "nan":
            metrics.write_text("epoch,train/loss,val/mAP_50_95\n4,nan,0\n")
        elif mode == "same":
            checkpoint.write_bytes(weights.read_bytes())
        else:
            raise RuntimeError("fixture failure")

    monkeypatch.setattr(box_training, "fit_engine", broken)
    with pytest.raises((ValueError, RuntimeError)):
        box_training.train_boxes(root, path, weights, output, "cpu")
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_cli_prepare_check_train_and_errors(fake_engine, tmp_path, capsys, monkeypatch):
    root, path, weights, output, _ = fake_engine
    base = [str(root), "--config", str(path)]
    engine = ["--weights", str(weights), "--device", "cpu"]
    assert box_cli.main(["prepare", *base, "--output", str(tmp_path / "prepared")]) == 0
    assert box_cli.main(["check", *base, *engine]) == 0
    assert box_cli.main(["train", *base, *engine, "--output", str(output)]) == 0
    assert box_cli.main(["train", *base, *engine, "--output", str(output)]) == 1
    assert "exists" in capsys.readouterr().err

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(box_cli, "train_boxes", interrupt)
    assert box_cli.main(["train", *base, *engine, "--output", str(output)]) == 130
    monkeypatch.setattr(box_cli, "download_archive", lambda p, s: p)
    assert box_cli.main(["weights", "--output", str(weights)]) == 0


def test_bbox_metrics_do_not_accept_segmentation_only(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,train/loss,val/segm_mAP_50_95\n0,1,0.5\n")
    with pytest.raises(ValueError):
        learning.summarize_metrics(path, segmentation=False)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_full_negative_reaches_training_with_its_review_and_group(review_case, tmp_path, split):
    selection, cache, review, root, document, _ = review_case
    document["records"][2].update(decision="negative", reviewed_scope=True)
    document["group_partitions"]["photo_2"] = split
    review.write_text(json.dumps(document))
    apply_review(selection, cache, review, root)
    config = {"batch_size": 2}
    pin(root, config)
    result = box_data.prepare_box_inputs(root, tmp_path / "prepared", config)
    target = "train" if split == "train" else "valid"
    assert result["splits"][target]["negative_photos"] == 1
    row = next(r for r in result["splits"][target]["records"] if r["id"] == "photo_2")
    assert row["boxes"] == [] and row["group"] == "photo_2"
    assert row["decision"] == "negative" and row["reviewed_scope"] is True
    assert (tmp_path / "prepared" / target / "photo_2.jpg").is_file()
