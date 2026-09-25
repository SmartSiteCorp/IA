"""Lancer un pilote par boîtes et conserver son état sans toucher aux anciens modèles."""

import fcntl
import os
import time
from pathlib import Path
from typing import Any

from smartsite_ia.box_data import (
    config_classes,
    corpus_photo_limit,
    inspect_corpus,
    load_box_config,
    prepare_inputs,
)
from smartsite_ia.curation import read_document
from smartsite_ia.learning import (
    file_hash,
    fit_engine,
    process_peak_rss,
    runtime,
    save_state,
    summarize_metrics,
    verify_training_data,
)
from smartsite_ia.model_assets import BOX_MODEL_NAME, BOX_PRETRAINED
from smartsite_ia.source import verify_archive


def check_box_training(root: Path, config_path: Path, weights: Path, device: str) -> dict[str, Any]:
    """Contrôler toutes les entrées et le matériel, sans commencer à apprendre."""
    config, sha = load_box_config(config_path)
    selection = inspect_corpus(root, config)
    verify_archive(weights, BOX_PRETRAINED)
    environment = runtime(device)
    return {
        "schema_version": 1,
        "status": "inputs_checked",
        "training_started": False,
        "model": BOX_MODEL_NAME,
        "config": config,
        "config_sha256": sha,
        "pretrained_sha256": BOX_PRETRAINED.sha256,
        "runtime": environment,
        # Les classes viennent de la configuration : une reprise les compare telles quelles.
        "class_names": list(config_classes(config)),
        "test_used": False,
        "qualified_for_smartsite": False,
        "selected": {
            split: {k: v for k, v in details.items() if k != "records"}
            for split, details in selection["splits"].items()
        },
    }


def resume_box_run(output: Path, checked: dict[str, Any]) -> dict[str, Any]:
    """Une reprise conserve données, configuration, matériel et versions du moteur."""
    previous, _ = read_document(output / "run.json")
    if previous.get("status") not in ("failed", "interrupted") or any(
        previous.get(key) != checked[key]
        for key in ("model", "config", "config_sha256", "pretrained_sha256", "class_names")
    ):
        raise ValueError("Resume requires a recorded interruption with unchanged inputs")
    for key in ("device", "versions"):
        if previous.get("runtime", {}).get(key) != checked["runtime"][key]:
            raise ValueError("Resume runtime differs from the interrupted run")
    verify_training_data(output, previous, corpus_photo_limit(checked["config"]))
    last = output / "checkpoints/last.ckpt"
    if last.parent.is_symlink() or file_hash(last) != previous.get("resume_checkpoint_sha256"):
        raise ValueError("No verified checkpoint to resume; keep this run and use a new output")
    return previous


def train_boxes(
    root: Path,
    config_path: Path,
    weights: Path,
    output: Path,
    device: str,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Conserver les poids du dernier passage ; ne jamais promouvoir ce pilote."""
    if output.is_symlink() or any(
        output.resolve().is_relative_to(p.resolve())
        for p in (root, config_path.parent, weights.parent)
    ):
        raise ValueError("Training output must be separate from all inputs")
    if resume and not output.is_dir():
        raise ValueError("Resume output is missing")
    if not resume and output.exists():
        raise FileExistsError("Training output exists; use --resume after a recorded interruption")
    checked = check_box_training(root, config_path, weights, device)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    lock = output / ".run.lock"
    if lock.is_symlink():
        raise ValueError("Training lock must not be linked")
    with lock.open("a") as stream:
        try:
            # Le verrou est libéré par le système même si le processus est arrêté.
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is already using this training output") from exc
        previous = resume_box_run(output, checked) if resume else None
        return run_box_attempt(root, weights, output, checked, previous)


def run_box_attempt(
    root: Path,
    weights: Path,
    output: Path,
    checked: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Le journal distingue préparation, apprentissage, interruption et réussite vérifiée."""
    report = dict(checked) if previous is None else dict(previous)
    if previous is not None:
        report.setdefault("attempts", []).append(
            {
                k: previous.get(k)
                for k in ("status", "error_type", "elapsed_seconds", "resume_checkpoint_sha256")
            }
        )
        report.pop("error_type", None)
    config = checked["config"]
    report.update(status="preparing", training_started=False, runtime=checked["runtime"])
    started = time.monotonic()
    save_state(output, report)
    try:
        if previous is None:
            prepare_inputs(root, output / "data", config)
            report.update(
                selection_sha256=file_hash(output / "data/selection.json"),
                input_annotations_sha256={
                    s: file_hash(output / f"data/{s}/_annotations.coco.json")
                    for s in ("train", "valid")
                },
            )
        report.update(status="training", training_started=True)
        save_state(output, report)
        fit_config = dict(config)
        if previous is not None:
            fit_config["_resume"] = str((output / "checkpoints/last.ckpt").resolve())
        old_safe = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            fit_engine(
                output / "data",
                output / "checkpoints",
                weights,
                fit_config,
                checked["runtime"]["device"],
                model_name=BOX_MODEL_NAME,
                class_names=tuple(checked["class_names"]),
            )
        finally:
            if old_safe is None:
                os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = old_safe
        metrics_path = output / "checkpoints/metrics.csv"
        metrics = summarize_metrics(metrics_path, segmentation=False)
        if metrics.get("last_validation_epoch", -1) + 1 != config["epochs"]:
            raise RuntimeError("Box training stopped before the requested final epoch")
        checkpoint = output / "checkpoints/checkpoint_best_total.pth"
        sha = file_hash(checkpoint)
        if sha == BOX_PRETRAINED.sha256:
            raise ValueError("Box checkpoint is unchanged from initial weights")
        verify_archive(weights, BOX_PRETRAINED)
        verify_training_data(output, report, corpus_photo_limit(config))
        report.update(
            status="completed",
            checkpoint="checkpoints/checkpoint_best_total.pth",
            checkpoint_sha256=sha,
            checkpoint_selection="last_epoch",
            last_epoch_metrics=metrics,
            metrics_sha256=file_hash(metrics_path),
        )
    except BaseException as exc:
        report.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        report.update(
            elapsed_seconds=time.monotonic() - started, process_peak_rss_bytes=process_peak_rss()
        )
        last = output / "checkpoints/last.ckpt"
        if last.is_file() and not last.parent.is_symlink():
            report["resume_checkpoint_sha256"] = file_hash(last)
        save_state(output, report)
    return report
