"""Piloter RF-DETR et garder les preuves de chaque essai, même en cas d'échec."""

import csv
import hashlib
import importlib
import math
import os
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from smartsite_ia.categories import get_class_names
from smartsite_ia.curation import digest, read_document
from smartsite_ia.importer import write_json
from smartsite_ia.model_assets import MODEL_NAME, MODEL_VERSION, PRETRAINED
from smartsite_ia.review import MAX_FILE_BYTES, read_local
from smartsite_ia.source import verify_archive
from smartsite_ia.training_data import load_training_config, prepare_training_inputs


def file_hash(path: Path, maximum: int = 2_000_000_000) -> str:
    """Les poids sont gros : on calcule leur empreinte sans les charger en mémoire."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError("Missing, linked or oversized model artifact")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_state(output: Path, report: dict[str, Any]) -> None:
    # Si le processus s'arrête pendant l'écriture, le dernier état reste lisible
    partial = output / "run.json.partial"
    write_json(partial, report)
    partial.replace(output / "run.json")


def runtime(device: str) -> dict[str, Any]:
    """Charger les dépendances lourdes seulement lorsqu'on demande le moteur."""
    if version("rfdetr") != MODEL_VERSION:
        raise ValueError("RF-DETR version differs from the pinned engine")
    torch = importlib.import_module("torch")
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if device not in available or not available[device]:
        raise ValueError(f"Requested device is unavailable: {device}")
    # Aucun service de suivi externe, les poids  sont déjà locaux.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return {
        "device": device,
        "available": available,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "rfdetr",
                "torch",
                "torchvision",
                "pytorch-lightning",
                "transformers",
                "numpy",
            )
        },
        "mps_fallback_requested": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "mps_allocator_environment": {
            name: os.environ.get(name)
            for name in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")
        },
    }


def fit_engine(
    data: Path,
    output: Path,
    weights: Path,
    config: dict[str, Any],
    device: str,
    *,
    model_name: str = MODEL_NAME,
    class_names: tuple[str, ...] | None = None,
) -> None:
    """Utiliser l'API du moteur, sans réécrire ses calculs d'apprentissage."""
    lightning = importlib.import_module("pytorch_lightning")
    lightning.seed_everything(config["seed"], workers=True)
    names = get_class_names(config) if class_names is None else class_names
    engine = getattr(importlib.import_module("rfdetr"), model_name)(
        pretrain_weights=str(weights.resolve()),
        device=device,
        amp=False,
        fused_optimizer=False,
        # Un parent adapté garde sa tête. Depuis les poids officiels, le moteur
        # ajuste lui-même le nombre de classes en lisant notre COCO préparé.
        **(
            {"num_classes": len(names)}
            if "initial_checkpoint_sha256" in config or "continuation_checkpoint_sha256" in config
            else {}
        ),
    )
    engine.train(
        dataset_dir=str(data.resolve()),
        output_dir=str(output.resolve()),
        epochs=config["epochs"],
        # Pour comparer deux apprentissages au même budget, on peut garder
        # uniquement le dernier passage. Le moteur sait déjà faire cette sélection.
        skip_best_epochs=(
            config["epochs"] - 1 if config.get("checkpoint_selection") == "last_epoch" else 0
        ),
        batch_size=config["batch_size"],
        grad_accum_steps=config["grad_accum_steps"],
        lr=config["lr"],
        resume=config.get("_resume"),
        device=device,
        seed=config["seed"],
        num_workers=0,
        pin_memory=False,
        use_ema=False,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        run_test=False,
        multi_scale=False,
        expanded_scales=False,
        scale_jitter=False,
        augmentation_backend="torchvision",
        compute_val_loss=True,
        log_per_class_metrics=True,
        progress_bar="tqdm",
        notes={"purpose": config["purpose"], "classes": list(names), "test_used": False},
    )


def summarize_metrics(path: Path, *, segmentation: bool = True) -> dict[str, float]:
    """Le CSV reste la référence complète ; le résumé garde la dernière valeur finie."""
    if path.stat().st_size > 10_000_000:
        raise ValueError("Training metrics file is too large")
    metric = "val/segm_mAP_50_95" if segmentation else "val/mAP_50_95"
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if value:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("Non-finite training metric")
                    result[key] = number
            if row.get(metric) and row.get("epoch"):
                result["last_validation_epoch"] = float(row["epoch"])
    if not result or "train/loss" not in result or metric not in result:
        raise ValueError("Training did not produce expected loss and detection metrics")
    return result


def train_model(
    root: Path,
    output: Path,
    config_path: Path,
    weights: Path | None,
    device: str,
    *,
    resume: bool = False,
    from_run: Path | None = None,
    continue_run: Path | None = None,
) -> dict[str, Any]:
    """Un dossier neuf par essai ; on ne déclare terminé qu'après les contrôles finaux."""
    config, config_hash = load_training_config(config_path)
    if continue_run is not None:
        from smartsite_ia.continuation import continuation_origin

        if weights is not None or from_run is not None:
            raise ValueError("Continuation cannot be mixed with a fresh initialization")
        weights, initialization = continuation_origin(continue_run, output, config, device)
    else:
        if "continuation_checkpoint_sha256" in config:
            raise ValueError("Continuation configuration requires its full parent state")
        weights, initialization = training_origin(config, weights, from_run, output)
    if (
        output.resolve().is_relative_to(root.resolve())
        or output.resolve() == weights.resolve().parent
    ):
        raise ValueError("Training output must be separate from its inputs")
    environment = runtime(device)
    previous = check_resume(output, config, config_hash, device) if resume else None
    if (
        previous
        and previous.get(
            "initialization",
            {"mode": "official_pretrained", "checkpoint_sha256": PRETRAINED.sha256},
        )
        != initialization
    ):
        raise ValueError("Resume initialization changed")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "model": MODEL_NAME,
        "config": config,
        "config_sha256": config_hash,
        "pretrained_sha256": PRETRAINED.sha256,
        "initialization": initialization,
        "runtime": environment,
        "class_names": list(get_class_names(config)),
        "test_used": False,
        "qualified_for_smartsite": False,
        "purpose": config["purpose"],
        "checkpoint_selection": config.get("checkpoint_selection", "best_validation"),
        "limits": [
            "Experimental run, not a qualified construction detector",
            "Internal validation from the same dam; no independent generalization evidence",
            "GPU execution is seeded but not guaranteed bit-for-bit deterministic",
        ],
    }
    if previous:
        report = previous
        report.setdefault("attempts", []).append(
            {
                "status": previous["status"],
                "error_type": previous.get("error_type"),
                "elapsed_seconds": previous.get("elapsed_seconds"),
                "resume_checkpoint_sha256": previous["resume_checkpoint_sha256"],
            }
        )
        report.pop("error_type", None)
        report.update(status="preparing", runtime=environment)
    started = time.monotonic()
    save_state(output, report)
    try:
        if resume:
            selection, _ = read_document(output / "data/selection.json")
        else:
            selection = prepare_training_inputs(root, output / "data", config)
        report.update(
            status="training",
            selected={
                k: {key: value for key, value in v.items() if key != "records"}
                for k, v in selection["splits"].items()
            },
            selection_sha256=file_hash(output / "data/selection.json"),
            input_annotations_sha256={
                split: file_hash(output / f"data/{split}/_annotations.coco.json")
                for split in ("train", "valid")
            },
        )
        save_state(output, report)
        fit_config = dict(config)
        if resume:
            fit_config["_resume"] = str((output / "checkpoints/last.ckpt").resolve())
        elif continue_run is not None:
            from smartsite_ia.continuation import copy_continuation_state

            # Seuls les chemins des sauvegardes sont remis à zéro
            # Les poids, moments de l'optimiseur et compteurs restent ceux du parent
            copied = output / "initial_state.ckpt"
            copy_continuation_state(continue_run, config, copied)
            fit_config["_resume"] = str(copied.resolve())
            report["initial_state_sha256"] = file_hash(copied)
            save_state(output, report)
        # Lightning restaure aussi l'optimiseur. On impose sa lecture sûre,
        # même si une dépendance demande implicitement une lecture pickle complète.
        old_safe_load = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            fit_engine(output / "data", output / "checkpoints", weights, fit_config, device)
        finally:
            if old_safe_load is None:
                os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = old_safe_load
        checkpoint = output / "checkpoints/checkpoint_best_total.pth"
        checksum = file_hash(checkpoint)
        if checksum == initialization["checkpoint_sha256"]:
            raise ValueError("Checkpoint is identical to its initial weights")
        if file_hash(weights) != initialization["checkpoint_sha256"]:
            raise ValueError("Initial checkpoint changed during training")
        metrics_path = output / "checkpoints/metrics.csv"
        last_metrics = summarize_metrics(metrics_path)
        if last_metrics.get("last_validation_epoch", -1) + 1 != config["epochs"]:
            raise RuntimeError("Training stopped before the requested final epoch")
        report.update(
            status="completed",
            checkpoint="checkpoints/checkpoint_best_total.pth",
            checkpoint_sha256=checksum,
            last_epoch_metrics=last_metrics,
            metrics_sha256=file_hash(metrics_path),
            elapsed_seconds=time.monotonic() - started,
            process_peak_rss_bytes=process_peak_rss(),
        )
        last = output / "checkpoints/last.ckpt"
        if last.is_file():
            report["resume_checkpoint_sha256"] = file_hash(last)
        save_state(output, report)
    except BaseException as error:
        # On garde les journaux et points de reprise d'un essai interrompu
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
            process_peak_rss_bytes=process_peak_rss(),
        )
        last = output / "checkpoints/last.ckpt"
        if last.is_file() and not last.parent.is_symlink():
            report["resume_checkpoint_sha256"] = file_hash(last)
        save_state(output, report)
        raise
    return report


def training_origin(
    config: dict[str, Any], weights: Path | None, parent: Path | None, output: Path
) -> tuple[Path, dict[str, Any]]:
    """Choisir des poids vérifiés, sans écrire dans l'essai qui nous sert de départ."""
    if (weights is None) == (parent is None):
        raise ValueError("Choose exactly one of pretrained weights or a completed parent run")
    pinned = config.get("initial_checkpoint_sha256")
    if parent is None:
        if pinned is not None:
            raise ValueError("A pinned initial checkpoint requires a parent run")
        assert weights is not None
        verify_archive(weights, PRETRAINED)
        return weights, {"mode": "official_pretrained", "checkpoint_sha256": PRETRAINED.sha256}
    if output.resolve().is_relative_to(parent.resolve()):
        raise ValueError("Fine-tuning output must be outside the parent run")
    report, checkpoint = verify_run(parent)
    if (
        pinned != report["checkpoint_sha256"]
        or tuple(report["class_names"]) != get_class_names(config)
        or report["config"]["corpus_sha256"] != config["corpus_sha256"]
        or report.get("runtime", {}).get("versions", {}).get("rfdetr") != MODEL_VERSION
    ):
        raise ValueError(
            "Parent checkpoint, classes, corpus or engine version differs from configuration"
        )
    return checkpoint, {
        "mode": "fine_tune",
        "checkpoint_sha256": pinned,
        "parent_run": str(parent.resolve()),
        "parent_report_sha256": file_hash(parent / "run.json"),
        "optimizer_restored": False,
        "scheduler_restored": False,
        "epoch_numbering": "Restarted at zero in this new run",
    }


def process_peak_rss() -> int:
    """Pic de mémoire du processus ; ce n'est pas un pic complet de mémoire GPU."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def check_resume(
    output: Path, config: dict[str, Any], config_hash: str, device: str
) -> dict[str, Any]:
    """Reprendre seulement un essai interrompu, avec exactement les mêmes entrées."""
    report, _ = read_document(output / "run.json")
    if (
        report.get("status") not in ("failed", "interrupted")
        or report.get("model") != MODEL_NAME
        or report.get("config_sha256") != config_hash
        or report.get("config") != config
        or report.get("class_names") != list(get_class_names(config))
        or report.get("runtime", {}).get("device") != device
    ):
        raise ValueError("Resume requires an interrupted run with unchanged configuration/device")
    last = output / "checkpoints/last.ckpt"
    if last.parent.is_symlink() or file_hash(last) != report.get("resume_checkpoint_sha256"):
        raise ValueError("Resume checkpoint missing or changed")
    verify_training_data(output, report)
    return report


def verify_training_data(output: Path, report: dict[str, Any]) -> None:
    """Les mêmes contrôles protègent une reprise et une continuation du parent."""
    data = output / "data"
    if data.is_symlink() or not data.resolve().is_relative_to(output.resolve()):
        raise ValueError("Resume data path escapes its run")
    if file_hash(data / "selection.json") != report.get("selection_sha256"):
        raise ValueError("Resume selection changed")
    selection, _ = read_document(data / "selection.json")
    for split in ("train", "valid"):
        if (data / split).is_symlink():
            raise ValueError("Resume split must not be linked")
        annotation = f"{split}/_annotations.coco.json"
        if file_hash(data / annotation) != report.get("input_annotations_sha256", {}).get(split):
            raise ValueError("Resume annotations changed")
        for record in selection["splits"][split]["records"]:
            raw = read_local(data, f"{split}/{record['id']}.jpg", MAX_FILE_BYTES)
            if digest(raw) != record["image_sha256"]:
                raise ValueError("Resume photo changed")


def verify_run(root: Path) -> tuple[dict[str, Any], Path]:
    report, _ = read_document(root / "run.json")
    if (
        report.get("status") != "completed"
        or report.get("model") != MODEL_NAME
        or report.get("class_names") != list(get_class_names(report.get("config", {})))
        or report.get("checkpoint") != "checkpoints/checkpoint_best_total.pth"
    ):
        raise ValueError("Expected a completed SmartSite training run")
    checkpoint = root / report["checkpoint"]
    if checkpoint.parent.is_symlink() or not checkpoint.resolve().is_relative_to(root.resolve()):
        raise ValueError("Trained checkpoint path escapes its run")
    if file_hash(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("Trained checkpoint changed since the run completed")
    return report, checkpoint
