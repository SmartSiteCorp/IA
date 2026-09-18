"""Piloter RF-DETR et garder les preuves de chaque essai, même en cas d'échec."""

import csv
import hashlib
import importlib
import math
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from smartsite_ia.curation import read_document
from smartsite_ia.importer import write_json
from smartsite_ia.model_assets import CLASS_NAMES, MODEL_NAME, MODEL_VERSION, PRETRAINED
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
    }


def fit_engine(
    data: Path, output: Path, weights: Path, config: dict[str, Any], device: str
) -> None:
    """Utiliser l'API du moteur, sans réécrire ses calculs d'apprentissage."""
    lightning = importlib.import_module("pytorch_lightning")
    lightning.seed_everything(config["seed"], workers=True)
    engine = importlib.import_module("rfdetr").RFDETRSegMedium(
        pretrain_weights=str(weights.resolve()), device=device, amp=False, fused_optimizer=False
    )
    engine.train(
        dataset_dir=str(data.resolve()),
        output_dir=str(output.resolve()),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        grad_accum_steps=config["grad_accum_steps"],
        lr=config["lr"],
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
        notes={"purpose": config["purpose"], "classes": list(CLASS_NAMES), "test_used": False},
    )


def summarize_metrics(path: Path) -> dict[str, float]:
    """Le CSV reste la référence complète ; le résumé garde la dernière valeur finie."""
    if path.stat().st_size > 10_000_000:
        raise ValueError("Training metrics file is too large")
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if value:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("Non-finite training metric")
                    result[key] = number
    if not result or "train/loss" not in result or "val/segm_mAP_50_95" not in result:
        raise ValueError("Training did not produce expected loss and segmentation metrics")
    return result


def train_model(
    root: Path, output: Path, config_path: Path, weights: Path, device: str
) -> dict[str, Any]:
    """Un dossier neuf par essai ; on ne déclare terminé qu'après les contrôles finaux."""
    config, config_hash = load_training_config(config_path)
    verify_archive(weights, PRETRAINED)
    if (
        output.resolve().is_relative_to(root.resolve())
        or output.resolve() == weights.resolve().parent
    ):
        raise ValueError("Training output must be separate from its inputs")
    environment = runtime(device)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "model": MODEL_NAME,
        "config": config,
        "config_sha256": config_hash,
        "pretrained_sha256": PRETRAINED.sha256,
        "runtime": environment,
        "class_names": list(CLASS_NAMES),
        "test_used": False,
        "qualified_for_smartsite": False,
        "purpose": config["purpose"],
        "limits": [
            "Small experimental run, not a qualified construction detector",
            "Internal validation from the same dam; no independent generalization evidence",
            "GPU execution is seeded but not guaranteed bit-for-bit deterministic",
        ],
    }
    started = time.monotonic()
    save_state(output, report)
    try:
        selection = prepare_training_inputs(root, output / "data", config)
        report.update(
            status="training",
            selected={
                k: {key: value for key, value in v.items() if key != "records"}
                for k, v in selection["splits"].items()
            },
            selection_sha256=file_hash(output / "data/selection.json"),
        )
        save_state(output, report)
        fit_engine(output / "data", output / "checkpoints", weights, config, device)
        checkpoint = output / "checkpoints/checkpoint_best_total.pth"
        checksum = file_hash(checkpoint)
        if checksum == PRETRAINED.sha256:
            raise ValueError("Checkpoint is identical to the unadapted pretrained file")
        metrics_path = output / "checkpoints/metrics.csv"
        report.update(
            status="completed",
            checkpoint="checkpoints/checkpoint_best_total.pth",
            checkpoint_sha256=checksum,
            last_epoch_metrics=summarize_metrics(metrics_path),
            metrics_sha256=file_hash(metrics_path),
            elapsed_seconds=time.monotonic() - started,
        )
        save_state(output, report)
    except BaseException as error:
        # On garde les journaux et points de reprise d'un essai interrompu
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        save_state(output, report)
        raise
    return report


def verify_run(root: Path) -> tuple[dict[str, Any], Path]:
    report, _ = read_document(root / "run.json")
    if (
        report.get("status") != "completed"
        or report.get("model") != MODEL_NAME
        or report.get("class_names") != list(CLASS_NAMES)
        or report.get("checkpoint") != "checkpoints/checkpoint_best_total.pth"
    ):
        raise ValueError("Expected a completed SmartSite training run")
    checkpoint = root / report["checkpoint"]
    if checkpoint.parent.is_symlink() or not checkpoint.resolve().is_relative_to(root.resolve()):
        raise ValueError("Trained checkpoint path escapes its run")
    if file_hash(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("Trained checkpoint changed since the run completed")
    return report, checkpoint
