"""Continuer un passage à la fois, avec l'optimiseur du parent et une validation native."""

import fcntl
import gc
import html
import importlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from smartsite_ia.annotations import unique_object
from smartsite_ia.curation import digest, read_document
from smartsite_ia.importer import write_json
from smartsite_ia.learning import (
    file_hash,
    runtime,
    train_model,
    verify_run,
    verify_training_data,
)
from smartsite_ia.review import read_local
from smartsite_ia.training_data import CORPUS_FILES


def read_full_state(parent: Path, expected_hash: str) -> dict[str, Any]:
    """Refuser un fichier de simples poids ou un état qui ne correspond plus au parent."""
    report, weights = verify_run(parent)
    path = parent / "checkpoints/last.ckpt"
    if (
        file_hash(path) != expected_hash
        or report.get("resume_checkpoint_sha256", expected_hash) != expected_hash
    ):
        raise ValueError("Full continuation checkpoint changed")
    torch = importlib.import_module("torch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    learned = torch.load(weights, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("epoch") != report["config"]["epochs"] - 1
        or type(state.get("global_step")) is not int
        or state["global_step"] < 1
        or not state.get("loops", {}).get("fit_loop")
        or len(state.get("optimizer_states", [])) != 1
        or not state["optimizer_states"][0].get("state")
        or not state["optimizer_states"][0].get("param_groups")
        or len(state.get("lr_schedulers", [])) != 1
        or state["lr_schedulers"][0].get("last_epoch") != state["global_step"]
    ):
        raise ValueError("Expected full optimizer/scheduler state at the parent's last epoch")
    expected = {f"model.{k}": v for k, v in learned["model"].items()}
    actual = state.get("state_dict", {})
    if set(actual) != set(expected) or any(
        not torch.equal(actual[k], v) for k, v in expected.items()
    ):
        raise ValueError("Full checkpoint weights differ from the verified last-epoch model")
    # Aucun EMA ou arrêt anticipé : juste les rappels de sauvegarde présents
    if any(
        not k.startswith(("ModelCheckpoint{", "BestModelCallback{"))
        for k in state.get("callbacks", {})
    ):
        raise ValueError("Unsupported stateful training callback")
    return state


def continuation_origin(
    parent: Path, output: Path, config: dict[str, Any], device: str
) -> tuple[Path, dict[str, Any]]:
    """Un seul passage supplémentaire, sans modifier les paramètres ni les données."""
    if output.resolve().is_relative_to(parent.resolve()) or parent.resolve().is_relative_to(
        output.resolve()
    ):
        raise ValueError("Continuation output must be separate from its parent")
    report, weights = verify_run(parent)
    previous = report["config"]
    ignored = {"epochs", "continuation_checkpoint_sha256"}
    if (
        {k: v for k, v in config.items() if k not in ignored}
        != {k: v for k, v in previous.items() if k not in ignored}
        or config["epochs"] != previous["epochs"] + 1
        or config.get("checkpoint_selection") != "last_epoch"
        or "initial_checkpoint_sha256" in config
        or report["runtime"]["device"] != device
        or report["runtime"]["versions"] != runtime(device)["versions"]
    ):
        raise ValueError(
            "Continuation requires unchanged configuration, classes, runtime and device"
        )
    raw = read_local(parent, "checkpoints/training_config.json", 1_000_000)
    engine, engine_hash = json.loads(raw, object_pairs_hook=unique_object), digest(raw)
    settings = engine["train_config"]
    # Le planning 'step' sans chauffe ne dépend pas du nombre total de passages.
    # Une courbe cosinus serait différente en allongeant ce nombre : on la refuse.
    supported = {
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
    if any(settings.get(k) != v for k, v in supported.items()) or any(
        settings.get(k) != previous[k]
        for k in ("epochs", "seed", "batch_size", "grad_accum_steps", "lr")
    ):
        raise ValueError("Parent engine settings do not support this continuation")
    verify_training_data(parent, report)
    checksum = config["continuation_checkpoint_sha256"]
    state = read_full_state(parent, checksum)
    return weights, {
        "mode": "continuation",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "parent_run": str(parent.resolve()),
        "parent_report_sha256": file_hash(parent / "run.json"),
        "parent_engine_config_sha256": engine_hash,
        "full_checkpoint_sha256": checksum,
        "start_epoch": previous["epochs"],
        "start_global_step": state["global_step"],
        "optimizer_restored": True,
        "scheduler_restored": True,
        "limits": [
            "Checkpoint callbacks reset in a copy to protect parent paths",
            "Random generators reseeded per segment; not bit-identical to uninterrupted training",
        ],
    }


def copy_continuation_state(parent: Path, config: dict[str, Any], destination: Path) -> None:
    """La copie garde les calculs, mais aucun ancien chemin de sauvegarde actif."""
    state = read_full_state(parent, config["continuation_checkpoint_sha256"])
    state["callbacks"] = {}
    torch = importlib.import_module("torch")
    with destination.open("xb") as stream:
        torch.save(state, stream)


def extension_plan(
    root: Path, parent: Path, path: Path, device: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Vérifier le budget figé et les fichiers avant de demander le moindre calcul."""
    plan, checksum = read_document(path)
    expected = {"schema_version", "parent_report_sha256", "parent_state_sha256", "target_epochs"}
    if (
        set(plan) != expected
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
        or type(plan["target_epochs"]) is not int
        or any(
            not isinstance(plan[k], str) or not re.fullmatch(r"[a-f0-9]{64}", plan[k])
            for k in ("parent_report_sha256", "parent_state_sha256")
        )
        or file_hash(parent / "run.json") != plan["parent_report_sha256"]
    ):
        raise ValueError("Invalid extension plan or changed parent report")
    report, _ = verify_run(parent)
    start = report["config"]["epochs"]
    if not start < plan["target_epochs"] <= min(start + 8, 10):
        raise ValueError("Extension budget must add 1 to 8 epochs, at most 10 total")
    for name in CORPUS_FILES:
        if file_hash(root / name) != report["config"]["corpus_sha256"][name]:
            raise ValueError("Extension corpus changed")
    config = {
        **report["config"],
        "epochs": start + 1,
        "continuation_checkpoint_sha256": plan["parent_state_sha256"],
    }
    # Le chemin de contrôle n'est jamais crée juste les contrôles sont exécutés
    _, origin = continuation_origin(parent, parent.parent / "_extension_check", config, device)
    protocol = {
        "plan_sha256": checksum,
        "plan": plan,
        "device": device,
        "parent": str(parent.resolve()),
        "corpus": str(root.resolve()),
        "first_epoch": start + 1,
        "target_epochs": plan["target_epochs"],
        "initialization": origin,
        "test_used": False,
        "preprocessing": "public-v1",
        "mask_threshold": 0.5,
        "display_threshold": 0.3,
        "selection": "Save every epoch; no automatic promotion",
    }
    return protocol, report


@contextmanager
def extension_lock(output: Path) -> Iterator[None]:
    """Deux terminaux ne doivent pas entraîner le même essai en même temps."""
    lock = output / ".extension.lock"
    if lock.is_symlink():
        raise ValueError("Extension lock must not be linked")
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This extension is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def extend_training(
    root: Path,
    parent: Path,
    output: Path,
    plan_path: Path,
    device: str,
    *,
    check: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Alterner un passage et sa validation ; une relance reprend les étapes vérifiées."""
    protocol, initial = extension_plan(root, parent, plan_path, device)
    if any(
        output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
        for p in (root, parent)
    ):
        raise ValueError("Extension output must be outside all inputs")
    if check:
        return {"status": "ready", "protocol": protocol, "training_started": False}
    if output.is_symlink():
        raise ValueError("Extension output must not be linked")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with extension_lock(output):
        journal = output / "extension.json"
        state = {"schema_version": 1, "protocol": protocol, "status": "running", "epochs": []}
        if resume:
            state, _ = read_document(journal)
            if state.get("protocol") != protocol:
                raise ValueError("Extension protocol changed")
        state["status"] = "running"
        _save_extension(output, state)
        current, report = parent, initial
        try:
            for epoch in range(initial["config"]["epochs"], protocol["target_epochs"] + 1):
                # La première ligne est le contrôle parent, évalué avec le même protocole.
                if epoch > initial["config"]["epochs"]:
                    config = {
                        **report["config"],
                        "epochs": epoch,
                        "continuation_checkpoint_sha256": file_hash(
                            current / "checkpoints/last.ckpt"
                        ),
                    }
                    config_path = output / f"epoch_{epoch:03}.json"
                    child = output / f"epoch_{epoch:03}"
                    if config_path.exists():
                        if read_document(config_path)[0] != config:
                            raise ValueError("Saved extension configuration changed")
                    else:
                        write_json(config_path, config)
                    if child.exists():
                        if child.is_symlink():
                            raise ValueError("Saved epoch directory must not be linked")
                        saved, _ = read_document(child / "run.json")
                        if saved.get("config") != config:
                            raise ValueError("Saved epoch configuration changed")
                        if saved.get("status") == "completed":
                            report, _ = verify_run(child)
                            origin = report.get("initialization", {})
                            if (
                                origin.get("parent_report_sha256")
                                != file_hash(current / "run.json")
                                or origin.get("full_checkpoint_sha256")
                                != config["continuation_checkpoint_sha256"]
                            ):
                                raise ValueError("Saved epoch parent changed")
                        else:
                            resumable = saved.get("status") in ("failed", "interrupted") and bool(
                                saved.get("resume_checkpoint_sha256")
                            )
                            if not resumable:
                                # Avant la fin du passage, le moteur n'a pas toujours
                                # de point de reprise. On garde la tentative et on
                                # recommence ce seul passage depuis le parent intact.
                                _archive_attempt(output, child, state)
                                _save_extension(output, state)
                            report = train_model(
                                root,
                                child,
                                config_path,
                                None,
                                device,
                                continue_run=current,
                                resume=resumable,
                            )
                    else:
                        report = train_model(
                            root, child, config_path, None, device, continue_run=current
                        )
                    current = child
                    gc.collect()
                _evaluate_epoch(root, current, output, epoch, device, report, state)
                _save_extension(output, state)
            state["status"] = "completed"
        except BaseException as error:
            state.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error_type=type(error).__name__,
            )
            _save_extension(output, state)
            raise
        state.pop("error_type", None)
        _save_extension(output, state)
        return state


def _archive_attempt(output: Path, child: Path, state: dict[str, Any]) -> None:
    attempts = state.setdefault("restarted_attempts", [])
    index = len(attempts) + 1
    destination = output / f"{child.name}_attempt_{index:03}"
    if destination.exists() or destination.is_symlink():
        raise ValueError("Archived attempt destination already exists")
    child.rename(destination)
    attempts.append(
        {
            "run": child.name,
            "archived_as": destination.name,
            "reason": "No recorded resumable state; restart only this epoch",
        }
    )


def _evaluate_epoch(
    root: Path,
    run: Path,
    output: Path,
    epoch: int,
    device: str,
    report: dict[str, Any],
    state: dict[str, Any],
) -> None:
    from smartsite_ia.threshold_study import load_study_inputs
    from smartsite_ia.validation import evaluate_validation

    target = output / f"validation_{epoch:03}"
    previous = next((r for r in state["epochs"] if r["epoch"] == epoch), None)
    if target.exists():
        result, _, _, _ = load_study_inputs(run, root, target, report["class_names"][0])
    else:
        if previous is not None:
            raise ValueError("Recorded epoch evaluation is missing")
        result = evaluate_validation(run, root, target, device)
    row = {
        "epoch": epoch,
        "run": str(run.resolve()),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "evaluation_sha256": file_hash(target / "report.json"),
        "counts": result["counts"],
        "coco": result["coco"],
    }
    if previous is not None:
        if previous != row:
            raise ValueError("Previously recorded epoch or evaluation changed")
    else:
        state["epochs"].append(row)
    print(f"Passage {epoch} évalué : {target / 'index.html'}", flush=True)
    gc.collect()


def _save_extension(output: Path, state: dict[str, Any]) -> None:
    from smartsite_ia.validation_html import frame

    partial = output / "extension.json.partial"
    write_json(partial, state)
    partial.replace(output / "extension.json")
    rows = []
    for row in state["epochs"]:
        details = " · ".join(
            f"{name} : {c['tp']} retrouvé(s), {c['fp']} fausse(s) proposition(s), "
            f"{c['fn']} manqué(s)"
            for name, c in row["counts"].items()
        )
        rows.append(
            f'<li><a href="validation_{row["epoch"]:03}/index.html">'
            f"Passage {row['epoch']}</a> — {html.escape(details)}</li>"
        )
    page = frame(
        "SmartSite — apprentissage prolongé",
        "<p>Un passage supplémentaire, puis une évaluation native sur la validation. "
        "Chaque modèle est conservé ; aucune promotion automatique. "
        "Les scores restent à 0,30.</p><ul>" + "".join(rows) + "</ul>",
    )
    (output / "index.html").write_text(page, encoding="utf-8")
