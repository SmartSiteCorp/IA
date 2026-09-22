"""Comparer des scores objets par classe, sans refaire l'inférence ni toucher au test."""

from pathlib import Path
from typing import Any

from smartsite_ia.annotations import number
from smartsite_ia.categories import get_class_names
from smartsite_ia.curation import digest, read_document, staged_output
from smartsite_ia.importer import write_json
from smartsite_ia.inference import inference_metadata
from smartsite_ia.learning import verify_run
from smartsite_ia.prediction import decode_photo, render_predictions
from smartsite_ia.review import MAX_FILE_BYTES, read_local
from smartsite_ia.saved_predictions import load_predictions
from smartsite_ia.validation import (
    DISPLAY_THRESHOLD,
    SCORE_FLOOR,
    load_validation,
    validation_protocol,
)
from smartsite_ia.validation_metrics import analyze_masks, coco_api, match_masks, summarize_counts

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def checked_thresholds(values: tuple[float, ...]) -> tuple[float, ...]:
    """La grille est fixée avant calcul et ne descend pas sous les sorties conservées."""
    if not 2 <= len(values) <= 20:
        raise ValueError("Expected between 2 and 20 thresholds")
    grid = tuple(sorted(number(v) for v in values))
    if len(set(grid)) != len(grid) or DISPLAY_THRESHOLD not in grid:
        raise ValueError("Thresholds must be unique and include baseline 0.3")
    if grid[0] < SCORE_FLOOR or grid[-1] >= 1:
        raise ValueError("Thresholds must cover saved scores only, below 1")
    return grid


def f1(counts: dict[str, Any]) -> float | None:
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 2 * counts["tp"] / denominator if denominator else None


def threshold_cases(
    references: list[dict[str, Any]], proposals: list[dict[str, Any]], grid: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Un seul appariement trié : ajouter un score plus faible ne déplace pas les précédents."""
    matches = match_masks(
        [r["segmentation"] for r in references],
        [{"segmentation": p["mask_rle"], "score": p["score"]} for p in proposals],
    )
    events = []
    for match in matches:
        proposal = proposals[match["prediction_index"]]
        reference = match["reference_index"]
        events.append(
            {
                "id": proposal["id"],
                "score": proposal["score"],
                "reference_id": references[reference]["id"] if reference is not None else None,
                "reason": match["reason"],
                "best_iou": match["best_iou"],
            }
        )
    baseline = {p["id"] for p in events if p["score"] >= DISPLAY_THRESHOLD}
    cases = []
    for threshold in grid:
        selected = [p for p in events if p["score"] >= threshold]
        ids = {p["id"] for p in selected}
        tp = sum(p["reason"] == "matched" for p in selected)
        cases.append(
            {
                "threshold": threshold,
                "counts": {"tp": tp, "fp": len(selected) - tp, "fn": len(references) - tp},
                "errors": {
                    reason: sum(p["reason"] == reason for p in selected)
                    for reason in ("duplicate", "insufficient_overlap", "no_overlap")
                },
                "added": [p for p in selected if p["id"] not in baseline],
                "removed": [p for p in events if p["id"] in baseline - ids],
            }
        )
    return cases


def load_study_inputs(
    run: Path, corpus: Path, evaluation: Path, target: str
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Rattacher les sorties à leurs poids, photos et annotations, avant les mesures."""
    training, _ = verify_run(run)
    source, checksum = read_document(evaluation / "report.json")
    names = get_class_names(source)
    document, records, annotation_hash = load_validation(corpus, training, names)
    settings = source.get("inference", {})
    if (
        target not in names
        or not set(names).issubset(get_class_names(training.get("config", {})))
        or source.get("status") != "completed"
        or source.get("split") != "valid"
        or source.get("test_used") is not False
        or source.get("protocol") != validation_protocol()
        or source.get("checkpoint_sha256") != training["checkpoint_sha256"]
        or source.get("annotations_sha256") != annotation_hash
        or not isinstance(settings, dict)
        or settings
        != inference_metadata(
            settings.get("preprocessing", ""), settings.get("mask_probability_threshold", 0)
        )
    ):
        raise ValueError(
            "Study requires matching completed validation, weights and inference settings"
        )
    old = source.get("records")
    if (
        not isinstance(old, list)
        or source.get("images") != len(records)
        or [r.get("id") for r in old if isinstance(r, dict)] != [r["id"] for r in records]
    ):
        raise ValueError("Saved validation records disagree with the pinned corpus")
    return (
        source,
        document,
        records,
        {"evaluation_report": checksum, "annotations": annotation_hash},
    )


def study_thresholds(
    run: Path,
    corpus: Path,
    evaluation: Path,
    output: Path,
    target: str = "surface_loss",
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Livrer toutes les variantes et les alertes ajoutées, sans sélectionner un seuil métier."""
    grid = checked_thresholds(thresholds)
    source, document, records, hashes = load_study_inputs(run, corpus, evaluation, target)
    names = get_class_names(source)
    label = names.index(target)
    images = {im["file_name"]: im for im in document["images"]}
    api = coco_api(document)
    rows = []
    with staged_output(output, [run, corpus, evaluation]) as stage:
        for record, old in zip(records, source["records"], strict=True):
            sid = record["id"]
            raw = read_local(corpus, f"valid/{sid}.jpg", MAX_FILE_BYTES)
            if digest(raw) != record["image_sha256"] or old.get("image_sha256") != digest(raw):
                raise ValueError(f"Study photo changed: {sid}")
            photo = decode_photo(raw)
            im = images[f"{sid}.jpg"]
            if (
                photo.size != (im["width"], im["height"])
                or old.get("image_id") != im["id"]
                or old.get("difficulty") != record["difficulty"]
            ):
                raise ValueError("Study image metadata disagree")
            proposals, hashes[f"predictions/{sid}"] = load_predictions(
                evaluation, sid, photo.size, names
            )
            refs = [{**a, "segmentation": api.annToRLE(a)} for a in api.imgToAnns[im["id"]]]
            predicted = [
                {
                    "category_id": p["class_id"] + 1,
                    "score": p["score"],
                    "segmentation": p["mask_rle"],
                }
                for p in proposals
            ]
            baseline = analyze_masks(refs, predicted, DISPLAY_THRESHOLD, names)
            if {n: r["counts"] for n, r in baseline.items()} != old.get("counts") or {
                n: r["errors"] for n, r in baseline.items()
            } != old.get("mask_errors"):
                raise ValueError(f"Saved baseline counts or errors disagree: {sid}")
            target_refs = [r for r in refs if r["category_id"] == label + 1]
            target_proposals = [p for p in proposals if p["class_id"] == label]
            variants = threshold_cases(target_refs, target_proposals, grid)
            if variants[grid.index(DISPLAY_THRESHOLD)]["counts"] != baseline[target]["counts"]:
                raise ValueError("Threshold matching disagrees with the baseline")
            row = {**record, "baseline_counts": old["counts"], "variants": variants}
            rows.append(row)
            folder = stage / sid
            folder.mkdir()
            (folder / "photo.jpg").write_bytes(raw)
            annotated = [{"class_id": label, "mask_rle": a["segmentation"]} for a in target_refs]
            render_predictions(photo, annotated, show_boxes=False, class_names=names).save(
                folder / "reference.jpg", quality=92
            )
            for index, variant in enumerate(variants):
                visible = [p for p in target_proposals if p["score"] >= variant["threshold"]]
                render_predictions(photo, visible, show_boxes=False, class_names=names).save(
                    folder / f"threshold_{index}.jpg", quality=92
                )
        if summarize_counts([r["baseline_counts"] for r in rows], names) != source["counts"]:
            raise ValueError("Saved aggregate baseline counts disagree")
        summaries = []
        for index, threshold in enumerate(grid):
            cases = [r["variants"][index] for r in rows]
            counts = summarize_counts(
                [{**r["baseline_counts"], target: r["variants"][index]["counts"]} for r in rows],
                names,
            )
            summaries.append(
                {
                    "threshold": threshold,
                    "counts": counts,
                    "target_f1": f1(counts[target]),
                    "changed_images": [
                        r["id"]
                        for r, c in zip(rows, cases, strict=True)
                        if c["added"] or c["removed"]
                    ],
                    "errors": {
                        reason: sum(c["errors"][reason] for c in cases)
                        for reason in ("duplicate", "insufficient_overlap", "no_overlap")
                    },
                    "images_with_false_proposals": sum(c["counts"]["fp"] > 0 for c in cases),
                    "without_target_annotation": {
                        "images": sum(c["counts"]["tp"] + c["counts"]["fn"] == 0 for c in cases),
                        "false_proposals": sum(
                            c["counts"]["fp"]
                            for c in cases
                            if c["counts"]["tp"] + c["counts"]["fn"] == 0
                        ),
                    },
                }
            )
        scored = [s["target_f1"] for s in summaries if s["target_f1"] is not None]
        best = max(scored) if scored else None
        result = {
            "schema_version": 1,
            "status": "completed",
            "split": "valid",
            "test_used": False,
            "qualified_for_smartsite": False,
            "selected_threshold": None,
            "target_class": target,
            "class_names": list(names),
            "images": len(rows),
            "thresholds": list(grid),
            "baseline_threshold": DISPLAY_THRESHOLD,
            "fixed_thresholds": {n: DISPLAY_THRESHOLD for n in names if n != target},
            "checkpoint_sha256": source["checkpoint_sha256"],
            "inference": source["inference"],
            "source_hashes": hashes,
            "implementation_sha256": {
                name: digest(Path(__file__).with_name(name).read_bytes())
                for name in (
                    "threshold_study.py",
                    "threshold_html.py",
                    "saved_predictions.py",
                    "validation_metrics.py",
                    "validation.py",
                    "prediction.py",
                )
            },
            "protocol": source["protocol"],
            "source_coco": source["coco"],
            "summaries": summaries,
            "records": rows,
            "best_observed_f1_thresholds": [
                s["threshold"] for s in summaries if best is not None and s["target_f1"] == best
            ],
            "limits": [
                "Descriptive validation sweep, not probability calibration or a deployed threshold",
                "Business costs and independent qualification remain undefined",
                "Saved outputs only; hashes pin this study's inputs, not their original creation",
                "Annotations remain unchanged; unannotated photos are not certified defect-free",
                "AP ranking unchanged; no new inference, weights, pixel threshold or preprocessing",
            ],
        }
        from smartsite_ia.threshold_html import write_threshold_pages

        write_json(stage / "report.json", result)
        write_threshold_pages(stage, result)
    return result
