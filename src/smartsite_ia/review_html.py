"""Une page de revue qui s'ouvre hors ligne, sans serveur ni script externe."""

from html import escape
from pathlib import Path
from string import Template
from typing import Any

LABELS = {
    "empty_annotations": "Sans annotation : à vérifier",
    "mask_iou_below_review_threshold": "Recouvrement faible : à revoir",
    "mask_difference_away_from_boundary": "Écart au-delà des bords",
    "source_classes_almost_fully_overlap": "Deux catégories presque entièrement superposées",
}


def write_review_page(destination: Path, report: dict[str, Any]) -> None:
    details, rows, pairs = [], [], []
    for record in report["records"]:
        sample_id = escape(record["id"], quote=True)
        flags = "; ".join(LABELS[f] for f in record["flags"]) or "Pas de signal automatique"
        rows.append(
            f'<tr><td><img loading="lazy" width="80" height="80" '
            f'src="thumbnails/{sample_id}.jpg" alt="Photo {sample_id}"></td>'
            f"<td>{sample_id}</td><td>{record['annotation_count']}</td>"
            f"<td>{escape(flags)}</td><td>{record['coco_disagreement_pixels']}</td></tr>"
        )
        if record["id"] in report["detailed_samples"]:
            figures = []
            for file, label in (
                ("photo.jpg", "Photo"),
                ("source.png", "Masque fourni"),
                ("coco.png", "Polygones source dessinés par COCO"),
                ("differences.png", "Écarts en rose sur la photo"),
            ):
                src = f"samples/{sample_id}/{file}"
                figures.append(
                    f'<figure><a href="{src}"><img loading="lazy" width="640" height="640" '
                    f'src="{src}" '
                    f'alt="{label} — {sample_id}"></a><figcaption>{label}</figcaption></figure>'
                )
            details.append(
                f'<article id="{sample_id}"><h3>{sample_id}</h3><p>{escape(flags)}. '
                f"Annotations : {record['annotation_count']}. "
                f"Pixels communs aux deux catégories : {record['cross_class_overlap_pixels']}."
                f'</p><div class="panels">'
                f"{''.join(figures)}</div></article>"
            )
    for pair in report["similar_pairs"]:
        left, right = escape(pair["left"]), escape(pair["right"])
        pairs.append(
            f'<tr><td><a href="samples/{left}/photo.jpg"><img loading="lazy" '
            f'width="160" height="160" '
            f'src="thumbnails/{left}.jpg" alt="{left}"></a>'
            f'<br>{left}</td><td><a href="samples/{right}/photo.jpg"><img loading="lazy" '
            f'width="160" height="160" '
            f'src="thumbnails/{right}.jpg" alt="{right}"></a><br>{right}</td>'
            f"<td>{escape(pair['kind'])}</td>"
            f"<td>{escape(pair['transform_right'])}</td>"
            f"<td>{pair['hamming_distance']} / 128</td>"
            f"<td>{pair['mean_absolute_error']:.2f} / 255</td></tr>"
        )
    template = Template(
        Path(__file__).with_name("review_template.html").read_text(encoding="utf-8")
    )
    html = template.substitute(
        images=report["images"],
        annotations=report["annotations"],
        pair_count=len(report["similar_pairs"]),
        details="".join(details),
        pairs="".join(pairs),
        rows="".join(rows),
        max_distance=report["parameters"]["max_hamming_distance"],
        max_error=report["parameters"]["max_rgb_mean_absolute_error"],
        manifest_sha256=escape(report["manifest_sha256"]),
    )
    (destination / "index.html").write_text(html, encoding="utf-8")
