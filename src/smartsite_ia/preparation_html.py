"""Écrire un bilan local lisible, sans script ni service externe."""

import json
from html import escape
from pathlib import Path
from typing import Any

SPLIT_LABELS = {
    "train": "Apprentissage",
    "valid": "Validation interne",
    "test": "Test interne",
    "quarantine": "À part — à revoir",
    "external_candidate": "Référence externe candidate",
    "external_variant": "Vue similaire conservée",
}


def write_preparation_page(destination: Path, report: dict[str, Any]) -> None:
    """Le détail reste disponible même si l'utilisateur ouvre la page hors ligne."""
    rows = []
    for record in report["records"]:
        sample_id = escape(record["id"])
        image = escape(record["image"], quote=True)
        preview = escape(record.get("preview", record["image"]), quote=True)
        mask_links = ""
        if "mask" in record:
            mask_links = (
                f'<br><a href="{escape(record["mask"], quote=True)}">Masque</a> · '
                f'<a href="{escape(record["preview"], quote=True)}">Superposition</a>'
            )
        rows.append(
            f'<tr id="image-{sample_id}"><td><a href="{image}">{sample_id}</a>{mask_links}</td>'
            f'<td><a href="{image}"><img src="{preview}" width="120" height="120" '
            f'loading="lazy" alt="Photo {sample_id}"></a></td>'
            f"<td>{escape(SPLIT_LABELS[record['split']])}</td>"
            f"<td>{escape(record['content_group'])}</td>"
            f"<td>{escape(record.get('reason', ''))}</td></tr>"
        )
    summary = {k: v for k, v in report.items() if k != "records"}
    page = """<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self';
style-src 'unsafe-inline'; base-uri 'none'">
<title>SmartSite — préparation du corpus</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172d3c}
table{width:100%;border-collapse:collapse}
td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}
img{object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f6;padding:20px}
a{color:#075d9d}.notice{background:#fff3d2;padding:16px}.cards{display:flex;gap:12px;flex-wrap:wrap}
.card{background:#eef3f6;padding:18px;border-radius:12px}.card strong{display:block;font-size:28px}
details{margin:20px 0}summary{cursor:pointer}nav{margin:20px 0}</style>
<h1>Préparation du corpus SmartSite</h1>
<p class="notice">Ce rapport décrit les données préparées, pas les résultats d'une IA.
Les limites et l'usage autorisé sont indiqués ci-dessous. Les groupes de contenu ne prouvent
pas l'indépendance des scènes. Les fichiers originaux sont conservés.</p>
<p><a href="report.json">Bilan JSON</a> · <a href="manifest.json">Manifeste</a> ·
<a href="policy.json">Décisions appliquées</a> · <a href="ATTRIBUTION.json">Attribution</a></p>
"""
    page += (
        '<nav><a href="#images">Voir les images</a> · <a href="#paires">Voir les paires</a></nav>'
    )
    counts = report.get("split_counts") or {
        **{k: v["images"] for k, v in report["splits"].items()},
        "quarantine": report["quarantined_images"],
    }
    page += (
        '<div class="cards">'
        + "".join(
            f'<div class="card"><strong>{count}</strong>{SPLIT_LABELS[split]}</div>'
            for split, count in counts.items()
        )
        + "</div>"
    )
    if report["dataset"] == "damsegment_v1":
        page += """<p><b>Usage : premiers essais d'entraînement.</b> Catégories retenues pour
SmartSite après inspection : 0 = fissure, 1 = perte de matière du béton. Cette interprétation
n'est pas une correspondance numérique confirmée par les auteurs. Les contours sont conservés.</p>
<p>Les trois partitions proviennent du même barrage. Les paires connues restent ensemble,
mais les scènes d'origine manquent : ces scores internes ne démontreront pas la performance
sur de nouveaux chantiers.</p>"""
    else:
        page += """<p><b>Usage : réserve externe à qualifier, exclue de l'apprentissage.</b>
Les photos sont orientées et conservées en PNG à leur résolution complète. Les masques blancs
désignent les fissures ; la superposition rouge permet de vérifier leur alignement.</p>
<p>Les images similaires sont regroupées. Une seule référence par groupe est proposée ;
les autres vues restent disponibles. Les paires dont les masques diffèrent sont mises à part.
La qualification finale et les exemples négatifs difficiles restent à compléter.</p>"""
    page += '<h2 id="paires">Paires et groupes conservés ensemble</h2>'
    pairs = report.get("similar_pairs", report.get("recomputed_similar_pairs", []))
    page += "<p>Comparaison de données : ces valeurs ne mesurent pas une prédiction IA.</p>"
    for pair in pairs:
        left, right = escape(pair["left"]), escape(pair["right"])
        detail = (
            f" — recouvrement des masques : {pair['binary_mask_iou']:.2%}"
            if "binary_mask_iou" in pair
            else ""
        )
        page += (
            f'<p><a href="#image-{left}">{left}</a> ↔ '
            f'<a href="#image-{right}">{right}</a>{detail}</p>'
        )
    page += "<details><summary>Traçabilité, paramètres et limites détaillés</summary>"
    page += f"<pre>{escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></details>"
    page += (
        '<h2 id="images">Images et décisions</h2><table><thead><tr><th>Image</th><th>Aperçu</th>'
    )
    page += "<th>Destination</th><th>Groupe de contenu</th><th>Motif</th></tr></thead><tbody>"
    page += "".join(rows) + "</tbody></table></html>"
    (destination / "index.html").write_text(page, encoding="utf-8")
