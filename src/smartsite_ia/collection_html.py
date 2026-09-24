"""Afficher les nouvelles données, leur origine et le travail d'annotation restant."""

import html
from pathlib import Path
from typing import Any


def write_collection_html(root: Path, report: dict[str, Any], classes: dict[str, str]) -> None:
    cards = []
    for row in report["records"]:
        sample_id, credit = row["id"], row["credit"]
        labels = ", ".join(classes[c] for c in sorted({b["class"] for b in row["boxes"]}))
        status = "À annoter" if not labels else "Boîtes de la source à revoir : " + labels
        if row["decision"] == "quarantine":
            status = "Mise de côté — " + status
        source = html.escape(credit["source_url"], quote=True)
        license_url = html.escape(credit["license_url"], quote=True)
        cards.append(f'''<article><h3>{html.escape(sample_id)}</h3>
<p class="status">{html.escape(status)}</p>
<a href="images/{sample_id}.jpg"><img loading="lazy" src="previews/{sample_id}.jpg"
alt="Photo {sample_id}, avec les annotations de la source lorsqu'elles existent"></a>
<p>{html.escape(row["note"])}</p>
<p>Groupe de précaution : <strong>{html.escape(row["group"])}</strong>.<br>
Lot chez l'auteur : {html.escape(row["source_split"])} ; affectation SmartSite en attente.</p>
<details><summary>Origine et droits</summary><p>{html.escape(credit["title"])}<br>
Auteur : {html.escape(credit["author"])}</p><p><a href="{source}">Page source</a> ·
<a href="{license_url}">{html.escape(credit["license"])}</a> ·
<a href="originals/{sample_id}.image">Original conservé</a></p>
<p>Copie de travail remise à l'endroit, convertie en JPEG sans métadonnées.
Aperçu réduit ; cadres de la source ajoutés lorsqu'ils existent. Les licences et attributions
des photos restent applicables à leurs copies et aperçus.</p></details></article>''')
    summary = report["summary"]
    exclusions = "".join(
        f"<li>{html.escape(r['id'])} : {html.escape(r['reason'])}</li>"
        for r in report["excluded_candidates"]
    )
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'">
<title>SmartSite — nouvelles photos à préparer</title><style>
body{{margin:0;background:#f4f7fa;color:#203640;font:16px/1.6 system-ui}}
main{{max-width:1240px;margin:auto;padding:32px 22px}}h1{{font-size:34px;line-height:1.2}}
.intro,article{{background:white;border:1px solid #d7e0e7;border-radius:12px;padding:22px}}
.intro{{margin:24px 0}}.notice{{background:#fff1cf;padding:16px;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}
article{{padding:16px}}h3{{font-size:17px}}img{{width:100%;height:310px;object-fit:contain;
background:#edf1f4}}a{{color:#075f90}}.status{{min-height:76px;font-weight:600}}
details{{font-size:14px;overflow-wrap:anywhere}}.muted{{color:#506878}}
@media(max-width:950px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:600px){{.grid{{grid-template-columns:1fr}}h1{{font-size:27px}}}}
</style></head><body><main><p class="muted">SmartSite · préparation des données</p>
<h1>De nouvelles surfaces pour la suite</h1><section class="intro">
<p><strong>{summary["images"]} photos</strong> · {summary["with_source_boxes"]} avec des boîtes
source à revoir · {summary["to_annotate"]} à annoter · {summary["quarantined"]} cas mis de côté.</p>
<p class="notice"><strong>Ce sont des données, pas des détections du modèle.</strong>
Rouge : moisissures suspectées. Orange : traces d'humidité possibles.
Bleu : revêtement écaillé. Aucun nouvel entraînement.</p>
<details><summary>Préparation, limites et prochaine étape</summary>
<p>{summary["orientation_corrected"]} orientations corrigées ; {summary["groups"]} groupes
de précaution. {len(report["source_split_conflicts"])} groupes croisent les lots train/validation
de l'auteur : ils devront rester ensemble dans notre future séparation.</p>
<p>Les cadres viennent des annotations des sources, selon la correspondance de catégories
documentée dans la sélection. Ils restent à corriger ; ils ne confirment aucune cause physique.
Aucune photo n'est encore approuvée pour l'apprentissage.</p>
<p>{html.escape(report["limitations"])}</p>
<details><summary>{summary["excluded_before_preparation"]}
photos écartées avant préparation</summary>
<ul>{exclusions}</ul></details>
<p>La suite : revoir les zones et leurs catégories, compléter les annotations et les confusions,
puis figer les groupes et les lots.
Une photo sans cadre n'est jamais considérée saine par défaut.</p></details>
<p><a href="report.json">Rapport complet</a> · <a href="review_queue.json">File de revue</a> ·
<a href="selection.json">Sélection reproductible</a></p></section>
<div class="grid">{"".join(cards)}</div></main></body></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
