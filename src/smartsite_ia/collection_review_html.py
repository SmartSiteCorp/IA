"""Montrer les corrections et leurs limites, sans les présenter comme des prédictions."""

import html
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from smartsite_ia.categories import (
    COLLECTION_COLORS,
    COLLECTION_LABELS,
    DEFECT_FAMILIES,
    FAMILY_LABELS,
)


def preview(root: Path, row: dict[str, Any]) -> None:
    """Dessiner seulement les rectangles proposés ; la photo d'origine est conservée."""
    with Image.open(root / row["image"]) as original:
        photo = original.convert("RGB")
    photo.thumbnail((1200, 1200))
    draw = ImageDraw.Draw(photo)
    for index, box in enumerate(row["boxes"], 1):
        x1, y1, x2, y2 = box["xyxy_normalized"]
        coords = (x1 * photo.width, y1 * photo.height, x2 * photo.width, y2 * photo.height)
        color = COLLECTION_COLORS[box["class"]]
        draw.rectangle(coords, outline=color, width=3)
        draw.text((coords[0] + 4, coords[1] + 4), str(index), fill=color, stroke_width=1)
    photo.save(root / "previews" / f"{row['id']}.jpg", quality=92)


def card(row: dict[str, Any]) -> str:
    name = row["id"]
    escape = html.escape
    labels = "".join(
        f"<li>{i}. {escape(COLLECTION_LABELS[b['class']])}</li>"
        for i, b in enumerate(row["boxes"], 1)
    )
    if not labels:
        # Une exclusion ambiguë n'est surtout pas un exemple sans défaut.
        message = (
            "Aucune des trois cibles visible dans ce recadrage revu."
            if row["is_crop"]
            else "Aucun rectangle retenu ; cette photo reste hors apprentissage."
        )
        if row["decision"] == "negative":
            message = "Photo entière revue sans cible visible dans le périmètre des trois classes."
        labels = f"<li>{message}</li>"
    families = sorted({FAMILY_LABELS[DEFECT_FAMILIES[b["class"]]] for b in row["boxes"]})
    status = "Écartée du lot d'essai" if row["decision"] == "excluded" else "Retenue pour l'essai"
    if row["is_crop"]:
        before = f'<p>Recadrage de <a href="#{row["parent_id"]}">{row["parent_id"]}</a>.<br>'
        before += "Ce n'est pas une nouvelle photo indépendante.</p>"
    else:
        caption = (
            "Avant : annotations de la source"
            if row["source_boxes"]
            else "Avant : photo sans annotation fournie"
        )
        before = f"""<figure><a href="source/previews/{name}.jpg">
<img loading="lazy" src="source/previews/{name}.jpg" alt="Annotations source de {name}"></a>
<figcaption>{caption}</figcaption></figure>"""
    credit = row["credit"]
    return f'''<article id="{name}"><h3>{escape(name)}</h3><p class="status">{status}
 · {escape(row["partition"])}</p><div class="comparison">{before}<figure>
<a href="previews/{name}.jpg"><img loading="lazy" src="previews/{name}.jpg"
alt="Propositions de l'assistant pour {name}"></a><figcaption>
Après : propositions de l'assistant, validation humaine en attente</figcaption></figure></div>
<p><strong>{escape(" · ".join(families))}</strong></p><ul>{labels}</ul>
<p>{escape(row["note"])}</p><p>Groupe conservé : <strong>{escape(row["group"])}</strong>.</p>
<p><a href="{row["image"]}">Voir la photo sans cadres</a></p>
<details><summary>Origine et droits</summary><p>{escape(credit["title"])}<br>
Auteur : {escape(credit["author"])}<br>
<a href="{escape(credit["source_url"], quote=True)}">Source</a>
 · <a href="{escape(credit["license_url"], quote=True)}">{escape(credit["license"])}</a></p>
<p>Copie orientée, aperçu réduit et rectangles ajoutés ; recadrage lorsque précisé.
Les droits de la photo restent applicables à ces adaptations.</p></details></article>'''


def write_review_gallery(root: Path, report: dict[str, Any]) -> None:
    records, summary = report["records"], report["summary"]
    for row in records:
        preview(root, row)
    kept = "".join(card(r) for r in records if r["decision"] != "excluded")
    excluded = "".join(card(r) for r in records if r["decision"] == "excluded")
    stats = "".join(
        f"<tr><th>{html.escape(split)}</th><td>{v['images']}</td><td>{v['groups']}</td>"
        f"<td>{v['negative_crops'] + v.get('negative_photos', 0)}</td></tr>"
        for split, v in summary["partitions"].items()
    )
    rules = "".join(
        f"<li><strong>{html.escape(COLLECTION_LABELS[name])}</strong> : {html.escape(rule)}</li>"
        for name, rule in report["protocol"]["class_rules"].items()
    )
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'">
<title>SmartSite — annotations revues</title><style>
body{{background:#f4f7fa;color:#203640;font:16px/1.6 system-ui;margin:0}}
main{{max-width:1100px;margin:auto;padding:26px 20px}}h1{{font-size:34px;line-height:1.2}}
article,.intro{{background:white;border:1px solid #d7e0e7;border-radius:12px;padding:22px;
margin:22px 0}}
.notice{{background:#fff1cf;padding:16px;border-radius:8px}}a{{color:#075f90}}
.comparison{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}figure{{margin:0}}
img{{width:100%;height:470px;object-fit:contain;background:#edf1f4}}
figcaption,details{{font-size:14px}}details{{overflow-wrap:anywhere}}.status{{font-weight:600}}
td,th{{padding:6px 18px;text-align:left;border-bottom:1px solid #d7e0e7}}
@media(max-width:650px){{.comparison{{grid-template-columns:1fr}}img{{height:350px}}
table{{width:100%;table-layout:fixed;font-size:13px}}
td,th{{padding:5px 3px;overflow-wrap:anywhere}}h1{{font-size:27px}}}}
</style></head><body><main><p>SmartSite · préparation de l'apprentissage</p>
<h1>Les annotations ont été revues</h1><section class="intro">
<p><strong>{summary["reviewed_photos"]} photos examinées</strong> :
{summary["candidate_photos"]} avec cibles ·
{summary.get("negative_photos", 0)} sans cible visible · {summary["excluded_photos"]} écartées.
{summary["negative_crops"]} recadrages sans cible visible ajoutés à l'apprentissage.</p>
<p class="notice"><strong>
Ce sont des corrections d'annotations, pas des détections du modèle.</strong>
Revue réalisée par l'assistant, sans validation experte ni confirmation terrain.
Rouge : moisissures suspectées · Orange : traces d'humidité possibles · Bleu : revêtement abîmé.</p>
<p>La perte de matière et l'écaillage partagent la famille « Éclats et dégradation du revêtement ».
Les sous-types restent distincts ; les anciens poids fissures/béton sont inchangés.</p>
<table><thead><tr><th>Lot pilote</th><th>Images</th><th>Groupes</th>
<th>Négatifs</th></tr></thead>
<tbody>{stats}</tbody></table><p>Les vues proches restent ensemble. Aucun test final créé.</p>
<details><summary>Règles et limites de ce petit lot</summary><ul>{rules}</ul>
<p>{html.escape(report["protocol"]["annotation_unit"])}</p>
<p>{html.escape(report["protocol"]["coverage"])}</p><p>{html.escape(report["protocol"]["limits"])}</p>
<p>Les rectangles donnent une zone à examiner, pas une surface exacte ni une gravité.
Ces quelques recadrages négatifs ne suffisent pas à mesurer les fausses alertes sur chantier.
</p></details>
<p><a href="report.json">Bilan et décisions</a> · <a href="review.json">Revue reproductible</a> ·
<a href="train/_annotations.coco.json">Boîtes d'apprentissage</a> ·
<a href="validation/_annotations.coco.json">Boîtes de validation</a> ·
<a href="source/">Collecte d'origine conservée</a></p></section>
<h2>Retenues pour un premier essai</h2>{kept}
<details><summary><strong>
{summary["excluded_photos"]} photos écartées : voir pourquoi</strong></summary>
<p>Les propositions éventuellement dessinées sur ces photos ne sont pas
exportées pour apprendre.</p>
{excluded}</details></main></body></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
