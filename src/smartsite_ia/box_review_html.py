"""Montrer les références et les sorties du détecteur sans confondre leurs rôles."""

import html
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from smartsite_ia.categories import COLLECTION_COLORS as COLORS
from smartsite_ia.categories import COLLECTION_LABELS


def render_boxes(photo: Image.Image, records: list[dict[str, Any]], prefix: str) -> Image.Image:
    """Dessiner sur un aperçu ; les coordonnées conservées restent celles de l'original."""
    result = photo.copy()
    result.thumbnail((1200, 1200))
    sx, sy = result.width / photo.width, result.height / photo.height
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=16)
    for box in records:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        coords = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        color = COLORS[box["class_name"]]
        draw.rectangle(coords, outline=color, width=3)
        label = f"{prefix}{box['id']}"
        if "score" in box:
            label += f" {box['score']:.0%}"
        width = draw.textlength(label, font=font)
        point = (min(coords[0], max(0, result.width - width - 4)), max(0, coords[1] - 20))
        draw.rectangle(draw.textbbox(point, label, font=font), fill="white")
        draw.text(point, label, font=font, fill=color)
    return result


def card(row: dict[str, Any]) -> str:
    escape = html.escape
    name = escape(row["id"], quote=True)
    references = "".join(
        f"<li><b>R{r['id']}</b> · {escape(COLLECTION_LABELS[r['class_name']])} · "
        f"{'manquée' if r['id'] in row['missed_reference_ids'] else 'retrouvée'}</li>"
        for r in row["references"]
    )
    pairs = {m["prediction_id"]: m for m in row["matches"]}
    predictions = []
    reasons = {
        "duplicate": "proposition en double",
        "insufficient_overlap": "recouvrement insuffisant avec une référence de cette classe",
        "no_overlap": "aucun recouvrement avec une référence de cette classe",
    }
    for prediction in row["predictions"]:
        match = pairs[prediction["id"]]
        status = (
            f"retrouve R{match['reference_id']}"
            if match["reference_id"] is not None
            else reasons[match["reason"]]
        )
        predictions.append(
            f"<li><b>P{prediction['id']}</b> · "
            f"{escape(COLLECTION_LABELS[prediction['class_name']])} · "
            f"score {prediction['score']:.1%} · {status}</li>"
        )
    totals = {k: sum(c[k] for c in row["counts"].values()) for k in ("tp", "fp", "fn")}
    credit = row["credit"]
    return f'''<article id="{name}"><h2>{name}</h2>
<p class="counts">{totals["tp"]} zone(s) retrouvée(s) · {totals["fn"]} manquée(s)
· {totals["fp"]} proposition(s) supplémentaire(s)</p><div class="photos">
<figure><a href="{name}/reference.jpg"><img loading="lazy" src="{name}/reference.jpg"
alt="Annotations de référence pour {name}"></a><figcaption>Annotations à confirmer</figcaption>
<ul>{references or "<li>Aucune référence dans cette image.</li>"}</ul></figure>
<figure><a href="{name}/prediction.jpg"><img loading="lazy" src="{name}/prediction.jpg"
alt="Prédictions du modèle pour {name}"></a><figcaption>Prédictions du modèle</figcaption>
<ul>{"".join(predictions) or "<li>Aucune proposition au seuil choisi.</li>"}</ul></figure></div>
<p><a href="{name}/photo.jpg">Photo sans cadres</a> ·
<a href="{name}/result.json">Détails des rectangles et des scores</a></p>
<details><summary>Note d'annotation et origine</summary><p>{escape(row["annotation_note"])}</p>
<p>Groupe : {escape(row["group"])}. {len(row["discarded"])} sortie(s) sans objet ou sans surface
écartée(s), tracée(s) dans les détails.</p><p>{escape(credit["title"])}<br>
Auteur : {escape(credit["author"])}<br>Source : {escape(credit["source_url"])}<br>
Licence : {escape(credit["license"])} — {escape(credit["license_url"])}</p>
<p>Aperçus réduits et rectangles ajoutés. Les droits d'origine restent applicables.</p>
</details></article>'''


def write_box_gallery(root: Path, report: dict[str, Any]) -> None:
    """Une galerie locale autonome, sans ressource distante ni script à exécuter."""
    cards = []
    for row in report["records"]:
        folder = root / row["id"]
        with Image.open(folder / "photo.jpg") as source:
            photo = source.convert("RGB")
        for kind, records, prefix in (
            ("reference", row["references"], "R"),
            ("prediction", row["predictions"], "P"),
        ):
            render_boxes(photo, records, prefix).save(folder / f"{kind}.jpg", quality=92)
        cards.append(card(row))
    summary = report["summary"]
    totals = {k: sum(c[k] for c in summary.values()) for k in ("tp", "fp", "fn")}
    stats = "".join(
        f"<tr><th>{html.escape(COLLECTION_LABELS[name])}</th><td>{row['tp']}</td>"
        f"<td>{row['fn']}</td><td>{row['fp']}</td></tr>"
        for name, row in summary.items()
    )
    threshold = report["protocol"]["score_threshold"]
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'">
<title>SmartSite — les prédictions du pilote</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f6f7;color:#19373e;font:16px/1.6 system-ui}}
main{{max-width:1200px;margin:auto;padding:28px 22px}}h1{{font-size:36px;line-height:1.2}}
h2{{font-size:22px;overflow-wrap:anywhere}}.eyebrow{{color:#296768;font-weight:700}}
article,.intro{{background:white;border:1px solid #d6e1e4;border-radius:14px;
padding:24px;margin:24px 0}}
.notice{{padding:16px;background:#fff3d9;border-radius:8px}}.counts{{font-weight:650}}
.photos{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}figure{{margin:0;min-width:0}}
img{{width:100%;height:430px;object-fit:contain;background:#edf2f4;border-radius:7px}}
figcaption{{font-weight:700;margin:10px 0}}a{{color:#006879}}li{{margin-bottom:8px}}
ul{{padding-left:20px;font-size:14px}}details{{overflow-wrap:anywhere;font-size:14px}}
table{{border-collapse:collapse;width:100%;text-align:left}}
td,th{{padding:12px 8px;border-bottom:1px solid #dce5e8}}
thead th{{font-size:14px}}.table{{overflow:auto}}.legend{{font-size:14px}}
@media(max-width:680px){{.photos{{grid-template-columns:1fr}}main{{padding:12px}}
article,.intro{{padding:16px}}h1{{font-size:29px}}img{{height:340px}}
table{{font-size:13px}}td,th{{padding:9px 4px}}}}
</style></head><body><main><p class="eyebrow">SmartSite · revue des résultats</p>
<h1>Ce que le modèle retrouve</h1><section class="intro">
<p>{len(report["records"])} photos de validation · modèle après
{report["training_epochs"]} passages d'apprentissage.</p>
<p class="counts">{totals["tp"]} zone(s) retrouvée(s) sur {totals["tp"] + totals["fn"]} annotées
· {totals["fp"]} proposition(s) supplémentaire(s).</p>
<p class="notice">Essai exploratoire. Les annotations ont été proposées par l'assistant et
restent à confirmer. Une proposition supplémentaire peut être une erreur du modèle ou une
limite de l'annotation. Aucune proposition ne signifie pas « surface conforme ».</p>
<div class="table"><table><thead><tr><th>Catégorie</th><th>Retrouvées</th><th>Manquées</th>
<th>En plus</th></tr></thead><tbody>{stats}</tbody></table></div>
<p>À gauche, les annotations (R). À droite, les prédictions (P).
Cliquer sur une image pour l'agrandir.</p><p class="legend">
<span style="color:{COLORS["mold_suspected"]}">■ Moisissures suspectées</span> ·
<span style="color:{COLORS["moisture_trace"]}">■ Traces d'humidité</span> ·
<span style="color:{COLORS["peeling_paint"]}">■ Peinture écaillée</span></p>
<details><summary>Comprendre les scores et les comptes</summary>
<p>Seuil de score commun : {threshold:.0%}, fixé pour cette inspection et non calibré.
Le score n'est ni une probabilité de défaut, ni une gravité.
Une référence est retrouvée si la catégorie correspond et si le recouvrement des rectangles
atteint 50 % (intersection divisée par union). Chaque référence ne compte qu'une fois.</p>
<p>Ces comptes utilisent un seuil fixe ; ils diffèrent du F1 du journal, qui choisit son seuil
sur la validation. Les huit photos ne constituent pas un test indépendant.
Le petit lot ne mesure pas les fausses alertes générales sur chantier.</p></details>
<p><a href="report.json">Rapport complet</a></p></section>{"".join(cards)}</main></body></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
