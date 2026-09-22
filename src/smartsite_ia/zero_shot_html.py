"""Montrer les propositions et les annotations source côte à côte."""

import html
import json
from pathlib import Path
from typing import Any


def write_probe_html(root: Path, report: dict[str, Any]) -> None:
    """Le rapport reste local, autonome et sans script ou image distante."""
    cards = []
    for row in report["records"]:
        sample_id = html.escape(row["id"], quote=True)
        counts = row["counts"]
        cards.append(
            f"<article><h2>{sample_id}</h2><p>{html.escape(row['review_note'])}</p>"
            '<div class="photos">'
            f'<figure><a href="{sample_id}/photo.jpg"><img loading="lazy" '
            f'src="{sample_id}/reference.jpg" alt="Annotations des auteurs"></a>'
            "<figcaption>Référence auteur « leakage » — vert</figcaption></figure>"
            f'<figure><a href="{sample_id}/prediction.jpg"><img loading="lazy" '
            f'src="{sample_id}/prediction.jpg" alt="Propositions du modèle"></a>'
            "<figcaption>Propositions à vérifier — orange</figcaption></figure></div>"
            f"<p>{len(row['predictions'])} proposition(s) · {counts['tp']} appariée(s) · "
            f"{counts['fp']} non appariée(s) · {counts['fn']} référence(s) manquée(s).</p>"
            f'<a href="{sample_id}/result.json">Détails et scores</a></article>'
        )
    summary = report["summary"]
    protocol = report["protocol"]
    metadata = html.escape(json.dumps(report["source"], ensure_ascii=False, indent=2))
    page = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartSite — essai de traces d’humidité</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f0f4f8;color:#163044;margin:0}}
main{{max-width:1400px;margin:auto;padding:24px}}h1{{font-size:30px}}
article,.intro{{background:white;padding:22px;border-radius:12px;margin:20px 0}}
.photos{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}figure{{margin:0}}
img{{width:100%;height:auto}}figcaption{{margin-top:8px;font-weight:600}}
p{{line-height:1.6}}a{{color:#075cac}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.warning{{border-left:5px solid #ed7425;padding-left:16px}}
@media(max-width:700px){{.photos{{grid-template-columns:1fr}}main{{padding:12px}}}}
</style><main><h1>Traces d’humidité : essai sans entraînement</h1>
<section class="intro"><p>{len(report["records"])} photos · Grounding DINO Tiny · CPU.</p>
<p>Consigne : <strong>{html.escape(protocol["prompt"])}</strong> · seuil objet
{protocol["box_threshold"]} · seuil texte {protocol["text_threshold"]}.</p>
<p><strong>{summary["tp"]}</strong> boîtes appariées aux annotations auteur,
<strong>{summary["fp"]}</strong> propositions non appariées,
<strong>{summary["fn"]}</strong> références manquées (IoU ≥ {protocol["match_iou"]}).</p>
<p>Ces comptes mesurent l’accord avec les annotations, pas la fiabilité sur un chantier.
Une image sans proposition ne signifie pas « aucun défaut ».</p>
<p class="warning">{html.escape(report["limitations"])}</p>
<p>Les scores ne sont pas des probabilités de fuite. Cliquer sur une référence ouvre la photo
sans annotation. <a href="report.json">Résultats complets</a>.</p>
<details><summary>Source, attribution et licence</summary><pre>{metadata}</pre></details>
</section>{"".join(cards)}</main></html>"""
    (root / "index.html").write_text(page, encoding="utf-8")
