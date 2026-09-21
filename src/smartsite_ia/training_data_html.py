"""Montrer ce qui entre dans l'apprentissage, sans présenter des photos comme des prédictions."""

import html
from pathlib import Path
from typing import Any

from smartsite_ia.categories import CLASS_LABELS, get_class_names


def write_training_data_page(output: Path, selection: dict[str, Any]) -> None:
    """Un bilan léger et quelques photos sans cible, choisies dans l'ordre des IDs."""
    labels = ", ".join(CLASS_LABELS[name] for name in get_class_names(selection))
    rows, examples = [], []
    for split, title in (("train", "Apprentissage"), ("valid", "Validation")):
        details = selection["splits"][split]
        rows.append(
            f"<tr><td>{title}</td><td>{details['images']}</td>"
            f"<td>{details['annotations']}</td><td>{details['excluded_annotations']}</td>"
            f"<td>{details['images_without_target_annotations']}</td></tr>"
        )
        # Ces exemples servent à revoir les données il peut y avoir des defauts 
        for sample_id in details["without_target_annotation_ids"][:3]:
            path = html.escape(f"{split}/{sample_id}.jpg", quote=True)
            examples.append(
                f'<figure><a href="{path}"><img loading="lazy" src="{path}" '
                f'alt="{title} : {html.escape(sample_id)}"></a>'
                f"<figcaption>{title} — {html.escape(sample_id)}<br>"
                "Sans annotation de la catégorie recherchée</figcaption></figure>"
            )
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'">
<title>SmartSite — données pour le spécialiste</title><style>
body{{font:17px system-ui;max-width:1150px;margin:36px auto;padding:0 22px;
color:#20323c;line-height:1.6}}
h1{{font-size:30px}}.notice{{padding:18px;background:#fff1ce;border-radius:10px}}
.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin:24px 0}}
th,td{{padding:12px;border-bottom:1px solid #dce4ea;text-align:left}}th{{background:#eef3f7}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}
figure{{margin:0}}img{{width:100%;height:auto}}a{{color:#1762a0}}figcaption{{margin:8px 0}}
@media(max-width:640px){{.grid{{grid-template-columns:1fr}}body{{padding:0 14px}}}}
</style></head><body><h1>Données prêtes pour l’apprentissage</h1>
<p>Catégories recherchées : <strong>{labels}</strong>.</p>
<p>Les photos et leurs groupes sont conservés. Seules les annotations des catégories choisies
deviennent des cibles. Les autres restent dans le corpus original.</p>
<p class="notice">Cette page présente des données, pas les détections d’un modèle.
Une photo sans annotation de fissure n’est pas une surface garantie sans fissure ni autre défaut.
Les photos du test réservé ne sont ni ouvertes ni copiées par cette préparation.</p>
<div class="table"><table><thead><tr><th>Lot</th><th>Photos</th><th>Annotations gardées</th>
<th>Annotations écartées de cet essai</th><th>Photos sans annotation cible</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p>Les contours retenus, les originaux, la validation et les groupes de photos similaires restent
inchangés. L’origine exacte de certaines scènes demeure inconnue ; ces lots ne prouvent pas
la fiabilité sur un autre chantier.</p>
<h2>Exemples sans annotation cible</h2>
<p>Au plus trois photos par lot, dans l’ordre des identifiants. Elles restent dans le corpus,
pour ne pas écarter les occasions de mesurer des fausses alertes.</p>
<div class="grid">{"".join(examples) or "<p>Aucun exemple dans cette sélection.</p>"}</div>
<p><a href="selection.json">Rapport complet : configuration, empreintes et liste des photos</a></p>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
