"""Des rapports lisibles, avec les mêmes photos avant et après l'apprentissage."""

import html
from pathlib import Path
from typing import Any

from smartsite_ia.model_assets import CLASS_NAMES

NAMES = {"crack": "Fissures", "surface_loss": "Pertes de matière"}
STYLE = """body{font:16px system-ui;background:#f4f6fa;color:#20323c;margin:0;padding:32px}
main{max-width:1450px;margin:auto}h1{font-size:32px}p{line-height:1.6;max-width:1050px}
.notice{background:#fff1ce;padding:18px;border-radius:12px}
table{border-collapse:collapse;background:white;width:100%;margin:24px 0}
th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd}.table{overflow-x:auto}
details{background:white;margin:14px 0;padding:16px;border-radius:12px}
summary{cursor:pointer;font-weight:650}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:16px}
figure{margin:0}img{width:100%;height:auto}figcaption{margin:8px 0;font-weight:600}
small{font-weight:400}a{color:#1762a0}@media(max-width:640px){body{padding:14px}.grid{grid-template-columns:1fr}
h1{font-size:25px}}
"""


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def frame(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy"
content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'">
<title>{title}</title><style>{STYLE}</style></head><body><main><h1>{title}</h1>
<p class="notice">Validation interne au même barrage, modèle expérimental. Ces résultats ne valident
pas un chantier. Aucun défaut détecté ne signifie pas que la surface est conforme.</p>
{body}<p><a href="report.json">Rapport complet et protocole</a></p></main></body></html>"""


def metric_table(reports: list[tuple[str, dict[str, Any]]]) -> str:
    rows = []
    for name in CLASS_NAMES:
        for label, report in reports:
            counts = report["counts"][name]
            ap = report["coco"]["per_class"][name]["mask_ap_50_95"]
            rows.append(f"""<tr><td>{NAMES[name]}</td><td>{label}</td><td>{percent(ap)}</td>
<td>{counts["tp"]}</td><td>{counts["fn"]}</td><td>{counts["fp"]}</td>
<td>{percent(counts["precision"])}</td><td>{percent(counts["recall"])}</td></tr>""")
    return (
        """<div class="table"><table><thead><tr>
<th>Défaut</th><th>Modèle</th><th>AP des masques</th>
<th>Trouvés</th><th>Manqués</th><th>Propositions sans correspondance</th>
<th>Précision</th><th>Rappel</th>
</tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></div>
<p>L'AP résume le classement et la précision des masques à plusieurs niveaux de recouvrement ;
ce n'est pas le pourcentage de photos correctes. Les comptages utilisent un score ≥ 0,30 et
un recouvrement des masques ≥ 50 %, avec un seul résultat par défaut de la même catégorie.
Le seuil est fixe, encore non calibré. Les fausses propositions sont définies par les annotations
fournies ; leur pertinence métier reste à revoir. Un tiret signifie que le ratio est
indéfini.</p>"""
    )


def gallery(records: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    cards = []
    for index, record in enumerate(records):
        sample_id = html.escape(record["id"])
        figures = "".join(
            f'''<figure><figcaption>{label}</figcaption><a href="{sample_id}/{filename}">
<img loading="lazy" src="{sample_id}/{filename}" alt="{label} — {sample_id}"></a></figure>'''
            for label, filename in columns
        )
        cards.append(f"""<details {"open" if index == 0 else ""}><summary>{sample_id} —
<small>{html.escape(record["difficulty"])}</small></summary>
<div class="grid">{figures}</div></details>""")
    return (
        "<p>Rouge : fissure · Bleu : perte de matière. "
        "Toutes les photos sont disponibles ci-dessous, dans le même ordre.</p>" + "".join(cards)
    )


def write_validation_page(output: Path, report: dict[str, Any]) -> None:
    body = (
        f"<p>{report['images']} photos de validation analysées. "
        "Les photos de test restent réservées.</p>"
    )
    body += metric_table([("Modèle évalué", report)])
    body += gallery(
        report["records"],
        [
            ("Photo", "photo.jpg"),
            ("Annotations fournies", "reference.jpg"),
            ("Prédictions", "prediction.jpg"),
        ],
    )
    (output / "index.html").write_text(
        frame("SmartSite — résultats de validation", body), encoding="utf-8"
    )


def write_comparison_page(output: Path, first: dict[str, Any], second: dict[str, Any]) -> None:
    body = (
        f"<p>Comparaison sur les mêmes {first['images']} photos, avec les mêmes règles. "
        "Les annotations fournies sont affichées séparément des prédictions.</p>"
    )
    body += metric_table([("Avant", first), ("Après", second)])
    body += gallery(
        first["records"],
        [
            ("Photo", "photo.jpg"),
            ("Annotations fournies", "reference.jpg"),
            ("Avant : premier essai", "before.jpg"),
            ("Après : entraînement élargi", "after.jpg"),
        ],
    )
    (output / "index.html").write_text(
        frame("SmartSite — avant et après entraînement", body), encoding="utf-8"
    )
