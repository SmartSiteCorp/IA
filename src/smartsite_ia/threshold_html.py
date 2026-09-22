"""Montrer le compromis entre défauts retrouvés et alertes, avec les photos modifiées."""

import html
from pathlib import Path
from typing import Any

from smartsite_ia.categories import CLASS_LABELS
from smartsite_ia.validation_html import frame, percent

REASONS = {
    "matched": "Référence retrouvée",
    "duplicate": "Doublon",
    "insufficient_overlap": "Contour insuffisamment recouvrant",
    "no_overlap": "Sans recouvrement",
}


def score_table(report: dict[str, Any]) -> str:
    rows = []
    target = report["target_class"]
    for index, variant in enumerate(report["summaries"]):
        counts = variant["counts"][target]
        rows.append(
            f'<tr><th><a href="threshold_{index}.html">{variant["threshold"]:g}</a></th>'
            f"<td>{counts['tp']}</td><td>{counts['fn']}</td><td>{counts['fp']}</td>"
            f"<td>{percent(counts['precision'])}</td><td>{percent(counts['recall'])}</td>"
            f"<td>{percent(variant['target_f1'])}</td>"
            f"<td>{variant['images_with_false_proposals']}</td>"
            f"<td>{len(variant['changed_images'])}</td></tr>"
        )
    return (
        '<div class="table"><table><thead><tr><th>Seuil à examiner</th><th>Retrouvés</th>'
        "<th>Manqués</th><th>Fausses propositions</th><th>Précision</th><th>Rappel</th>"
        "<th>F1</th><th>Photos avec fausse proposition</th><th>Photos modifiées</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def change_table(case: dict[str, Any]) -> str:
    rows = []
    for kind, title in (("added", "Ajoutée"), ("removed", "Retirée")):
        for item in case[kind]:
            reference = item["reference_id"] if item["reference_id"] is not None else "—"
            rows.append(
                f"<tr><td>{title}</td><td>{item['id']}</td><td>{item['score']:.4f}</td>"
                f"<td>{REASONS[item['reason']]}</td><td>{reference}</td></tr>"
            )
    if not rows:
        return "<p>Aucun changement d’affichage pour cette photo.</p>"
    return (
        '<div class="table"><table><thead><tr><th>Changement</th><th>Proposition</th>'
        "<th>Score</th><th>Statut selon les annotations</th><th>Référence</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def photo_card(row: dict[str, Any], index: int, baseline: int) -> str:
    sid = html.escape(row["id"])
    case = row["variants"][index]
    figures = "".join(
        f'<figure><figcaption>{label}</figcaption><a href="{sid}/{file}">'
        f'<img loading="lazy" width="640" height="640" src="{sid}/{file}" '
        f'alt="{label} — {sid}"></a></figure>'
        for label, file in [
            ("Photo originale", "photo.jpg"),
            ("Annotations de la classe étudiée", "reference.jpg"),
            ("Seuil de référence 0,30", f"threshold_{baseline}.jpg"),
            (f"Seuil étudié {case['threshold']:g}", f"threshold_{index}.jpg"),
        ]
    )
    c = case["counts"]
    return (
        f'<details id="{sid}"><summary>{sid} — {c["tp"]} retrouvé(s), {c["fn"]} manqué(s), '
        f'{c["fp"]} fausse(s) proposition(s)</summary><div class="grid">{figures}</div>'
        + change_table(case)
        + "</details>"
    )


def write_threshold_pages(output: Path, report: dict[str, Any]) -> None:
    """Les pages restent autonomes et utilisables sans script ni service externe."""
    target = report["target_class"]
    baseline = report["thresholds"].index(report["baseline_threshold"])
    best = report["best_observed_f1_thresholds"]
    description = (
        f"<p>{report['images']} photos, mêmes poids et mêmes contours. "
        f"Seul le score objet de la classe « {CLASS_LABELS[target]} » varie. "
        "Les autres catégories restent à 0,30 ; le seuil des pixels reste inchangé.</p>"
        "<p>Un score plus faible montre davantage de propositions. Une fausse proposition "
        "est définie par les annotations : doublon, contour insuffisant ou absence de référence "
        "recouverte. Cela ne tranche pas son utilité sur chantier.</p>"
    )
    best_text = ", ".join(f"{s:g}" for s in best) if best else "indéfini"
    baseline_counts = report["summaries"][baseline]["counts"][target]
    tradeoffs = "".join(
        f"<li>Seuil {s['threshold']:g} : "
        f"{s['counts'][target]['tp'] - baseline_counts['tp']:+d} défaut(s) retrouvé(s), "
        f"{s['counts'][target]['fp'] - baseline_counts['fp']:+d} fausse(s) proposition(s).</li>"
        for s in report["summaries"]
        if s["threshold"] in best
    )
    body = description + score_table(report)
    body += (
        f"<p><strong>Meilleur F1 observé sur cette grille : seuil(s) {best_text}.</strong> "
        "Le F1 résume précision et rappel avec le même poids ; il ne représente pas le coût métier "
        "d’un défaut manqué ou d’une fausse alerte. "
        "Aucun seuil de production n’est sélectionné.</p>"
        f"<p>Écart des meilleurs F1 par rapport au seuil actuel 0,30 :</p><ul>{tradeoffs}</ul>"
        "<h2>Comment lire les résultats</h2><p>Cliquer sur un seuil pour voir toutes les photos "
        "dont l’affichage change, avec chaque proposition ajoutée ou retirée. "
        "Le tableau porte toujours sur le lot complet. Les images sans annotation de cette classe "
        "ne sont pas certifiées saines. Le détail par photo est aussi dans le rapport JSON.</p>"
        "<p>L’appariement reste un-à-un, dans l’ordre des scores, avec IoU de masque ≥ 0,50. "
        "L’AP du modèle reste celle de l’évaluation source : aucun nouveau poids ni classement "
        "n’a été calculé. Cette étude n’est pas une calibration des scores en probabilités.</p>"
        "<p>La validation a déjà servi à choisir des réglages. Il faudra une évaluation réservée "
        "après gel des choix, puis des données indépendantes adaptées au chantier.</p>"
        '<p>Photos : <a href="https://data.mendeley.com/datasets/z5z6gtt5t4/1">DamSegment v1</a>, '
        "Vahidreza Gharehbaghi, Caroline R. Bennett, Rémy Lequesne, Hang Zhao, Jian Li — "
        '<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. '
        "Originaux conservés, superpositions dérivées pour cette étude.</p>"
    )
    (output / "index.html").write_text(
        frame("SmartSite — étude des seuils par catégorie", body), encoding="utf-8"
    )
    for index, summary in enumerate(report["summaries"]):
        selected = [
            r
            for r in report["records"]
            if r["id"] in summary["changed_images"] or index == baseline
        ]
        color = "Bleu" if target == "surface_loss" else "Rouge"
        detail = '<p><a href="index.html">Retour au bilan de tous les seuils</a></p>' + description
        detail += (
            f"<p>{color} : {CLASS_LABELS[target]}. Les autres catégories ne sont pas "
            "dessinées ici pour garder la comparaison lisible ; "
            "leurs résultats restent inchangés.</p>"
        )
        detail += (
            f"<p>{len(selected)} photos affichées. Cliquer sur une fiche, "
            "puis sur une image pour l’agrandir.</p>"
        )
        detail += "".join(photo_card(r, index, baseline) for r in selected)
        (output / f"threshold_{index}.html").write_text(
            frame(f"SmartSite — seuil {summary['threshold']:g}", detail), encoding="utf-8"
        )
