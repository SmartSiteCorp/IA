"""Exporter un dossier lisible hors ligne, avec des preuves et du texte échappé."""

import html
import json
from pathlib import Path
from string import Template
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from smartsite_ia import finding_validation as check
from smartsite_ia.categories import FAMILY_LABELS
from smartsite_ia.curation import staged_output
from smartsite_ia.finding_events import replay
from smartsite_ia.finding_store import evidence_hashes, read_case, unlinked
from smartsite_ia.prediction import decode_photo
from smartsite_ia.review import MAX_FILE_BYTES, read_local

STATES = {
    "proposed": "À vérifier",
    "confirmed": "Confirmé par une personne",
    "rejected": "Rejeté par une personne",
    "indeterminate": "Indéterminé",
}
CORRECTIONS = {
    "not_assessed": "Reprise non évaluée",
    "planned": "Reprise prévue",
    "in_progress": "Reprise en cours",
    "resolved": "Réparation déclarée résolue",
    "not_required": "Aucune reprise demandée",
}
OUTCOMES = {
    "defects_found": "Zones suspectes proposées",
    "no_target_defect_in_readable_area": "Aucune cible détectée dans la zone lisible",
    "partial_analysis": "Analyse partielle — aucun constat automatique",
    "not_assessable": "Photo non évaluable — détection non exécutée",
}
QUALITY = {
    "usable": "Exploitable",
    "partial": "Partiellement exploitable",
    "unusable": "Non évaluable",
}
COLORS = {
    "crack": "#d63224",
    "surface_loss": "#1870c5",
    "peeling_paint": "#1870c5",
    "moisture_trace": "#a96600",
    "mold_suspected": "#a82c79",
}


def escaped(value: Any) -> str:
    return html.escape("Non renseigné" if value is None else str(value), quote=True)


def draw_findings(photo: Image.Image, views: list[dict[str, Any]]) -> Image.Image:
    """Les cadres concernent la photo orientée ; aucune position 3D n'est simulée."""
    result = photo.copy()
    result.thumbnail((1400, 1400))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=16)
    sx, sy = result.width / photo.width, result.height / photo.height
    for index, view in enumerate(views, 1):
        finding = view["finding"]
        x, y, u, v = finding["location"]["bbox_xyxy"]
        color = COLORS[finding["defect"]["class"]]
        if view["verification"]["state"] == "rejected":
            color = "#66717c"
        draw.rectangle((x * sx, y * sy, u * sx, v * sy), outline=color, width=3)
        label = f"C{index}"
        point = (min(x * sx, result.width - 35), max(0, y * sy - 20))
        draw.rectangle(draw.textbbox(point, label, font=font), fill="white")
        draw.text(point, label, font=font, fill=color)
    return result


def render_page(record: dict[str, Any], current: dict[str, Any]) -> str:
    """Le HTML est un instantané : le JSON et le journal restent la source des décisions."""
    analysis = record["analysis"]
    quality = analysis["quality"]
    context = analysis["request"]["context"]
    cards = []
    for i, view in enumerate(current["findings"], 1):
        finding = view["finding"]
        verification = view["verification"]
        correction = view["correction"]
        review = ""
        if verification["state"] != "proposed":
            category_label = (
                "Catégorie retenue"
                if verification["state"] == "confirmed"
                else "Catégorie proposée"
            )
            review = f"""<h3>Retour humain</h3><p>{escaped(verification["note"])}</p>
<p>{category_label} : {escaped(FAMILY_LABELS[verification["effective_class"]])} ·
Tâche : {escaped(verification["task_id"])} ·
Gravité renseignée : {escaped(verification["severity"])}</p>
<p>Par {escaped(verification["actor"])},
enregistré le {escaped(verification["recorded_at"])}.</p>"""
        proofs = "".join(
            f'<a href="evidence/{sha}.png">Photo de preuve {n}</a> '
            for n, sha in enumerate(correction.get("evidence_sha256", []), 1)
        )
        score = finding["confidence"]["model_score"]
        score_text = (
            "Signalement manuel, sans score de modèle"
            if score is None
            else f"Score du modèle : {score:.3f} (ni probabilité vérifiée, ni gravité)"
        )
        observability = finding.get("observability")
        readable = (
            "Ajout humain : lisibilité de la zone à vérifier."
            if observability is None
            else "Part du cadre dans une zone illisible : "
            f"{observability['unreadable_box_fraction']:.1%}."
        )
        cards.append(f'''<article id="{finding["id"]}">
<h2>C{i} · {escaped(finding["defect"]["label"])}</h2>
<p class="state">{STATES[verification["state"]]}</p><p>{score_text}</p>
<p>Identifiant : <code>{finding["id"]}</code></p>
<p>Rectangle en pixels : {escaped(finding["location"]["bbox_xyxy"])}.
Position 3D : indisponible.</p>
<p>{readable}</p>
{review}<h3>{CORRECTIONS[correction["status"]]}</h3><p>{escaped(correction.get("note"))}</p>{proofs}
<p>Les coordonnées et la catégorie initiales sont conservées,
même après un rejet ou une correction humaine.</p></article>''')
    history = "".join(
        f"<li><b>Révision {e['revision']} · {escaped(e['request']['kind'])}</b> — "
        f"{escaped(e['request']['actor'])} · {escaped(e['recorded_at'])}"
        f"<pre>{escaped(json.dumps(e['request'], ensure_ascii=False, indent=2))}</pre></li>"
        for e in record["events"]
    )
    template = Template(Path(__file__).with_name("finding_template.html").read_text())
    values = {
        "revision": current["revision"],
        "outcome": OUTCOMES[analysis["outcome"]],
        "site": escaped(context["site_id"]),
        "zone": escaped(context["zone_id"]),
        "task": escaped(context["task_id"]),
        "capture": escaped(analysis["request"]["capture_id"]),
        "captured_at": escaped(analysis["request"]["captured_at"]),
        "analysed_at": escaped(analysis["analysed_at"]),
        "quality": QUALITY[quality["verdict"]],
        "quality_reasons": escaped("; ".join(quality["reasons"]) or "Pas de limitation détectée."),
        "withheld": len(analysis["withheld"]),
        "manual": current["human_reports"],
        "model": escaped(analysis["model"]["run"]),
        "threshold": analysis["model"]["score_threshold"],
        "cards": "".join(cards) or "<section>Aucun constat dans ce dossier.</section>",
        "history": history or "<li>Aucun retour humain enregistré.</li>",
    }
    return template.substitute(values)


def export_case(root: Path, output: Path, site_id: str) -> dict[str, Any]:
    """Exporter un instantané portable, contrôlable et sans dépendance à SmartSite."""
    unlinked(output)
    record = read_case(root, site_id)
    current = replay(record["analysis"], record["events"])
    raw = read_local(root, "source.bin", MAX_FILE_BYTES)
    photo = decode_photo(raw)
    with staged_output(output, [root]) as stage:
        (stage / "record.json").write_bytes(check.canonical(record) + b"\n")
        (stage / "current.json").write_bytes(check.canonical(current) + b"\n")
        (stage / "source.bin").write_bytes(raw)
        photo.save(stage / "photo.png")
        draw_findings(photo, current["findings"]).save(stage / "annotated.jpg", quality=92)
        for sha in sorted(evidence_hashes(record["events"])):
            (stage / "evidence").mkdir(exist_ok=True)
            proof = read_local(root, f"evidence/{sha}.bin", MAX_FILE_BYTES)
            (stage / "evidence" / f"{sha}.bin").write_bytes(proof)
            decode_photo(proof).save(stage / "evidence" / f"{sha}.png")
        (stage / "index.html").write_text(render_page(record, current), encoding="utf-8")
        read_case(stage, site_id)
    return current
