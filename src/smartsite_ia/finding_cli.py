"""Commandes locales pour créer, contrôler et relire un constat sans SmartSite."""

import argparse
import json
import sys
from pathlib import Path

from smartsite_ia import finding_validation as check
from smartsite_ia.annotations import unique_object
from smartsite_ia.curation import digest
from smartsite_ia.finding_engine import analyse_photo
from smartsite_ia.finding_events import replay
from smartsite_ia.finding_html import export_case
from smartsite_ia.finding_store import BusyError, IntegrityError, read_case, record_event, unlinked
from smartsite_ia.review import MAX_FILE_BYTES, read_local


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyse = commands.add_parser(
        "analyse", help="Analyser une photo locale avec des poids Nano vérifiés"
    )
    analyse.add_argument("photo", type=Path)
    analyse.add_argument("--run", type=Path, required=True)
    analyse.add_argument("--output", type=Path, required=True)
    analyse.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    analyse.add_argument("--threshold", type=float, default=0.3)
    for name in ("zone", "task", "capture", "captured-at", "media"):
        analyse.add_argument("--" + name)
    show = commands.add_parser("show", help="Vérifier les preuves et lire l'état courant")
    export = commands.add_parser("export", help="Créer un nouvel export portable HTML/JSON")
    export.add_argument("--output", type=Path, required=True)
    record = commands.add_parser("record", help="Enregistrer un retour humain depuis un JSON")
    record.add_argument("--request", type=Path, required=True)
    record.add_argument("--revision", type=int, required=True)
    record.add_argument("--evidence", type=Path, action="append", default=[])
    for command in (show, export, record):
        command.add_argument("dossier", type=Path)
    for command in (analyse, show, export, record):
        command.add_argument("--site", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyse":
            result = analyse_photo(
                args.photo,
                run=args.run,
                output=args.output,
                device=args.device,
                threshold=args.threshold,
                request={
                    "context": {"site_id": args.site, "zone_id": args.zone, "task_id": args.task},
                    "capture_id": args.capture,
                    "captured_at": args.captured_at,
                    "media_id": args.media,
                },
            )
            response = {
                "analysis_id": result["analysis"]["id"],
                "revision": len(result["events"]),
                "outcome": result["analysis"]["outcome"],
                "findings": len(result["analysis"]["findings"]),
                "notification_sent": False,
            }
        elif args.command == "export":
            response = export_case(args.dossier, args.output, args.site)
        elif args.command == "show":
            result = read_case(args.dossier, args.site)
            response = replay(result["analysis"], result["events"])
        else:
            unlinked(args.request)
            request = json.loads(
                read_local(args.request.parent, args.request.name, 32 * 1024),
                object_pairs_hook=unique_object,
            )
            check.fields(
                request, ["request_id", "kind", "finding_id", "actor", "payload"], ["occurred_at"]
            )
            attachments = {}
            if len(args.evidence) > 4:
                raise ValueError("At most four evidence photos per event")
            for path in args.evidence:
                unlinked(path)
                raw = read_local(path.parent, path.name, MAX_FILE_BYTES)
                attachments[digest(raw)] = raw
            if attachments:
                if (
                    request["kind"] != "correction"
                    or not isinstance(request["payload"], dict)
                    or not isinstance(request["payload"].get("evidence_sha256"), list)
                ):
                    raise ValueError("Evidence photos require a correction request")
                request["payload"]["evidence_sha256"] = sorted(
                    set(request["payload"]["evidence_sha256"]) | attachments.keys()
                )
            result = record_event(
                args.dossier,
                args.site,
                request,
                expected_revision=args.revision,
                evidence=attachments,
            )
            response = replay(result["analysis"], result["events"])
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "error": {
                        "code": "interrupted",
                        "message": "Interruption : vérifier le dossier avant de relancer.",
                    }
                }
            ),
            file=sys.stderr,
        )
        return 130
    except (
        ValueError,
        OSError,
        RuntimeError,
        ImportError,
        KeyError,
        TypeError,
        RecursionError,
    ) as exc:
        code = (
            "conflict"
            if isinstance(exc, check.ConflictError)
            else "integrity_error"
            if isinstance(exc, IntegrityError)
            else "busy"
            if isinstance(exc, BusyError)
            else "io_error"
            if isinstance(exc, OSError)
            else "engine_error"
            if isinstance(exc, (RuntimeError, ImportError))
            else "invalid_input"
        )
        # Ne pas recopier une note, une URL ou un chemin potentiellement sensible dans les traces.
        messages = {
            "conflict": "Chantier, demande ou révision incompatible avec le dossier.",
            "integrity_error": "Le dossier ou ses preuves sont absents, modifiés ou invalides.",
            "busy": "Une autre opération écrit ce dossier. Réessayer ensuite.",
            "io_error": "Accès local impossible. Vérifier les chemins et les droits.",
            "engine_error": "Moteur local indisponible. Vérifier installation et versions.",
            "invalid_input": "Entrée invalide. Vérifier le contrat et les limites des fichiers.",
        }
        print(
            json.dumps({"error": {"code": code, "message": messages[code]}}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(response, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
