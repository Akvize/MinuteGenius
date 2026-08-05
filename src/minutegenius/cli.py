"""
Point d'entrée unifié : minutegenius {meeting,transcribe,vtt} ...

Chaque sous-commande délègue directement au argparse du module correspondant
(meeting.py, transcribe.py, vtt.py), qui reste utilisable indépendamment.
"""
import sys

USAGE = """usage: minutegenius <command> [options]

Commands:
  meeting      Génère un compte-rendu intelligent à partir d'une vidéo de réunion
  transcribe   Transcrit + diarise un fichier audio/vidéo (sans le pipeline complet)
  vtt          Convertit un fichier de sous-titres WebVTT en Markdown

Run 'minutegenius <command> --help' for command-specific options.
"""

COMMANDS = {"meeting", "transcribe", "vtt"}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command, rest = sys.argv[1], sys.argv[2:]
    if command not in COMMANDS:
        print(f"Erreur : commande inconnue '{command}'.\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    # Chaque sous-module parse son propre sys.argv — on le reconstruit
    # pour que ses messages d'usage/erreur restent cohérents (argparse
    # affiche prog=sys.argv[0]).
    sys.argv = [f"minutegenius {command}"] + rest

    if command == "meeting":
        from minutegenius.meeting import main as run
    elif command == "transcribe":
        from minutegenius.transcribe import main as run
    else:
        from minutegenius.vtt import main as run

    run()


if __name__ == "__main__":
    main()
