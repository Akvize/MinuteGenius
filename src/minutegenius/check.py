"""
Vérifie que l'environnement est prêt pour les commandes meeting/transcribe :
dépendances système (ffmpeg, argmax-cli / whisperkit-cli), variables d'environnement
requises, et connectivité aux APIs configurées (AI_API/AI_API2, pyannote.ai).

Usage: minutegenius check [--lexicon fichier.md]
"""
import argparse
import os
import shutil
import sys

import requests
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from minutegenius import stt_backend
from minutegenius.meeting import get_client, get_client_image


def check_env_vars(*names: str) -> None:
    """Vérifie que les variables d'environnement requises sont définies, avec message clair sinon."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Erreur : variable(s) d'environnement manquante(s) : {', '.join(missing)}.", file=sys.stderr)
        print("Renseignez-les dans votre .env (voir .env.example).", file=sys.stderr)
        sys.exit(1)


def check_ai_connection(client: OpenAI, model: str, label: str) -> None:
    """Fait un appel minimal pour vérifier la connexion avant de lancer le pipeline complet.

    Évite de découvrir une clé invalide ou un endpoint injoignable après plusieurs
    minutes d'extraction de frames/audio (voir analyze_frame, qui n'a pas de retry).
    """
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            timeout=15,
        )
    except AuthenticationError:
        print(f"Erreur : clé API invalide pour {label} (AI_KEY/AI_KEY2).", file=sys.stderr)
        sys.exit(1)
    except NotFoundError:
        print(f"Erreur : modèle '{model}' introuvable sur {label} (AI_MODEL_*).", file=sys.stderr)
        sys.exit(1)
    except RateLimitError:
        print(f"Avertissement : limite de débit atteinte lors du test de connexion à {label}, on continue.", file=sys.stderr)
    except APIConnectionError:
        print(f"Erreur : impossible de joindre {label} ({client.base_url}).", file=sys.stderr)
        print("Vérifiez AI_API/AI_API2 et votre connexion réseau.", file=sys.stderr)
        sys.exit(1)
    except APIStatusError as e:
        print(f"Erreur : {label} a répondu avec le statut {e.status_code} ({e.message}).", file=sys.stderr)
        sys.exit(1)


def check_pyannote_connection() -> None:
    """Vérifie PYANNOTE_API_KEY via l'endpoint /test de pyannote.ai."""
    check_env_vars("PYANNOTE_API_KEY")
    try:
        resp = requests.get(
            f"{stt_backend.PYANNOTE_BASE_URL}/test",
            headers={"Authorization": f"Bearer {os.environ['PYANNOTE_API_KEY']}"},
            timeout=stt_backend.PYANNOTE_HTTP_TIMEOUT_S,
        )
    except requests.RequestException as e:
        print(f"Erreur : impossible de joindre pyannote.ai ({e}).", file=sys.stderr)
        sys.exit(1)
    if resp.status_code == 401:
        print("Erreur : clé API invalide pour pyannote.ai (PYANNOTE_API_KEY).", file=sys.stderr)
        sys.exit(1)
    if not resp.ok:
        print(f"Erreur : pyannote.ai a répondu avec le statut {resp.status_code} ({resp.text}).", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vérifie la configuration et la connectivité requises par meeting/transcribe"
    )
    parser.add_argument("--lexicon", type=str, default=None,
                        help="Vérifie aussi AI_MODEL_WRITER (utilisé par l'étape 6 de correction)")
    args = parser.parse_args()

    print("Dépendances système :")
    for tool, required in (("ffmpeg", True), ("ffprobe", True), *((t, False) for t in stt_backend.LOCAL_TRANSCRIBER_CANDIDATES)):
        found = shutil.which(tool) is not None
        marker = "✓" if found else ("✗" if required else "–")
        print(f"  {marker} {tool}{'' if found else ' (introuvable)' if required else ' (absent, optionnel)'}")
        if required and not found:
            print(f"Erreur : '{tool}' est requis mais introuvable sur le PATH.", file=sys.stderr)
            sys.exit(1)
    print()

    print("Connexion aux services IA (vision/écriture) :")
    check_env_vars("AI_KEY", "AI_API", "AI_MODEL_IMAGE")
    image_model = os.environ.get("AI_MODEL_IMAGE2") or os.environ["AI_MODEL_IMAGE"]
    image_label = "AI_API2 (vision)" if os.environ.get("AI_API2") else "AI_API (vision)"
    check_ai_connection(get_client_image(), image_model, image_label)
    print(f"  ✓ {image_label} ({image_model})")

    if args.lexicon and os.path.exists(os.path.expanduser(args.lexicon)):
        check_env_vars("AI_MODEL_WRITER")
        check_ai_connection(get_client(), os.environ["AI_MODEL_WRITER"], "AI_API (writer)")
        print(f"  ✓ AI_API (writer) ({os.environ['AI_MODEL_WRITER']})")
    print()

    print("Transcription + diarisation :")
    transcriber = stt_backend.find_local_transcriber()
    if transcriber:
        print(f"  ✓ {transcriber} détecté (local) — le fallback cloud pyannote.ai ne sera pas utilisé.")
    else:
        check_pyannote_connection()
        print("  ✓ pyannote.ai (PYANNOTE_API_KEY)")
    print()

    print("→ configuration OK")


if __name__ == "__main__":
    main()
