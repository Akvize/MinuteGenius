"""
Check that the environment is ready for the meeting/transcribe commands:
system dependencies (ffmpeg, argmax-cli / whisperkit-cli), required environment
variables, and connectivity to the configured APIs (AI_API/AI_API2, pyannote.ai).

Usage: minutegenius check [--lexicon lexicon.md]
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
    """Check that the required environment variables are set, with a clear message otherwise."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"Error: missing environment variable(s): {', '.join(missing)}.", file=sys.stderr)
        print("Set them in your .env (see .env.example).", file=sys.stderr)
        sys.exit(1)


def check_ai_connection(client: OpenAI, model: str, label: str) -> None:
    """Make a minimal call to check the connection before running the full pipeline.

    Avoids discovering an invalid key or unreachable endpoint after several minutes
    of frame/audio extraction (see analyze_frame, which has no retry).
    """
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            timeout=15,
        )
    except AuthenticationError:
        print(f"Error: invalid API key for {label} (AI_KEY/AI_KEY2).", file=sys.stderr)
        sys.exit(1)
    except NotFoundError:
        print(f"Error: model '{model}' not found on {label} (AI_MODEL_*).", file=sys.stderr)
        sys.exit(1)
    except RateLimitError:
        print(f"Warning: rate limit hit while testing the connection to {label}, continuing.", file=sys.stderr)
    except APIConnectionError:
        print(f"Error: cannot reach {label} ({client.base_url}).", file=sys.stderr)
        print("Check AI_API/AI_API2 and your network connection.", file=sys.stderr)
        sys.exit(1)
    except APIStatusError as e:
        print(f"Error: {label} responded with status {e.status_code} ({e.message}).", file=sys.stderr)
        sys.exit(1)


def check_pyannote_connection() -> None:
    """Check PYANNOTE_API_KEY using pyannote.ai's /test endpoint."""
    check_env_vars("PYANNOTE_API_KEY")
    try:
        resp = requests.get(
            f"{stt_backend.PYANNOTE_BASE_URL}/test",
            headers={"Authorization": f"Bearer {os.environ['PYANNOTE_API_KEY']}"},
            timeout=stt_backend.PYANNOTE_HTTP_TIMEOUT_S,
        )
    except requests.RequestException as e:
        print(f"Error: cannot reach pyannote.ai ({e}).", file=sys.stderr)
        sys.exit(1)
    if resp.status_code == 401:
        print("Error: invalid API key for pyannote.ai (PYANNOTE_API_KEY).", file=sys.stderr)
        sys.exit(1)
    if not resp.ok:
        print(f"Error: pyannote.ai responded with status {resp.status_code} ({resp.text}).", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the configuration and connectivity required by meeting/transcribe"
    )
    parser.add_argument("--lexicon", type=str, default=None,
                        help="Also check AI_MODEL_WRITER (used by step 6, the write-up)")
    args = parser.parse_args()

    print("System dependencies:")
    for tool, required in (("ffmpeg", True), ("ffprobe", True), *((t, False) for t in stt_backend.LOCAL_TRANSCRIBER_CANDIDATES)):
        found = shutil.which(tool) is not None
        marker = "✓" if found else ("✗" if required else "–")
        print(f"  {marker} {tool}{'' if found else ' (not found)' if required else ' (not installed, optional)'}")
        if required and not found:
            print(f"Error: '{tool}' is required but was not found on PATH.", file=sys.stderr)
            sys.exit(1)
    print()

    print("AI services (vision/writer):")
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

    print("Transcription + diarization:")
    transcriber = stt_backend.find_local_transcriber()
    if transcriber:
        print(f"  ✓ {transcriber} found (local) — the pyannote.ai cloud fallback won't be used.")
    else:
        check_pyannote_connection()
        print("  ✓ pyannote.ai (PYANNOTE_API_KEY)")
    print()

    print("→ configuration OK")


if __name__ == "__main__":
    main()
