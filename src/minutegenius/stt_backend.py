"""
Backend de transcription : argmax-cli en local (argmax-oss-swift : WhisperKit +
SpeakerKit, Mac Apple Silicon) si disponible, sinon fallback cloud pyannote.ai —
transcription + diarisation en un seul appel (STT Orchestration) — voir
transcribe_cloud(). Le cloud peut aussi être forcé explicitement via
transcribe_audio(force_cloud=True), même si un transcripteur local est disponible.

`argmax-cli` et `whisperkit-cli` sont le même binaire (cible ArgmaxCLI du dépôt
argmax-oss-swift) : mêmes flags, même sortie RTTM étendue :
    SPEAKER <fichier> 1 <start> <durée> <texte…> <NA> <A|B|…> <NA> <NA>
On préfère `argmax-cli` (build depuis les sources) et on retombe sur
`whisperkit-cli` (Homebrew) s'il n'est pas installé.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


LOCAL_TRANSCRIBER_CANDIDATES = ("argmax-cli", "whisperkit-cli")


def find_local_transcriber() -> str | None:
    """Premier transcripteur local présent dans le PATH (argmax-cli, puis whisperkit-cli)."""
    for name in LOCAL_TRANSCRIBER_CANDIDATES:
        if shutil.which(name):
            return name
    return None


def local_transcriber_available() -> bool:
    return find_local_transcriber() is not None


def build_transcribe_cmd(
    transcriber: str,
    audio_path: Path,
    language: str | None,
    num_speakers: int | None = None,
    model: str | None = None,
) -> list[str]:
    """Commande de transcription avec diarisation.

    language : code langue ("fr", "en"…) ; "auto" ou None = détection automatique
    (on omet --language, utile pour les réunions bilingues FR/EN).
    """
    cmd = [
        transcriber, "transcribe",
        "--audio-path", str(audio_path),
        "--diarization",
        "--skip-special-tokens",
    ]
    if language and language.lower() != "auto":
        cmd += ["--language", language]
    if model:
        cmd += ["--model", model]
    if num_speakers:
        cmd += ["--diarization-num-speakers", str(num_speakers)]
    return cmd


_HYPHEN_SPACE = re.compile(r"(?<=\w) -(?=\w)")


def clean_text(text: str) -> str:
    """Corrige les espaces parasites de la tokenisation Whisper.

    "J 'ai" -> "J'ai", "Est -ce" -> "Est-ce", "C'est -à -dire" -> "C'est-à-dire".
    Les tirets de dialogue ("- Tu réponds?") et les tirets espacés ("a - b") sont conservés.
    """
    text = text.replace(" '", "'")
    return _HYPHEN_SPACE.sub("-", text)


def _spinner(stop_event: threading.Event):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    start = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - start)
        m, s = divmod(elapsed, 60)
        print(f"\r  {frames[i % len(frames)]}  En cours... {m:02d}:{s:02d}", end="", flush=True)
        i += 1
        time.sleep(0.1)
    print("\r" + " " * 30 + "\r", end="", flush=True)


def run_local_transcriber(
    transcriber: str, audio_path: Path, language: str | None, num_speakers: int | None, model: str | None,
) -> str:
    cmd = build_transcribe_cmd(transcriber, audio_path, language, num_speakers, model)

    print(f"Transcription de {audio_path.name} ({transcriber})...")
    stop = threading.Event()
    t = threading.Thread(target=_spinner, args=(stop,), daemon=True)
    t.start()

    t_start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t_start

    stop.set()
    t.join()

    if result.returncode != 0:
        print(result.stderr[-500:], file=sys.stderr)
        sys.exit(1)

    m, s = divmod(int(elapsed), 60)
    print(f"  Terminé en {m:02d}:{s:02d}")
    return result.stdout


def format_as_whisperkit_lines(segments: list[Segment], audio_filename: str) -> str:
    """Sérialise des segments au format texte brut d'argmax-cli/whisperkit-cli, pour rester
    compatible avec parse_diarization_output()/_parse_speaker_line() existants."""
    lines = []
    for seg in segments:
        text = " ".join(seg.text.split())
        duration = seg.end - seg.start
        lines.append(
            f"SPEAKER {audio_filename} 1 {seg.start:.3f} {duration:.3f} {text} NA {seg.speaker} NA NA"
        )
    return "\n".join(lines)


PYANNOTE_BASE_URL = "https://api.pyannote.ai/v1"
PYANNOTE_POLL_INTERVAL_S = 5
PYANNOTE_POLL_TIMEOUT_S = 1800  # 30 min
PYANNOTE_HTTP_TIMEOUT_S = 30
PYANNOTE_UPLOAD_TIMEOUT_S = 600  # generous — uploads the full original audio file
# faster-whisper-large-v3-turbo : 99+ langues (vs. 27 pour parakeet-tdt-0.6b-v3,
# l'autre choix possible). L'API pyannote.ai n'accepte pas de langue en paramètre
# (détection automatique) : ce modèle maximise les chances de bien couvrir la
# langue réellement parlée, --language n'étant pas transmis sur ce chemin cloud.
# Configurable via PYANNOTE_TRANSCRIPTION_MODEL (voir .env.example).
PYANNOTE_TRANSCRIPTION_MODEL_DEFAULT = "faster-whisper-large-v3-turbo"


def _pyannote_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['PYANNOTE_API_KEY']}"}


def transcribe_pyannote(audio_path: Path, num_speakers: int | None) -> list[Segment]:
    """Transcrit et diarise un fichier audio en un seul appel pyannote.ai (STT
    Orchestration : transcription attribuée par locuteur). Retourne les segments
    de tour de parole (turnLevelTranscription), déjà alignés avec le locuteur."""
    media_key = f"media://minutegenius-{uuid.uuid4().hex}"

    resp = requests.post(
        f"{PYANNOTE_BASE_URL}/media/input", json={"url": media_key}, headers=_pyannote_headers(),
        timeout=PYANNOTE_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    upload_url = resp.json()["url"]

    with open(audio_path, "rb") as f:
        put_resp = requests.put(upload_url, data=f, timeout=PYANNOTE_UPLOAD_TIMEOUT_S)
    put_resp.raise_for_status()

    transcription_model = os.environ.get("PYANNOTE_TRANSCRIPTION_MODEL") or PYANNOTE_TRANSCRIPTION_MODEL_DEFAULT
    body: dict = {
        "url": media_key,
        "transcription": True,
        "transcriptionConfig": {"model": transcription_model},
    }
    if num_speakers:
        body["numSpeakers"] = num_speakers
    resp = requests.post(
        f"{PYANNOTE_BASE_URL}/diarize", json=body, headers=_pyannote_headers(),
        timeout=PYANNOTE_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    job_id = resp.json()["jobId"]

    output = _poll_pyannote_job(job_id)
    turns = output.get("turnLevelTranscription")
    if turns is None:
        raise RuntimeError(
            f"Réponse pyannote.ai inattendue pour le job {job_id} : "
            "'turnLevelTranscription' absent (transcription non activée côté API ?)."
        )
    return [
        Segment(start=t["start"], end=t["end"], speaker=t.get("speaker") or "SPEAKER_00", text=t["text"].strip())
        for t in turns
        if t["text"].strip()
    ]


def _poll_pyannote_job(job_id: str) -> dict:
    deadline = time.time() + PYANNOTE_POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.get(
            f"{PYANNOTE_BASE_URL}/jobs/{job_id}", headers=_pyannote_headers(),
            timeout=PYANNOTE_HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        job = resp.json()
        status = job["status"]
        if status == "succeeded":
            return job["output"]
        if status in ("failed", "canceled"):
            error = (job.get("output") or {}).get("error", "raison inconnue")
            raise RuntimeError(f"Job pyannote.ai {job_id} : statut {status} ({error})")
        time.sleep(PYANNOTE_POLL_INTERVAL_S)
    raise TimeoutError(f"Job pyannote.ai {job_id} n'a pas terminé après {PYANNOTE_POLL_TIMEOUT_S}s")


def _require_cloud_env_vars() -> None:
    """Vérifie les clés cloud avant toute dépense (appels réseau, API payante)."""
    if not os.environ.get("PYANNOTE_API_KEY"):
        print("Erreur : variable d'environnement manquante : PYANNOTE_API_KEY", file=sys.stderr)
        print("Configure-la dans .env (voir .env.example) avant de relancer.", file=sys.stderr)
        sys.exit(1)


def transcribe_cloud(audio_path: Path, num_speakers: int | None, *, forced: bool = False) -> str:
    _require_cloud_env_vars()
    if forced:
        print("Transcription cloud demandée (--cloud) — pyannote.ai (transcription + diarisation)...")
    else:
        print("argmax-cli / whisperkit-cli introuvable — bascule sur le fallback cloud (pyannote.ai, transcription + diarisation)...")
    segments = transcribe_pyannote(audio_path, num_speakers)
    if not segments:
        print("Erreur : aucun segment de transcription obtenu (audio vide ou silencieux ?).", file=sys.stderr)
        sys.exit(1)
    return format_as_whisperkit_lines(segments, audio_path.name)


def transcribe_audio(
    audio_path: Path, language: str | None, num_speakers: int | None, model: str | None, force_cloud: bool = False,
) -> str:
    """Point d'entrée unique : argmax-cli / whisperkit-cli si disponible, sinon fallback cloud.

    force_cloud=True force le cloud même si un transcripteur local est disponible.
    """
    if force_cloud:
        return transcribe_cloud(audio_path, num_speakers, forced=True)
    transcriber = find_local_transcriber()
    if transcriber:
        return run_local_transcriber(transcriber, audio_path, language, num_speakers, model)
    return transcribe_cloud(audio_path, num_speakers)
