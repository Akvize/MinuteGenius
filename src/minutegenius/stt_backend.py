"""
Transcription backend: local argmax-cli (argmax-oss-swift: WhisperKit +
SpeakerKit, Apple Silicon Mac) when available, otherwise the pyannote.ai cloud
fallback — transcription + diarization in a single call (STT Orchestration), see
transcribe_cloud(). The cloud path can also be forced explicitly with
transcribe_audio(force_cloud=True), even when a local transcriber is available.

`argmax-cli` and `whisperkit-cli` are the same binary (ArgmaxCLI target of the
argmax-oss-swift repo): same flags, same extended RTTM output:
    SPEAKER <file> 1 <start> <duration> <text…> <NA> <A|B|…> <NA> <NA>
We prefer `argmax-cli` (built from source) and fall back to `whisperkit-cli`
(Homebrew) when it isn't installed.
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
    """First local transcriber found on PATH (argmax-cli, then whisperkit-cli)."""
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
    """Transcription command with diarization.

    language: language code ("fr", "en"…); "auto" or None = automatic detection
    (--language is omitted, useful for bilingual meetings).
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
    """Fix stray spaces left by Whisper tokenization (mostly in French text).

    "J 'ai" -> "J'ai", "Est -ce" -> "Est-ce", "C'est -à -dire" -> "C'est-à-dire".
    Dialogue dashes ("- Tu réponds?") and spaced dashes ("a - b") are left untouched.
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
        print(f"\r  {frames[i % len(frames)]}  Working... {m:02d}:{s:02d}", end="", flush=True)
        i += 1
        time.sleep(0.1)
    print("\r" + " " * 30 + "\r", end="", flush=True)


def run_local_transcriber(
    transcriber: str, audio_path: Path, language: str | None, num_speakers: int | None, model: str | None,
) -> str:
    cmd = build_transcribe_cmd(transcriber, audio_path, language, num_speakers, model)

    print(f"Transcribing {audio_path.name} ({transcriber})...")
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
    print(f"  Done in {m:02d}:{s:02d}")
    return result.stdout


def format_as_whisperkit_lines(segments: list[Segment], audio_filename: str) -> str:
    """Serialize segments in the raw argmax-cli/whisperkit-cli text format, so they stay
    compatible with the existing parse_diarization_output()/_parse_speaker_line()."""
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
# faster-whisper-large-v3-turbo: 99+ languages (vs. 27 for parakeet-tdt-0.6b-v3,
# the other available option). The pyannote.ai API takes no language parameter
# (automatic detection): this model maximizes the chances of covering the language
# actually spoken, since --language isn't passed on this cloud path.
# Configurable via PYANNOTE_TRANSCRIPTION_MODEL (see .env.example).
PYANNOTE_TRANSCRIPTION_MODEL_DEFAULT = "faster-whisper-large-v3-turbo"


def _pyannote_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['PYANNOTE_API_KEY']}"}


def transcribe_pyannote(audio_path: Path, num_speakers: int | None) -> list[Segment]:
    """Transcribe and diarize an audio file in a single pyannote.ai call (STT
    Orchestration: speaker-attributed transcription). Returns the speaker-turn
    segments (turnLevelTranscription), already aligned with their speaker."""
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
            f"Unexpected pyannote.ai response for job {job_id}: "
            "'turnLevelTranscription' missing (transcription not enabled on the API side?)."
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
            error = (job.get("output") or {}).get("error", "unknown reason")
            raise RuntimeError(f"pyannote.ai job {job_id}: status {status} ({error})")
        time.sleep(PYANNOTE_POLL_INTERVAL_S)
    raise TimeoutError(f"pyannote.ai job {job_id} did not finish within {PYANNOTE_POLL_TIMEOUT_S}s")


def _require_cloud_env_vars() -> None:
    """Check cloud credentials before spending anything (network calls, paid API)."""
    if not os.environ.get("PYANNOTE_API_KEY"):
        print("Error: missing environment variable: PYANNOTE_API_KEY", file=sys.stderr)
        print("Set it in .env (see .env.example) and try again.", file=sys.stderr)
        sys.exit(1)


def transcribe_cloud(audio_path: Path, num_speakers: int | None, *, forced: bool = False) -> str:
    _require_cloud_env_vars()
    if forced:
        print("Cloud transcription requested (--cloud) — pyannote.ai (transcription + diarization)...")
    else:
        print("argmax-cli / whisperkit-cli not found — falling back to the cloud (pyannote.ai, transcription + diarization)...")
    segments = transcribe_pyannote(audio_path, num_speakers)
    if not segments:
        print("Error: no transcription segments returned (empty or silent audio?).", file=sys.stderr)
        sys.exit(1)
    return format_as_whisperkit_lines(segments, audio_path.name)


def transcribe_audio(
    audio_path: Path, language: str | None, num_speakers: int | None, model: str | None, force_cloud: bool = False,
) -> str:
    """Single entry point: argmax-cli / whisperkit-cli when available, otherwise the cloud fallback.

    force_cloud=True forces the cloud path even when a local transcriber is available.
    """
    if force_cloud:
        return transcribe_cloud(audio_path, num_speakers, forced=True)
    transcriber = find_local_transcriber()
    if transcriber:
        return run_local_transcriber(transcriber, audio_path, language, num_speakers, model)
    return transcribe_cloud(audio_path, num_speakers)
