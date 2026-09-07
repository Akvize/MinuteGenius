"""
Backend de transcription : WhisperKit (local, Mac) si disponible, sinon
fallback cloud (OpenAI Whisper + pyannote.ai) — voir transcribe_cloud().
"""
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


def whisperkit_available() -> bool:
    return shutil.which("whisperkit-cli") is not None


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


def run_whisperkit(audio_path: Path, language: str, num_speakers: int | None, model: str | None) -> str:
    cmd = [
        "whisperkit-cli", "transcribe",
        "--audio-path", str(audio_path),
        "--language", language,
        "--diarization",
        "--skip-special-tokens",
    ]
    if model:
        cmd += ["--model", model]
    if num_speakers:
        cmd += ["--diarization-num-speakers", str(num_speakers)]

    print(f"Transcription de {audio_path.name}...")
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
    """Sérialise des segments au format texte brut de WhisperKit, pour rester
    compatible avec parse_diarization_output()/_parse_speaker_line() existants."""
    lines = []
    for seg in segments:
        text = " ".join(seg.text.split())
        duration = seg.end - seg.start
        lines.append(
            f"SPEAKER {audio_filename} 1 {seg.start:.3f} {duration:.3f} {text} NA {seg.speaker} NA NA"
        )
    return "\n".join(lines)


def merge_diarization_transcript(diarization: list[dict], transcript_segments: list[dict]) -> list[Segment]:
    """Associe à chaque segment de transcript le locuteur dont le tour de parole
    chevauche le plus (fallback : locuteur le plus proche par milieu de segment)."""
    merged = []
    for ts in transcript_segments:
        text = ts["text"].strip()
        if not text:
            continue

        best_speaker, best_overlap = None, 0.0
        for turn in diarization:
            overlap = min(turn["end"], ts["end"]) - max(turn["start"], ts["start"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn["speaker"]

        if best_speaker is None and diarization:
            mid = (ts["start"] + ts["end"]) / 2
            best_speaker = min(
                diarization, key=lambda t: abs((t["start"] + t["end"]) / 2 - mid)
            )["speaker"]

        merged.append(Segment(start=ts["start"], end=ts["end"], speaker=best_speaker or "SPEAKER_00", text=text))
    return merged
