"""
Transcription + diarisation avec argmax-cli / whisperkit-cli (ou fallback cloud : pyannote.ai).
Usage: python transcribe.py [--audio audio.wav] [--speakers 4] [--language auto|fr|en…]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from minutegenius.stt_backend import Segment, clean_text, transcribe_audio

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("Erreur : ffmpeg est requis pour convertir les fichiers audio/vidéo.")
        print("Installe-le avec : brew install ffmpeg (macOS) ou le gestionnaire de paquets de ton OS (Linux/Windows).")
        sys.exit(1)


def extract_audio(input_path: Path) -> Path:
    check_ffmpeg()
    wav_path = input_path.with_suffix(".wav")
    print(f"Extraction audio de {input_path.name} → {wav_path.name}...")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ar", "16000", "-ac", "1",
        str(wav_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Erreur ffmpeg :\n{result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)
    print(f"  → {wav_path.name} ({wav_path.stat().st_size // 1024 // 1024} Mo)\n")
    return wav_path


def resolve_audio(input_path: Path) -> Path:
    suffix = input_path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return input_path
    if suffix in VIDEO_EXTENSIONS:
        return extract_audio(input_path)
    print(f"Erreur : format '{suffix}' non supporté.")
    print(f"Formats acceptés : {', '.join(sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))}")
    sys.exit(1)


def _find_start_idx(parts: list[str]) -> int | None:
    """Trouve l'index du premier float dans deux floats consécutifs (start + duration).

    Robuste aux noms de fichier avec espaces : on ne suppose pas de position fixe.
    """
    for i in range(1, len(parts) - 4):
        try:
            if "." not in parts[i] or "." not in parts[i + 1]:
                continue
            float(parts[i])
            float(parts[i + 1])
            return i
        except (ValueError, IndexError):
            continue
    return None


def parse_diarization_output(raw: str) -> list[Segment]:
    """
    Parse la sortie RTTM étendue (argmax-cli / whisperkit-cli) :
    SPEAKER <file (peut avoir des espaces)> <ch> <start> <dur> <text> <NA> <label> <NA> <NA>
    """
    segments = []
    for line in raw.splitlines():
        if not line.startswith("SPEAKER"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        start_idx = _find_start_idx(parts)
        if start_idx is None:
            continue
        try:
            start = float(parts[start_idx])
            duration = float(parts[start_idx + 1])
            end = start + duration
            speaker = parts[-3]
            text = " ".join(parts[start_idx + 2:-4]).strip()
            text = clean_text(text)
            segments.append(Segment(start=start, end=end, speaker=speaker, text=text))
        except (ValueError, IndexError):
            continue

    return sorted(segments, key=lambda s: s.start)


def format_timestamp_short(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def write_txt(segments: list[Segment], output: Path, speakers_map: dict[str, str]):
    lines = []
    prev_speaker = None
    for seg in segments:
        label = speakers_map.get(seg.speaker, f"Speaker {seg.speaker}")
        if seg.speaker != prev_speaker:
            lines.append(f"\n[{format_timestamp_short(seg.start)} → {format_timestamp_short(seg.end)}] {label}")
            prev_speaker = seg.speaker
        else:
            lines.append(f"[{format_timestamp_short(seg.start)} → {format_timestamp_short(seg.end)}]")
        lines.append(seg.text)
    output.write_text("\n".join(lines).strip())
    print(f"  → {output} ({output.stat().st_size:,} octets)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="audio.wav")
    parser.add_argument("--output", default="transcript.txt")
    parser.add_argument("--language", default="auto",
                        help="Code langue (fr, en…) ou 'auto' pour la détection automatique (défaut : auto)")
    parser.add_argument("--speakers", type=int, default=None,
                        help="Nombre de locuteurs (auto si non spécifié)")
    parser.add_argument("--model", default=None,
                        help="Modèle Whisper, ex : large-v3-v20240930_626MB (auto si non spécifié)")
    parser.add_argument("--speakers-map", default=None,
                        help="Noms des locuteurs, ex: \"A:Mickael,B:Slimane\"")
    parser.add_argument("--cloud", action="store_true",
                        help="Force la transcription cloud (pyannote.ai) même si argmax-cli/whisperkit-cli est disponible")
    args = parser.parse_args()

    input_path = Path(args.audio)
    if not input_path.exists():
        print(f"Fichier introuvable : {input_path}")
        sys.exit(1)

    speakers_map = {}
    if args.speakers_map:
        for pair in args.speakers_map.split(","):
            key, _, name = pair.partition(":")
            speakers_map[key.strip()] = name.strip()

    audio_path = resolve_audio(input_path)
    raw = transcribe_audio(audio_path, args.language, args.speakers, args.model, force_cloud=args.cloud)
    segments = parse_diarization_output(raw)

    if not segments:
        print("Aucun segment parsé — sortie brute :")
        print(raw)
        sys.exit(1)

    print(f"\n{len(segments)} segments, {len(set(s.speaker for s in segments))} locuteurs détectés\n")

    out = Path(args.output)
    if out.is_dir():
        txt_path = out / input_path.with_suffix(".txt").name
    else:
        txt_path = out.with_suffix(".txt")
    write_txt(segments, txt_path, speakers_map)


if __name__ == "__main__":
    main()
