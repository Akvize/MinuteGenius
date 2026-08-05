"""
Convertit un fichier de sous-titres WebVTT (.vtt) en Markdown.
Usage: python vtt.py transcript.vtt [--output out.md]
"""
import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TIMESTAMP_RE = re.compile(
    r"(?P<start>(?:\d+:)?\d{2}:\d{2}[.,]\d{3})\s*-->\s*(?P<end>(?:\d+:)?\d{2}:\d{2}[.,]\d{3})"
)
VOICE_TAG_RE = re.compile(r"<v(?:\.[^ >]+)*\s+([^>]*)>")
TAG_RE = re.compile(r"<[^>]*>")
PARAGRAPH_GAP = 4.0  # secondes : au-delà, on saute un paragraphe (sous-titres sans locuteur)


@dataclass
class Cue:
    start: float
    end: float
    speaker: str | None
    text: str


def parse_timestamp(raw: str) -> float:
    raw = raw.replace(",", ".")
    parts = raw.split(":")
    if len(parts) == 3:
        h, m, s = parts
    else:
        h = "0"
        m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def clean_text(raw: str) -> str:
    text = TAG_RE.sub("", raw)
    text = html.unescape(text)
    return " ".join(text.split())


def parse_vtt(content: str) -> list[Cue]:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", content.strip())

    cues: list[Cue] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        ts_line_idx = None
        for i, line in enumerate(lines[:2]):
            if TIMESTAMP_RE.search(line):
                ts_line_idx = i
                break
        if ts_line_idx is None:
            continue  # bloc d'en-tête (WEBVTT, NOTE, STYLE...) ou identifiant sans timestamp

        match = TIMESTAMP_RE.search(lines[ts_line_idx])
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))

        raw_text = " ".join(lines[ts_line_idx + 1:]).strip()
        if not raw_text:
            continue

        speaker = None
        voice_match = VOICE_TAG_RE.search(raw_text)
        if voice_match:
            speaker = voice_match.group(1).strip()

        text = clean_text(raw_text)
        if not text:
            continue

        # Sous-titres auto-générés : la même phrase peut se répéter cue après cue
        # pendant l'affichage progressif (karaoke). On ignore le doublon exact.
        if cues and cues[-1].speaker == speaker and cues[-1].text == text:
            cues[-1].end = end
            continue

        cues.append(Cue(start=start, end=end, speaker=speaker, text=text))

    return cues


def build_markdown(cues: list[Cue]) -> str:
    if not cues:
        return ""

    has_speakers = any(c.speaker for c in cues)
    parts = []

    if has_speakers:
        prev_speaker = object()
        for cue in cues:
            label = cue.speaker or "?"
            if cue.speaker != prev_speaker:
                parts.append(f"\n**[{format_timestamp(cue.start)} → {format_timestamp(cue.end)}] {label}**\n")
                prev_speaker = cue.speaker
            parts.append(cue.text)
    else:
        paragraph_start = cues[0].start
        parts.append(f"\n**[{format_timestamp(paragraph_start)}]**\n")
        prev_end = cues[0].end
        for cue in cues:
            if cue.start - prev_end > PARAGRAPH_GAP:
                parts.append(f"\n\n**[{format_timestamp(cue.start)}]**\n")
            parts.append(cue.text)
            prev_end = cue.end

    return "\n".join(parts).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Convertit un fichier WebVTT (.vtt) en Markdown.")
    parser.add_argument("source", help="Fichier .vtt source")
    parser.add_argument("--output", "-o", default=None, help="Fichier ou dossier de sortie")
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"❌ Fichier introuvable : {src}", file=sys.stderr)
        sys.exit(1)

    out_arg = Path(args.output) if args.output else None
    if out_arg is None:
        out_path = src.with_suffix(".md")
    elif out_arg.is_dir():
        out_path = out_arg / src.with_suffix(".md").name
    else:
        out_path = out_arg

    print(f"📄 Extraction : {src} …")
    cues = parse_vtt(src.read_text(encoding="utf-8"))
    if not cues:
        print("❌ Aucun cue trouvé dans le fichier VTT.", file=sys.stderr)
        sys.exit(1)

    markdown = build_markdown(cues)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"✅ Terminé → {out_path} ({len(cues)} cue(s))")


if __name__ == "__main__":
    main()
