"""
Meeting video analysis: keyframe extraction, shared-screen description and speaker identification.
Usage: python meeting.py <video.mp4> [--scene-threshold 0.4] [--output-dir ./meeting_out]
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from minutegenius import stt_backend

load_dotenv()

DEFAULT_SCENE_THRESHOLD = 0.4

_client: OpenAI | None = None
_client_image: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["AI_KEY"],
            base_url=os.environ["AI_API"],
        )
    return _client


def get_client_image() -> OpenAI:
    """Client dedicated to vision tasks (steps 2 and 4). Uses AI_KEY2/AI_API2 when set."""
    global _client_image
    if _client_image is None:
        key = os.environ.get("AI_KEY2") or os.environ["AI_KEY"]
        api = os.environ.get("AI_API2") or os.environ["AI_API"]
        _client_image = OpenAI(api_key=key, base_url=api)
    return _client_image


def analyze_frame(image_path: Path) -> dict:
    """Analyze a frame: detect screen sharing and describe the shared content.

    Returns: {"shared_content": bool, "clearly_visible": str|null, "partially_visible": str|null}
    """
    model = os.environ.get("AI_MODEL_IMAGE2") or os.environ["AI_MODEL_IMAGE"]
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    ext = image_path.suffix.lstrip(".").lower()
    media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    response = get_client_image().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are analyzing a frame from an online meeting.\n\n"
                            "First, determine whether content is being shared on screen "
                            "(PowerPoint slide, document, web browser, spreadsheet, diagram...) or not.\n\n"
                            "If **no content is shared** (participants on video with no screen share, "
                            "or the screen share has ended), reply only with:\n"
                            '{"shared_content": false, "clearly_visible": null, "partially_visible": null}\n\n'
                            "If **content is shared**, describe only the shared content itself. "
                            "Completely ignore everything that belongs to the video-call interface: "
                            "participant tiles, displayed names, taskbar, timestamps, "
                            "meeting software UI. These are not part of the presented content.\n\n"
                            "- `clearly_visible`: **free-text string** describing what is sharp, readable and certain in the shared document — "
                            "title, date, names, figures, structure, key elements. "
                            "This field must be a string, not a nested JSON object. "
                            "Keep the language of the document as it appears on screen. "
                            "Do not complete or infer anything that is not clearly visible.\n"
                            "- `partially_visible`: what is visible but blurry, cut off or uncertain. "
                            "May be null.\n\n"
                            "Reply ONLY with valid JSON, no markdown:\n"
                            '{"shared_content": true, "clearly_visible": "<description>", "partially_visible": "<description or null>"}'
                        ),
                    },
                ],
            }
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # Escape stray backslashes that break JSON (e.g. Windows paths)
    import re
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    # strict=False: tolerate raw control characters (unescaped newlines) that the
    # vision model sometimes inserts in string values.
    return json.loads(raw, strict=False)


def analyze_frames(frames: list[dict], output_path: Path) -> list[dict]:
    """Analyze every frame with the vision model in parallel. Saves the intermediate result."""
    results: list[dict | None] = [None] * len(frames)
    done = 0
    total = len(frames)

    def _analyze(frame):
        return {**frame, **analyze_frame(Path(frame["path"]))}

    workers = int(os.environ.get("AI_PARALLEL_IMAGES", "3"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze, frame): frame["index"] for frame in frames}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done += 1
            print(f"  {done}/{total} frames analyzed...", end="\r", flush=True)
    print()
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def _wav_is_valid(wav_path: Path) -> bool:
    """Check that a cached WAV is usable (an interrupted extraction leaves a truncated
    file with no audio data, whose duration ffprobe cannot read)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    try:
        return float(result.stdout.strip()) > 0
    except ValueError:
        return False


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """Extract the audio track as 16kHz mono WAV for argmax-cli, kept as an artifact in output_dir."""
    wav_path = output_dir / f"{video_path.stem}_audio.wav"
    if wav_path.exists():
        if _wav_is_valid(wav_path):
            return wav_path
        print("  Invalid cached audio (previous extraction interrupted) — re-extracting...", file=sys.stderr)
        wav_path.unlink()
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", str(wav_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-500:], file=sys.stderr)
        sys.exit(1)
    return wav_path


def transcribe_video(
    video_path: Path, output_path: Path, output_dir: Path, language: str = "auto", num_speakers: int | None = None,
    force_cloud: bool = False, model: str | None = None,
) -> str:
    """Extract the audio, then run transcription + diarization (argmax-cli or cloud fallback). Saves the raw output."""
    wav_path = extract_audio(video_path, output_dir)
    print(f"  Audio extracted: {wav_path}")

    raw = stt_backend.transcribe_audio(wav_path, language, num_speakers, model=model, force_cloud=force_cloud)

    output_path.write_text(raw, encoding="utf-8")
    return raw


def _parse_speaker_line(parts: list[str]) -> tuple[float, float, str, list[str]] | None:
    """Parse an already-split SPEAKER line. Robust to filenames containing spaces.

    argmax-cli / whisperkit-cli format: SPEAKER <file (may contain spaces)> <ch> <start> <dur> <text…> <NA> <speaker> <NA> <NA>
    We look for two consecutive floats from position 1 (after "SPEAKER") — that's start+dur.
    """
    # parts[-3] = speaker, parts[-1] and parts[-2] = NA, parts[-4] = NA
    if len(parts) < 9:
        return None
    speaker = parts[-3]
    # Find start and duration: two consecutive floats, both with a decimal point.
    # Requiring a '.' avoids matching integers from the filename or the channel.
    start_idx = None
    for i in range(1, len(parts) - 4):
        try:
            if "." not in parts[i] or "." not in parts[i + 1]:
                continue
            float(parts[i])
            float(parts[i + 1])
            start_idx = i
            break
        except (ValueError, IndexError):
            continue
    if start_idx is None:
        return None
    start = float(parts[start_idx])
    duration = float(parts[start_idx + 1])
    text_parts = parts[start_idx + 2:-4]
    return start, duration, speaker, text_parts


def parse_transcript(transcript_path: Path) -> list[dict]:
    """Parse the raw transcript (argmax-cli / whisperkit-cli) into a list of {speaker, start, end, text} segments."""
    segments = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("SPEAKER"):
            continue
        parsed = _parse_speaker_line(line.split())
        if parsed is None:
            continue
        start, duration, speaker, text_parts = parsed
        text = stt_backend.clean_text(" ".join(text_parts))
        if text.strip():
            segments.append({"speaker": speaker, "start": start, "end": start + duration, "text": text.strip()})
    return sorted(segments, key=lambda s: s["start"])


def parse_segments(transcript_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Return {speaker: [(start, duration), ...]} from the raw argmax-cli / whisperkit-cli output."""
    segments: dict[str, list[tuple[float, float]]] = {}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("SPEAKER"):
            continue
        parsed = _parse_speaker_line(line.split())
        if parsed is None:
            continue
        start, duration, speaker, _ = parsed
        segments.setdefault(speaker, []).append((start, duration))
    return segments


def prefetch_speaker_frames(video_path: Path, all_segments: dict[str, list[tuple[float, float]]], frames_dir: Path, max_times: int = 3, frames_per_time: int = 3) -> dict[str, list[list[Path]]]:
    """Extract candidate frames for speaker identification, in parallel.

    For each speaker, takes the max_times longest segments and extracts
    frames_per_time consecutive frames, 1 second apart, from the middle of each.
    Returns {speaker: [[time1_f1, time1_f2, time1_f3], [time2_f1, ...], ...]}.
    """
    duration = _video_duration(video_path)
    max_ts = max(duration - 0.1, 0.0)

    tasks: list[tuple[str, int, int, float, Path]] = []
    for speaker, segments in all_segments.items():
        top = sorted(segments, key=lambda s: s[1], reverse=True)[:max_times]
        for t_idx, (start, duration_seg) in enumerate(top):
            mid = start + duration_seg / 2
            for f_idx in range(frames_per_time):
                ts = min(mid + f_idx, max_ts)
                frame_path = frames_dir / f"speaker_{speaker}_time{t_idx+1}_frame{f_idx+1}_{ts:.2f}s.jpg"
                tasks.append((speaker, t_idx, f_idx, ts, frame_path))

    print(f"  Extracting {len(tasks)} frames in parallel...", flush=True)

    def _extract(task):
        speaker, t_idx, f_idx, ts, frame_path = task
        if not frame_path.exists():
            extract_frame(video_path, ts, frame_path)
        return speaker, t_idx, f_idx, frame_path

    results: dict[str, list[list[Path | None]]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_extract, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            speaker, t_idx, f_idx, frame_path = future.result()
            if speaker not in results:
                results[speaker] = [[None] * frames_per_time for _ in range(max_times)]
            results[speaker][t_idx][f_idx] = frame_path
            done += 1
            print(f"  {done}/{len(tasks)} frames extracted...", end="\r", flush=True)
    print()
    return results


def identify_speaker(frames_by_time: list[list[Path]], known: dict[str, str]) -> tuple[str | None, list[str]]:
    """Identify a speaker by trying each group of consecutive frames until one gives a result.

    Also returns the union of participants visible on all frames tried, to enrich the
    list of names offered for validation even when no speaker could be singled out.
    """
    roster: list[str] = []
    for t_idx, frames in enumerate(frames_by_time):
        result, visible = _identify_speaker_once([f for f in frames if f and f.exists()], known, attempt=t_idx + 1)
        for name in visible:
            if name not in roster:
                roster.append(name)
        if result:
            return result, roster
    return None, roster


def _identify_speaker_once(frames: list[Path], known: dict[str, str], attempt: int = 1) -> tuple[str | None, list[str]]:
    """Identify a speaker by sending their frames in a single vision call.

    Returns (identified_name_or_None, visible_participants): the second value lists
    every name seen on the participant tiles (phase 1.1), even when no speaker could
    be singled out, so these names can still be offered during manual validation.
    """
    model = os.environ.get("AI_MODEL_IMAGE2") or os.environ["AI_MODEL_IMAGE"]

    valid_frames = [f for f in frames if f is not None and Path(f).exists()]
    if not valid_frames:
        return None, []

    if known:
        known_str = "Speakers already identified in this meeting:\n" + "\n".join(
            f"- {name}" for name in known.values()
        )
    else:
        known_str = "No speaker identified yet."

    content = []
    for i, frame_path in enumerate(valid_frames):
        image_b64 = base64.b64encode(frame_path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        content.append({"type": "text", "text": f"Image {i+1}/{len(valid_frames)}"})

    content.append({"type": "text", "text": (
        f"You are analyzing {len(valid_frames)} consecutive frames from an online meeting, "
        "all corresponding to the same unknown speaker. "
        "Follow this 3-phase reasoning:\n\n"
        f"{known_str}\n\n"
        "**Phase 1 — Identify participants and rule out known speakers**\n"
        "1.1. List every participant visible on the tiles (names displayed in the interface).\n"
        "1.2. Rule out those already in the list of identified speakers above.\n\n"
        "**Phase 2 — Visual analysis, ignoring what the interface shows**\n"
        "In this phase, completely ignore interface highlighting (borders, colored backgrounds, highlighted names). "
        "In this meeting, participants may share a microphone, so the interface may highlight "
        "the wrong person.\n"
        "2.1. Among the remaining participants, rule out those whose movements are incompatible with speaking "
        "(closed mouth, static posture, hand on chin or face).\n"
        "2.2. If someone remains whose movements are clearly consistent with speaking "
        "(mouth clearly open on at least 2 frames, visible articulation, active gestures) → "
        "that is the speaker. Use 'high' if the signal is clear and unambiguous, 'medium' if it is likely "
        "but another person could also fit.\n\n"
        "**Phase 3 — Take the interface into account (only if phase 2 was inconclusive)**\n"
        "3.1. Check whether someone among the remaining participants is explicitly highlighted by the interface "
        "(colored border, colored background behind the name — not just an avatar with no visual indicator). "
        "If that person has no active video (avatar with initials) → that is the speaker (confidence: high).\n\n"
        "In all other cases, return null.\n\n"
        "Reply ONLY with valid JSON, no markdown:\n"
        '{"visible_participants": ["<all names listed in phase 1.1>"], '
        '"name": "<identified first and/or last name, or null>", "confidence": "high|medium", '
        '"reasoning": "<summary of your phase-by-phase reasoning>"}'
    )})

    API_TIMEOUT = 120
    MAX_RETRIES = 3

    for retry in range(MAX_RETRIES):
        try:
            response = get_client_image().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                timeout=API_TIMEOUT,
            )
            break
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                print(f"    timeout/error ({retry+1}/{MAX_RETRIES}), retrying...", flush=True)
                time.sleep(2)
            else:
                print(f"    failed after {MAX_RETRIES} attempts: {e}")
                return None, []

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        result = json.loads(raw.strip())
        name = result.get("name")
        confidence = result.get("confidence", "?")
        reasoning = result.get("reasoning", "")
        visible = [v for v in (result.get("visible_participants") or []) if isinstance(v, str) and v.strip()]
        if name and name != "null":
            if name in known.values():
                print(f"→ {name} already identified, marked as unknown — {reasoning}")
                return None, visible
            print(f"→ {name} ({confidence}) — {reasoning}")
            return name, visible
        else:
            print(f"→ undetermined — {reasoning}")
            return None, visible
    except (json.JSONDecodeError, AttributeError):
        print("→ parsing error")

    return None, []


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def validate_speakers_interactive(
    speakers_map: dict[str, str],
    all_segments: dict[str, list[tuple[float, float]]],
    output_path: Path,
    known_roster: list[str] | None = None,
) -> dict[str, str]:
    """Interactive validation of the speaker → name mapping, from most to least talkative."""
    # Sort by decreasing speaking time
    ordered = sorted(
        speakers_map.items(),
        key=lambda x: sum(d for _, d in all_segments.get(x[0], [])),
        reverse=True,
    )

    # All known names (to offer a menu), plus participants seen on the tiles
    # but never picked as the main speaker of a segment.
    all_names = [name for _, name in ordered]
    for name in known_roster or []:
        if name not in all_names:
            all_names.append(name)

    # Restore the terminal to cooked mode (ffmpeg/argmax-cli may leave it in raw mode)
    try:
        subprocess.run(["stty", "sane"], check=False)
    except Exception:
        pass

    print()
    print("=" * 60)
    print("Validate identified speakers")
    print("  y       → confirm the proposed identification")
    print("  <n>     → pick number n from the list")
    print("  <name>  → type a name not in the list")
    print("  q       → quit and keep the current state")
    print("=" * 60)

    updated = dict(speakers_map)

    for spk_id, current_name in ordered:
        segs = all_segments.get(spk_id, [])
        total_sec = sum(d for _, d in segs)

        # Timestamp to check: middle of the longest segment
        best_seg = max(segs, key=lambda s: s[1]) if segs else None
        if best_seg:
            check_ts = best_seg[0] + best_seg[1] / 2
            ts_str = _fmt_time(check_ts)
        else:
            ts_str = "?"

        print()
        print(f"Speaker {spk_id}  ({total_sec:.0f}s of speech)  →  check around {ts_str} in the video")
        print(f"  Proposed identification: {current_name}")
        print("  All participants:")
        for i, name in enumerate(all_names, 1):
            marker = "◀" if name == current_name else " "
            print(f"    {i:2d}. {name} {marker}")

        while True:
            try:
                raw_answer = input("  Your choice [y / number / name / q]: ").strip("\r\n \t")
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted.")
                output_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
                return updated

            answer = raw_answer.lower()

            if answer == "q":
                print("  → Validation stopped, current state saved.")
                output_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
                return updated
            elif answer in ("y", ""):
                break  # keep the proposed name
            elif answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(all_names):
                    updated[spk_id] = all_names[idx]
                    print(f"  → Corrected: {all_names[idx]}")
                    break
                else:
                    print(f"  Invalid number (1–{len(all_names)}), try again.")
            else:
                # Free text: the participant was not offered in the list.
                updated[spk_id] = raw_answer
                if raw_answer not in all_names:
                    all_names.append(raw_answer)
                print(f"  → Corrected (free text): {raw_answer}")
                break

    output_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
    print()
    print("Validation complete.")
    return updated


def generate_timeline(analyzed_frames: list[dict], transcript_path: Path, speakers_map: dict[str, str], output_path: Path) -> None:
    """Generate a raw chronological timeline: speech and screen content interleaved by timestamp."""
    segments = parse_transcript(transcript_path)
    frames_with_content = [f for f in analyzed_frames if f.get("shared_content")]

    # Build a list of events sorted by timestamp
    events: list[dict] = []
    for seg in segments:
        name = speakers_map.get(seg["speaker"], seg["speaker"])
        events.append({"ts": seg["start"], "type": "speech", "speaker": name, "text": seg["text"]})
    for frame in frames_with_content:
        cv = frame.get("clearly_visible") or ""
        if cv:
            events.append({"ts": frame["timestamp"], "type": "screen", "text": cv if isinstance(cv, str) else json.dumps(cv, ensure_ascii=False)})

    events.sort(key=lambda e: e["ts"])

    lines = ["# Raw meeting timeline\n"]
    for ev in events:
        t = _fmt_time(ev["ts"])
        if ev["type"] == "speech":
            lines.append(f"[{t}] **{ev['speaker']}**: {ev['text']}")
        else:
            lines.append(f"[{t}] 🖥 {ev['text']}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {output_path}")


def identify_speakers(video_path: Path, transcript_path: Path, output_path: Path, frames_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Identify all speakers and save the {speaker_id: name} mapping.

    Also returns the roster of every participant seen on the tiles,
    including those never picked as the main speaker of a segment.
    """
    all_segments = parse_segments(transcript_path)
    ordered = sorted(all_segments.items(), key=lambda x: sum(d for _, d in x[1]), reverse=True)

    # Pre-extract all candidate frames in parallel
    prefetched = prefetch_speaker_frames(video_path, dict(ordered), frames_dir)

    known: dict[str, str] = {}
    roster: list[str] = []
    for speaker, segments in ordered:
        total = sum(d for _, d in segments)
        print(f"  {speaker} ({total:.0f}s) ...", end=" ", flush=True)
        frames = prefetched.get(speaker, [])
        name, visible = identify_speaker(frames, known)
        known[speaker] = name if name else speaker
        for v in visible:
            if v not in roster:
                roster.append(v)
        if not name:
            print("→ unknown")

    output_path.write_text(json.dumps(known, indent=2, ensure_ascii=False))
    return known, roster


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    """Extract a single frame at the given timestamp."""
    cmd = [
        "ffmpeg", "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        str(output_path), "-y",
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _video_duration(video_path: Path) -> float:
    """Return the video duration in seconds, via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _dhash(image_path: Path, size: int = 8) -> int:
    """Compute the perceptual dhash of an image (size*size bits)."""
    img = Image.open(image_path).convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = list(img.getdata()) if not hasattr(img, "get_flattened_data") else list(img.get_flattened_data())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def extract_interval_frames(video_path: Path, output_dir: Path, interval: int = 5, hamming_threshold: int = 12) -> list[dict]:
    """Extract one frame every `interval` seconds and drop near-duplicates using dhash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = _video_duration(video_path)
    timestamps = [i * interval for i in range(int(duration // interval) + 1) if i * interval <= duration]
    total = len(timestamps)
    print(f"  {total} frames to extract (every {interval}s, duration {duration:.0f}s)...")

    # Parallel extraction
    def _extract(args):
        idx, ts = args
        path = output_dir / f"frame_{idx:04d}_{ts:.2f}s.jpg"
        if not path.exists():
            extract_frame(video_path, ts, path)
        return idx, ts, path

    raw_frames: list[tuple[int, float, Path]] = [None] * total
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_extract, (i, ts)): i for i, ts in enumerate(timestamps)}
        done = 0
        for future in as_completed(futures):
            idx, ts, path = future.result()
            raw_frames[idx] = (idx, ts, path)
            done += 1
            print(f"  {done}/{total} extracted...", end="\r", flush=True)
    print()

    # dhash deduplication: sliding window — a frame is only dropped once the next one is known
    print("  Deduplicating with dhash...")
    kept: list[dict] = []
    removed = 0

    # candidate = frame "pending" a decision, with its hash
    candidate: tuple[int, float, Path, int] | None = None  # (idx, ts, path, hash)

    for idx, ts, path in raw_frames:
        h = _dhash(path)
        if candidate is None:
            # First frame: always kept
            candidate = (idx, ts, path, h)
        elif _hamming(h, candidate[3]) <= hamming_threshold:
            # Same as the candidate: drop the candidate, the current frame becomes the new candidate
            candidate[2].unlink()
            removed += 1
            candidate = (idx, ts, path, h)
        else:
            # Different: the candidate is a keyframe, keep it
            _, c_ts, c_path, _ = candidate
            kept.append({"index": len(kept), "timestamp": c_ts, "path": str(c_path)})
            candidate = (idx, ts, path, h)

    # Last pending frame: always kept
    if candidate is not None:
        _, c_ts, c_path, _ = candidate
        kept.append({"index": len(kept), "timestamp": c_ts, "path": str(c_path)})

    print(f"  {removed} duplicates removed → {len(kept)} frames kept")
    return kept


def _parse_report_blocks(report_path: Path) -> list[dict]:
    """Split the report into {header, presented, transcription, raw_lines} blocks.

    Each block corresponds to a ## heading in the report.
    """
    lines = report_path.read_text(encoding="utf-8").splitlines()
    blocks = []
    current: dict | None = None

    for line in lines:
        if line.startswith("## "):
            if current:
                blocks.append(current)
            current = {"header": line, "presented": None, "transcription": [], "section": None}
        elif current is None:
            # Before the first block (H1 title)
            blocks.append({"header": line, "presented": None, "transcription": [], "section": "preamble"})
            current = None
        elif line == "### What is presented":
            current["section"] = "presented"
        elif line == "### Transcription":
            current["section"] = "transcription"
        elif current["section"] == "presented" and line.strip():
            current["presented"] = (current["presented"] or "") + line + "\n"
        elif current["section"] == "transcription":
            current["transcription"].append(line)

    if current:
        blocks.append(current)
    return blocks


def _correct_block(transcription_lines: list[str], presented: str | None, lexicon: str) -> str:
    """Send a transcription block to the LLM for correction."""
    model = os.environ["AI_MODEL_WRITER"]

    context_slide = (
        f"Content shown on screen during this sequence:\n{presented.strip()}\n\n"
        if presented else ""
    )
    raw_text = "\n".join(transcription_lines).strip()
    if not raw_text:
        return raw_text

    prompt = (
        "Your task is to turn a raw meeting transcript into structured content "
        "for a knowledge base (RAG). The goal is an information-dense document "
        "suited to semantic search.\n\n"
        "REFERENCE LEXICON — use it to fix speech-recognition errors:\n"
        f"{lexicon}\n\n"
        f"{context_slide}"
        "RAW TRANSCRIPT:\n"
        f"{raw_text}\n\n"
        "INSTRUCTIONS:\n"
        "- Write a structured markdown summary of what was said.\n"
        "- Write in the language mainly spoken in the transcript.\n"
        "- If several distinct topics are covered in the block, create one subsection (###) per topic.\n"
        "- Stay detailed: keep details, figures, examples and nuances — do not lose any information.\n"
        "- Attribute statements to their speakers naturally "
        "(e.g. 'Alice Dupont explains that...', 'According to Bob Martin,...').\n"
        "- Fix mishearings using the lexicon (e.g. 'Kubernetiss' → 'Kubernetes', 'Jean Dupond' → 'Jean Dupont').\n"
        "- Do not use ## or # headings — start directly with the content.\n"
        "- Reply only with the markdown content, no commentary."
    )

    API_TIMEOUT = 120
    MAX_RETRIES = 3
    for retry in range(MAX_RETRIES):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=API_TIMEOUT,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                print(f"    failed after {MAX_RETRIES} attempts: {e}")
                return raw_text  # Keep the original on failure


def generate_minutes(timeline_path: Path, lexicon_path: Path, output_path: Path, max_blocks: int | None = None) -> None:
    """Generate structured minutes by sending the full timeline to the LLM in a single call."""
    model = os.environ["AI_MODEL_WRITER"]
    lexicon = lexicon_path.read_text(encoding="utf-8")
    timeline = timeline_path.read_text(encoding="utf-8")

    prompt = (
        "Your task is to write the minutes of a meeting from its raw timeline.\n\n"
        "The timeline contains two kinds of timestamped events:\n"
        "- `[HH:MM] **First Last**: text` — speech (raw transcript, may contain speech-recognition errors)\n"
        "- `[HH:MM] 🖥 description` — analyzed screenshot (what was on screen at that moment)\n\n"
        "REFERENCE LEXICON — use it to fix speech-recognition errors:\n"
        f"{lexicon}\n\n"
        "RAW TIMELINE:\n"
        f"{timeline}\n\n"
        "INSTRUCTIONS:\n"
        "- Write structured minutes in markdown.\n"
        "- Write in the language mainly spoken in the meeting.\n"
        "- Organize by main topics discussed, not by speaker or strict chronological order.\n"
        "- For each topic, naturally weave in what was shown on screen to enrich the context.\n"
        "- Attribute statements to their speakers naturally (e.g. 'Alice Dupont explains that...').\n"
        "- Keep details, figures, examples and nuances — do not lose any important information.\n"
        "- Fix mishearings using the lexicon.\n"
        "- Start with a 3-5 line executive summary.\n"
        "- Reply only with the markdown content, no commentary."
    )

    API_TIMEOUT = 600
    MAX_RETRIES = 3
    print(f"  Sending the timeline to the LLM ({len(timeline)} characters)...", flush=True)
    for retry in range(MAX_RETRIES):
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=API_TIMEOUT,
                extra_body={"thinking": {"type": "disabled"}},
            )
            result = response.choices[0].message.content.strip()
            output_path.write_text(result, encoding="utf-8")
            print(f"  → {output_path}")
            return
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                print(f"    error ({retry+1}/{MAX_RETRIES}), retrying... {e}")
                time.sleep(5)
            else:
                print(f"    failed after {MAX_RETRIES} attempts: {e}")
                raise


def video_temp_dir(video_path: Path) -> Path:
    """Temporary directory (frames) dedicated to this video, under the system tmp.

    Named after the video's absolute path: stable across reruns of the same video
    (allows resuming after a crash), but isolated per video so several analyses can
    run in parallel without collisions, even with identical filenames.
    """
    digest = hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:10]
    temp_dir = Path(tempfile.gettempdir()) / f"meeting_{video_path.stem}_{digest}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp_files(temp_dir: Path) -> None:
    """Delete the temporary directory (frames) once the pipeline has finished.

    Prevents resuming (--from-step) on a later rerun, but avoids piling up
    gigabytes of frames for every analyzed video.
    """
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"  Temporary files cleaned up ({temp_dir}).")


def main():
    parser = argparse.ArgumentParser(description="Generate meeting minutes from a meeting video")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument("--scene-threshold", type=float, default=DEFAULT_SCENE_THRESHOLD,
                        help=f"ffmpeg scene-detection threshold (default: {DEFAULT_SCENE_THRESHOLD})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <video_stem>_meeting/)")
    parser.add_argument("--language", default="auto",
                        help="Transcription language (fr, en…) or 'auto' for automatic detection (default: auto)")
    parser.add_argument("--model", default=None,
                        help="Whisper model, e.g. large-v3-v20240930_626MB (auto if omitted)")
    parser.add_argument("--speakers", type=int, default=None, help="Number of speakers (auto-detected if omitted)")
    parser.add_argument("--cloud", action="store_true",
                        help="Force cloud transcription (pyannote.ai) even if argmax-cli/whisperkit-cli is available")
    parser.add_argument("--lexicon", type=str, default=None, help="Path to the markdown lexicon used for corrections (step 6)")
    parser.add_argument("--max-clean-blocks", type=int, default=None, help="Maximum number of blocks to correct (step 6, for testing)")
    parser.add_argument("--from-step", type=int, default=None, metavar="N",
                        help="Rerun from step N onward (deletes the outputs of steps ≥ N)")
    parser.add_argument("--only-step", type=int, default=None, metavar="N",
                        help="Rerun only step N (deletes only its output)")
    parser.add_argument("--revalidate", action="store_true",
                        help="Rerun only the interactive speaker validation, from the identification "
                             "already cached (step 4), without making any new API calls")
    parser.add_argument("--clean", action="store_true",
                        help="Delete intermediate files (extracted frames, cache) at the end of the "
                             "pipeline. By default they are kept to allow fast reruns "
                             "(--from-step / --only-step).")
    parser.add_argument("--extract-audio-only", action="store_true",
                        help="Extract only the audio track (16kHz mono WAV) and exit, "
                             "without running the rest of the pipeline (frames, transcription, etc.).")
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or video_path.parent / f"{video_path.stem}_meeting"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.extract_audio_only:
        print(f"Video    : {video_path}")
        print(f"Output   : {output_dir}")
        print()
        print("Extracting the audio track...")
        wav_path = extract_audio(video_path, output_dir)
        print(f"  → {wav_path}")
        return

    temp_dir = video_temp_dir(video_path)

    print(f"Video    : {video_path}")
    print(f"Output   : {output_dir}")
    print(f"Temp     : {temp_dir}")
    print(f"Threshold: {args.scene_threshold}")
    print()

    # Delete the outputs of the steps to rerun
    participants_json = output_dir / f"{video_path.stem}_participants.json"
    step_outputs = {
        2: [output_dir / f"{video_path.stem}_frames_analyzed.json"],
        3: [output_dir / f"{video_path.stem}_transcript_raw.txt"],
        4: [output_dir / f"{video_path.stem}_speaker_map.json", participants_json],
        5: [output_dir / f"{video_path.stem}_timeline.md"],
        6: [output_dir / f"{video_path.stem}_minutes.md"],
    }
    if args.from_step:
        for step, paths in step_outputs.items():
            if step >= args.from_step:
                for path in paths:
                    if path.exists():
                        path.unlink()
                        print(f"  ↺ step {step}: {path.name} deleted")
    elif args.only_step:
        for path in step_outputs.get(args.only_step, []):
            if path.exists():
                path.unlink()
                print(f"  ↺ step {args.only_step}: {path.name} deleted")

    if not args.lexicon:
        print("⚠️  No lexicon provided (--lexicon). Step 6 (write-up) will be skipped,")
        print("   and names and domain terms in the transcript won't be corrected.")
        try:
            answer = input("   Continue without a lexicon? [y/N]: ").strip("\r\n \t").lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if answer != "y":
            print("Aborted. Rerun with --lexicon <file>.")
            sys.exit(0)
        print()

    # Step 1 — keyframe extraction (only if step 2 isn't already cached:
    # re-extracting from the video is pointless if the analysis already exists).
    analyzed_json = output_dir / f"{video_path.stem}_frames_analyzed.json"
    if analyzed_json.exists():
        print("Step 1: skipped, the analysis (step 2) is already cached.")
    else:
        frames_json = temp_dir / "frames.json"
        if frames_json.exists():
            print("Step 1: frames already extracted, loading...")
            frames = json.loads(frames_json.read_text())
        else:
            print("Step 1: extracting one frame every 5s + dhash deduplication...")
            frames = extract_interval_frames(video_path, temp_dir / "frames")
            frames_json.write_text(json.dumps(frames, indent=2, ensure_ascii=False))
            print(f"  → {frames_json}")
    print()

    # Step 2 — vision analysis of each frame
    if analyzed_json.exists():
        print("Step 2: frames already analyzed, loading...")
        analyzed = json.loads(analyzed_json.read_text())
    else:
        print(f"Step 2: vision analysis of {len(frames)} frames...")
        analyzed = analyze_frames(frames, analyzed_json)
        projected = sum(1 for f in analyzed if f.get("shared_content"))
        print(f"  → {projected}/{len(analyzed)} frames with shared content")
        print(f"  → {analyzed_json}")
    print()

    # Step 3 — transcription (argmax-cli or cloud fallback)
    transcript_raw = output_dir / f"{video_path.stem}_transcript_raw.txt"
    if transcript_raw.exists():
        print("Step 3: transcription already done, loading...")
    else:
        print("Step 3: transcription + diarization...")
        transcribe_video(video_path, transcript_raw, output_dir, language=args.language, num_speakers=args.speakers,
                         force_cloud=args.cloud, model=args.model)
        print(f"  → {transcript_raw}")
    print()

    # Step 4 — speaker identification
    speaker_map_json = output_dir / f"{video_path.stem}_speaker_map.json"
    known_roster: list[str] = []
    if args.revalidate:
        if not speaker_map_json.exists():
            print(f"Error: --revalidate requires an existing identification ({speaker_map_json.name} not found).",
                  file=sys.stderr)
            print("Run the command without --revalidate first.", file=sys.stderr)
            sys.exit(1)
        print("Step 4: manual re-validation (existing identification reloaded, no API calls)...")
        speakers_map = json.loads(speaker_map_json.read_text())
        if participants_json.exists():
            known_roster = json.loads(participants_json.read_text())
        needs_validation = True
    elif speaker_map_json.exists():
        print("Step 4: speakers already identified, loading...")
        speakers_map = json.loads(speaker_map_json.read_text())
        needs_validation = False
    else:
        print("Step 4: identifying speakers...")
        speakers_map, known_roster = identify_speakers(video_path, transcript_raw, speaker_map_json, temp_dir / "frames")
        participants_json.write_text(json.dumps(known_roster, indent=2, ensure_ascii=False))
        print(f"  → {speaker_map_json}")
        needs_validation = True
    print()
    for spk, name in speakers_map.items():
        print(f"  {spk} → {name}")

    # Interactive validation (only after a fresh identification or --revalidate)
    if needs_validation:
        all_segments = parse_segments(transcript_raw)
        old_map = dict(speakers_map)
        speakers_map = validate_speakers_interactive(speakers_map, all_segments, speaker_map_json, known_roster)
        report_invalidated = speakers_map != old_map
    else:
        report_invalidated = False
    print()

    # Step 5 — markdown timeline generation
    timeline_md = output_dir / f"{video_path.stem}_timeline.md"
    if timeline_md.exists() and not report_invalidated:
        print("Step 5: timeline already generated.")
    else:
        print("Step 5: generating the timeline...")
        generate_timeline(analyzed, transcript_raw, speakers_map, timeline_md)
    print()

    # Step 6 — meeting minutes (LLM write-up + lexicon)
    minutes_md = output_dir / f"{video_path.stem}_minutes.md"
    lexicon_path = Path(args.lexicon).expanduser().resolve() if args.lexicon else None
    if args.only_step and args.only_step != 6:
        print("Step 6: skipped (--only-step).")
    elif minutes_md.exists():
        print("Step 6: minutes already generated.")
    elif lexicon_path is None or not lexicon_path.exists():
        print("Step 6: skipped (no lexicon provided via --lexicon).")
    else:
        print("Step 6: generating the minutes...")
        generate_minutes(timeline_md, lexicon_path, minutes_md, max_blocks=args.max_clean_blocks)
    print()

    # Cleanup — the final report is written, the extracted frames are no longer needed
    if args.clean:
        cleanup_temp_files(temp_dir)
    else:
        print(f"  Temporary files kept ({temp_dir}) — rerun with --clean to delete them.")


if __name__ == "__main__":
    main()
