"""
Analyse de réunion vidéo : extraction de frames clés, identification des slides et des locuteurs.
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
    """Client dédié aux tâches vision (étape 4). Utilise AI_KEY2/AI_API2 si définis."""
    global _client_image
    if _client_image is None:
        key = os.environ.get("AI_KEY2") or os.environ["AI_KEY"]
        api = os.environ.get("AI_API2") or os.environ["AI_API"]
        _client_image = OpenAI(api_key=key, base_url=api)
    return _client_image


def analyze_frame(image_path: Path) -> dict:
    """Analyse une frame : détecte un partage d'écran et décrit son contenu.

    Retourne : {"shared_content": bool, "clearly_visible": str|null, "partially_visible": str|null}
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
                            "Tu analyses une frame issue d'une réunion en ligne.\n\n"
                            "Commence par déterminer si un contenu est partagé à l'écran "
                            "(slide PowerPoint, document, navigateur web, tableau, schéma...) ou non.\n\n"
                            "Si **aucun contenu n'est partagé** (participants en visioconférence sans partage, "
                            "ou fin de partage d'écran) : réponds uniquement :\n"
                            '{"shared_content": false, "clearly_visible": null, "partially_visible": null}\n\n'
                            "Si un **contenu est partagé**, décris uniquement le contenu partagé lui-même. "
                            "Ignore totalement tout ce qui appartient à l'interface de visioconférence : "
                            "vignettes des participants, noms affichés, barre des tâches, horodatage, "
                            "interface du logiciel de réunion. Ces éléments ne font pas partie du contenu présenté.\n\n"
                            "- `clearly_visible` : **string de texte libre** décrivant ce qui est net, lisible, certain dans le document partagé — "
                            "titre, date, noms, chiffres, structure, éléments clés. "
                            "Ce champ doit être une chaîne de caractères, pas un objet JSON imbriqué. "
                            "Conserve la langue du document tel qu'il apparaît à l'écran. "
                            "Ne complète pas, n'infère pas ce qui n'est pas clairement visible.\n"
                            "- `partially_visible` : ce qui est visible mais flou, tronqué ou incertain. "
                            "Peut être null.\n\n"
                            "Réponds UNIQUEMENT en JSON valide, sans markdown :\n"
                            '{"shared_content": true, "clearly_visible": "<description>", "partially_visible": "<description ou null>"}'
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
    # Échappe les backslashes isolés qui cassent le JSON (ex: chemins Windows)
    import re
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    # strict=False : tolère les caractères de contrôle bruts (retours à la ligne non échappés)
    # que le modèle vision insère parfois dans les valeurs de string.
    return json.loads(raw, strict=False)


def analyze_frames(frames: list[dict], output_path: Path) -> list[dict]:
    """Analyse chaque frame avec le modèle vision en parallèle. Sauvegarde le résultat intermédiaire."""
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
            print(f"  {done}/{total} frames analysées...", end="\r", flush=True)
    print()
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def _wav_is_valid(wav_path: Path) -> bool:
    """Vérifie qu'un WAV en cache est exploitable (une extraction interrompue laisse
    un fichier tronqué, sans données audio, que ffprobe ne peut pas dater)."""
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
    """Extrait la piste audio en WAV 16kHz mono pour WhisperKit, conservée comme artefact dans output_dir."""
    wav_path = output_dir / f"{video_path.stem}_audio.wav"
    if wav_path.exists():
        if _wav_is_valid(wav_path):
            return wav_path
        print("  Audio en cache invalide (extraction précédente interrompue) — ré-extraction...", file=sys.stderr)
        wav_path.unlink()
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", str(wav_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-500:], file=sys.stderr)
        sys.exit(1)
    return wav_path


def transcribe_video(
    video_path: Path, output_path: Path, output_dir: Path, language: str = "fr", num_speakers: int | None = None,
    force_cloud: bool = False,
) -> str:
    """Extrait l'audio puis lance la transcription+diarisation (WhisperKit ou fallback cloud). Sauvegarde la sortie brute."""
    wav_path = extract_audio(video_path, output_dir)
    print(f"  Audio extrait : {wav_path}")

    raw = stt_backend.transcribe_audio(wav_path, language, num_speakers, model=None, force_cloud=force_cloud)

    output_path.write_text(raw, encoding="utf-8")
    return raw


def _parse_speaker_line(parts: list[str]) -> tuple[float, float, str, list[str]] | None:
    """Parse une ligne SPEAKER déjà splittée. Robuste aux noms de fichier avec espaces.

    Format WhisperKit : SPEAKER <file (peut avoir des espaces)> <ch> <start> <dur> <texte…> <NA> <speaker> <NA> <NA>
    On cherche deux floats consécutifs depuis la position 1 (après "SPEAKER") — c'est start+dur.
    """
    # parts[-3] = speaker, parts[-1] et parts[-2] = NA, parts[-4] = NA
    if len(parts) < 9:
        return None
    speaker = parts[-3]
    # Trouver start et duration : deux floats consécutifs où la duration a un point décimal.
    # Exiger un '.' dans la duration évite de matcher des entiers du nom de fichier ou du channel.
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
    """Parse le transcript WhisperKit en liste de segments {speaker, start, end, text}."""
    segments = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("SPEAKER"):
            continue
        parsed = _parse_speaker_line(line.split())
        if parsed is None:
            continue
        start, duration, speaker, text_parts = parsed
        text = " ".join(text_parts).replace(" '", "'")
        if text.strip():
            segments.append({"speaker": speaker, "start": start, "end": start + duration, "text": text.strip()})
    return sorted(segments, key=lambda s: s["start"])


def parse_segments(transcript_path: Path) -> dict[str, list[tuple[float, float]]]:
    """Retourne {speaker: [(start, duration), ...]} depuis la sortie brute WhisperKit."""
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
    """Extrait en parallèle les frames candidates pour l'identification des locuteurs.

    Pour chaque locuteur, prend les max_times segments les plus longs et extrait
    frames_per_time frames consécutives à 1 seconde d'intervalle depuis le milieu de chacun.
    Retourne {speaker: [[time1_f1, time1_f2, time1_f3], [time2_f1, ...], ...]}.
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

    print(f"  Extraction de {len(tasks)} frames en parallèle...", flush=True)

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
            print(f"  {done}/{len(tasks)} frames extraites...", end="\r", flush=True)
    print()
    return results


def identify_speaker(frames_by_time: list[list[Path]], known: dict[str, str]) -> tuple[str | None, list[str]]:
    """Identifie un locuteur en essayant chaque groupe de frames consécutives jusqu'à trouver un résultat.

    Renvoie aussi l'union des participants visibles sur toutes les frames essayées, pour
    enrichir le référentiel de noms proposables même quand aucun locuteur n'a pu être isolé.
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
    """Identifie un locuteur en envoyant ses frames en un seul appel vision.

    Renvoie (nom_identifié_ou_None, participants_visibles) : la seconde valeur liste
    tous les noms vus sur les vignettes (phase 1.1), même quand aucun locuteur n'a pu
    être isolé, pour que ces noms restent proposables lors de la validation manuelle.
    """
    model = os.environ.get("AI_MODEL_IMAGE2") or os.environ["AI_MODEL_IMAGE"]

    valid_frames = [f for f in frames if f is not None and Path(f).exists()]
    if not valid_frames:
        return None, []

    if known:
        known_str = "Locuteurs déjà identifiés dans cette réunion :\n" + "\n".join(
            f"- {name}" for name in known.values()
        )
    else:
        known_str = "Aucun locuteur encore identifié."

    content = []
    for i, frame_path in enumerate(valid_frames):
        image_b64 = base64.b64encode(frame_path.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        content.append({"type": "text", "text": f"Image {i+1}/{len(valid_frames)}"})

    content.append({"type": "text", "text": (
        f"Tu analyses {len(valid_frames)} frames consécutives d'une réunion en ligne, "
        "toutes correspondant au même locuteur inconnu. "
        "Suis ce raisonnement en 3 phases :\n\n"
        f"{known_str}\n\n"
        "**Phase 1 — Identification et élimination des connus**\n"
        "1.1. Liste tous les participants visibles sur les vignettes (noms affichés dans l'interface).\n"
        "1.2. Élimine ceux qui sont déjà dans la liste des locuteurs identifiés ci-dessus.\n\n"
        "**Phase 2 — Analyse visuelle, sans tenir compte de ce que montre l'interface**\n"
        "À cette phase, ignore totalement les mises en valeur de l'interface (bordures, fonds colorés, noms surlignés). "
        "Dans cette réunion, des participants peuvent partager un micro, donc l'interface peut mettre en valeur "
        "la mauvaise personne.\n"
        "2.1. Parmi les restants, élimine ceux dont les mouvements sont incompatibles avec une prise de parole "
        "(bouche fermée, posture statique, main sur le menton ou le visage).\n"
        "2.2. S'il reste quelqu'un avec des mouvements clairement compatibles avec la parole "
        "(bouche nettement ouverte sur au moins 2 frames, articulation visible, gestes actifs) → "
        "c'est le locuteur. Indique 'high' si le signal est net et sans ambiguïté, 'medium' si c'est probable "
        "mais qu'une autre personne pourrait aussi convenir.\n\n"
        "**Phase 3 — Prise en compte de l'interface (seulement si phase 2 n'a pas conclu)**\n"
        "3.1. Cherche si quelqu'un parmi les restants est explicitement mis en valeur par l'interface "
        "(bordure colorée, fond coloré derrière le nom — pas un simple avatar sans indicateur visuel). "
        "Si cette personne n'a pas de vidéo active (avatar avec initiales) → c'est le locuteur (confidence: high).\n\n"
        "Dans tous les autres cas, renvoie null.\n\n"
        "Réponds UNIQUEMENT en JSON valide, sans markdown :\n"
        '{"visible_participants": ["<tous les noms listés en phase 1.1>"], '
        '"name": "<prénom et/ou nom identifié ou null>", "confidence": "high|medium", '
        '"reasoning": "<résumé de ton raisonnement phase par phase>"}'
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
                print(f"    timeout/erreur ({retry+1}/{MAX_RETRIES}), retry...", flush=True)
                time.sleep(2)
            else:
                print(f"    échec après {MAX_RETRIES} tentatives : {e}")
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
                print(f"→ {name} déjà identifié, classé inconnu — {reasoning}")
                return None, visible
            print(f"→ {name} ({confidence}) — {reasoning}")
            return name, visible
        else:
            print(f"→ indéterminé — {reasoning}")
            return None, visible
    except (json.JSONDecodeError, AttributeError):
        print("→ erreur parsing")

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
    """Validation interactive du mapping locuteur → nom, du plus au moins bavard."""
    # Trier par temps de parole décroissant
    ordered = sorted(
        speakers_map.items(),
        key=lambda x: sum(d for _, d in all_segments.get(x[0], [])),
        reverse=True,
    )

    # Liste de tous les noms connus (pour proposer un menu), enrichie des participants
    # vus sur les vignettes mais jamais retenus comme locuteur principal d'un segment.
    all_names = [name for _, name in ordered]
    for name in known_roster or []:
        if name not in all_names:
            all_names.append(name)

    # Restaurer le terminal en mode cooked (ffmpeg/whisperkit peuvent le laisser en raw)
    try:
        subprocess.run(["stty", "sane"], check=False)
    except Exception:
        pass

    print()
    print("=" * 60)
    print("Validation des locuteurs identifiés")
    print("  y       → confirmer l'identification proposée")
    print("  <n>     → choisir le n° dans la liste")
    print("  q       → quitter et conserver l'état actuel")
    print("=" * 60)

    updated = dict(speakers_map)

    for spk_id, current_name in ordered:
        segs = all_segments.get(spk_id, [])
        total_sec = sum(d for _, d in segs)

        # Timestamp de vérification : milieu du segment le plus long
        best_seg = max(segs, key=lambda s: s[1]) if segs else None
        if best_seg:
            check_ts = best_seg[0] + best_seg[1] / 2
            ts_str = _fmt_time(check_ts)
        else:
            ts_str = "?"

        print()
        print(f"Locuteur {spk_id}  ({total_sec:.0f}s de parole)  →  vérifier vers {ts_str} dans la vidéo")
        print(f"  Identification proposée : {current_name}")
        print("  Tous les participants :")
        for i, name in enumerate(all_names, 1):
            marker = "◀" if name == current_name else " "
            print(f"    {i:2d}. {name} {marker}")

        while True:
            try:
                raw_answer = input("  Votre choix [y / numéro / nom / q] : ").strip("\r\n \t")
            except (EOFError, KeyboardInterrupt):
                print("\nInterrompu.")
                output_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
                return updated

            answer = raw_answer.lower()

            if answer == "q":
                print("  → Validation interrompue, état courant sauvegardé.")
                output_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
                return updated
            elif answer in ("y", ""):
                break  # on garde le nom proposé
            elif answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(all_names):
                    updated[spk_id] = all_names[idx]
                    print(f"  → Corrigé : {all_names[idx]}")
                    break
                else:
                    print(f"  Numéro invalide (1–{len(all_names)}), recommencez.")
            else:
                # Saisie libre : le participant n'était pas proposé dans la liste.
                updated[spk_id] = raw_answer
                if raw_answer not in all_names:
                    all_names.append(raw_answer)
                print(f"  → Corrigé (saisie libre) : {raw_answer}")
                break

    output_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
    print()
    print("Validation terminée.")
    return updated


def generate_timeline(analyzed_frames: list[dict], transcript_path: Path, speakers_map: dict[str, str], output_path: Path) -> None:
    """Génère une timeline chronologique brute : speech et écrans entrelacés par timestamp."""
    segments = parse_transcript(transcript_path)
    frames_with_content = [f for f in analyzed_frames if f.get("shared_content")]

    # Construire une liste d'événements triés par timestamp
    events: list[dict] = []
    for seg in segments:
        name = speakers_map.get(seg["speaker"], seg["speaker"])
        events.append({"ts": seg["start"], "type": "speech", "speaker": name, "text": seg["text"]})
    for frame in frames_with_content:
        cv = frame.get("clearly_visible") or ""
        if cv:
            events.append({"ts": frame["timestamp"], "type": "screen", "text": cv if isinstance(cv, str) else json.dumps(cv, ensure_ascii=False)})

    events.sort(key=lambda e: e["ts"])

    lines = ["# Timeline brute de la réunion\n"]
    for ev in events:
        t = _fmt_time(ev["ts"])
        if ev["type"] == "speech":
            lines.append(f"[{t}] **{ev['speaker']}** : {ev['text']}")
        else:
            lines.append(f"[{t}] 🖥 {ev['text']}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {output_path}")


def identify_speakers(video_path: Path, transcript_path: Path, output_path: Path, frames_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Identifie tous les locuteurs et sauvegarde le mapping {speaker_id: name}.

    Renvoie aussi le référentiel de tous les participants aperçus sur les vignettes,
    y compris ceux jamais retenus comme locuteur principal d'un segment.
    """
    all_segments = parse_segments(transcript_path)
    ordered = sorted(all_segments.items(), key=lambda x: sum(d for _, d in x[1]), reverse=True)

    # Pré-extraction parallèle de toutes les frames candidates
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
            print("→ inconnu")

    output_path.write_text(json.dumps(known, indent=2, ensure_ascii=False))
    return known, roster


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    """Extrait une frame à un timestamp donné."""
    cmd = [
        "ffmpeg", "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        str(output_path), "-y",
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _video_duration(video_path: Path) -> float:
    """Retourne la durée de la vidéo en secondes via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _dhash(image_path: Path, size: int = 8) -> int:
    """Calcule le dhash perceptuel d'une image (size*size bits)."""
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
    """Extrait une frame toutes les `interval` secondes, supprime les quasi-doublons par dhash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = _video_duration(video_path)
    timestamps = [i * interval for i in range(int(duration // interval) + 1) if i * interval <= duration]
    total = len(timestamps)
    print(f"  {total} frames à extraire ({interval}s d'intervalle, durée {duration:.0f}s)...")

    # Extraction parallèle
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
            print(f"  {done}/{total} extraites...", end="\r", flush=True)
    print()

    # Déduplication dhash : sliding window — on ne supprime une frame que quand la suivante est connue
    print("  Déduplication par dhash...")
    kept: list[dict] = []
    removed = 0

    # candidate = frame "en attente" de décision, avec son hash
    candidate: tuple[int, float, Path, int] | None = None  # (idx, ts, path, hash)

    for idx, ts, path in raw_frames:
        h = _dhash(path)
        if candidate is None:
            # Première frame : toujours gardée
            candidate = (idx, ts, path, h)
        elif _hamming(h, candidate[3]) <= hamming_threshold:
            # Pareille que le candidat : on supprime le candidat, la frame courante devient le nouveau candidat
            candidate[2].unlink()
            removed += 1
            candidate = (idx, ts, path, h)
        else:
            # Différente : le candidat est une key frame, on le sauve
            _, c_ts, c_path, _ = candidate
            kept.append({"index": len(kept), "timestamp": c_ts, "path": str(c_path)})
            candidate = (idx, ts, path, h)

    # Dernière frame en attente : toujours sauvegardée
    if candidate is not None:
        _, c_ts, c_path, _ = candidate
        kept.append({"index": len(kept), "timestamp": c_ts, "path": str(c_path)})

    print(f"  {removed} doublons supprimés → {len(kept)} frames conservées")
    return kept


def _parse_report_blocks(report_path: Path) -> list[dict]:
    """Découpe le rapport en blocs {header, presented, transcription, raw_lines}.

    Chaque bloc correspond à un ## du rapport.
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
            # Avant le premier bloc (titre H1)
            blocks.append({"header": line, "presented": None, "transcription": [], "section": "preamble"})
            current = None
        elif line == "### Ce qui est présenté":
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
    """Envoie un bloc de transcription au LLM pour correction."""
    model = os.environ["AI_MODEL_WRITER"]

    context_slide = (
        f"Contenu affiché à l'écran pendant cette séquence :\n{presented.strip()}\n\n"
        if presented else ""
    )
    raw_text = "\n".join(transcription_lines).strip()
    if not raw_text:
        return raw_text

    prompt = (
        "Tu es chargé de transformer une retranscription brute de réunion en contenu structuré "
        "pour une base de connaissance (RAG). L'objectif est de produire un document dense en information, "
        "exploitable par recherche sémantique.\n\n"
        "LEXIQUE DE RÉFÉRENCE — utilise-le pour corriger les erreurs de reconnaissance phonétique :\n"
        f"{lexicon}\n\n"
        f"{context_slide}"
        "TRANSCRIPTION BRUTE :\n"
        f"{raw_text}\n\n"
        "INSTRUCTIONS :\n"
        "- Produis un compte-rendu structuré en markdown de ce qui a été dit.\n"
        "- Si plusieurs sujets distincts sont abordés dans le bloc, crée une sous-section (###) par sujet.\n"
        "- Reste verbeux : conserve les détails, chiffres, exemples, nuances — ne perds aucune information.\n"
        "- Attribue les propos à leurs auteurs de façon naturelle "
        "(ex: 'Alice Dupont explique que...', 'Selon Bob Martin,...').\n"
        "- Corrige les mishearings avec le lexique (ex: 'Kubernetiss' → 'Kubernetes', 'Jean Dupond' → 'Jean Dupont').\n"
        "- Ne mets pas de titre de niveau ## ou # — commence directement par le contenu.\n"
        "- Réponds uniquement avec le contenu markdown, sans commentaire."
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
                print(f"    échec après {MAX_RETRIES} tentatives : {e}")
                return raw_text  # Conserve l'original en cas d'échec


def generate_minutes(timeline_path: Path, lexicon_path: Path, output_path: Path, max_blocks: int | None = None) -> None:
    """Génère un compte-rendu structuré en envoyant la timeline complète au LLM en un seul appel."""
    model = os.environ["AI_MODEL_WRITER"]
    lexicon = lexicon_path.read_text(encoding="utf-8")
    timeline = timeline_path.read_text(encoding="utf-8")

    prompt = (
        "Tu es chargé de rédiger le compte-rendu d'une réunion à partir de sa timeline brute.\n\n"
        "La timeline contient deux types d'événements horodatés :\n"
        "- `[HH:MM] **Prénom Nom** : texte` — prise de parole (transcription brute, peut contenir des erreurs phonétiques)\n"
        "- `[HH:MM] 🖥 description` — capture d'écran analysée (ce qui était affiché à ce moment)\n\n"
        "LEXIQUE DE RÉFÉRENCE — utilise-le pour corriger les erreurs de reconnaissance phonétique :\n"
        f"{lexicon}\n\n"
        "TIMELINE BRUTE :\n"
        f"{timeline}\n\n"
        "INSTRUCTIONS :\n"
        "- Rédige un compte-rendu structuré en markdown.\n"
        "- Organise par grandes thématiques abordées, pas par interlocuteur ni par ordre chronologique strict.\n"
        "- Pour chaque thématique, intègre naturellement ce qui était affiché à l'écran pour enrichir le contexte.\n"
        "- Attribue les propos à leurs auteurs de façon naturelle (ex: 'Alice Dupont explique que...').\n"
        "- Conserve les détails, chiffres, exemples et nuances — ne perds aucune information importante.\n"
        "- Corrige les mishearings avec le lexique.\n"
        "- Commence par un résumé exécutif de 3-5 lignes.\n"
        "- Réponds uniquement avec le contenu markdown, sans commentaire."
    )

    API_TIMEOUT = 600
    MAX_RETRIES = 3
    print(f"  Envoi de la timeline au LLM ({len(timeline)} caractères)...", flush=True)
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
                print(f"    erreur ({retry+1}/{MAX_RETRIES}), retry... {e}")
                time.sleep(5)
            else:
                print(f"    échec après {MAX_RETRIES} tentatives : {e}")
                raise


def video_temp_dir(video_path: Path) -> Path:
    """Dossier temporaire (frames, audio) dédié à cette vidéo, sous le tmp du système.

    Nommé d'après le chemin absolu de la vidéo : stable entre deux relances de la même
    vidéo (permet la reprise après un crash), mais isolé par vidéo pour supporter
    plusieurs analyses en parallèle sans collision, même à nom de fichier identique.
    """
    digest = hashlib.sha1(str(video_path.resolve()).encode()).hexdigest()[:10]
    temp_dir = Path(tempfile.gettempdir()) / f"meeting_{video_path.stem}_{digest}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp_files(temp_dir: Path) -> None:
    """Supprime le dossier temporaire (frames, audio) une fois le pipeline terminé.

    Casse la reprise (--from-step) en cas de future relance, mais évite d'accumuler
    des Go de frames/audio pour chaque vidéo analysée.
    """
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"  Fichiers temporaires nettoyés ({temp_dir}).")


def main():
    parser = argparse.ArgumentParser(description="Analyse de réunion vidéo")
    parser.add_argument("video", type=Path, help="Fichier vidéo d'entrée")
    parser.add_argument("--scene-threshold", type=float, default=DEFAULT_SCENE_THRESHOLD,
                        help=f"Seuil de détection de scène ffmpeg (défaut: {DEFAULT_SCENE_THRESHOLD})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Répertoire de sortie (défaut: <video_stem>_meeting/)")
    parser.add_argument("--language", default="fr", help="Langue pour la transcription (défaut: fr)")
    parser.add_argument("--speakers", type=int, default=None, help="Nombre de locuteurs (auto si non spécifié)")
    parser.add_argument("--cloud", action="store_true",
                        help="Force la transcription cloud (pyannote.ai) même si whisperkit-cli est disponible")
    parser.add_argument("--lexicon", type=str, default=None, help="Chemin vers le lexique markdown pour la correction (étape 6)")
    parser.add_argument("--max-clean-blocks", type=int, default=None, help="Nombre maximum de blocs à corriger (étape 6, pour test)")
    parser.add_argument("--from-step", type=int, default=None, metavar="N",
                        help="Relancer à partir de l'étape N (supprime les sorties des étapes ≥ N)")
    parser.add_argument("--only-step", type=int, default=None, metavar="N",
                        help="Relancer uniquement l'étape N (supprime uniquement sa sortie)")
    parser.add_argument("--revalidate", action="store_true",
                        help="Relance uniquement la validation interactive des locuteurs à partir de "
                             "l'identification déjà en cache (étape 4), sans réémettre d'appels cloud "
                             "(pyannote.ai / OpenRouter)")
    parser.add_argument("--clean", action="store_true",
                        help="Supprime les fichiers intermédiaires (frames extraites, cache) à la fin du "
                             "pipeline. Par défaut ils sont conservés pour permettre une relance rapide "
                             "(--from-step / --only-step).")
    parser.add_argument("--extract-audio-only", action="store_true",
                        help="Extrait uniquement la piste audio (WAV 16kHz mono) et quitte, "
                             "sans lancer le reste du pipeline (frames, transcription, etc.).")
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve()
    if not video_path.exists():
        print(f"Erreur : fichier introuvable : {video_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or video_path.parent / f"{video_path.stem}_meeting"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.extract_audio_only:
        print(f"Vidéo    : {video_path}")
        print(f"Sortie   : {output_dir}")
        print()
        print("Extraction de la piste audio...")
        wav_path = extract_audio(video_path, output_dir)
        print(f"  → {wav_path}")
        return

    temp_dir = video_temp_dir(video_path)

    print(f"Vidéo    : {video_path}")
    print(f"Sortie   : {output_dir}")
    print(f"Temp     : {temp_dir}")
    print(f"Seuil    : {args.scene_threshold}")
    print()

    # Suppression des sorties des étapes à relancer
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
                        print(f"  ↺ étape {step} : {path.name} supprimé")
    elif args.only_step:
        for path in step_outputs.get(args.only_step, []):
            if path.exists():
                path.unlink()
                print(f"  ↺ étape {args.only_step} : {path.name} supprimé")

    if not args.lexicon:
        print("⚠️  Aucun lexique fourni (--lexicon). L'étape 6 de correction sera ignorée,")
        print("   la retranscription sera moins propre (noms propres, termes métier non corrigés).")
        try:
            answer = input("   Continuer sans lexique ? [y/N] : ").strip("\r\n \t").lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if answer != "y":
            print("Abandon. Relancez avec --lexicon <fichier>.")
            sys.exit(0)
        print()

    # Étape 1 — extraction des frames clés (seulement si l'étape 2 n'est pas déjà en
    # cache : ré-extraire du fichier vidéo est inutile si l'analyse existe déjà).
    analyzed_json = output_dir / f"{video_path.stem}_frames_analyzed.json"
    if analyzed_json.exists():
        print("Étape 1 : ignorée, l'analyse (étape 2) est déjà en cache.")
    else:
        frames_json = temp_dir / "frames.json"
        if frames_json.exists():
            print("Étape 1 : frames déjà extraites, chargement...")
            frames = json.loads(frames_json.read_text())
        else:
            print("Étape 1 : extraction une frame toutes les 5s + déduplication dhash...")
            frames = extract_interval_frames(video_path, temp_dir / "frames")
            frames_json.write_text(json.dumps(frames, indent=2, ensure_ascii=False))
            print(f"  → {frames_json}")
    print()

    # Étape 2 — analyse vision de chaque frame
    if analyzed_json.exists():
        print("Étape 2 : frames déjà analysées, chargement...")
        analyzed = json.loads(analyzed_json.read_text())
    else:
        print(f"Étape 2 : analyse vision des {len(frames)} frames...")
        analyzed = analyze_frames(frames, analyzed_json)
        projected = sum(1 for f in analyzed if f.get("shared_content"))
        print(f"  → {projected}/{len(analyzed)} frames avec contenu projeté")
        print(f"  → {analyzed_json}")
    print()

    # Étape 3 — transcription (WhisperKit ou fallback cloud)
    transcript_raw = output_dir / f"{video_path.stem}_transcript_raw.txt"
    if transcript_raw.exists():
        print("Étape 3 : transcription déjà effectuée, chargement...")
    else:
        print("Étape 3 : transcription + diarisation...")
        transcribe_video(video_path, transcript_raw, output_dir, language=args.language, num_speakers=args.speakers, force_cloud=args.cloud)
        print(f"  → {transcript_raw}")
    print()

    # Étape 4 — identification des locuteurs
    speaker_map_json = output_dir / f"{video_path.stem}_speaker_map.json"
    known_roster: list[str] = []
    if args.revalidate:
        if not speaker_map_json.exists():
            print(f"Erreur : --revalidate nécessite une identification existante ({speaker_map_json.name} introuvable).",
                  file=sys.stderr)
            print("Lancez d'abord le script sans --revalidate.", file=sys.stderr)
            sys.exit(1)
        print("Étape 4 : re-validation manuelle (identification existante rechargée, aucun appel cloud)...")
        speakers_map = json.loads(speaker_map_json.read_text())
        if participants_json.exists():
            known_roster = json.loads(participants_json.read_text())
        needs_validation = True
    elif speaker_map_json.exists():
        print("Étape 4 : locuteurs déjà identifiés, chargement...")
        speakers_map = json.loads(speaker_map_json.read_text())
        needs_validation = False
    else:
        print("Étape 4 : identification des locuteurs...")
        speakers_map, known_roster = identify_speakers(video_path, transcript_raw, speaker_map_json, temp_dir / "frames")
        participants_json.write_text(json.dumps(known_roster, indent=2, ensure_ascii=False))
        print(f"  → {speaker_map_json}")
        needs_validation = True
    print()
    for spk, name in speakers_map.items():
        print(f"  {spk} → {name}")

    # Validation interactive (uniquement après une identification fraîche)
    if needs_validation:
        all_segments = parse_segments(transcript_raw)
        old_map = dict(speakers_map)
        speakers_map = validate_speakers_interactive(speakers_map, all_segments, speaker_map_json, known_roster)
        report_invalidated = speakers_map != old_map
    else:
        report_invalidated = False
    print()

    # Étape 5 — génération de la timeline markdown
    timeline_md = output_dir / f"{video_path.stem}_timeline.md"
    if timeline_md.exists() and not report_invalidated:
        print("Étape 5 : timeline déjà générée.")
    else:
        print("Étape 5 : génération de la timeline...")
        generate_timeline(analyzed, transcript_raw, speakers_map, timeline_md)
    print()

    # Étape 6 — compte-rendu de réunion (synthèse LLM + lexique)
    minutes_md = output_dir / f"{video_path.stem}_minutes.md"
    lexicon_path = Path(args.lexicon).expanduser().resolve() if args.lexicon else None
    if args.only_step and args.only_step != 6:
        print("Étape 6 : ignorée (--only-step).")
    elif minutes_md.exists():
        print("Étape 6 : compte-rendu déjà généré.")
    elif lexicon_path is None or not lexicon_path.exists():
        print("Étape 6 : ignorée (pas de lexique fourni via --lexicon).")
    else:
        print("Étape 6 : génération du compte-rendu...")
        generate_minutes(timeline_md, lexicon_path, minutes_md, max_blocks=args.max_clean_blocks)
    print()

    # Nettoyage — le rapport final est généré, les frames extraites ne servent plus
    if args.clean:
        cleanup_temp_files(temp_dir)
    else:
        print(f"  Fichiers temporaires conservés ({temp_dir}) — relancez avec --clean pour les supprimer.")


if __name__ == "__main__":
    main()
