# Cloud Transcription Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Windows/Linux users (and Mac users without WhisperKit installed) run the full MinuteGenius pipeline by falling back to a cloud transcription+diarization path — OpenAI's Whisper API for the transcript, pyannote.ai for speaker diarization — whenever `whisperkit-cli` isn't on `PATH`. WhisperKit stays the default and only path on Mac; nothing changes for existing Mac users.

**Architecture:** A new module, `src/minutegenius/stt_backend.py`, becomes the single place that knows how to get a diarized transcript out of an audio file. It exposes one function, `transcribe_audio(...)`, with the exact same signature/return type (`str`, WhisperKit's raw `SPEAKER ...` line format) that `run_whisperkit()` used to have. Internally it picks a backend: `whisperkit-cli` if present, otherwise a cloud path that (1) transcodes the audio to a small mono MP3, (2) sends it to OpenAI Whisper (`whisper-1`, `verbose_json`) for segment-level text+timestamps, (3) sends it to pyannote.ai for a `[{start, end, speaker}]` diarization, (4) merges the two by timestamp overlap, and (5) re-serializes the merged result into WhisperKit's own text format. Because step 5 reconstructs the exact line shape WhisperKit already produces, every existing consumer (`transcribe.py`'s `parse_diarization_output`, `meeting.py`'s `_parse_speaker_line`/`parse_transcript`/`parse_segments`, and the on-disk transcript caching in `meeting.py`'s step 3) needs **zero changes**.

**Tech Stack:** Python 3.11+, `openai` SDK (already a dependency) pointed at the real OpenAI API for Whisper, `requests` (new dependency) for pyannote.ai's plain REST API, `ffmpeg`/`ffprobe` (already required) for audio transcoding/chunking/duration probing, `pytest` with monkeypatching for tests (no real network calls in the test suite).

**Spec:** No separate spec doc — this plan was scoped directly in conversation. Key decisions locked in during that conversation, recorded here so they don't need to be re-derived:
- WhisperKit stays the primary/preferred backend everywhere it's available (it's free, on-device, and already diarizes) — the cloud path is a fallback only.
- Diarization fallback uses **pyannote.ai** (hosted diarization-only API from the pyannote.audio team), not a bundled transcribe+diarize service like Deepgram/AssemblyAI, and not a locally-run `pyannote.audio` (which would add heavy `torch` deps and be slow on CPU).
- Transcription fallback uses **OpenAI's Whisper API** (`whisper-1` model specifically — it's the only OpenAI transcription model that supports `verbose_json`/segment timestamps; `gpt-4o-transcribe` variants don't).
- The two cloud results are merged by timestamp overlap (max-overlap wins; nearest-midpoint as a fallback when a transcript segment has no diarization overlap at all).

## Global Constraints

- Python `>=3.11` (per `pyproject.toml` `requires-python`).
- Ruff config already in `pyproject.toml`: `select = ["E", "F", "I", "W"]`, `ignore = ["E501"]` (long lines OK — don't rewrap for line length).
- Tests are plain `pytest` functions with bare `assert` (see `tests/test_transcribe.py`, `tests/test_meeting.py`) — no test classes, no fixtures beyond `tmp_path`/`monkeypatch`. No new test framework or mocking library (use stdlib `unittest.mock.patch`/`monkeypatch`, not `requests_mock` or `responses`).
- User-facing `print()` output in this codebase is in French — match that style for any new user-facing messages (error messages, progress notices).
- `OPENAI_API_KEY` and `PYANNOTE_API_KEY` are separate from the existing `AI_KEY`/`AI_API` env vars (those point at an arbitrary OpenAI-*compatible* endpoint chosen by the user — OpenRouter, a local Ollama server, etc. — which is not guaranteed to expose OpenAI's real Whisper transcription endpoint). Do not reuse `AI_KEY`/`AI_API` for Whisper.
- OpenAI's `/v1/audio/transcriptions` endpoint has a hard 25 MB file-size limit and accepts only `mp3, mp4, mpeg, mpga, m4a, wav, webm` (verified against current OpenAI docs on 2026-09-07 — re-verify if this plan is executed much later). `ogg`/`opus` is **not** in that list — don't use it for the upload file.
- Existing whisperkit-based tests (`tests/test_transcribe.py::*`, `tests/test_meeting.py::*`) must keep passing unmodified — they exercise pure parsing functions (`_find_start_idx`, `parse_diarization_output`, `_parse_speaker_line`, `parse_segments`) that this plan does not change.

---

## File Structure

- **Create** `src/minutegenius/stt_backend.py` — new module owning all transcription-backend logic: `Segment` dataclass, backend selection, WhisperKit subprocess call (moved from `transcribe.py`), the full cloud fallback (transcode → chunk → OpenAI Whisper → pyannote.ai → merge → re-serialize).
- **Create** `tests/test_stt_backend.py` — unit tests for every function above, all network/subprocess calls mocked.
- **Modify** `src/minutegenius/transcribe.py` — remove `run_whisperkit` (moved out); import `Segment` from `stt_backend` instead of defining it locally; `main()` calls `stt_backend.transcribe_audio(...)` instead of `run_whisperkit(...)`.
- **Modify** `src/minutegenius/meeting.py` — `transcribe_video()` (lines 146–167) delegates to `stt_backend.transcribe_audio(...)` instead of building its own `whisperkit-cli` subprocess call.
- **Modify** `pyproject.toml` — add `requests` to `dependencies` (done in Task 4, the first task that needs it — not deferred to the docs task).
- **Modify** `.env.example` — document `OPENAI_API_KEY` / `PYANNOTE_API_KEY` as fallback-only config.
- **Modify** `README.md` — add a "Windows/Linux (no WhisperKit)" subsection under Requirements explaining the fallback, its cost/privacy tradeoff, and the two new env vars.
- **Modify** `.github/workflows/ci.yml` — update the comment explaining why CI runs on `ubuntu-latest`; the cloud fallback logic is now real, tested, cross-platform code, not just "parsing functions."

---

### Task 1: Extract `stt_backend.py` with `Segment` + WhisperKit call (pure refactor)

No behavior change — this task only moves code so later tasks have a home for the cloud path, and unblocks the circular-import problem (`stt_backend` must not import from `transcribe`, so `Segment` has to live in `stt_backend`).

**Files:**
- Create: `src/minutegenius/stt_backend.py`
- Modify: `src/minutegenius/transcribe.py`
- Test: `tests/test_stt_backend.py` (new)

**Interfaces:**
- Produces: `stt_backend.Segment` (dataclass: `start: float`, `end: float`, `speaker: str`, `text: str`), `stt_backend.whisperkit_available() -> bool`, `stt_backend.run_whisperkit(audio_path: Path, language: str, num_speakers: int | None, model: str | None) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stt_backend.py
import shutil
from pathlib import Path

from minutegenius import stt_backend


def test_whisperkit_available_true(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/whisperkit-cli")
    assert stt_backend.whisperkit_available() is True


def test_whisperkit_available_false(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert stt_backend.whisperkit_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stt_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'minutegenius.stt_backend'`

- [ ] **Step 3: Create `stt_backend.py` and move the WhisperKit logic into it**

```python
# src/minutegenius/stt_backend.py
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
```

- [ ] **Step 4: Update `transcribe.py` to import from `stt_backend` instead of defining locally**

In `src/minutegenius/transcribe.py`:
- Remove the `@dataclass class Segment: ...` block (lines 18–23).
- Remove the `spinner()` function (lines 26–36) — it's now `stt_backend._spinner`.
- Remove the `run_whisperkit()` function (lines 74–105).
- Remove the now-unused imports `threading` and `time` (only `spinner`/`run_whisperkit` used them) and `from dataclasses import dataclass` (only the removed `Segment` class used it) — leaving these in place trips ruff's `F401` (unused import), which is part of this repo's enabled lint rules (`select = ["E", "F", "I", "W"]`). Keep `argparse`, `shutil`, `subprocess`, `sys`, and `from pathlib import Path` — all still used by `check_ffmpeg`/`extract_audio`/`resolve_audio`/`main`.
- Add near the top, after the remaining imports:

```python
from minutegenius.stt_backend import Segment, run_whisperkit  # noqa: F401 (Segment used in type hints below)
```

`run_whisperkit` stays imported under the same name for now — Task 7 will replace its call site with `stt_backend.transcribe_audio`. Keep `main()` unchanged in this task.

- [ ] **Step 5: Run tests to verify everything passes**

Run: `pytest tests/ -v`
Expected: PASS — the new `test_stt_backend.py` tests pass, and the existing `test_transcribe.py`/`test_meeting.py` tests are unaffected since `_find_start_idx`/`parse_diarization_output` weren't touched.

- [ ] **Step 6: Commit**

```bash
git add src/minutegenius/stt_backend.py src/minutegenius/transcribe.py tests/test_stt_backend.py
git commit -m "refactor: extract Segment + WhisperKit call into stt_backend module"
```

---

### Task 2: WhisperKit-format serialization (`format_as_whisperkit_lines`)

This is the piece that lets the cloud path reuse every existing parser unchanged: it turns a `list[Segment]` back into the exact text format `parse_diarization_output`/`_parse_speaker_line` already know how to read.

**Files:**
- Modify: `src/minutegenius/stt_backend.py`
- Test: `tests/test_stt_backend.py`

**Interfaces:**
- Consumes: `stt_backend.Segment` (from Task 1).
- Produces: `stt_backend.format_as_whisperkit_lines(segments: list[Segment], audio_filename: str) -> str`, consumed by `transcribe_cloud` in Task 6 and by `transcribe.py`/`meeting.py`'s existing parsers unchanged.

- [ ] **Step 1: Write the failing test — round-trip through the real parser**

```python
# tests/test_stt_backend.py (append)
from minutegenius.stt_backend import Segment, format_as_whisperkit_lines
from minutegenius.transcribe import parse_diarization_output


def test_format_as_whisperkit_lines_roundtrips_through_existing_parser():
    segments = [
        Segment(start=0.0, end=2.5, speaker="SPEAKER_00", text="hello there"),
        Segment(start=2.5, end=3.7, speaker="SPEAKER_01", text="hi"),
    ]
    raw = format_as_whisperkit_lines(segments, "audio.wav")
    parsed = parse_diarization_output(raw)

    assert len(parsed) == 2
    assert parsed[0].speaker == "SPEAKER_00"
    assert parsed[0].start == 0.0
    assert parsed[0].end == 2.5
    assert parsed[0].text == "hello there"
    assert parsed[1].speaker == "SPEAKER_01"
    assert parsed[1].text == "hi"


def test_format_as_whisperkit_lines_normalizes_whitespace_in_text():
    segments = [Segment(start=0.0, end=1.0, speaker="SPEAKER_00", text="hello\nworld   again")]
    raw = format_as_whisperkit_lines(segments, "audio.wav")
    parsed = parse_diarization_output(raw)
    assert parsed[0].text == "hello world again"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stt_backend.py -v -k format_as_whisperkit`
Expected: FAIL with `ImportError: cannot import name 'format_as_whisperkit_lines'`

- [ ] **Step 3: Implement**

```python
# src/minutegenius/stt_backend.py (append)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stt_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/minutegenius/stt_backend.py tests/test_stt_backend.py
git commit -m "feat: serialize merged cloud segments back to WhisperKit's text format"
```

---

### Task 3: Merge diarization + transcript by timestamp overlap

**Files:**
- Modify: `src/minutegenius/stt_backend.py`
- Test: `tests/test_stt_backend.py`

**Interfaces:**
- Consumes: `stt_backend.Segment` (Task 1).
- Produces: `stt_backend.merge_diarization_transcript(diarization: list[dict], transcript_segments: list[dict]) -> list[Segment]`, consumed by `transcribe_cloud` in Task 6.
  - `diarization` items: `{"start": float, "end": float, "speaker": str}` (this is pyannote.ai's `output.diarization` shape, Task 4).
  - `transcript_segments` items: `{"start": float, "end": float, "text": str}` (this is the shape Task 5's OpenAI Whisper call produces).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stt_backend.py (append)
from minutegenius.stt_backend import merge_diarization_transcript


def test_merge_assigns_speaker_by_max_overlap():
    diarization = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    transcript = [
        {"start": 0.1, "end": 1.9, "text": "hello there"},
        {"start": 2.1, "end": 4.9, "text": "hi back"},
    ]
    merged = merge_diarization_transcript(diarization, transcript)
    assert [s.speaker for s in merged] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s.text for s in merged] == ["hello there", "hi back"]


def test_merge_picks_speaker_with_greater_overlap_when_segment_spans_two_turns():
    diarization = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    transcript = [{"start": 0.5, "end": 2.0, "text": "mostly speaker 01"}]
    merged = merge_diarization_transcript(diarization, transcript)
    assert merged[0].speaker == "SPEAKER_01"


def test_merge_falls_back_to_nearest_speaker_when_no_overlap():
    diarization = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 10.0, "end": 11.0, "speaker": "SPEAKER_01"},
    ]
    transcript = [{"start": 9.0, "end": 9.5, "text": "gap segment"}]
    merged = merge_diarization_transcript(diarization, transcript)
    assert merged[0].speaker == "SPEAKER_01"


def test_merge_skips_empty_text_segments():
    diarization = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    transcript = [{"start": 0.0, "end": 1.0, "text": "   "}]
    assert merge_diarization_transcript(diarization, transcript) == []


def test_merge_defaults_to_speaker_00_when_diarization_is_empty():
    merged = merge_diarization_transcript([], [{"start": 0.0, "end": 1.0, "text": "solo"}])
    assert merged[0].speaker == "SPEAKER_00"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stt_backend.py -v -k merge_`
Expected: FAIL with `ImportError: cannot import name 'merge_diarization_transcript'`

- [ ] **Step 3: Implement**

```python
# src/minutegenius/stt_backend.py (append)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stt_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/minutegenius/stt_backend.py tests/test_stt_backend.py
git commit -m "feat: merge pyannote.ai diarization with Whisper transcript by overlap"
```

---

### Task 4: pyannote.ai diarization client

Reference: `POST /v1/media/input` → presigned upload URL → `PUT` the file → `POST /v1/diarize` → `{jobId, status}` → poll `GET /v1/jobs/{jobId}` until `status == "succeeded"` → `output.diarization` is `list[{speaker, start, end, confidence}]`. Auth: `Authorization: Bearer <PYANNOTE_API_KEY>` on every call. (Verified against `docs.pyannote.ai` on 2026-09-07 — re-check `https://docs.pyannote.ai/api-reference/diarize` and `.../get-job` if this plan is executed much later, in case the API shape has moved on.)

**Files:**
- Modify: `src/minutegenius/stt_backend.py`
- Modify: `pyproject.toml` — add `requests` to `dependencies` (needed now, not deferred to Task 8: this task is the first to `import requests`, and `uv sync` won't install an undeclared package, so the tests below fail with `ModuleNotFoundError` until this is done here).
- Test: `tests/test_stt_backend.py`

**Interfaces:**
- Produces: `stt_backend.diarize_pyannote(audio_path: Path, num_speakers: int | None) -> list[dict]` (each item `{"start": float, "end": float, "speaker": str}` — trimmed down from pyannote.ai's full `confidence`-carrying shape since `merge_diarization_transcript` only needs these three fields), consumed by `transcribe_cloud` in Task 6.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stt_backend.py (append)
import pytest

from minutegenius import stt_backend


class _FakeResponse:
    def __init__(self, json_body, status_ok=True):
        self._json_body = json_body
        self._status_ok = status_ok

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")


def test_diarize_pyannote_full_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake audio bytes")

    calls = []

    def fake_post(url, json=None, headers=None, **kwargs):
        calls.append(("POST", url, json, headers))
        if url.endswith("/media/input"):
            return _FakeResponse({"url": "https://upload.example.com/presigned"})
        if url.endswith("/diarize"):
            assert json["numSpeakers"] == 2
            return _FakeResponse({"jobId": "job-123", "status": "created"})
        raise AssertionError(f"unexpected POST {url}")

    def fake_put(url, data=None, **kwargs):
        calls.append(("PUT", url))
        return _FakeResponse({})

    def fake_get(url, headers=None, **kwargs):
        assert url.endswith("/jobs/job-123")
        return _FakeResponse({
            "jobId": "job-123",
            "status": "succeeded",
            "output": {"diarization": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5, "confidence": {}}]},
        })

    monkeypatch.setattr(stt_backend.requests, "post", fake_post)
    monkeypatch.setattr(stt_backend.requests, "put", fake_put)
    monkeypatch.setattr(stt_backend.requests, "get", fake_get)
    monkeypatch.setattr(stt_backend.time, "sleep", lambda _: None)

    result = stt_backend.diarize_pyannote(audio_path, num_speakers=2)

    assert result == [{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}]
    assert any(method == "PUT" and u == "https://upload.example.com/presigned" for method, u, *_ in calls)
    for method, url, *_ in calls:
        if method == "POST":
            assert "Bearer test-key" in _headers_for(calls, url)


def _headers_for(calls, url):
    for entry in calls:
        if entry[0] == "POST" and entry[1] == url:
            return entry[3]["Authorization"]
    return ""


def test_diarize_pyannote_raises_on_job_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake audio bytes")
    monkeypatch.setattr(stt_backend.requests, "post", lambda url, json=None, headers=None, **kw: (
        _FakeResponse({"url": "https://upload.example.com/presigned"}) if url.endswith("/media/input")
        else _FakeResponse({"jobId": "job-123", "status": "created"})
    ))
    monkeypatch.setattr(stt_backend.requests, "put", lambda url, data=None, **kw: _FakeResponse({}))
    monkeypatch.setattr(stt_backend.requests, "get", lambda url, headers=None, **kw: _FakeResponse({
        "jobId": "job-123", "status": "failed", "output": {"error": "boom"},
    }))
    monkeypatch.setattr(stt_backend.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError):
        stt_backend.diarize_pyannote(audio_path, num_speakers=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stt_backend.py -v -k pyannote`
Expected: FAIL with `AttributeError: module 'minutegenius.stt_backend' has no attribute 'diarize_pyannote'` (and no `requests` attribute yet either)

- [ ] **Step 3: Declare the `requests` dependency**

In `pyproject.toml`, add it to `dependencies` (alphabetical, matching the existing order):

```toml
dependencies = [
    "openai",
    "pillow>=12.3.0",
    "python-dotenv",
    "requests",
]
```

Then run `uv sync --extra dev` so the package is actually installed in the environment before the next step tries to import it.

- [ ] **Step 4: Implement**

```python
# src/minutegenius/stt_backend.py — add imports at top
import os
import uuid

import requests

# ... (after the existing imports/dataclass)

PYANNOTE_BASE_URL = "https://api.pyannote.ai/v1"
PYANNOTE_POLL_INTERVAL_S = 5
PYANNOTE_POLL_TIMEOUT_S = 1800  # 30 min


def _pyannote_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['PYANNOTE_API_KEY']}"}


def diarize_pyannote(audio_path: Path, num_speakers: int | None) -> list[dict]:
    """Diarise un fichier audio via pyannote.ai. Retourne [{start, end, speaker}, ...]."""
    media_key = f"media://minutegenius-{uuid.uuid4().hex}"

    resp = requests.post(
        f"{PYANNOTE_BASE_URL}/media/input", json={"url": media_key}, headers=_pyannote_headers()
    )
    resp.raise_for_status()
    upload_url = resp.json()["url"]

    with open(audio_path, "rb") as f:
        put_resp = requests.put(upload_url, data=f)
    put_resp.raise_for_status()

    body: dict = {"url": media_key}
    if num_speakers:
        body["numSpeakers"] = num_speakers
    resp = requests.post(f"{PYANNOTE_BASE_URL}/diarize", json=body, headers=_pyannote_headers())
    resp.raise_for_status()
    job_id = resp.json()["jobId"]

    diarization = _poll_pyannote_job(job_id)
    return [{"start": t["start"], "end": t["end"], "speaker": t["speaker"]} for t in diarization]


def _poll_pyannote_job(job_id: str) -> list[dict]:
    deadline = time.time() + PYANNOTE_POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.get(f"{PYANNOTE_BASE_URL}/jobs/{job_id}", headers=_pyannote_headers())
        resp.raise_for_status()
        job = resp.json()
        status = job["status"]
        if status == "succeeded":
            return job["output"]["diarization"]
        if status in ("failed", "canceled"):
            error = job.get("output", {}).get("error", "raison inconnue")
            raise RuntimeError(f"Job pyannote.ai {job_id} : statut {status} ({error})")
        time.sleep(PYANNOTE_POLL_INTERVAL_S)
    raise TimeoutError(f"Job pyannote.ai {job_id} n'a pas terminé après {PYANNOTE_POLL_TIMEOUT_S}s")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_stt_backend.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/minutegenius/stt_backend.py tests/test_stt_backend.py
git commit -m "feat: add pyannote.ai diarization client"
```

---

### Task 5: OpenAI Whisper transcription with size-safe chunking

OpenAI's endpoint caps uploads at 25 MB. Transcoding to mono MP3 at 24 kbps keeps ~2 hours of audio under ~21.6 MB (24 kbit/s × 7200 s ÷ 8 = 21 600 000 bytes), so chunking only kicks in for recordings longer than `OPENAI_CHUNK_SECONDS`.

**Files:**
- Modify: `src/minutegenius/stt_backend.py`
- Test: `tests/test_stt_backend.py`

**Interfaces:**
- Produces: `stt_backend.transcribe_openai_whisper(audio_path: Path, language: str, temp_dir: Path) -> list[dict]` (each item `{"start": float, "end": float, "text": str}`), consumed by `transcribe_cloud` in Task 6.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stt_backend.py (append)
class _FakeSegment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeTranscriptionResponse:
    def __init__(self, segments):
        self.segments = segments


class _FakeAudioTranscriptions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, model, file, language, response_format):
        self.calls.append({"model": model, "language": language, "response_format": response_format})
        return self._responses.pop(0)


def test_prepare_upload_audio_invokes_ffmpeg_with_low_bitrate_mono(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"fake mp3")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(stt_backend.subprocess, "run", fake_run)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake wav")

    out_path = stt_backend._prepare_upload_audio(audio_path, tmp_path)

    assert out_path.exists()
    cmd = calls[0]
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
    assert "-b:a" in cmd and cmd[cmd.index("-b:a") + 1] == "24k"


def test_split_audio_if_needed_returns_single_chunk_for_short_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(stt_backend, "_probe_duration", lambda path: 60.0)
    mp3_path = tmp_path / "audio_upload.mp3"
    mp3_path.write_bytes(b"fake")

    chunks = stt_backend._split_audio_if_needed(mp3_path, tmp_path)

    assert chunks == [(mp3_path, 0.0)]


def test_split_audio_if_needed_splits_long_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(stt_backend, "_probe_duration", lambda path: 3 * stt_backend.OPENAI_CHUNK_SECONDS)

    def fake_run(cmd, capture_output, text):
        for i in range(3):
            (tmp_path / f"audio_upload_chunk_{i:03d}.mp3").write_bytes(b"fake")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(stt_backend.subprocess, "run", fake_run)
    mp3_path = tmp_path / "audio_upload.mp3"
    mp3_path.write_bytes(b"fake")

    chunks = stt_backend._split_audio_if_needed(mp3_path, tmp_path)

    assert len(chunks) == 3
    assert [offset for _, offset in chunks] == [0.0, stt_backend.OPENAI_CHUNK_SECONDS, 2 * stt_backend.OPENAI_CHUNK_SECONDS]


def test_transcribe_openai_whisper_offsets_segments_across_chunks(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    fake_client = _FakeAudioTranscriptions([
        _FakeTranscriptionResponse([_FakeSegment(0.0, 1.0, "chunk one")]),
        _FakeTranscriptionResponse([_FakeSegment(0.0, 1.0, "chunk two")]),
    ])

    class _FakeClient:
        class audio:
            transcriptions = fake_client

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(stt_backend, "_get_openai_whisper_client", lambda: _FakeClient())
    monkeypatch.setattr(stt_backend, "_prepare_upload_audio", lambda path, tmp: tmp_path / "audio_upload.mp3")
    monkeypatch.setattr(
        stt_backend, "_split_audio_if_needed",
        lambda mp3_path, tmp: [(tmp_path / "chunk0.mp3", 0.0), (tmp_path / "chunk1.mp3", 100.0)],
    )
    for name in ("chunk0.mp3", "chunk1.mp3"):
        (tmp_path / name).write_bytes(b"fake")

    segments = stt_backend.transcribe_openai_whisper(audio_path, "fr", tmp_path)

    assert segments == [
        {"start": 0.0, "end": 1.0, "text": "chunk one"},
        {"start": 100.0, "end": 101.0, "text": "chunk two"},
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stt_backend.py -v -k "upload_audio or split_audio or transcribe_openai"`
Expected: FAIL — `_prepare_upload_audio`/`_split_audio_if_needed`/`transcribe_openai_whisper`/`OPENAI_CHUNK_SECONDS` don't exist yet.

- [ ] **Step 3: Implement**

```python
# src/minutegenius/stt_backend.py — add import at top
from openai import OpenAI

# ... (after the pyannote.ai section)

OPENAI_CHUNK_SECONDS = 7200  # 2h — 24kbps mono MP3 stays well under the 25MB API limit
OPENAI_UPLOAD_BITRATE_KBPS = 24

_openai_client: OpenAI | None = None


def _get_openai_whisper_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _prepare_upload_audio(audio_path: Path, temp_dir: Path) -> Path:
    """Réencode en MP3 mono basse résolution : format accepté par l'API OpenAI,
    taille prévisible (~10.8MB/heure), suffisant pour de la voix."""
    mp3_path = temp_dir / f"{audio_path.stem}_upload.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ac", "1", "-b:a", f"{OPENAI_UPLOAD_BITRATE_KBPS}k",
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-500:], file=sys.stderr)
        sys.exit(1)
    return mp3_path


def _probe_duration(audio_path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-500:], file=sys.stderr)
        sys.exit(1)
    return float(result.stdout.strip())


def _split_audio_if_needed(mp3_path: Path, temp_dir: Path) -> list[tuple[Path, float]]:
    """Découpe en morceaux de OPENAI_CHUNK_SECONDS si l'audio dépasse cette durée.
    Retourne [(chemin_du_morceau, offset_en_secondes), ...]."""
    duration = _probe_duration(mp3_path)
    if duration <= OPENAI_CHUNK_SECONDS:
        return [(mp3_path, 0.0)]

    pattern = temp_dir / f"{mp3_path.stem}_chunk_%03d.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-f", "segment", "-segment_time", str(OPENAI_CHUNK_SECONDS),
        "-c", "copy", str(pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-500:], file=sys.stderr)
        sys.exit(1)

    chunks = sorted(temp_dir.glob(f"{mp3_path.stem}_chunk_*.mp3"))
    return [(chunk, i * OPENAI_CHUNK_SECONDS) for i, chunk in enumerate(chunks)]


def transcribe_openai_whisper(audio_path: Path, language: str, temp_dir: Path) -> list[dict]:
    """Transcrit via l'API OpenAI Whisper. Retourne [{start, end, text}, ...] avec
    les timestamps ramenés à l'échelle du fichier d'origine (offsets de chunk inclus)."""
    mp3_path = _prepare_upload_audio(audio_path, temp_dir)
    chunks = _split_audio_if_needed(mp3_path, temp_dir)

    segments: list[dict] = []
    for chunk_path, offset in chunks:
        print(f"  Transcription OpenAI Whisper ({chunk_path.name})...")
        with open(chunk_path, "rb") as f:
            response = _get_openai_whisper_client().audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                response_format="verbose_json",
            )
        segments += [
            {"start": seg.start + offset, "end": seg.end + offset, "text": seg.text}
            for seg in response.segments
        ]
    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stt_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/minutegenius/stt_backend.py tests/test_stt_backend.py
git commit -m "feat: add OpenAI Whisper transcription with size-safe chunking"
```

---

### Task 6: Public dispatcher — `transcribe_audio()`

Wires Tasks 1–5 together and is the one function `transcribe.py`/`meeting.py` will call.

**Files:**
- Modify: `src/minutegenius/stt_backend.py`
- Test: `tests/test_stt_backend.py`

**Interfaces:**
- Produces: `stt_backend.transcribe_audio(audio_path: Path, language: str, num_speakers: int | None, model: str | None) -> str` — same signature/return type `run_whisperkit` used to have. This is what Task 7 wires into `transcribe.py`/`meeting.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stt_backend.py (append)
def test_transcribe_audio_uses_whisperkit_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(stt_backend, "whisperkit_available", lambda: True)
    monkeypatch.setattr(stt_backend, "run_whisperkit", lambda *a, **kw: "WHISPERKIT_OUTPUT")

    result = stt_backend.transcribe_audio(tmp_path / "a.wav", "fr", None, None)

    assert result == "WHISPERKIT_OUTPUT"


def test_transcribe_audio_falls_back_to_cloud_when_whisperkit_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(stt_backend, "whisperkit_available", lambda: False)
    monkeypatch.setattr(stt_backend, "transcribe_cloud", lambda *a, **kw: "CLOUD_OUTPUT")

    result = stt_backend.transcribe_audio(tmp_path / "a.wav", "fr", 3, None)

    assert result == "CLOUD_OUTPUT"


def test_transcribe_cloud_wires_whisper_and_pyannote_through_merge_and_format(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(
        stt_backend, "transcribe_openai_whisper",
        lambda path, language, temp_dir: [{"start": 0.0, "end": 1.0, "text": "hello"}],
    )
    monkeypatch.setattr(
        stt_backend, "diarize_pyannote",
        lambda path, num_speakers: [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
    )

    raw = stt_backend.transcribe_cloud(audio_path, "fr", None)

    assert "SPEAKER_00" in raw
    assert "hello" in raw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stt_backend.py -v -k "transcribe_audio or transcribe_cloud"`
Expected: FAIL — `transcribe_audio`/`transcribe_cloud` don't exist yet.

- [ ] **Step 3: Implement**

```python
# src/minutegenius/stt_backend.py — add import at top
import tempfile

# ... (append at the end of the file)

def transcribe_cloud(audio_path: Path, language: str, num_speakers: int | None) -> str:
    print("whisperkit-cli introuvable — bascule sur le fallback cloud (OpenAI Whisper + pyannote.ai)...")
    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)
        transcript_segments = transcribe_openai_whisper(audio_path, language, temp_dir)
        print("  Diarisation pyannote.ai...")
        diarization = diarize_pyannote(audio_path, num_speakers)

    segments = merge_diarization_transcript(diarization, transcript_segments)
    return format_as_whisperkit_lines(segments, audio_path.name)


def transcribe_audio(audio_path: Path, language: str, num_speakers: int | None, model: str | None) -> str:
    """Point d'entrée unique : WhisperKit si disponible, sinon fallback cloud."""
    if whisperkit_available():
        return run_whisperkit(audio_path, language, num_speakers, model)
    return transcribe_cloud(audio_path, language, num_speakers)
```

Note: `diarize_pyannote` in this task receives the original `audio_path` (the extracted WAV), not the transcoded MP3 — pyannote.ai's docs don't document a file-size limit as of this writing, and full-fidelity audio gives diarization the best signal. If very long recordings turn out to hit an undocumented pyannote.ai limit, revisit by passing `transcribe_openai_whisper`'s already-transcoded MP3 instead (verify current limits at `https://docs.pyannote.ai/api-reference/diarize` first).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stt_backend.py -v`
Expected: PASS — full test suite for the module is green.

- [ ] **Step 5: Commit**

```bash
git add src/minutegenius/stt_backend.py tests/test_stt_backend.py
git commit -m "feat: add transcribe_audio() dispatcher (WhisperKit or cloud fallback)"
```

---

### Task 7: Wire real call sites (`transcribe.py`, `meeting.py`)

**Files:**
- Modify: `src/minutegenius/transcribe.py:147` (the `run_whisperkit(...)` call inside `main()` — line number as of Task 6's HEAD; Task 1's refactor shifted it from the plan's original `:203` reference, so match by content, not line number, if this drifts again)
- Modify: `src/minutegenius/meeting.py:146-167` (`transcribe_video()`)

**Interfaces:**
- Consumes: `stt_backend.transcribe_audio` (Task 6).

- [ ] **Step 1: Update `transcribe.py`**

In `src/minutegenius/transcribe.py`, change the import added in Task 1:

```python
from minutegenius.stt_backend import Segment, transcribe_audio
```

And in `main()`, replace:

```python
    raw = run_whisperkit(audio_path, args.language, args.speakers, args.model)
```

with:

```python
    raw = transcribe_audio(audio_path, args.language, args.speakers, args.model)
```

- [ ] **Step 2: Update `meeting.py`**

In `src/minutegenius/meeting.py`, add near the top imports:

```python
from minutegenius import stt_backend
```

Replace the body of `transcribe_video()` (currently lines 146–167):

```python
def transcribe_video(video_path: Path, output_path: Path, temp_dir: Path, language: str = "fr", num_speakers: int | None = None) -> str:
    """Extrait l'audio puis lance la transcription+diarisation (WhisperKit ou fallback cloud). Sauvegarde la sortie brute."""
    wav_path = extract_audio(video_path, temp_dir)
    print(f"  Audio extrait : {wav_path}")

    raw = stt_backend.transcribe_audio(wav_path, language, num_speakers, model=None)

    output_path.write_text(raw, encoding="utf-8")
    return raw
```

This removes the inline `whisperkit-cli` subprocess call and its duplicate of `_find_start_idx`-style logic that used to live only in `run_whisperkit` — `meeting.py`'s own parsing functions (`parse_transcript`, `parse_segments`, `_parse_speaker_line`) are untouched and keep reading the same file format.

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: PASS — all of `test_transcribe.py`, `test_meeting.py`, `test_vtt.py`, `test_stt_backend.py`.

- [ ] **Step 4: Manual smoke test on Mac (whisperkit path unchanged)**

Run: `uv run minutegenius transcribe --audio <a short local recording>.wav --output /tmp/out.txt`
Expected: Behaves exactly as before this plan — WhisperKit runs, transcript is written. This confirms the refactor didn't change Mac behavior.

- [ ] **Step 5: Commit**

```bash
git add src/minutegenius/transcribe.py src/minutegenius/meeting.py
git commit -m "refactor: route transcribe.py and meeting.py through stt_backend.transcribe_audio"
```

---

### Task 8: Config, dependencies, and docs

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `CONTRIBUTING.md`

**Interfaces:** none (docs/config only).

- [ ] **Step 1: Verify the `requests` dependency is already declared**

Task 4 already added `requests` to `pyproject.toml`'s `dependencies` (it needed the package before this task runs). Confirm with `grep requests pyproject.toml` — no edit needed here; this step is just a checkpoint, not new work.

- [ ] **Step 2: Document the two new env vars in `.env.example`**

Append:

```
# --- Fallback transcription (Windows/Linux, or Mac without WhisperKit) ---
# Used only when `whisperkit-cli` is not found on PATH. Requires two separate
# cloud accounts — this is independent of AI_API/AI_KEY above, since that
# endpoint may not be OpenAI's real API and won't expose the Whisper
# transcription endpoint.
#
# OpenAI API key, for the Whisper transcription itself.
#OPENAI_API_KEY=sk-...
#
# pyannote.ai API key, for speaker diarization (who-said-what).
# Get one at https://www.pyannote.ai
#PYANNOTE_API_KEY=...
```

- [ ] **Step 3: Fix the now-contradictory Requirements bullet, then add a README section**

`README.md`'s `## Requirements` section (around line 90-97) currently opens with:

```markdown
- **macOS on Apple Silicon.** WhisperKit uses CoreML to run on the Neural
  Engine — this is the one piece that makes the pipeline fast and fully
  on-device, and it's currently macOS-only. There's no Linux/Windows support
  planned for the transcription step in the near term (contributions
  swapping in a cross-platform backend are welcome — see
  [CONTRIBUTING.md](CONTRIBUTING.md)).
```

That claim ("no Linux/Windows support planned") is now false — this plan adds exactly that. Replace it with:

```markdown
- **macOS on Apple Silicon (recommended), or any OS via a cloud fallback.**
  WhisperKit uses CoreML to run fast and fully on-device on the Neural
  Engine, but it's macOS-only. On Windows, Linux, or an Intel Mac,
  MinuteGenius automatically falls back to a cloud transcription+diarization
  path instead — see "Windows / Linux (no WhisperKit)" below.
```

Then add a new subsection right after that `## Requirements` bullet list, before `## Install` (adjust heading level to match surrounding structure — this repo uses `###` for subsections under a `##` section):

```markdown
### Windows / Linux (no WhisperKit)

WhisperKit only runs on Apple Silicon (CoreML). On Windows, Linux, or an
Intel Mac, MinuteGenius automatically falls back to a cloud path when
`whisperkit-cli` isn't found on `PATH`:

- **Transcription** via the OpenAI Whisper API (`whisper-1`).
- **Diarization** (who-said-what) via [pyannote.ai](https://www.pyannote.ai).

Set `OPENAI_API_KEY` and `PYANNOTE_API_KEY` in your `.env` (see
`.env.example`) to use it. This is independent of `AI_API`/`AI_KEY`, which
configure the summarization model and may point at a non-OpenAI provider.

Tradeoffs versus WhisperKit: per-minute cloud cost on both APIs, meeting
audio leaves the machine (a consideration for confidential meetings), and it
requires network access. WhisperKit remains free, on-device, and the
default whenever it's installed — this path only kicks in when it's absent.
```

- [ ] **Step 4: Update the CI comment**

In `.github/workflows/ci.yml`, replace:

```yaml
    # Unit tests only exercise the pure parsing/formatting functions — they don't
    # need ffmpeg/whisperkit-cli, so a Linux runner is enough even though the
    # tool itself is macOS-only (see README).
```

with:

```yaml
    # Unit tests cover the pure parsing/formatting functions plus the cloud
    # transcription fallback in stt_backend.py (network and subprocess calls
    # mocked) — none of it needs ffmpeg/whisperkit-cli or real API keys, so a
    # Linux runner is enough even though whisperkit-cli itself is macOS-only
    # (see README).
```

- [ ] **Step 5: Update the equivalent stale note in `CONTRIBUTING.md`**

`CONTRIBUTING.md` (around line 19-21) has the same under-inclusive claim in its "Running tests" section:

```markdown
The test suite only covers pure parsing/formatting functions (WebVTT parsing,
WhisperKit output parsing, timestamp formatting) — it doesn't require
`ffmpeg`/`whisperkit-cli` or an API key.
```

Replace with:

```markdown
The test suite covers pure parsing/formatting functions (WebVTT parsing,
WhisperKit output parsing, timestamp formatting) plus the cloud transcription
fallback in `stt_backend.py` (network and subprocess calls mocked) — it
doesn't require `ffmpeg`/`whisperkit-cli` or real API keys.
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example README.md .github/workflows/ci.yml CONTRIBUTING.md
git commit -m "docs: document the cloud transcription fallback and its config"
```

---

### Task 9: Manual end-to-end verification (not automated)

The mocked unit tests in Tasks 1–6 prove the wiring is correct, but nothing in this plan exercises real OpenAI/pyannote.ai calls — that needs real API keys and costs real (small) money, so it's a manual step rather than a CI test.

- [ ] **Step 1:** On a machine without `whisperkit-cli` on `PATH` (or a Linux/Windows box), set `OPENAI_API_KEY` and `PYANNOTE_API_KEY` in `.env`.
- [ ] **Step 2:** Run `uv run minutegenius transcribe --audio <a short 1-2 minute multi-speaker recording> --output /tmp/out.txt --speakers 2`.
- [ ] **Step 3:** Confirm: the console prints the "bascule sur le fallback cloud" notice, no crash, and `/tmp/out.txt` contains a transcript with plausible speaker labels and text matching what was said.
- [ ] **Step 4:** Run the full `uv run minutegenius meeting <a short recording>.mp4` pipeline and confirm step 3 (transcription) and step 4 (speaker identification) both complete without error, producing a `_speakers.json` with sensible names/labels.
- [ ] **Step 5:** If available, repeat Step 2 with a recording longer than 2 hours to exercise the chunking path in `_split_audio_if_needed` — confirm segments across the chunk boundary have correct, monotonically increasing timestamps (no jump/overlap at the boundary).
