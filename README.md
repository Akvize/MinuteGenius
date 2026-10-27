# MinuteGenius

**Turn a meeting recording into minutes someone actually wants to read.**

Most transcription tools give you a wall of text: who said what, roughly, with
the phonetic mistakes and filler words intact. That's a transcript, not a
report. MinuteGenius produces something different — a structured,
readable, *intelligent* account of the meeting: organized by topic (not by
whoever spoke longest), attributed to the right person, cross-referenced with
whatever was shared on screen, and corrected against your own vocabulary of
names and jargon.

It's built from a stack of existing tools (on-device speech recognition,
speaker diarization, a vision model, an LLM) — the value isn't any single
piece, it's the pipeline that turns their combined output into a document a
stakeholder can read top to bottom and actually understand what happened.

## Before / after

The raw material MinuteGenius works from is two separate, messy things: the
diarized speech (full of filler words and false starts, and blind to
anything visual), and — independently — a factual description of whatever
was on screen at that moment (from the vision model, not from anyone
describing it out loud). Neither one alone is a report. For example, the
speech track alone might contain just this:

```
[03:14] SPEAKER_00: so yeah this is kind of the the starting point uh
the the initial dashboard uh so you can step back before before going
into the detail
```

That line means nothing on its own — "the initial dashboard" showing what?
Separately, for that same moment, the vision model produced this from the
shared screen:

```
[03:10] 🖥  Dashboard. KPIs shown: "On-time deliveries: 48.5%",
            "Average receipt lead time: 77.3 days". Charts breaking down
            purchase orders (PO) and work orders (WO) by status, plus a
            list of packing lists (PKL) pending validation.
```

MinuteGenius merges the two — matching what was said to what was on screen
at that exact timestamp — and writes it up as one coherent paragraph that
keeps the actual figures and labels, not a paraphrase of them:

```markdown
### Initial dashboard walkthrough

Alice Dupont opened with the home screen, presented as an "initial
dashboard" meant to give a step-back view before diving into detail. It
displayed KPIs such as on-time deliveries at 48.5% and an average receipt
lead time of 77.3 days, along with charts breaking down purchase orders
(PO) and work orders (WO), and a list of packing lists (PKL) pending
validation.
```

Without the screen content, this paragraph couldn't exist — the speech
alone never states what the dashboard actually showed.

## How it works

![How MinuteGenius works: the recording is split into what was shown and what was said, then merged into one timeline and written up as a report](docs/minutegenius-pipeline.svg)

Given a meeting video, MinuteGenius runs a 6-step pipeline:

1. **Keyframe extraction** — samples the video and dedupes near-identical
   frames (perceptual hashing), so slides/shared screens are captured without
   thousands of redundant images.
2. **Screen content analysis** — a vision model describes what's actually
   presented in each keyframe (slide content, doc, dashboard...), ignoring
   video-call chrome (participant tiles, names, toolbars).
3. **Transcription + diarization** — `argmax-cli` ([argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift):
   WhisperKit + SpeakerKit) runs fully on-device (CoreML / Neural Engine) to
   transcribe and separate speakers on Mac, with automatic language detection
   (handy for bilingual meetings); falls back to a cloud path ([pyannote.ai](https://www.pyannote.ai))
   elsewhere — see Requirements.
4. **Speaker identification** — for each detected speaker, the vision model
   looks at frames from their longest speaking turns to put a name on them
   (lip movement + who the call UI highlights), with an interactive
   validation step before moving on.
5. **Timeline assembly** — speech and on-screen content are merged into a
   single chronological timeline.
6. **Structured write-up** — an LLM rewrites the raw timeline into an
   organized report: grouped by theme, phonetic mistakes corrected against a
   lexicon you provide, screen content woven in as context, opening with an
   executive summary.

Each step's output is cached on disk, so you can inspect, fix, or resume from
any step (`--from-step` / `--only-step`) without re-running the whole
pipeline.

## Requirements

- **macOS on Apple Silicon (recommended), or any OS via a cloud fallback.**
  argmax-cli (WhisperKit + SpeakerKit) uses CoreML to run fast and fully
  on-device on the Neural Engine, but it's macOS-only (macOS 14+). On Windows,
  Linux, or an Intel Mac, MinuteGenius automatically falls back to a cloud
  transcription+diarization path instead — see "Windows / Linux (no local
  transcriber)" below.
- `ffmpeg` (required on every platform):
  ```bash
  brew install ffmpeg   # macOS
  # or apt install ffmpeg / your OS's package manager on Linux/Windows
  ```
- `argmax-cli` from [argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift)
  (macOS only, optional — enables the fast on-device path; skip it to use the
  cloud fallback). Build it from source (Xcode 16+):
  ```bash
  git clone https://github.com/argmaxinc/argmax-oss-swift && cd argmax-oss-swift
  swift build -c release --product argmax-cli
  cp .build/release/argmax-cli "$(brew --prefix)/bin/"   # already on PATH, no sudo
  ```
  Alternatively, `brew install whisperkit-cli` installs the same binary under
  another name; MinuteGenius uses it when `argmax-cli` isn't on `PATH`.
- An API key for any OpenAI-compatible LLM endpoint (used for steps 2, 4 and
  6 — vision analysis, speaker ID, and write-up). Any provider works:
  OpenRouter, Groq, a local Ollama/vLLM server, etc.

### Windows / Linux (no local transcriber)

argmax-cli / whisperkit-cli only run on Apple Silicon (CoreML). On Windows,
Linux, or an Intel Mac, MinuteGenius automatically falls back to a cloud path
when neither `argmax-cli` nor `whisperkit-cli` is found on `PATH`:

- **Transcription + diarization**, in a single call, via [pyannote.ai](https://www.pyannote.ai)'s
  STT Orchestration API (speaker-attributed transcription).

Set `PYANNOTE_API_KEY` in your `.env` (see `.env.example`) to use it. This is
independent of `AI_API`/`AI_KEY`, which configure the summarization model and
may point at a different provider.

Tradeoffs versus the local path: per-minute cloud cost, meeting audio leaves
the machine (a consideration for confidential meetings), and it requires
network access. The local transcriber remains free, on-device, and the default
whenever it's installed — this path only kicks in when it's absent (or when
you pass `--cloud`).

## Install

```bash
git clone https://github.com/Akvize/MinuteGenius
cd minutegenius
uv sync
cp .env.example .env   # fill in AI_API / AI_KEY
```

## Usage

### Verify your setup — `check`

Validates your `.env` and tests connectivity to every configured service
(AI_API/AI_API2, pyannote.ai) before you spend time/money on a real run:

```bash
uv run minutegenius check
# also test AI_MODEL_WRITER, used by step 6's lexicon-based correction:
uv run minutegenius check --lexicon my-lexicon.md
```

### Full pipeline — `meeting`

```bash
uv run minutegenius meeting recording.mp4 --lexicon my-lexicon.md
```

See [`examples/lexicon.example.md`](examples/lexicon.example.md) for the
expected lexicon format — it's plain markdown listing participant names,
domain jargon, and known mishearings; there's no strict schema.

Useful options:

```bash
--output-dir ./out          # defaults to <video_stem>_meeting/
--language fr                # force the transcription language (default: auto-detect)
--model large-v3-v20240930_626MB  # Whisper model (best multilingual accuracy; auto if omitted)
--speakers 4                  # number of speakers, auto-detected if omitted
--cloud                       # force the pyannote.ai cloud path
--from-step 4                 # rerun from step N onward (e.g. after fixing a mis-ID)
--only-step 6                 # rerun a single step (e.g. after editing the lexicon)
--clean                       # delete intermediary files (extracted frames cache) when done
--extract-audio-only          # only extract the audio track (*_audio.wav) and exit
```

The pipeline writes intermediate JSON/markdown files at each step
(`*_frames_analyzed.json`, `*_transcript_raw.txt`, `*_speaker_map.json`,
`*_participants.json`, `*_timeline.md`, `*_minutes.md`) as well as the
extracted audio track (`*_audio.wav`) so you can inspect or fix any of them by
hand before continuing. `*_timeline.md` is the raw chronological log (speech +
on-screen content interleaved); `*_minutes.md` is the final LLM-written,
lexicon-corrected meeting summary.

By default the extracted-frames cache (used to resume with `--from-step` /
`--only-step` without re-decoding the video) is kept on disk between runs;
pass `--clean` to delete it once the pipeline finishes.

### Just the transcript — `transcribe`

If you only want a diarized transcript without the full report pipeline
(no screen analysis, no LLM write-up, no API key needed):

```bash
uv run minutegenius transcribe --audio recording.mov --output transcript.txt

# with known speaker names (labels are A, B… with argmax-cli,
# SPEAKER_00, SPEAKER_01… with the cloud fallback)
uv run minutegenius transcribe --audio recording.mov \
  --speakers-map "A:Alice,B:Bob"
```

### WebVTT → Markdown — `vtt`

Converts an existing `.vtt` subtitle file (e.g. exported from Zoom/Meet/Teams)
into readable markdown, grouped by speaker where available:

```bash
uv run minutegenius vtt captions.vtt --output notes.md
```

## Limitations

- Fully on-device transcription requires macOS on Apple Silicon; other
  platforms use the paid pyannote.ai cloud fallback (see [Requirements](#requirements)).
- Speaker identification relies on visual cues (lip movement, UI highlighting)
  — it works well for typical video calls but isn't infallible; that's why
  there's an interactive validation step before the report is generated.
- Quality of the final report scales with the lexicon you provide — without
  one, names and jargon won't get corrected.

## License

[MIT](LICENSE).

## Credits

On-device speech recognition and diarization powered by
[argmax-oss-swift](https://github.com/argmaxinc/argmax-oss-swift) (WhisperKit +
SpeakerKit, by Argmax).

Built by [Akvize](https://akvize.com).
