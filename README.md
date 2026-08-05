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

Given a meeting video, MinuteGenius runs a 6-step pipeline:

1. **Keyframe extraction** — samples the video and dedupes near-identical
   frames (perceptual hashing), so slides/shared screens are captured without
   thousands of redundant images.
2. **Screen content analysis** — a vision model describes what's actually
   presented in each keyframe (slide content, doc, dashboard...), ignoring
   video-call chrome (participant tiles, names, toolbars).
3. **Transcription + diarization** — [WhisperKit](https://github.com/argmaxinc/WhisperKit)
   runs fully on-device (CoreML / Neural Engine) to transcribe and separate
   speakers.
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

- **macOS on Apple Silicon.** WhisperKit uses CoreML to run on the Neural
  Engine — this is the one piece that makes the pipeline fast and fully
  on-device, and it's currently macOS-only. There's no Linux/Windows support
  planned for the transcription step in the near term (contributions
  swapping in a cross-platform backend are welcome — see
  [CONTRIBUTING.md](CONTRIBUTING.md)).
- `ffmpeg` and [`whisperkit-cli`](https://github.com/argmaxinc/WhisperKit):
  ```bash
  brew install ffmpeg whisperkit-cli
  ```
- An API key for any OpenAI-compatible LLM endpoint (used for steps 2, 4 and
  6 — vision analysis, speaker ID, and write-up). Any provider works:
  [Scaleway](https://www.scaleway.com/en/generative-apis/), OpenRouter, Groq,
  a local Ollama/vLLM server, etc.

## Install

```bash
git clone https://github.com/Akvize/MinuteGenius
cd minutegenius
uv sync
cp .env.example .env   # fill in AI_API / AI_KEY
```

## Usage

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
--language fr                # transcription language (default: fr)
--speakers 4                  # number of speakers, auto-detected if omitted
--from-step 4                 # rerun from step N onward (e.g. after fixing a mis-ID)
--only-step 6                 # rerun a single step (e.g. after editing the lexicon)
```

The pipeline writes intermediate JSON/markdown files at each step
(`*_frames_analyzed.json`, `*_transcript_raw.txt`, `*_speakers.json`,
`*_report.md`, `*_report_clean.md`) so you can inspect or fix any of them by
hand before continuing.

### Just the transcript — `transcribe`

If you only want a diarized transcript without the full report pipeline
(no screen analysis, no LLM write-up, no API key needed):

```bash
uv run minutegenius transcribe --audio recording.mov --output transcript.txt

# with known speaker names
uv run minutegenius transcribe --audio recording.mov \
  --speakers-map "SPEAKER_00:Alice,SPEAKER_01:Bob"
```

### WebVTT → Markdown — `vtt`

Converts an existing `.vtt` subtitle file (e.g. exported from Zoom/Meet/Teams)
into readable markdown, grouped by speaker where available:

```bash
uv run minutegenius vtt captions.vtt --output notes.md
```

## Limitations

- macOS on Apple Silicon only (see [Requirements](#requirements)).
- Speaker identification relies on visual cues (lip movement, UI highlighting)
  — it works well for typical video calls but isn't infallible; that's why
  there's an interactive validation step before the report is generated.
- Quality of the final report scales with the lexicon you provide — without
  one, names and jargon won't get corrected.

## License

[MIT](LICENSE).

## Credits

Speech recognition and diarization powered by
[WhisperKit](https://github.com/argmaxinc/WhisperKit) (Argmax).

Built by [Akvize](https://akvize.com).
