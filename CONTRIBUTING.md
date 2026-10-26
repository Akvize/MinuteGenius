# Contributing

Thanks for considering a contribution to MinuteGenius!

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # only needed to run the `meeting` command end-to-end
```

## Running tests

```bash
uv run pytest -q
uv run ruff check .
```

The test suite covers pure parsing/formatting functions (WebVTT parsing,
argmax-cli / WhisperKit output parsing and command building, timestamp formatting) plus the cloud transcription
fallback in `stt_backend.py` (network and subprocess calls mocked) — it
doesn't require `ffmpeg`/`argmax-cli` or real API keys.

## Scope

This project intentionally does one thing: turn a meeting recording into a
high-quality, structured report. Please keep PRs focused — if you're solving
a different problem (e.g. generic audio transcription with no meeting
context), it's probably a better fit as its own tool or an issue for
discussion first.

## Reporting issues

Please include:
- OS/chip (macOS version, Apple Silicon vs Intel)
- `argmax-cli --version` or `whisperkit-cli --version` (if installed) and `ffmpeg -version`
- The exact command you ran and the full error output
