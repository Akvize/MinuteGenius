import shutil
from pathlib import Path

import pytest

from minutegenius import stt_backend
from minutegenius.stt_backend import Segment, format_as_whisperkit_lines, merge_diarization_transcript
from minutegenius.transcribe import parse_diarization_output


def test_whisperkit_available_true(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/whisperkit-cli")
    assert stt_backend.whisperkit_available() is True


def test_whisperkit_available_false(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert stt_backend.whisperkit_available() is False


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
