import shutil

import pytest

from minutegenius import stt_backend
from minutegenius.stt_backend import Segment, format_as_whisperkit_lines
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


class _FakeResponse:
    def __init__(self, json_body, status_ok=True):
        self._json_body = json_body
        self._status_ok = status_ok

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")


def _headers_for(calls, url):
    for entry in calls:
        if entry[0] == "POST" and entry[1] == url:
            return entry[3]["Authorization"]
    return ""


def test_transcribe_pyannote_full_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio bytes")

    calls = []

    def fake_post(url, json=None, headers=None, **kwargs):
        calls.append(("POST", url, json, headers))
        if url.endswith("/media/input"):
            return _FakeResponse({"url": "https://upload.example.com/presigned"})
        if url.endswith("/diarize"):
            assert json["numSpeakers"] == 2
            assert json["transcription"] is True
            assert json["transcriptionConfig"] == {"model": stt_backend.PYANNOTE_TRANSCRIPTION_MODEL_DEFAULT}
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
            "output": {
                "diarization": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5, "confidence": {}}],
                "turnLevelTranscription": [
                    {"start": 0.0, "end": 1.5, "text": "hello there", "speaker": "SPEAKER_00"},
                    {"start": 1.5, "end": 2.0, "text": "  ", "speaker": "SPEAKER_01"},
                ],
            },
        })

    monkeypatch.setattr(stt_backend.requests, "post", fake_post)
    monkeypatch.setattr(stt_backend.requests, "put", fake_put)
    monkeypatch.setattr(stt_backend.requests, "get", fake_get)
    monkeypatch.setattr(stt_backend.time, "sleep", lambda _: None)

    result = stt_backend.transcribe_pyannote(audio_path, num_speakers=2)

    assert result == [Segment(start=0.0, end=1.5, speaker="SPEAKER_00", text="hello there")]
    assert any(method == "PUT" and u == "https://upload.example.com/presigned" for method, u, *_ in calls)
    for method, url, *_ in calls:
        if method == "POST":
            assert "Bearer test-key" in _headers_for(calls, url)


def test_transcribe_pyannote_uses_transcription_model_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    monkeypatch.setenv("PYANNOTE_TRANSCRIPTION_MODEL", "parakeet-tdt-0.6b-v3")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio bytes")

    def fake_post(url, json=None, headers=None, **kwargs):
        if url.endswith("/media/input"):
            return _FakeResponse({"url": "https://upload.example.com/presigned"})
        assert json["transcriptionConfig"] == {"model": "parakeet-tdt-0.6b-v3"}
        return _FakeResponse({"jobId": "job-123", "status": "created"})

    monkeypatch.setattr(stt_backend.requests, "post", fake_post)
    monkeypatch.setattr(stt_backend.requests, "put", lambda url, data=None, **kw: _FakeResponse({}))
    monkeypatch.setattr(stt_backend.requests, "get", lambda url, headers=None, **kw: _FakeResponse({
        "jobId": "job-123", "status": "succeeded", "output": {"turnLevelTranscription": []},
    }))
    monkeypatch.setattr(stt_backend.time, "sleep", lambda _: None)

    stt_backend.transcribe_pyannote(audio_path, num_speakers=None)


def test_transcribe_pyannote_raises_on_job_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.wav"
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
        stt_backend.transcribe_pyannote(audio_path, num_speakers=None)


def test_transcribe_pyannote_raises_cleanly_when_job_output_is_null(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio bytes")
    monkeypatch.setattr(stt_backend.requests, "post", lambda url, json=None, headers=None, **kw: (
        _FakeResponse({"url": "https://upload.example.com/presigned"}) if url.endswith("/media/input")
        else _FakeResponse({"jobId": "job-123", "status": "created"})
    ))
    monkeypatch.setattr(stt_backend.requests, "put", lambda url, data=None, **kw: _FakeResponse({}))
    monkeypatch.setattr(stt_backend.requests, "get", lambda url, headers=None, **kw: _FakeResponse({
        "jobId": "job-123", "status": "failed", "output": None,
    }))
    monkeypatch.setattr(stt_backend.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError):
        stt_backend.transcribe_pyannote(audio_path, num_speakers=None)


def test_transcribe_pyannote_raises_when_turn_level_transcription_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio bytes")
    monkeypatch.setattr(stt_backend.requests, "post", lambda url, json=None, headers=None, **kw: (
        _FakeResponse({"url": "https://upload.example.com/presigned"}) if url.endswith("/media/input")
        else _FakeResponse({"jobId": "job-123", "status": "created"})
    ))
    monkeypatch.setattr(stt_backend.requests, "put", lambda url, data=None, **kw: _FakeResponse({}))
    monkeypatch.setattr(stt_backend.requests, "get", lambda url, headers=None, **kw: _FakeResponse({
        "jobId": "job-123", "status": "succeeded",
        "output": {"diarization": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]},
    }))
    monkeypatch.setattr(stt_backend.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError):
        stt_backend.transcribe_pyannote(audio_path, num_speakers=None)


def test_transcribe_pyannote_defaults_missing_speaker_to_speaker_00(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake audio bytes")
    monkeypatch.setattr(stt_backend.requests, "post", lambda url, json=None, headers=None, **kw: (
        _FakeResponse({"url": "https://upload.example.com/presigned"}) if url.endswith("/media/input")
        else _FakeResponse({"jobId": "job-123", "status": "created"})
    ))
    monkeypatch.setattr(stt_backend.requests, "put", lambda url, data=None, **kw: _FakeResponse({}))
    monkeypatch.setattr(stt_backend.requests, "get", lambda url, headers=None, **kw: _FakeResponse({
        "jobId": "job-123", "status": "succeeded",
        "output": {"turnLevelTranscription": [{"start": 0.0, "end": 1.0, "text": "overlap", "speaker": None}]},
    }))
    monkeypatch.setattr(stt_backend.time, "sleep", lambda _: None)

    result = stt_backend.transcribe_pyannote(audio_path, num_speakers=None)

    assert result == [Segment(start=0.0, end=1.0, speaker="SPEAKER_00", text="overlap")]


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


def test_transcribe_audio_force_cloud_skips_whisperkit_even_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(stt_backend, "whisperkit_available", lambda: True)
    monkeypatch.setattr(
        stt_backend, "run_whisperkit",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("run_whisperkit should not be called")),
    )
    calls = []
    monkeypatch.setattr(
        stt_backend, "transcribe_cloud",
        lambda *a, **kw: calls.append((a, kw)) or "CLOUD_OUTPUT",
    )

    result = stt_backend.transcribe_audio(tmp_path / "a.wav", "fr", 3, None, force_cloud=True)

    assert result == "CLOUD_OUTPUT"
    assert calls == [((tmp_path / "a.wav", 3), {"forced": True})]


def test_transcribe_cloud_forced_message_does_not_claim_whisperkit_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")
    monkeypatch.setattr(stt_backend, "transcribe_pyannote", lambda path, num_speakers: [])

    with pytest.raises(SystemExit):
        stt_backend.transcribe_cloud(audio_path, None, forced=True)

    out = capsys.readouterr().out
    assert "introuvable" not in out
    assert "--cloud" in out


def test_transcribe_cloud_wires_pyannote_through_format(monkeypatch, tmp_path):
    monkeypatch.setenv("PYANNOTE_API_KEY", "test-key")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(
        stt_backend, "transcribe_pyannote",
        lambda path, num_speakers: [Segment(start=0.0, end=1.0, speaker="SPEAKER_00", text="hello")],
    )

    raw = stt_backend.transcribe_cloud(audio_path, None)

    assert "SPEAKER_00" in raw
    assert "hello" in raw


def test_transcribe_cloud_exits_cleanly_when_api_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("PYANNOTE_API_KEY", raising=False)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    with pytest.raises(SystemExit):
        stt_backend.transcribe_cloud(audio_path, None)
