import shutil

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
