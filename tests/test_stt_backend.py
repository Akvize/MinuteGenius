import shutil

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
