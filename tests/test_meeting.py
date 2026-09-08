import subprocess

from minutegenius.meeting import _fmt_time, _hamming, _parse_speaker_line, _wav_is_valid, parse_segments


def test_parse_speaker_line_simple_filename():
    parts = "SPEAKER audio.wav 1 12.34 2.50 hello world NA SPEAKER_00 NA NA".split()
    result = _parse_speaker_line(parts)
    assert result is not None
    start, duration, speaker, text_parts = result
    assert start == 12.34
    assert duration == 2.50
    assert speaker == "SPEAKER_00"
    assert text_parts == ["hello", "world"]


def test_parse_speaker_line_filename_with_spaces():
    parts = "SPEAKER my meeting recording.wav 1 12.34 2.50 hello NA SPEAKER_00 NA NA".split()
    result = _parse_speaker_line(parts)
    assert result is not None
    start, duration, speaker, _ = result
    assert start == 12.34
    assert speaker == "SPEAKER_00"


def test_parse_speaker_line_too_short_returns_none():
    assert _parse_speaker_line(["SPEAKER", "a.wav"]) is None


def test_fmt_time_under_an_hour():
    assert _fmt_time(65) == "01:05"


def test_fmt_time_over_an_hour():
    assert _fmt_time(3725) == "01:02:05"


def test_hamming_distance():
    assert _hamming(0b1010, 0b1010) == 0
    assert _hamming(0b1010, 0b0010) == 1
    assert _hamming(0b1111, 0b0000) == 4


def test_parse_segments_groups_by_speaker(tmp_path):
    transcript = tmp_path / "raw.txt"
    transcript.write_text(
        "SPEAKER audio.wav 1 0.00 2.50 hello NA SPEAKER_00 NA NA\n"
        "SPEAKER audio.wav 1 2.50 1.00 hi NA SPEAKER_01 NA NA\n"
        "SPEAKER audio.wav 1 3.50 1.00 again NA SPEAKER_00 NA NA\n"
    )
    segments = parse_segments(transcript)
    assert set(segments.keys()) == {"SPEAKER_00", "SPEAKER_01"}
    assert segments["SPEAKER_00"] == [(0.00, 2.50), (3.50, 1.00)]


def test_wav_is_valid_true_for_real_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, capture_output, text: type("R", (), {"returncode": 0, "stdout": "3.4\n"})(),
    )
    assert _wav_is_valid(tmp_path / "audio.wav") is True


def test_wav_is_valid_false_when_ffprobe_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, capture_output, text: type("R", (), {"returncode": 1, "stdout": ""})(),
    )
    assert _wav_is_valid(tmp_path / "audio.wav") is False


def test_wav_is_valid_false_for_zero_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, capture_output, text: type("R", (), {"returncode": 0, "stdout": "N/A\n"})(),
    )
    assert _wav_is_valid(tmp_path / "audio.wav") is False
