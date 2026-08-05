from minutegenius.transcribe import _find_start_idx, parse_diarization_output


def test_find_start_idx_simple_filename():
    # SPEAKER <file> <ch> <start> <dur> <text...> <NA> <label> <NA> <NA>
    parts = "SPEAKER audio.wav 1 12.34 2.50 hello world NA SPEAKER_00 NA NA".split()
    assert _find_start_idx(parts) == 3


def test_find_start_idx_filename_with_spaces():
    parts = "SPEAKER my meeting recording.wav 1 12.34 2.50 hello NA SPEAKER_00 NA NA".split()
    idx = _find_start_idx(parts)
    assert idx is not None
    assert float(parts[idx]) == 12.34
    assert float(parts[idx + 1]) == 2.50


def test_find_start_idx_returns_none_when_no_floats():
    parts = "SPEAKER audio.wav 1 hello world NA SPEAKER_00 NA NA".split()
    assert _find_start_idx(parts) is None


def test_parse_diarization_output_basic():
    raw = (
        "SPEAKER audio.wav 1 0.00 2.50 hello there NA SPEAKER_00 NA NA\n"
        "SPEAKER audio.wav 1 2.50 1.20 hi NA SPEAKER_01 NA NA\n"
    )
    segments = parse_diarization_output(raw)
    assert len(segments) == 2
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[0].start == 0.0
    assert segments[0].end == 2.5
    assert segments[0].text == "hello there"
    assert segments[1].speaker == "SPEAKER_01"


def test_parse_diarization_output_ignores_non_speaker_lines():
    raw = "some banner text\nSPEAKER audio.wav 1 0.00 1.00 hi NA SPEAKER_00 NA NA\n"
    segments = parse_diarization_output(raw)
    assert len(segments) == 1


def test_parse_diarization_output_sorted_by_start():
    raw = (
        "SPEAKER audio.wav 1 5.00 1.00 second NA SPEAKER_00 NA NA\n"
        "SPEAKER audio.wav 1 0.00 1.00 first NA SPEAKER_01 NA NA\n"
    )
    segments = parse_diarization_output(raw)
    assert [s.text for s in segments] == ["first", "second"]
