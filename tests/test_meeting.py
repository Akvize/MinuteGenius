import subprocess
from pathlib import Path
from types import SimpleNamespace

from minutegenius import meeting
from minutegenius.meeting import (
    _fmt_time,
    _hamming,
    _parse_speaker_line,
    _wav_is_valid,
    analyze_frame,
    parse_segments,
    validate_speakers_interactive,
)


def test_analyze_frame_tolerates_unescaped_control_char_in_string(monkeypatch, tmp_path):
    """Le modèle vision insère parfois un vrai retour à la ligne dans une valeur de string
    au lieu de l'échapper en \\n, ce qui casse json.loads en mode strict."""
    raw_content = (
        '{"shared_content": true, "clearly_visible": "Titre\ndetails", "partially_visible": null}'
    )
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw_content))]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )
    monkeypatch.setattr(meeting, "get_client_image", lambda: fake_client)
    monkeypatch.setenv("AI_MODEL_IMAGE", "fake-model")

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")  # JPEG minimal valide pour read_bytes()

    result = analyze_frame(image_path)

    assert result == {
        "shared_content": True,
        "clearly_visible": "Titre\ndetails",
        "partially_visible": None,
    }


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


def test_validate_speakers_interactive_accepts_free_text_name(monkeypatch, tmp_path):
    """Un participant absent de la liste proposée (jamais isolé comme locuteur principal
    par le modèle vision) doit pouvoir être saisi librement plutôt que rejeté."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    answers = iter(["Anthime Ferrieu"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    speakers_map = {"SPEAKER_00": "Ferdinand Rousseau"}
    all_segments = {"SPEAKER_00": [(0.0, 10.0)]}
    output_path = tmp_path / "speakers.json"

    updated = validate_speakers_interactive(speakers_map, all_segments, output_path)

    assert updated["SPEAKER_00"] == "Anthime Ferrieu"


def test_validate_speakers_interactive_lists_known_roster_as_menu_option(monkeypatch, tmp_path, capsys):
    """Les participants vus sur les vignettes mais jamais retenus comme locuteur principal
    (known_roster) doivent apparaître dans le menu numéroté, pas seulement en saisie libre."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "2")

    speakers_map = {"SPEAKER_00": "Ferdinand Rousseau"}
    all_segments = {"SPEAKER_00": [(0.0, 10.0)]}
    output_path = tmp_path / "speakers.json"

    updated = validate_speakers_interactive(
        speakers_map, all_segments, output_path, known_roster=["Anthime Ferrieu"]
    )

    assert "2. Anthime Ferrieu" in capsys.readouterr().out
    assert updated["SPEAKER_00"] == "Anthime Ferrieu"
