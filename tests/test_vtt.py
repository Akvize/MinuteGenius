from minutegenius.vtt import build_markdown, clean_text, parse_timestamp, parse_vtt


def test_parse_timestamp_with_hours():
    assert parse_timestamp("01:02:03.400") == 3723.4


def test_parse_timestamp_without_hours():
    assert parse_timestamp("02:03.400") == 123.4


def test_parse_timestamp_comma_decimal():
    assert parse_timestamp("00:01:00,000") == 60.0


def test_clean_text_strips_tags_and_entities():
    assert clean_text("<v Alice>Hello&nbsp;world</v>") == "Hello world"


def test_parse_vtt_basic_cues_with_speakers():
    content = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "<v Alice>Hello there</v>\n\n"
        "00:00:02.500 --> 00:00:04.000\n"
        "<v Bob>Hi Alice</v>\n"
    )
    cues = parse_vtt(content)
    assert len(cues) == 2
    assert cues[0].speaker == "Alice"
    assert cues[0].text == "Hello there"
    assert cues[1].speaker == "Bob"


def test_parse_vtt_merges_duplicate_karaoke_cues():
    content = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "<v Alice>Hello</v>\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "<v Alice>Hello</v>\n"
    )
    cues = parse_vtt(content)
    assert len(cues) == 1
    assert cues[0].end == 2.0


def test_parse_vtt_skips_header_blocks_without_timestamp():
    content = "WEBVTT\nKind: captions\n\n00:00:00.000 --> 00:00:01.000\nHello\n"
    cues = parse_vtt(content)
    assert len(cues) == 1
    assert cues[0].text == "Hello"


def test_build_markdown_empty_cues():
    assert build_markdown([]) == ""


def test_build_markdown_groups_by_speaker():
    content = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "<v Alice>Hello there</v>\n\n"
        "00:00:02.500 --> 00:00:04.000\n"
        "<v Alice>How are you</v>\n\n"
        "00:00:04.500 --> 00:00:06.000\n"
        "<v Bob>Doing fine</v>\n"
    )
    cues = parse_vtt(content)
    md = build_markdown(cues)
    assert md.count("**[") == 2  # one header for the two consecutive Alice cues, one for Bob
    assert "Alice" in md
    assert "Bob" in md
