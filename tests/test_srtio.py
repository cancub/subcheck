import textwrap

from subcheck.srtio import Cue, cues_overlapping, read_srt

SAMPLE = textwrap.dedent(
    """\
    1
    00:00:01,000 --> 00:00:03,500
    <i>Hello</i> there,
    my friend.

    2
    00:00:05,000 --> 00:00:07,000
    {\\an8}How are you?

    3
    00:00:06,000 --> 00:00:06,000
    zero-length, must be skipped

    4
    00:01:00,000 --> 00:01:02,500
    Later scene [music]
    """
)


def test_read_srt_basics(tmp_path):
    f = tmp_path / "t.srt"
    f.write_text(SAMPLE, encoding="utf-8")
    cues = read_srt(str(f))
    assert len(cues) == 3  # zero-length cue skipped
    assert cues[0].start == 1.0 and cues[0].end == 3.5
    # tags stripped, lines merged into one text block
    assert "Hello there, my friend." in cues[0].text
    assert "<i>" not in cues[0].text and "{\\an8}" not in cues[1].text
    # sorted by start
    assert [c.start for c in cues] == [1.0, 5.0, 60.0]


def test_overlap_windowing():
    cs = [
        Cue(1, 0.0, 5.0, "ends at window start"),
        Cue(2, 4.0, 8.0, "straddles start"),
        Cue(3, 8.0, 12.0, "inside"),
        Cue(4, 9.0, 14.0, "straddles end"),
        Cue(5, 15.0, 20.0, "after"),
    ]
    got = cues_overlapping(cs, 5.0, 15.0)
    assert [c.index for c in got] == [2, 3, 4]
    # a cue whose interval ends exactly at the window start is excluded
    assert [c.index for c in cues_overlapping(cs, 8.0, 15.0)] == [3, 4]
    # cue 5 starts exactly at the window end -> no overlap
    assert cues_overlapping(cs, 14.5, 15.0) == []
