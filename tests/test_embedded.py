"""Embedded-subtitle stream discovery/extraction and video-shift tests.

Needs working ffmpeg/ffprobe (SUBCHECK_FFMPEG_BIN / SUBCHECK_FFPROBE_BIN or
PATH) — skipped otherwise. Test media is generated on the fly with ffmpeg.
"""

import os
import shutil
import subprocess

import pytest

from subcheck import cli
from subcheck.embedded import (
    SubtitleStream,
    choose_subtitle_stream,
    extract_subtitle_stream,
    probe_subtitle_streams,
    remux_with_subtitles,
    select_subtitle_stream,
)
from subcheck.srtio import read_srt

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") and not os.environ.get("SUBCHECK_FFMPEG_BIN"),
    reason="ffmpeg not available",
)

SRT_CONTENT = """\
1
00:00:00,500 --> 00:00:02,000
Are you watching closely?

2
00:00:02,500 --> 00:00:04,000
Every magic trick.
"""


def _ffmpeg():
    return os.environ.get("SUBCHECK_FFMPEG_BIN", "ffmpeg")


def _make_video_with_subs(tmp_path, name="clip.mkv", subs=SRT_CONTENT):
    srt = tmp_path / "in.srt"
    srt.write_text(subs, encoding="utf-8")
    out = tmp_path / name
    subprocess.run(
        [
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=6",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-i", str(srt),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
            "-c:v", "mpeg4", "-q:v", "5",
            "-c:a", "pcm_s16le",
            "-c:s", "srt",
            "-t", "6",
            "-metadata:s:s:0", "language=eng",
            "-disposition:s:0", "default",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return str(out), str(srt)


def test_probe_finds_embedded_stream(tmp_path):
    video, _ = _make_video_with_subs(tmp_path)
    streams = probe_subtitle_streams(video)
    assert len(streams) == 1
    st = streams[0]
    assert st.codec in ("subrip", "srt")
    assert st.language == "eng"
    assert st.default is True
    assert st.text_capable is True
    assert "eng" in st.desc


def test_select_by_index_language_default(tmp_path):
    video, _ = _make_video_with_subs(tmp_path)
    streams = probe_subtitle_streams(video)
    st = select_subtitle_stream(streams, language="eng")
    assert st.index == streams[0].index
    with pytest.raises(ValueError):
        select_subtitle_stream(streams, language="deu")
    with pytest.raises(ValueError):
        select_subtitle_stream(streams, index=9)


def test_select_rejects_bitmap_only(tmp_path):
    # pure python: no need for a real bitmap track
    bitmap = [SubtitleStream(0, "hdmv_pgs_subtitle", language="eng", text_capable=False)]
    with pytest.raises(ValueError, match="bitmap"):
        select_subtitle_stream(bitmap)
    with pytest.raises(ValueError, match="no embedded"):
        select_subtitle_stream([])


def test_extract_matches_original_timing(tmp_path):
    video, _ = _make_video_with_subs(tmp_path)
    streams = probe_subtitle_streams(video)
    out = tmp_path / "extracted.srt"
    extract_subtitle_stream(video, streams[0], str(out))
    cues = read_srt(str(out))
    assert len(cues) == 2
    assert cues[0].start == pytest.approx(0.5, abs=0.05)
    assert cues[0].end == pytest.approx(2.0, abs=0.05)
    assert "closely" in cues[0].text


def test_second_subtitle_track_uses_type_ordinal(tmp_path):
    """ffmpeg -map 0:s:N counts per type — a stream with global index 3 is
    subtitle #1, and must still be selectable by its global index."""
    a = tmp_path / "a.srt"
    a.write_text(SRT_CONTENT, encoding="utf-8")
    b = tmp_path / "b.srt"
    b.write_text("1\n00:00:10,000 --> 00:00:11,500\nSecond language.\n", encoding="utf-8")
    out = tmp_path / "two.mkv"
    subprocess.run(
        [
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            # finite lavfi inputs — infinite ones would encode forever
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=16",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=16",
            "-i", str(a), "-i", str(b),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0", "-map", "3:0",
            "-c:v", "mpeg4", "-q:v", "5", "-c:a", "pcm_s16le", "-c:s", "srt",
            "-metadata:s:s:0", "language=eng",
            "-metadata:s:s:1", "language=fre",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    streams = probe_subtitle_streams(str(out))
    assert [s.index for s in streams] == [2, 3]
    assert [s.ordinal for s in streams] == [0, 1]
    assert [s.language for s in streams] == ["eng", "fre"]
    # global index 3 -> the fre track, extracted via ordinal 1
    chosen = select_subtitle_stream(streams, index=3)
    assert chosen.language == "fre"
    extracted = tmp_path / "fre.srt"
    extract_subtitle_stream(str(out), chosen, str(extracted))
    cues = read_srt(str(extracted))
    assert cues[0].start == pytest.approx(10.0, abs=0.05)
    assert "Second" in cues[0].text


def test_shift_accepts_video_with_embedded_subs(tmp_path):
    video, _ = _make_video_with_subs(tmp_path)
    out = tmp_path / "fixed.srt"
    rc = cli.main(["shift", video, "--by", "1.0", "--out", str(out)])
    assert rc == 0
    cues = read_srt(str(out))
    assert cues[0].start == pytest.approx(1.5, abs=0.05)
    assert cues[1].start == pytest.approx(3.5, abs=0.05)


def test_shift_video_remux(tmp_path):
    video, _ = _make_video_with_subs(tmp_path)
    remuxed = tmp_path / "fixed.mkv"
    rc = cli.main(
        ["shift", video, "--by", "0.5", "--remux", str(remuxed), "--out",
         str(tmp_path / "fixed.srt")]
    )
    assert rc == 0
    assert remuxed.exists()
    streams = probe_subtitle_streams(str(remuxed))
    assert any(s.codec in ("subrip", "srt") for s in streams)


def test_streams_cli_lists_tracks(tmp_path, capsys):
    video, _ = _make_video_with_subs(tmp_path)
    rc = cli.main(["streams", video])
    out = capsys.readouterr().out
    assert rc == 0
    assert "subtitle stream" in out and "eng" in out


def _two_stream_streams():
    return [
        SubtitleStream(2, "subrip", language="eng", default=True, ordinal=0),
        SubtitleStream(3, "subrip", language="fre", ordinal=1),
    ]


def test_choose_prompts_and_honours_answer(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "3")
    assert choose_subtitle_stream(_two_stream_streams(), interactive=True).index == 3
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert choose_subtitle_stream(_two_stream_streams(), interactive=True).index == 2


def test_choose_never_prompts_with_explicit_index(monkeypatch):
    def fail(_):
        raise AssertionError("should not prompt when --sub-stream is given")

    monkeypatch.setattr("builtins.input", fail)
    chosen = choose_subtitle_stream(_two_stream_streams(), index=3, interactive=True)
    assert chosen.index == 3 and chosen.language == "fre"


def test_choose_batch_mode_uses_default_rule(monkeypatch):
    def fail(_):
        raise AssertionError("no prompt in batch mode")

    monkeypatch.setattr("builtins.input", fail)
    chosen = choose_subtitle_stream(_two_stream_streams(), interactive=False)
    assert chosen.index == 2  # default-marked stream


def test_shift_video_prompts_for_stream_when_multiple(tmp_path, monkeypatch):
    """CLI-level: a video with two text tracks asks which one to fix."""
    a = tmp_path / "a.srt"
    a.write_text(SRT_CONTENT, encoding="utf-8")
    b = tmp_path / "b.srt"
    b.write_text("1\n00:00:10,000 --> 00:00:11,500\nSecond language.\n", encoding="utf-8")
    video = tmp_path / "two.mkv"
    subprocess.run(
        [
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=16",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=16",
            "-i", str(a), "-i", str(b),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0", "-map", "3:0",
            "-c:v", "mpeg4", "-q:v", "5", "-c:a", "pcm_s16le", "-c:s", "srt",
            "-metadata:s:s:0", "language=eng",
            "-metadata:s:s:1", "language=fre",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    answers = iter(["3"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    # fake a terminal so the CLI asks (pytest's stdin is not a tty)
    fake_tty = type("FakeTTY", (), {"isatty": lambda self: True})()
    monkeypatch.setattr("subcheck.embedded.sys.stdin", fake_tty)
    fixed = tmp_path / "fre-fixed.srt"
    rc = cli.main(["shift", str(video), "--by", "1.0", "--out", str(fixed)])
    assert rc == 0
    cues = read_srt(str(fixed))
    assert len(cues) == 1  # only the fre track's line
    assert "Second" in cues[0].text
    assert cues[0].start == pytest.approx(11.0, abs=0.05)


def test_shift_rejects_file_without_subtitles(tmp_path, capsys):
    rc = cli.main(["shift", str(tmp_path / "plain.txt"), "--by", "1.0"])
    assert rc == 1
    assert "error" in capsys.readouterr().err
