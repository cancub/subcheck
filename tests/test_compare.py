import json

import pytest

from subcheck.runner import CompareEntry, run_compare

GOOD_SRT = """\
1
00:01:41,000 --> 00:01:45,000
Hello there, my friend.

2
00:33:21,000 --> 00:33:25,000
Open the door now.
"""

# same dialogue, but each cue shifted +2 s (and window 2 text differs)
SHIFTED_SRT = """\
1
00:01:43,000 --> 00:01:47,000
Hello there, my friend.

2
00:33:23,000 --> 00:33:27,000
Open the door now.
"""

GIBBERISH_SRT = """\
1
00:01:41,000 --> 00:01:45,000
blah blah blah blah

2
00:33:21,000 --> 00:33:25,000
blah blah blah blah
"""


def _windows_json(path, starts=(100.0, 2000.0)):
    """Two transcribed windows whose speech matches GOOD_SRT's cues exactly."""
    data = {
        "tool": "subcheck test",
        "media": "movie.mp4",
        "total_duration_s": 3600.0,
        "snippet_duration_s": 10.0,
        "settings": {"snippets": 2, "model": "test"},
        "windows": [
            {
                "start": starts[0],
                "end": starts[0] + 10.0,
                "status": "scored",
                "wav": "",
                "asr_segments": [
                    {"start": starts[0] + 1.0, "end": starts[0] + 5.0,
                     "text": "Hello there, my friend."}
                ],
            },
            {
                "start": starts[1],
                "end": starts[1] + 10.0,
                "status": "scored",
                "wav": "",
                "asr_segments": [
                    {"start": starts[1] + 1.0, "end": starts[1] + 5.0,
                     "text": "Open the door now."}
                ],
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_compare_ranks_perfect_subtitles_first(tmp_path):
    wj = _windows_json(tmp_path / "windows.json")
    good = _write(tmp_path, "good.srt", GOOD_SRT)
    shifted = _write(tmp_path, "shifted.srt", SHIFTED_SRT)
    rep = run_compare([good, shifted], windows_json=wj)

    assert rep.mode == "json"
    assert rep.window_count == 2
    assert [e.rank for e in rep.entries] == [1, 2]
    assert rep.entries[0].subtitle == good
    # perfect file: 100 everywhere
    s = rep.entries[0].report.summary
    assert s["text_accuracy_mean"] == 100.0 and s["composite_mean"] == 100.0
    # shifted file must rank last and score far worse
    s2 = rep.entries[1].report.summary
    assert s2["composite_mean"] < 50.0


def test_compare_error_file_keeps_ranking_others(tmp_path):
    wj = _windows_json(tmp_path / "windows.json")
    good = _write(tmp_path, "good.srt", GOOD_SRT)
    rep = run_compare([good, str(tmp_path / "missing.srt"), good], windows_json=wj)
    assert rep.error is None
    ok = [e for e in rep.entries if e.report]
    broken = [e for e in rep.entries if e.error]
    assert len(ok) == 2 and len(broken) == 1
    assert all(e.rank is not None for e in ok)
    assert broken[0].rank is None
    # stable order of equal scores follows input order
    assert ok[0].rank == 1 and ok[1].rank == 2


def test_compare_gibberish_loses_to_good(tmp_path):
    wj = _windows_json(tmp_path / "windows.json")
    good = _write(tmp_path, "good.srt", GOOD_SRT)
    gib = _write(tmp_path, "gib.srt", GIBBERISH_SRT)
    rep = run_compare([gib, good], windows_json=wj)
    assert rep.entries[0].subtitle == good  # re-sorted by score
    assert rep.entries[1].subtitle == gib
    assert rep.entries[0].report.summary["composite_mean"] == 100.0
    # text is garbage (0 %) but timing is perfect (IoU 100 %)
    # -> composite = 0.75*0 + 0.25*100 = 25, still far below the good file
    assert rep.entries[1].report.summary["composite_mean"] == 25.0


def test_compare_requires_inputs():
    with pytest.raises(ValueError):
        run_compare([], windows_json="x.json")
    with pytest.raises(ValueError):
        run_compare(["a.srt"], media=None, windows_json=None)


def test_compare_json_matches_rescore_numbers(tmp_path):
    wj = _windows_json(tmp_path / "windows.json")
    good = _write(tmp_path, "good.srt", GOOD_SRT)
    rep = run_compare([good], windows_json=wj)
    # drift of a perfect file: zero -> good match
    ds = rep.entries[0].report.drift_summary
    assert ds.category == "good"
    assert rep.entries[0].report.drift_estimate == 0.0
    assert rep.to_dict()["mode"] == "json"
    assert len(rep.to_dict()["shared_windows"]) == 2
