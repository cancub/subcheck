"""Autosub tests — fully offline: cache logic, filename metadata and the
verify-and-pick machinery are exercised with synthetic data; no OpenSubtitles
API calls are made."""

import json
import os
from pathlib import Path

import pytest

from subcheck import autosub
from subcheck.autosub import (
    Candidate,
    describe_quota,
    fetch_candidates,
    get_cached_file,
    get_cached_search,
    load_credentials,
    parse_video_name,
    run_autosub,
    save_cached_file,
    save_cached_search,
)
from subcheck.runner import run_compare
from subcheck.tools import ToolError

SRT_A = """\
1
00:01:41,000 --> 00:01:45,000
Hello there, my friend.

2
00:33:21,000 --> 00:33:25,000
Open the door now.
"""

SRT_B_SHIFTED = """\
1
00:01:43,000 --> 00:01:47,000
Hello there, my friend.

2
00:33:23,000 --> 00:33:27,000
Open the door now.
"""


def _windows_json(path):
    data = {
        "media": "movie.mp4",
        "total_duration_s": 3600.0,
        "snippet_duration_s": 10.0,
        "settings": {"snippets": 2, "model": "test"},
        "windows": [
            {"start": 100.0, "end": 110.0, "status": "scored", "wav": "",
             "asr_segments": [
                 {"start": 101.0, "end": 105.0, "text": "Hello there, my friend."}]},
            {"start": 2000.0, "end": 2010.0, "status": "scored", "wav": "",
             "asr_segments": [
                 {"start": 2001.0, "end": 2005.0, "text": "Open the door now."}]},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ name parsing
def test_parse_episode_name():
    m = parse_video_name(
        "Steven.Universe.S01E23.Monster.Buddies.1080p.BluRay.AV1.Opus.2.0-RAV1NE.mkv")
    assert m["title"] == "Steven Universe"
    assert m["season"] == 1 and m["episode"] == 23 and m["episode_of"] is True


def test_parse_movie_name():
    m = parse_video_name("The.Prestige.2006.1080p.BluRay.x264-[YTS.AM].mp4")
    assert m["title"] == "The Prestige"
    assert m["season"] is None


def test_parse_alt_episode_marker():
    m = parse_video_name("Futurama - 3x09 (720p).mkv")
    assert m["season"] == 3 and m["episode"] == 9
    assert "Futurama" in m["title"]


# ------------------------------------------------------------------ credentials
def _only_tmp_dotenv(monkeypatch, tmp_path):
    """Neutralise the real project .env (and ~/.config) so tests only see the
    .env file placed in tmp_path."""
    real_load = autosub._load_dotenv
    tmp_env = str(tmp_path / ".env")
    monkeypatch.setattr(
        autosub, "_load_dotenv",
        lambda path: real_load(path) if path == tmp_env else None,
    )
    for k in ("OPENSUBTITLES_API_KEY", "OPENSUBTITLES_USERNAME",
              "OPENSUBTITLES_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    monkeypatch.chdir(tmp_path)


def test_load_credentials_from_dotenv(monkeypatch, tmp_path):
    _only_tmp_dotenv(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text(
        'OPENSUBTITLES_API_KEY="abc"\n'
        "# comment\n"
        "OPENSUBTITLES_USERNAME=user\n"
        "OPENSUBTITLES_PASSWORD=pw\n",
        encoding="utf-8",
    )
    creds = load_credentials()
    assert creds == {"api_key": "abc", "username": "user", "password": "pw"}


def test_load_credentials_missing_raises(monkeypatch, tmp_path):
    _only_tmp_dotenv(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="credentials"):
        load_credentials()


# ------------------------------------------------------------------ cache tiers
def test_cache_search_and_file_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBCHECK_CACHE_DIR", str(tmp_path / "cache"))
    results = [{"id": "1", "file_id": 42, "release": "Rel", "file_name": "f.srt"}]
    p = save_cached_search("Steven Universe", 1, 23, "en", results)
    assert p.exists()
    assert get_cached_search("Steven Universe", 1, 23, "en") == results
    # a different query must not hit the cache
    assert get_cached_search("Other Show", 1, 23, "en") is None

    save_cached_file(42, b"1\n00:00:01,000 --> 00:00:02,000\nHi.\n")
    assert get_cached_file(42) is not None


def test_broken_cached_file_is_purged(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBCHECK_CACHE_DIR", str(tmp_path / "cache"))
    d = tmp_path / "cache"
    save_cached_file(7, b"\x00\xff\xfe garbage not text")
    assert (d / "file_7.srt").exists()
    assert get_cached_file(7) is None          # invalid -> dropped
    assert not (d / "file_7.srt").exists()     # so the next pull re-downloads


def test_fetch_candidates_uses_cache_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("SUBCHECK_CACHE_DIR", str(tmp_path / "cache"))
    content = b"1\n00:00:00,500 --> 00:00:02,000\nAre you watching closely?\n"
    save_cached_search("Steven Universe", 1, 23, "en", [
        {"id": "1", "file_id": 11, "release": "BluRay", "file_name": "x.srt"},
        {"id": "2", "file_id": 22, "release": "HDTV", "file_name": "y.srt"},
    ])
    save_cached_file(11, content)
    meta = {"title": "Steven Universe", "season": 1, "episode": 23,
            "episode_of": True}
    workdir = tmp_path / "work"
    workdir.mkdir()
    cands, quota, downloads_new, cache_hits = fetch_candidates(
        "Steven.Universe.S01E23.mkv", meta, limit=1, workdir=str(workdir))
    assert len(cands) == 1            # limit honoured, no API access needed
    assert cands[0].from_cache is True
    assert cands[0].file_id == 11
    assert open(cands[0].path, "rb").read() == content
    # all-cache run: no API session, no quota used
    assert quota is None
    assert downloads_new == 0 and cache_hits == 1


def test_quota_description_lines():
    assert "no API session" in describe_quota(None)
    q = {
        "username": "bob", "level": "VIP",
        "allowed_downloads": 20, "remaining_downloads": 17,
        "downloads_count": 3, "reset_time": "2026-09-06T00:00:00Z",
    }
    s = describe_quota(q)
    assert "bob (VIP)" in s and "3 of 20" in s
    assert "17 remaining" in s and "resets at" in s


# ------------------------------------------------- verify, fix and pick (offline)
def test_evaluate_pick_adopts_verified_shift(tmp_path):
    """A candidate whose drift is diagnosed as constant should be picked in
    its *shifted* form only after the shift verifiably improves the score."""
    wj = _windows_json(tmp_path / "windows.json")
    a = tmp_path / "perfect.srt"
    a.write_text(SRT_A, encoding="utf-8")
    b = tmp_path / "shifted2s.srt"
    b.write_text(SRT_B_SHIFTED, encoding="utf-8")
    compare = run_compare([str(a), str(b)], windows_json=wj)

    candidates = [
        Candidate(file_id=1, release="perfect", file_name="perfect.srt",
                  path=str(a)),
        Candidate(file_id=2, release="shifted", file_name="shifted2s.srt",
                  path=str(b)),
    ]
    variants = tmp_path / "variants"
    variants.mkdir()
    chosen = autosub.evaluate_and_pick(compare, candidates,
                                       variants_dir=str(variants))
    assert chosen is not None
    cand_a, cand_b = candidates
    # perfect file: as-is, no fix needed
    assert cand_a.best_mode == "as-is"
    assert cand_a.composite == 100.0
    # per-snippet drift samples feed the histogram (one value per window)
    assert cand_a.drift_samples == [0.0, 0.0]
    assert cand_b.drift_samples == pytest.approx([2.0, 2.0])
    # shifted file: diagnosis suggests a shift; verify it before adopting
    assert cand_b.best_mode == "shift"
    assert cand_b.best_by == pytest.approx(-2.0, abs=0.2)
    assert cand_b.composite_after == 100.0
    assert cand_b.composite_after > cand_b.composite
    # ranking picks the perfect file on the tie (input order, stable sort)
    assert chosen is cand_a and cand_a.rank == 1 and cand_b.rank == 2


def test_autosub_offline_e2e_without_video_errors_cleanly(tmp_path, monkeypatch):
    """Fetching is cache-backed; with a missing video the run must fail
    loudly at the probe stage (before any ASR)."""
    monkeypatch.setenv("SUBCHECK_CACHE_DIR", str(tmp_path / "cache"))
    save_cached_search("Demo", 1, 2, "en",
                       [{"id": "1", "file_id": 5, "release": "R",
                         "file_name": "f.srt"}])
    save_cached_file(5, b"1\n00:00:01,000 --> 00:00:02,000\nHi.\n")
    with pytest.raises(ToolError):
        run_autosub(
            str(tmp_path / "missing.mkv"),
            title="Demo", season=1, episode=2, embed=False,
            workdir=str(tmp_path / "work"),
        )


def test_autosub_embed_in_place(tmp_path, monkeypatch):
    from subcheck.autosub import _embed

    recorded = []

    def fake_remux(video, srt_path, out_path, language="", title=""):
        recorded.append((video, srt_path, out_path, language, title))
        return out_path

    monkeypatch.setattr("subcheck.embedded.remux_with_subtitles", fake_remux)

    video = str(tmp_path / "movie.mkv")
    srt = str(tmp_path / "subs.srt")

    # Without overwrite, default is <video>.autosub.mkv
    out, status = _embed(video, srt, "en", embed_out=None, overwrite=False)
    assert status == "ok"
    assert out.endswith("movie.autosub.mkv")

    # With overwrite, default is movie.mkv itself
    out, status = _embed(video, srt, "en", embed_out=None, overwrite=True)
    assert status == "ok"
    assert out == video
    assert recorded[-1][0] == video and recorded[-1][2] == video

    # Explicit embed_out == video without overwrite is rejected
    out, status = _embed(video, srt, "en", embed_out=video, overwrite=False)
    assert "refusing" in status


def test_extract_embedded_candidate(tmp_path, monkeypatch):
    from subcheck.autosub import extract_embedded_candidate
    from subcheck.embedded import SubtitleStream

    video = tmp_path / "test.mkv"
    video.write_bytes(b"dummy")

    fake_stream = SubtitleStream(index=2, codec="subrip", language="eng")

    monkeypatch.setattr("subcheck.embedded.probe_subtitle_streams", lambda v: [fake_stream])
    monkeypatch.setattr(
        "subcheck.embedded.extract_subtitle_stream",
        lambda v, s, out: (Path(out).write_text(SRT_A, encoding="utf-8"), out)[1],
    )

    cand = extract_embedded_candidate(str(video), "en", str(tmp_path))
    assert cand is not None
    assert cand.file_id == 0
    assert "embedded stream #2 (subrip, eng)" in cand.release
    assert cand.from_cache is True
    assert os.path.isfile(cand.path)

    # When no streams match language
    cand_none = extract_embedded_candidate(str(video), "de", str(tmp_path))
    assert cand_none is None


def test_autosub_adopts_embedded_directly_without_api_call(tmp_path, monkeypatch):
    video = tmp_path / "Show.S01E02.mkv"
    video.write_bytes(b"dummy")

    emb_srt = tmp_path / "embedded.srt"
    emb_srt.write_text(SRT_A, encoding="utf-8")
    emb_cand = Candidate(
        file_id=0,
        release="embedded stream #0 (subrip, eng)",
        file_name="embedded_0.srt",
        path=str(emb_srt),
        from_cache=True,
    )

    monkeypatch.setattr("subcheck.autosub.extract_embedded_candidate", lambda *args, **kwargs: emb_cand)

    def fail_fetch(*args, **kwargs):
        pytest.fail("fetch_candidates should not be called when embedded candidate matches!")

    monkeypatch.setattr("subcheck.autosub.fetch_candidates", fail_fetch)

    # Mock compare & evaluate
    from subcheck.runner import CompareReport

    def fake_compare(*args, **kwargs):
        return CompareReport(mode="live", records=[None] * 5, media=str(video))

    def fake_eval(compare, candidates, **kwargs):
        for c in candidates:
            c.composite = 95.0
            c.composite_after = 95.0
            c.rank = 1
        return candidates[0]

    monkeypatch.setattr("subcheck.autosub.run_compare", fake_compare)
    monkeypatch.setattr("subcheck.autosub.evaluate_and_pick", fake_eval)
    monkeypatch.setattr("subcheck.autosub._embed", lambda *args, **kwargs: ("out.mkv", "ok"))

    res = run_autosub(str(video), language="en", workdir=str(tmp_path / "work"))
    assert res.error == ""
    assert res.chosen is emb_cand
    assert res.chosen.file_id == 0
    assert res.downloads_new == 0
    assert any("using embedded subtitle stream" in note for note in res.notes)
    assert any("skipped OpenSubtitles search" in note for note in res.notes)


def test_autosub_falls_back_to_opensubtitles_when_embedded_scores_low(tmp_path, monkeypatch):
    video = tmp_path / "Show.S01E02.mkv"
    video.write_bytes(b"dummy")

    emb_srt = tmp_path / "embedded.srt"
    emb_srt.write_text(SRT_A, encoding="utf-8")
    emb_cand = Candidate(
        file_id=0,
        release="embedded stream #0",
        file_name="embedded_0.srt",
        path=str(emb_srt),
        from_cache=True,
    )

    os_srt = tmp_path / "os.srt"
    os_srt.write_text(SRT_A, encoding="utf-8")
    os_cand = Candidate(
        file_id=999,
        release="OpenSubtitles Release",
        file_name="os.srt",
        path=str(os_srt),
        from_cache=False,
    )

    monkeypatch.setattr("subcheck.autosub.extract_embedded_candidate", lambda *args, **kwargs: emb_cand)
    monkeypatch.setattr(
        "subcheck.autosub.fetch_candidates",
        lambda *args, **kwargs: ([os_cand], {"remaining_downloads": 10}, 1, 0),
    )

    from subcheck.runner import CompareReport

    def fake_compare(*args, **kwargs):
        return CompareReport(mode="live", records=[None] * 5, media=str(video))

    eval_count = 0

    def fake_eval(compare, candidates, **kwargs):
        nonlocal eval_count
        eval_count += 1
        if eval_count == 1:
            # First eval on embedded only: low score 50%
            candidates[0].composite = 50.0
            candidates[0].composite_after = 50.0
            candidates[0].rank = 1
            return candidates[0]
        # Second eval on both: os_cand wins
        for c in candidates:
            if c.file_id == 0:
                c.composite = 50.0
                c.composite_after = 50.0
                c.rank = 2
            else:
                c.composite = 98.0
                c.composite_after = 98.0
                c.rank = 1
        return os_cand

    monkeypatch.setattr("subcheck.autosub.run_compare", fake_compare)
    monkeypatch.setattr("subcheck.autosub.score_subtitle_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("subcheck.autosub.evaluate_and_pick", fake_eval)
    monkeypatch.setattr("subcheck.autosub._embed", lambda *args, **kwargs: ("out.mkv", "ok"))

    res = run_autosub(str(video), language="en", workdir=str(tmp_path / "work"))
    assert res.error == ""
    assert res.chosen is os_cand
    assert res.chosen.file_id == 999
    assert len(res.candidates) == 2
    assert any("falling back to OpenSubtitles" in note for note in res.notes)


def test_autosub_skip_embedded_and_refresh(tmp_path, monkeypatch):
    video = tmp_path / "Show.S01E02.mkv"
    video.write_bytes(b"dummy")

    extracted_called = False

    def fake_extract(*args, **kwargs):
        nonlocal extracted_called
        extracted_called = True
        return None

    monkeypatch.setattr("subcheck.autosub.extract_embedded_candidate", fake_extract)
    monkeypatch.setattr(
        "subcheck.autosub.fetch_candidates",
        lambda *args, **kwargs: ([], None, 0, 0),
    )

    # When skip_embedded is True, extract_embedded_candidate is not even called
    res = run_autosub(str(video), skip_embedded=True, workdir=str(tmp_path / "work"))
    assert not extracted_called
    assert res.error == "no subtitle candidates found"


def test_autosub_refresh_searches_opensubtitles_even_if_embedded_good(tmp_path, monkeypatch):
    video = tmp_path / "Show.S01E02.mkv"
    video.write_bytes(b"dummy")

    emb_srt = tmp_path / "embedded.srt"
    emb_srt.write_text(SRT_A, encoding="utf-8")
    emb_cand = Candidate(
        file_id=0,
        release="embedded stream #0",
        file_name="embedded_0.srt",
        path=str(emb_srt),
        from_cache=True,
    )

    fetch_called = False

    def fake_fetch(*args, **kwargs):
        nonlocal fetch_called
        fetch_called = True
        return [], None, 0, 0

    monkeypatch.setattr("subcheck.autosub.extract_embedded_candidate", lambda *args, **kwargs: emb_cand)
    monkeypatch.setattr("subcheck.autosub.fetch_candidates", fake_fetch)

    from subcheck.runner import CompareReport

    monkeypatch.setattr(
        "subcheck.autosub.run_compare",
        lambda *args, **kwargs: CompareReport(mode="live", records=[None] * 5, media=str(video)),
    )

    def fake_eval(compare, candidates, **kwargs):
        candidates[0].composite = 95.0
        candidates[0].composite_after = 95.0
        candidates[0].rank = 1
        return candidates[0]

    monkeypatch.setattr("subcheck.autosub.evaluate_and_pick", fake_eval)
    monkeypatch.setattr("subcheck.autosub._embed", lambda *args, **kwargs: ("out.mkv", "ok"))

    # With refresh=True, fetch_candidates MUST be called even if embedded candidate scored 95%
    res = run_autosub(str(video), refresh=True, workdir=str(tmp_path / "work"))
    assert fetch_called is True
    assert res.chosen is emb_cand  # emb_cand wins since fetch returned no other candidates
    assert any("--refresh requested" in note for note in res.notes)



