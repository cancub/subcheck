"""Autosub tests — fully offline: cache logic, filename metadata and the
verify-and-pick machinery are exercised with synthetic data; no OpenSubtitles
API calls are made."""

import json

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
