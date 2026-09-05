"""End-to-end subtitle acquisition: `subcheck autosub`.

Fetch subtitle candidates for a video from OpenSubtitles (two-tier cache,
credentials from .env / config), verify every candidate against the video's
own audio with the compare engine (one ASR pass over shared windows), pick
the candidate that fits best — trying its recommended constant/timebase fix
first — and optionally mux the corrected subtitles back into the video
(mkv/webm/mp4, lossless video/audio copy).

OpenSubtitles access follows the original standalone downloader's logic:
credentials from .env / env vars / config, plus the two-tier query and
file cache under .cache/opensubtitles.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .runner import CheckReport, CompareReport, run_compare, score_subtitle_file
from .srtio import shift_srt

# --------------------------------------------------------------------------- #
# video filename -> (title, season, episode) metadata

_EP_RE = re.compile(r"(?i)\bs(\d{1,2})[ ._-]?e(\d{1,2})\b")
_EP_ALT_RE = re.compile(r"(?i)\b(\d{1,2})x(\d{1,2})\b")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_JUNK_TOKENS = {
    "1080p", "720p", "480p", "576p", "2160p", "4k", "8k", "uhd", "hdr",
    "bluray", "brrip", "bdrip", "webrip", "webdl", "web-dl", "web", "hdtv",
    "hdtvrip", "dvdrip", "x264", "x265", "h264", "h265", "hevc", "avc", "av1",
    "vp9", "aac", "ac3", "eac3", "dts", "dtshd", "dts-hd", "truehd", "opus",
    "mp3", "flac", "10bit", "8bit", "repack", "proper", "remux", "extended",
    "unrated", "imax", "yify", "yts", "rarbg", "ettv", "eztv", "amzn",
    "nf", "dsnp", "hmax", "am", "mkv", "mp4", "eng", "multi",
}


def parse_video_name(path: str) -> Dict[str, object]:
    """Best-effort (title, season, episode) from the file name.

    ``Steven.Universe.S01E23.Monster.Buddies.1080p.BluRay...mkv`` ->
    ``{"title": "Steven Universe", "season": 1, "episode": 23,
    "episode_of": True}``. Without an episode marker the file is treated as a
    movie (title only).
    """
    stem = os.path.basename(str(path).split("?")[0])
    stem = os.path.splitext(stem)[0]
    m = _EP_RE.search(stem) or _EP_ALT_RE.search(stem)
    if m:
        season, episode = int(m.group(1)), int(m.group(2))
        return {
            "title": _clean_title(stem[: m.start()]),
            "season": season,
            "episode": episode,
            "episode_of": True,
        }
    return {
        "title": _clean_title(stem),
        "season": None,
        "episode": None,
        "episode_of": False,
    }


def _clean_title(raw: str) -> str:
    raw = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", raw)
    out = []
    for tok in re.split(r"[^0-9A-Za-z']+", raw):
        low = tok.lower()
        if not tok or low in _JUNK_TOKENS or _YEAR_RE.match(tok):
            continue
        out.append(tok)
    return " ".join(out) or ""


# --------------------------------------------------------------------------- #
# credentials (.env / env vars / ~/.config)

def _load_dotenv(path: Optional[str]) -> None:
    if not path or not os.path.isfile(path):
        return
    try:
        for line in open(path, encoding="utf-8").read().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except OSError:
        pass


def load_credentials() -> Dict[str, str]:
    """API key + login from env vars, the project .env, ./.env or
    ~/.config/opensubtitlescom/config.json."""
    project_dotenv = str(Path(__file__).resolve().parents[1] / ".env")
    _load_dotenv(project_dotenv)
    _load_dotenv(os.path.join(os.getcwd(), ".env"))

    def get(key: str) -> Optional[str]:
        return os.environ.get(key) or None

    api_key = get("OPENSUBTITLES_API_KEY")
    username = get("OPENSUBTITLES_USERNAME")
    password = get("OPENSUBTITLES_PASSWORD")
    if not (api_key and username and password):
        cfg = Path("~/.config/opensubtitlescom/config.json").expanduser()
        if cfg.exists():
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            api_key = api_key or data.get("api_key")
            username = username or data.get("username")
            password = password or data.get("password")
    if not (api_key and username and password):
        raise RuntimeError(
            "Missing OpenSubtitles credentials — configure .env (see "
            ".env.example) with OPENSUBTITLES_API_KEY / _USERNAME / _PASSWORD."
        )
    return {"api_key": api_key, "username": username, "password": password}


# --------------------------------------------------------------------------- #
# two-tier cache (.cache/opensubtitles: search_*.json + file_*.srt)

def cache_dir() -> Path:
    override = os.environ.get("SUBCHECK_CACHE_DIR")
    if override:
        d = Path(override)
    else:
        d = Path(__file__).resolve().parents[1] / ".cache" / "opensubtitles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _search_key(query: str, season: Optional[int], episode: Optional[int],
                language: str) -> Path:
    raw = f"{query.lower()}:{season}:{episode}:{language}".encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:16]
    return cache_dir() / f"search_{h}.json"


def get_cached_search(query: str, season: Optional[int], episode: Optional[int],
                      language: str) -> Optional[List[dict]]:
    p = _search_key(query, season, episode, language)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None


def save_cached_search(query: str, season: Optional[int], episode: Optional[int],
                       language: str, results: List[dict]) -> Path:
    p = _search_key(query, season, episode, language)
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def get_cached_file(file_id: int) -> Optional[bytes]:
    """Cached subtitle bytes for a file_id, or None. Invalid/empty cache
    entries are dropped so the next pull re-downloads (this is the
    "previous pull didn't work" escape hatch for cached subs)."""
    p = cache_dir() / f"file_{file_id}.srt"
    if p.exists():
        try:
            content = p.read_bytes()
        except OSError:
            return None
        if content and _decodes(content):
            return content
        try:
            p.unlink()
        except OSError:
            pass
    return None


def save_cached_file(file_id: int, content: bytes) -> Path:
    p = cache_dir() / f"file_{file_id}.srt"
    p.write_bytes(content)
    return p


def _decodes(content: bytes) -> bool:
    if b"\x00" in content:      # binary garbage, not subtitle text
        return False
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return bool(content.decode(enc).strip())
        except UnicodeDecodeError:
            continue
    return False


def describe_quota(q: Optional[dict]) -> str:
    """One-line human summary of an OpenSubtitles quota dict (from
    OpenSubtitlesClient.quota)."""
    if not q:
        return "quota: no API session (everything served from cache)"
    who = f"{q.get('username') or '?'} ({q.get('level') or 'User'})"
    bits = []
    remaining = q.get("remaining_downloads")
    used = q.get("downloads_count")
    allowed = q.get("allowed_downloads")
    reset = q.get("reset_time")
    if used is not None and allowed is not None:
        bits.append(f"{used} of {allowed} downloads used today")
    if remaining is not None:
        bits.append(f"{remaining} remaining")
    if reset:
        bits.append(f"quota resets at {reset}")
    return "quota: " + who + (" — " + ", ".join(bits) if bits else "")


# --------------------------------------------------------------------------- #
# OpenSubtitles API access (opensubtitlescom, lazily imported)

class OpenSubtitlesError(RuntimeError):
    pass


class OpenSubtitlesClient:
    """Thin wrapper around the opensubtitlescom package (API v1)."""

    def __init__(self, creds: Dict[str, str]) -> None:
        try:
            from opensubtitlescom import OpenSubtitles
        except ImportError as exc:  # pragma: no cover - env problem
            raise OpenSubtitlesError(
                "the 'opensubtitlescom' package is required for autosub "
                "(pip install opensubtitlescom)"
            ) from exc
        self._client = OpenSubtitles("subcheck v0.1", creds["api_key"])
        self._client.login(creds["username"], creds["password"])
        self._client_creds = creds

    @property
    def downloads_remaining(self) -> Optional[int]:
        try:
            return int(self._client.user_downloads_remaining)
        except Exception:
            return None

    def quota(self) -> dict:
        """Account status from the API (user_info) — mirrors the reference
        check_quota.py output: level, allowed/remaining downloads, downloads
        used today and when the quota resets."""
        info = self._client.user_info() or {}
        data = info.get("data", {}) or {}
        return {
            "username": self._client_creds.get("username", ""),
            "level": data.get("level", "User"),
            "allowed_downloads": data.get("allowed_downloads", 20),
            "remaining_downloads": data.get("remaining_downloads"),
            "downloads_count": data.get("downloads_count", 0),
            "reset_time": data.get("reset_time"),
        }

    def search(self, query: str, season: Optional[int], episode: Optional[int],
               language: str) -> List[dict]:
        kwargs = dict(query=query, languages=language)
        if season is not None:
            kwargs["season_number"] = season
            kwargs["episode_number"] = episode
        res = self._client.search(**kwargs)
        results = []
        for item in res.data:
            results.append(
                {
                    "id": item.id,
                    "file_id": item.file_id,
                    "file_name": item.file_name,
                    "release": item.release,
                    "download_count": item.download_count,
                    "ratings": item.ratings,
                    "hearing_impaired": item.hearing_impaired,
                    "fps": item.fps,
                }
            )
        return results

    def download(self, file_id: int) -> bytes:
        content = self._client.download(file_id)
        if not isinstance(content, bytes):
            content = bytes(content)
        return content


# --------------------------------------------------------------------------- #
# candidate discovery -> local srt files

@dataclass
class Candidate:
    """One fetched subtitle candidate on disk."""

    file_id: int
    release: str
    file_name: str
    path: str
    from_cache: bool = False
    error: str = ""            # download/validation failure, when any
    # verification results (filled by the compare stage)
    composite: Optional[float] = None          # as-is composite (0-100)
    text_accuracy: Optional[float] = None
    drift: str = ""                            # short drift classification
    best_mode: str = "as-is"                   # "as-is" | "shift" | "timebase"
    best_by: float = 0.0
    best_scale: float = 1.0
    composite_after: Optional[float] = None    # after the applied fix
    text_after: Optional[float] = None
    rank: Optional[int] = None
    drift_samples: List[float] = field(default_factory=list)
    # per-snippet drift of this candidate (median matched offset per window,
    # seconds; + = subtitles run late) — the values behind the histogram

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "release": self.release,
            "file_name": self.file_name,
            "cached": self.from_cache,
            "error": self.error,
            "composite": None if self.composite is None else round(self.composite, 2),
            "text_accuracy": (None if self.text_accuracy is None
                              else round(self.text_accuracy, 2)),
            "drift": self.drift,
            "best_mode": self.best_mode,
            "fix_by_s": round(self.best_by, 3),
            "fix_scale": round(self.best_scale, 6),
            "composite_after": (None if self.composite_after is None
                                else round(self.composite_after, 2)),
            "text_after": (None if self.text_after is None
                           else round(self.text_after, 2)),
            "rank": self.rank,
            "drift_samples": [round(d, 3) for d in self.drift_samples],
        }


def fetch_candidates(
    video: str,
    meta: Dict[str, object],
    *,
    language: str = "en",
    limit: int = 3,
    refresh: bool = False,
    workdir: str,
):
    """Search + download up to *limit* candidates into *workdir*.

    Both tiers of the cache are honoured (0 quota on hits); ``refresh``
    re-queries and re-downloads even when cache entries exist. A broken
    cached file is purged automatically (see get_cached_file). Download
    failures mark that candidate and the rest are still fetched.

    Returns ``(candidates, quota, downloads_new, cache_hits)`` where *quota*
    is the account status dict (None when no API session was needed) — it is
    printed on connect and refreshed after every real download.
    """
    query = str(meta["title"])
    season = meta.get("season")
    episode = meta.get("episode")
    cand_dir = os.path.join(workdir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    results: Optional[List[dict]] = None
    client: Optional[OpenSubtitlesClient] = None
    quota: Optional[dict] = None
    if not refresh:
        results = get_cached_search(query, season, episode, language)
    if results is None:
        client = OpenSubtitlesClient(load_credentials())
        quota = client.quota()
        print(describe_quota(quota))
        print(f"Searching OpenSubtitles for {query!r}"
              + (f" S{season:02d}E{episode:02d}" if season is not None else "")
              + f" [{language}]...")
        results = client.search(query, season, episode, language)
        save_cached_search(query, season, episode, language, results)
    if not results:
        return [], quota, 0, 0

    out: List[Candidate] = []
    downloads_new = 0
    cache_hits = 0
    for item in results[:limit]:
        file_id = int(item["file_id"])
        dest = os.path.join(cand_dir, f"cand_{len(out):02d}_file_{file_id}.srt")
        content = None
        cached = False
        if not refresh:
            content = get_cached_file(file_id)
            cached = content is not None
        if content is None:
            try:
                if client is None:
                    client = OpenSubtitlesClient(load_credentials())
                    quota = client.quota()
                    print(describe_quota(quota))
                print(f"Downloading subtitle file {file_id} "
                      f"('{item.get('release', '')}')...")
                content = client.download(file_id)
                save_cached_file(file_id, content)
                downloads_new += 1
                quota = client.quota()  # refresh: the download counts at once
                print(describe_quota(quota))
            except Exception as exc:  # noqa: BLE001 - quota/network errors
                out.append(Candidate(
                    file_id=file_id,
                    release=str(item.get("release", "")),
                    file_name=str(item.get("file_name", "")),
                    path=dest,
                    error=f"download failed: {exc}",
                ))
                continue
        else:
            cache_hits += 1
        try:
            Path(dest).write_bytes(content)
        except OSError as exc:
            out.append(Candidate(
                file_id=file_id,
                release=str(item.get("release", "")),
                file_name=str(item.get("file_name", "")),
                path=dest,
                error=f"write failed: {exc}",
            ))
            continue
        out.append(Candidate(
            file_id=file_id,
            release=str(item.get("release", "")),
            file_name=str(item.get("file_name", "")),
            path=dest,
            from_cache=cached,
        ))
    return out, quota, downloads_new, cache_hits


# --------------------------------------------------------------------------- #
# verify candidates against the video, pick the best (with its drift fix)

def evaluate_and_pick(
    compare: CompareReport,
    candidates: List[Candidate],
    *,
    variants_dir: str,
    expand_numbers: bool = True,
) -> Optional[Candidate]:
    """Fill each candidate with compare results; for fixable candidates
    (consistent drift / bad fps) apply + verify the suggested fix and adopt
    it when it improves the score. Returns the winning candidate."""
    by_path: Dict[str, object] = {}
    for e in compare.entries:
        by_path[e.subtitle] = e

    for cand in candidates:
        entry = by_path.get(cand.path)
        if entry is None:
            continue
        if entry.report is None:
            cand.error = entry.error or "could not be scored"
            continue
        rep: CheckReport = entry.report
        summ = rep.summary
        cand.composite = summ.get("composite_mean")
        cand.text_accuracy = summ.get("text_accuracy_mean")
        ds = rep.drift_summary
        cand.drift = ds.label if ds else ""
        cand.composite_after = cand.composite
        cand.text_after = cand.text_accuracy
        cand.drift_samples = [
            o.score.drift_seconds
            for o in rep.windows
            if o.score and not o.rolled and o.score.drift_seconds is not None
        ]
        if ds is None or cand.composite is None:
            continue
        if ds.fix_by is None and ds.fix_scale is None:
            continue
        # verify the diagnosed fix on a copy: keep it only if it helps
        label = "timebase" if ds.fix_scale is not None else "shift"
        variant = os.path.join(variants_dir, f"fixed_{cand.file_id}.srt")
        try:
            shift_srt(cand.path, variant, by=ds.fix_by or 0.0,
                      scale=ds.fix_scale or 1.0)
            _adopt_if_better(cand, variant, label, ds.fix_by or 0.0,
                             ds.fix_scale or 1.0, compare, expand_numbers)
        except Exception:  # noqa: BLE001 - keep as-is on any transform error
            pass

    # Even when the drift classifier is unsure (few windows, borderline
    # fps-vs-constant), scan constant shifts and adopt whichever verifiably
    # scores best — "which candidate would be good after a common shift?".
    # Verification happens on the same windows, so a wrong guess never wins.
    grid = [round(k * 0.5, 2) for k in range(-16, 17)]  # -8.0 ... +8.0 s
    for cand in candidates:
        if cand.error or cand.composite is None:
            continue
        if cand.composite_after is not None and cand.composite_after >= 95.0:
            continue  # already excellent as-is (or after its diagnosed fix)
        for by in grid:
            variant = os.path.join(variants_dir, f"grid_{cand.file_id}_{by:+.1f}.srt")
            try:
                shift_srt(cand.path, variant, by=by, scale=1.0)
                _adopt_if_better(cand, variant, "shift", by, 1.0,
                                 compare, expand_numbers)
            except Exception:  # noqa: BLE001 - skip broken candidates
                break

    ranked = sorted(candidates, key=lambda c: (
        0.0 if c.error else (c.composite_after if c.composite_after is not None
                             else -1.0)
    ), reverse=True)
    for i, cand in enumerate(ranked, start=1):
        if not cand.error and cand.composite_after is not None:
            cand.rank = i
    return next((c for c in ranked if c.rank == 1), None)


def _adopt_if_better(cand: Candidate, variant_srt: str, label: str, by: float,
                     scale: float, compare: CompareReport,
                     expand_numbers: bool) -> None:
    """Score *variant_srt* on the shared windows; adopt it when it beats the
    candidate's current best achievable score."""
    rep2 = score_subtitle_file(
        variant_srt,
        compare.records,
        media=compare.media,
        total_duration=compare.total_duration,
        snippet_duration=compare.snippet_duration,
        settings=compare.settings,
        expand_numbers=expand_numbers,
    )
    c2 = rep2.summary.get("composite_mean")
    if c2 is not None and (cand.composite_after is None or c2 > cand.composite_after):
        cand.composite_after = c2
        cand.text_after = rep2.summary.get("text_accuracy_mean")
        cand.best_mode = label
        cand.best_by = by
        cand.best_scale = scale


# --------------------------------------------------------------------------- #
# orchestration

@dataclass
class AutoSubResult:
    """Outcome of one `subcheck autosub` run."""

    video: str
    meta: dict
    language: str
    limit: int
    candidates: List[Candidate] = field(default_factory=list)
    windows_count: int = 0
    quota: Optional[dict] = None      # account status (None when no API session)
    downloads_new: int = 0            # downloads performed this run (quota used)
    cache_hits: int = 0               # candidates served from the file cache
    chosen: Optional[Candidate] = None
    final_srt: str = ""
    embed_status: str = ""       # "ok" | "skipped" | "failed"
    embed_out: str = ""
    notes: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        chosen_index = None
        if self.chosen is not None:
            for i, c in enumerate(self.candidates):
                if c is self.chosen:
                    chosen_index = i
        return {
            "tool": f"subcheck {__version__} (autosub)",
            "video": self.video,
            "metadata": self.meta,
            "language": self.language,
            "limit": self.limit,
            "windows_count": self.windows_count,
            "candidates": [c.to_dict() for c in self.candidates],
            "chosen_index": chosen_index,
            "quota": self.quota,
            "downloads_new": self.downloads_new,
            "cache_hits": self.cache_hits,
            "final_srt": self.final_srt,
            "embed_status": self.embed_status,
            "embed_out": self.embed_out,
            "notes": self.notes,
            "error": self.error,
        }


def run_autosub(
    video: str,
    *,
    language: str = "en",
    limit: int = 3,
    refresh: bool = False,
    title: Optional[str] = None,
    season: Optional[int] = None,
    episode: Optional[int] = None,
    snippets: int = 10,
    duration: float = 10.0,
    min_gap: float = 30.0,
    seed: Optional[int] = None,
    model: str = "large-v3",
    device: str = "auto",
    compute_type: Optional[str] = None,
    download_root: Optional[str] = None,
    beam_size: int = 5,
    embed: bool = True,
    embed_out: Optional[str] = None,
    final_srt: Optional[str] = None,
    workdir: Optional[str] = None,
    expand_numbers: bool = True,
) -> AutoSubResult:
    """Full E2E: fetch candidates -> compare against the video's audio ->
    pick the best (fixing it when the drift diagnosis suggests so) -> embed
    it into a copy of the video when possible."""
    meta = parse_video_name(video)
    if title:
        meta["title"] = title
    if season is not None:
        meta["season"], meta["episode_of"] = season, True
    if episode is not None:
        meta["episode"] = episode
    if not meta["title"]:
        raise ValueError(
            f"could not derive a title from {os.path.basename(video)!r} — "
            "pass --title"
        )

    cleanup = None
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        base = workdir
    else:
        base = tempfile.mkdtemp(prefix="subcheck-autosub-")
        cleanup = base

    result = AutoSubResult(video=video, meta=meta, language=language, limit=limit)
    try:
        candidates, quota, downloads_new, cache_hits = fetch_candidates(
            video, meta, language=language, limit=limit, refresh=refresh,
            workdir=base,
        )
        result.candidates = candidates
        result.quota = quota
        result.downloads_new = downloads_new
        result.cache_hits = cache_hits
        broken = [c for c in candidates if c.error]
        ok = [c for c in candidates if not c.error]
        if not candidates:
            result.error = "no subtitle candidates found"
            return result
        result.notes.append(
            f"fetched {len(ok)} candidate(s)"
            + (f"; {len(broken)} failed ({broken[0].error}" if broken else "")
            + (")" if broken else "")
        )
        result.notes.append(
            describe_quota(quota)
            + (f" — {downloads_new} downloaded, {cache_hits} from cache"
               if downloads_new or cache_hits else "")
        )
        if not ok:
            result.error = "every subtitle candidate failed to download"
            return result

        compare = run_compare(
            [c.path for c in ok],
            media=video,
            snippets=snippets,
            duration=duration,
            min_gap=min_gap,
            seed=seed,
            model=model,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            beam_size=beam_size,
            expand_numbers=expand_numbers,
        )
        result.windows_count = compare.window_count
        variants = os.path.join(base, "variants")
        os.makedirs(variants, exist_ok=True)
        chosen = evaluate_and_pick(compare, candidates, variants_dir=variants,
                                   expand_numbers=expand_numbers)
        result.chosen = chosen
        if chosen is None:
            result.error = "none of the candidates could be scored"
            return result

        # persist the winner (with its fix applied) as <video-stem>.autosub.srt
        source_path = chosen.path
        if chosen.best_mode != "as-is":
            fixed = os.path.join(variants, "best_fixed.srt")
            shift_srt(chosen.path, fixed, by=chosen.best_by,
                      scale=chosen.best_scale)
            source_path = fixed
        if final_srt is None:
            dest_dir = os.path.dirname(os.path.abspath(video))
            if not os.path.isdir(dest_dir):
                dest_dir = os.getcwd()
            stem = os.path.splitext(os.path.basename(video.split("?")[0]))[0]
            final_srt = os.path.join(dest_dir, stem + ".autosub.srt")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(final_srt)),
                        exist_ok=True)
            shutil.copyfile(source_path, final_srt)
        except OSError as exc:
            result.error = f"could not write final subtitles: {exc}"
            return result
        result.final_srt = final_srt

        if embed:
            result.embed_out, result.embed_status = _embed(
                video, final_srt, language, embed_out
            )
        else:
            result.embed_status = "skipped"
        return result
    finally:
        if cleanup and os.path.isdir(cleanup):
            shutil.rmtree(cleanup, ignore_errors=True)


def _embed(video: str, srt_path: str, language: str,
           embed_out: Optional[str]) -> "tuple[str, str]":
    from .embedded import remux_with_subtitles

    if video.startswith(("http://", "https://")):
        return "", "skipped (remote video — no local container to remux)"
    ext = os.path.splitext(video)[1].lower()
    if embed_out is None:
        if ext not in (".mkv", ".webm", ".mp4", ".mov", ".m4v"):
            return "", f"skipped (cannot embed text subs in a {ext or '?'} container)"
        stem = os.path.splitext(os.path.basename(video))[0]
        embed_out = os.path.join(os.path.dirname(os.path.abspath(video)),
                                 stem + ".autosub" + ext)
    if os.path.abspath(embed_out) == os.path.abspath(video):
        return "", "skipped (refusing to overwrite the source video — use --embed-out)"
    try:
        remux_with_subtitles(video, srt_path, embed_out,
                             language=language,
                             title=f"{language} (autosub)")
        return embed_out, "ok"
    except Exception as exc:  # noqa: BLE001 - report, don't abort
        return "", f"failed ({exc})"
