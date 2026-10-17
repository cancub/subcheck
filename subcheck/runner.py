"""Pipeline orchestration: probe -> sample -> slice -> transcribe -> offset ->
window the subtitles -> score; handles silent/music re-rolls and drift flags."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

from . import __version__
from .probe import probe_duration
from .sampling import SamplingError, Snippet, SnippetSampler
from .scoring import (
    CATEGORY_BAD,
    CATEGORY_CONSISTENT,
    CATEGORY_FPS,
    DriftSummary,
    SnippetScore,
    score_window,
    summarize,
    summarize_drift,
)
from .slicing import extract_audio
from .srtio import Cue, cues_overlapping, read_srt
from .transcribe import Segment, Transcriber

@dataclass
class WindowOutcome:
    """Outcome of checking one sampled window."""

    snippet: Snippet
    wav_path: str
    status: str
    rolled: bool = False                 # True when the window was re-rolled away
    score: Optional[SnippetScore] = None

    def to_dict(self) -> dict:
        s = self.score
        return {
            "start": round(self.snippet.start, 3),
            "end": round(self.snippet.end, 3),
            "status": self.status,
            "rolled": self.rolled,
            "wav": self.wav_path,
            **(s.to_dict() if s else {}),
        }


@dataclass
class CheckReport:
    """Full report of one `subcheck check` run."""

    media: str
    subtitle: str
    total_duration: float
    snippet_duration: float
    windows: List[WindowOutcome] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    drift_estimate: Optional[float] = None          # pooled mean of matched offsets
    drift_summary: Optional[DriftSummary] = None    # mean/std + classification
    fix_command: Optional[str] = None               # e.g. when drift is consistent
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        drift: dict = {}
        if self.drift_summary is not None:
            drift = self.drift_summary.to_dict()
            if self.fix_command:
                drift["fix_command"] = self.fix_command
        return {
            "tool": f"subcheck {__version__}",
            "media": self.media,
            "subtitle": self.subtitle,
            "total_duration_s": round(self.total_duration, 3),
            "snippet_duration_s": self.snippet_duration,
            "settings": self.settings,
            "windows": [w.to_dict() for w in self.windows],
            "summary": self.summary,
            "drift_estimate_s": None if self.drift_estimate is None else round(self.drift_estimate, 3),
            "drift_summary": drift,
            "flags": self.flags,
        }


@dataclass
class Options:
    """Everything that tunes a check run (mirrors the CLI)."""

    media: str
    subtitle: Optional[str] = None  # explicit SRT; None = auto-resolve
    snippets: int = 10
    duration: float = 10.0
    min_gap: float = 30.0
    seed: Optional[int] = None
    model: str = "large-v3"
    device: str = "auto"
    compute_type: Optional[str] = None
    download_root: Optional[str] = None
    beam_size: int = 5
    language: Optional[str] = None
    max_rerolls: int = 3
    workdir: Optional[str] = None
    expand_numbers: bool = True
    # embedded-subtitle source selection (used when subtitle is None or
    # use_embedded is set)
    use_embedded: bool = False
    sub_stream: Optional[int] = None
    sub_language: Optional[str] = None


def _find_sidecar_srt(media: str) -> Optional[str]:
    """An .srt with the same base name next to the media (external check)."""
    import glob

    base = os.path.splitext(media)[0]
    hits = sorted(glob.glob(base + "*.srt")) + sorted(glob.glob(base + ".*.srt"))
    return hits[0] if hits else None


def _resolve_subtitle_source(
    opts: Options, workdir: str
) -> "tuple[str, dict]":
    """Decide which subtitle file to check and how it was obtained.

    Returns (srt_path, source_info). Resolution order:
      1. opts.subtitle (explicit external file)
      2. sidecar .srt next to the media (unless --embedded forces step 3)
      3. an embedded text subtitle stream of the media, extracted to *workdir*
    """
    if opts.subtitle and opts.use_embedded:
        raise ValueError(
            "cannot combine an explicit subtitle file with --embedded"
        )
    if opts.subtitle:
        if not os.path.exists(opts.subtitle):
            raise FileNotFoundError(
                f"subtitle file not found: {opts.subtitle}"
            )
        return opts.subtitle, {"source": "external", "subtitle_file": opts.subtitle}
    if not opts.use_embedded:
        sidecar = _find_sidecar_srt(opts.media)
        if sidecar:
            return sidecar, {"source": "sidecar", "subtitle_file": sidecar}
    # embedded: needs a probeable media file
    from .embedded import (
        choose_subtitle_stream,
        extract_subtitle_stream,
        probe_subtitle_streams,
    )

    streams = probe_subtitle_streams(opts.media)
    stream = choose_subtitle_stream(
        streams, index=opts.sub_stream, language=opts.sub_language
    )
    os.makedirs(workdir, exist_ok=True)
    out_path = os.path.join(workdir, f"embedded_{stream.index}.srt")
    extract_subtitle_stream(opts.media, stream, out_path)
    text_streams = sum(1 for s in streams if s.text_capable)
    return out_path, {
        "source": "embedded",
        "subtitle_file": out_path,
        "embedded_stream": stream.index,
        "embedded_codec": stream.codec,
        "embedded_language": stream.language,
        "embedded_streams_total": len(streams),
        "embedded_text_streams": text_streams,
        "embedded_label": f"{os.path.basename(opts.media)} — embedded "
        f"{stream.desc}",
    }


def run_check(opts: Options) -> CheckReport:
    media = opts.media

    cleanup = None
    if opts.workdir:
        workdir = opts.workdir
        os.makedirs(workdir, exist_ok=True)
    else:
        workdir = tempfile.mkdtemp(prefix="subcheck-")
        cleanup = workdir

    subtitle_path, source_info = _resolve_subtitle_source(opts, workdir)

    report = CheckReport(
        media=media,
        subtitle=media if source_info.get("source") == "embedded" else subtitle_path,
        total_duration=probe_duration(media),
        snippet_duration=opts.duration,
        settings={
            "snippets": opts.snippets,
            "min_gap": opts.min_gap,
            "seed": opts.seed,
            "model": opts.model,
            "device": opts.device,
            "beam_size": opts.beam_size,
            "language": opts.language,
            "max_rerolls": opts.max_rerolls,
        },
    )
    report.settings.update(source_info)
    total = report.total_duration
    cues: List[Cue] = read_srt(subtitle_path)
    if not cues:
        raise ValueError(
            f"subtitle file {subtitle_path!r} contains no usable cues — "
            "cannot compare."
        )
    report.settings["cues_total"] = len(cues)
    report.settings["cues_speech"] = sum(
        1 for c in cues if c.end - c.start >= 0.1
    )

    sampler = SnippetSampler(
        total=total,
        duration=opts.duration,
        count=opts.snippets,
        min_gap=opts.min_gap,
        seed=opts.seed,
    )
    try:
        initial = sampler.sample_all()
    except SamplingError as exc:
        raise ValueError(str(exc)) from None

    transcriber = Transcriber(
        model=opts.model,
        device=opts.device,
        compute_type=opts.compute_type,
        download_root=opts.download_root,
        beam_size=opts.beam_size,
        language=opts.language,
    )
    report.settings["model_repo"] = transcriber.repo
    report.settings["device"] = transcriber.device
    report.settings["compute_type"] = transcriber.compute_type

    try:
        queue: List[Snippet] = list(initial)
        outcomes: List[WindowOutcome] = []
        rerolls_left = opts.max_rerolls
        while queue:
            snippet = queue.pop(0)
            wav = os.path.join(workdir, f"snippet_{snippet.index:02d}.wav")
            extract_audio(media, snippet, wav)

            # stage 4/5: transcribe (local times) then offset to the global timeline
            local_segments = transcriber.transcribe_wav(wav)
            global_segments = [
                Segment(s.start + snippet.start, s.end + snippet.start, s.text)
                for s in local_segments
            ]
            window_cues = cues_overlapping(cues, snippet.start, snippet.end)
            score = score_window(
                global_segments,
                window_cues,
                snippet.start,
                snippet.end,
                expand_numbers=opts.expand_numbers,
                cue_pool=cues,
            )

            outcome = WindowOutcome(
                snippet=snippet,
                wav_path=wav,
                status=score.status,
                score=score,
            )

            # Edge case: subtitles exist but nothing was recognised — usually a
            # music/silent interlude or an ASR miss. Re-roll a fresh window
            # instead of penalising the file (bounded by max_rerolls).
            if score.status == "subs-no-speech" and rerolls_left > 0:
                extra = sampler.draw_extra()
                if extra is not None:
                    rerolls_left -= 1
                    outcome.rolled = True
                    outcome.score = None
                    outcome.status = "rolled"
                    outcomes.append(outcome)
                    queue.append(extra)
                    continue
            outcomes.append(outcome)

        report.windows = sorted(
            (o for o in outcomes), key=lambda o: o.snippet.start
        )
        finalize(report)
        return report
    finally:
        if cleanup and os.path.isdir(cleanup):
            shutil.rmtree(cleanup, ignore_errors=True)


def finalize(report: CheckReport) -> None:
    """Aggregate per-window scores and run the flag/drift/verdict logic."""
    report.summary = summarize([o.score for o in report.windows if o.score])
    _flag_and_drift(report)


# ----------------------------------------------------------------- window records
# A "window record" is a shared sampling unit: (snippet, global segments, wav).
# Records are what every subtitle candidate gets scored against — identical
# for all files, which is what makes a comparison fair.

WindowRecord = tuple  # (Snippet, List[Segment], str)


def load_window_records(windows_json: str):
    """Load window records + run metadata from a report JSON."""
    import json

    with open(windows_json, encoding="utf-8") as fh:
        data = json.load(fh)
    records = []
    for w in data.get("windows", []):
        wstart = float(w["start"])
        wend = float(w["end"])
        snip = Snippet(index=len(records), start=wstart, duration=wend - wstart)
        segments = [
            Segment(float(s["start"]), float(s["end"]), s["text"])
            for s in w.get("asr_segments", [])
        ]
        records.append((snip, segments, w.get("wav", "")))
    return records, data


def score_subtitle_file(
    subtitle: str,
    records,
    *,
    media: str = "",
    total_duration: float = 0.0,
    snippet_duration: float = 10.0,
    settings: Optional[dict] = None,
    expand_numbers: bool = True,
) -> CheckReport:
    """Score one SRT file against the given window records."""
    cues: List[Cue] = read_srt(subtitle)
    if not cues:
        raise ValueError(f"subtitle file {subtitle!r} contains no usable cues")

    outcomes: List[WindowOutcome] = []
    for snip, segments, wav in records:
        window_cues = cues_overlapping(cues, snip.start, snip.end)
        score = score_window(
            segments,
            window_cues,
            snip.start,
            snip.end,
            expand_numbers=expand_numbers,
            cue_pool=cues,
        )
        outcomes.append(
            WindowOutcome(snippet=snip, wav_path=wav, status=score.status, score=score)
        )
    report = CheckReport(
        media=media,
        subtitle=subtitle,
        total_duration=total_duration,
        snippet_duration=snippet_duration,
        windows=outcomes,
        settings=dict(settings or {}) | {"cues_total": len(cues), "mode": "rescore"},
    )
    finalize(report)
    return report


def rescore_windows(
    windows_json: str,
    subtitle: str,
    expand_numbers: bool = True,
) -> CheckReport:
    """Re-score previously transcribed windows against a different SRT.

    Loads a report JSON produced by ``subcheck check --json`` and re-runs only
    the comparison stage (no probing/slicing/ASR) — useful to compare several
    subtitle candidates for the same video.
    """
    records, data = load_window_records(windows_json)
    return score_subtitle_file(
        subtitle,
        records,
        media=data.get("media", ""),
        total_duration=float(data.get("total_duration_s", 0.0) or 0.0),
        snippet_duration=float(data.get("snippet_duration_s", 10.0)),
        settings=dict(data.get("settings", {})),
        expand_numbers=expand_numbers,
    )


# --------------------------------------------------------------- multi-file compare
@dataclass
class CompareEntry:
    """One subtitle file inside a comparison."""

    subtitle: str
    report: Optional[CheckReport] = None
    error: Optional[str] = None
    rank: Optional[int] = None  # 1 = best composite score; None = failed

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "subtitle": self.subtitle,
            "error": self.error,
            "report": None if self.report is None else self.report.to_dict(),
        }


@dataclass
class CompareReport:
    """Result of comparing several subtitle files against identical windows."""

    mode: str                        # "live" (fresh ASR pass) or "json" (reuse)
    records: list                    # shared WindowRecords
    entries: List[CompareEntry] = field(default_factory=list)
    media: str = ""
    total_duration: float = 0.0
    snippet_duration: float = 10.0
    settings: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def window_count(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict:
        return {
            "tool": f"subcheck {__version__}",
            "mode": self.mode,
            "media": self.media,
            "total_duration_s": round(self.total_duration, 3),
            "snippet_duration_s": self.snippet_duration,
            "settings": self.settings,
            "shared_windows": [
                {
                    "start": round(snip.start, 3),
                    "end": round(snip.end, 3),
                    "wav": wav,
                    "asr_segments": [
                        {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                        for s in segs
                    ],
                }
                for snip, segs, wav in self.records
            ],
            "files": [e.to_dict() for e in self.entries],
            "error": self.error,
        }


def _capture_live_windows(media: str, opts: Options):
    """Sample + transcribe windows once (no per-file re-rolls, so every
    candidate is judged on the very same windows). Returns (records, total,
    workdir_cleanup)."""
    total = probe_duration(media)
    sampler = SnippetSampler(
        total=total,
        duration=opts.duration,
        count=opts.snippets,
        min_gap=opts.min_gap,
        seed=opts.seed,
    )
    try:
        snippets = sampler.sample_all()
    except SamplingError as exc:
        raise ValueError(str(exc)) from None

    cleanup = None
    if opts.workdir:
        workdir = opts.workdir
        os.makedirs(workdir, exist_ok=True)
    else:
        workdir = tempfile.mkdtemp(prefix="subcheck-")
        cleanup = workdir

    transcriber = Transcriber(
        model=opts.model,
        device=opts.device,
        compute_type=opts.compute_type,
        download_root=opts.download_root,
        beam_size=opts.beam_size,
        language=opts.language,
    )
    records = []
    model_info = {}
    try:
        for snippet in snippets:
            wav = os.path.join(workdir, f"snippet_{snippet.index:02d}.wav")
            extract_audio(media, snippet, wav)
            local_segments = transcriber.transcribe_wav(wav)
            global_segments = [
                Segment(s.start + snippet.start, s.end + snippet.start, s.text)
                for s in local_segments
            ]
            records.append((snippet, global_segments, wav))
        model_info = {
            "model_repo": transcriber.repo,
            "device": transcriber.device,
            "compute_type": transcriber.compute_type,
        }
    finally:
        # keep the wavs when the caller asked for a workdir, else clean up
        if cleanup and os.path.isdir(cleanup):
            shutil.rmtree(cleanup, ignore_errors=True)
    return records, total, model_info


def run_compare(
    subtitles: List[str],
    *,
    media: Optional[str] = None,
    windows_json: Optional[str] = None,
    duration: float = 10.0,
    min_gap: float = 30.0,
    snippets: int = 10,
    seed: Optional[int] = None,
    model: str = "large-v3",
    device: str = "auto",
    compute_type: Optional[str] = None,
    download_root: Optional[str] = None,
    beam_size: int = 5,
    language: Optional[str] = None,
    workdir: Optional[str] = None,
    expand_numbers: bool = True,
) -> CompareReport:
    """Score *every* subtitle file against the *same* sampled windows.

    mode "live": one ffprobe/ffmpeg/ASR pass over the media, then each SRT is
    scored against those windows; mode "json": windows come from a previous
    ``subcheck check --json`` report (no media/ASR needed).
    """
    if not subtitles:
        raise ValueError("no subtitle files given to compare")
    if windows_json:
        records, data = load_window_records(windows_json)
        report = CompareReport(
            mode="json",
            records=records,
            media=data.get("media", ""),
            total_duration=float(data.get("total_duration_s", 0.0) or 0.0),
            snippet_duration=float(data.get("snippet_duration_s", duration)),
            settings=dict(data.get("settings", {})) | {"mode": "json"},
        )
    else:
        if not media:
            raise ValueError("compare needs either a video (media) or --from windows.json")
        opts = Options(
            media=media,
            subtitle=subtitles[0],  # only used as a placeholder
            snippets=snippets,
            duration=duration,
            min_gap=min_gap,
            seed=seed,
            model=model,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
            beam_size=beam_size,
            language=language,
            workdir=workdir,
        )
        records, total, model_info = _capture_live_windows(media, opts)
        report = CompareReport(
            mode="live",
            records=records,
            media=media,
            total_duration=total,
            snippet_duration=duration,
            settings={
                "snippets": snippets,
                "min_gap": min_gap,
                "seed": seed,
                "model": model,
                "device": device,
                "beam_size": beam_size,
                "language": language,
                "mode": "live",
                **model_info,
            },
        )

    for subtitle in subtitles:
        try:
            entry_report = score_subtitle_file(
                subtitle,
                records,
                media=report.media,
                total_duration=report.total_duration,
                snippet_duration=report.snippet_duration,
                settings=report.settings,
                expand_numbers=expand_numbers,
            )
            report.entries.append(CompareEntry(subtitle=subtitle, report=entry_report))
        except Exception as exc:  # noqa: BLE001 - keep comparing the others
            report.entries.append(CompareEntry(subtitle=subtitle, error=str(exc)))

    # rank by composite score (best first); failed files rank last (no rank)
    def score_key(entry: CompareEntry):
        if entry.report is None:
            return -1.0
        mean = entry.report.summary.get("composite_mean")
        return -1.0 if mean is None else float(mean)

    report.entries.sort(key=score_key, reverse=True)
    for i, entry in enumerate(report.entries, start=1):
        if entry.report is not None and entry.report.summary.get("composite_mean") is not None:
            entry.rank = i
    if all(e.error for e in report.entries):
        report.error = "none of the subtitle files could be scored"
    return report


def _flag_and_drift(report: CheckReport) -> None:
    """Per-window status notes + drift statistics flags (factual wording:
    numbers and applicable fix commands, no judgement)."""
    scored = [o for o in report.windows if o.score and not o.rolled]

    rolled = sum(1 for o in report.windows if o.rolled)
    if rolled:
        report.flags.append(
            f"{rolled} window(s) were re-rolled (subtitles present, no speech "
            "recognised)."
        )

    silent = [o for o in scored if o.score.status == "silent-match"]
    if silent:
        report.flags.append(
            f"{len(silent)} window(s) had no speech and no subtitles — counted "
            "as perfect 100 % matches."
        )

    missing = [o for o in scored if o.score.status == "speech-no-subs"]
    if missing:
        report.flags.append(
            f"{len(missing)} window(s) contain speech but no subtitle text "
            "within the window."
        )

    no_speech = [o for o in scored if o.score.status == "subs-no-speech"]
    if no_speech:
        report.flags.append(
            f"{len(no_speech)} window(s) have subtitle text but no speech was "
            "recognised."
        )

    # drift statistics over the sampled clips: each window contributes its
    # median matched (segment, cue) offset; we report the mean/std over those,
    # fit drift vs movie-time (to catch wrong-fps timelines) and classify
    samples = [
        (o.snippet.start, o.score.drift_seconds)
        for o in scored
        if o.score.drift_seconds is not None
    ]
    medians = [m for _, m in samples]
    xs = [x for x, _ in samples]
    matched_lines_total = sum(len(o.score.drift_deltas) for o in scored)
    drift = summarize_drift(
        medians,
        xs=xs,
        matched_windows=len(medians),
        matched_lines=matched_lines_total,
    )
    report.drift_summary = drift
    report.drift_estimate = drift.mean
    if drift.category == CATEGORY_CONSISTENT:
        report.fix_command = (
            f'subcheck shift "{report.subtitle}" --by {drift.fix_by:.3f}'
        )
        report.flags.append(
            f"drift: mean {drift.mean:+.2f}s, std {drift.std:.2f}s "
            f"({drift.matched_lines} matched line(s)) — constant offset; "
            f"apply a constant shift:"
            f"\n        {report.fix_command}"
        )
    elif drift.category == CATEGORY_FPS:
        report.fix_command = (
            f'subcheck shift "{report.subtitle}"'
            f" --scale {drift.fix_scale:.6f} --by {drift.fix_by:.3f}"
        )
        report.flags.append(
            f"drift slope {drift.slope_s_per_s:+.6f} s/s of movie "
            f"(timeline speed off by ≈{abs(drift.rate_delta_pct):.2f}%) — "
            f"apply a timebase correction:"
            f"\n        {report.fix_command}"
        )
    elif drift.category == CATEGORY_BAD and drift.matched_lines == 0:
        report.flags.append(
            "no matched text between the speech and the subtitles — "
            "drift not measurable."
        )

    if report.summary.get("composite_mean") is None:
        report.flags.append("no window could be scored.")
