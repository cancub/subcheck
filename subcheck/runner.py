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

# text accuracy / IoU thresholds used for the constant-offset-drift flag
DRIFT_TEXT_MIN = 90.0   # text must be this good ...
DRIFT_IOU_MAX = 20.0    # ... while overlap is worse than this (%)


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
    subtitle: str
    snippets: int = 3
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


def run_check(opts: Options) -> CheckReport:
    media, subtitle = opts.media, opts.subtitle
    if not os.path.exists(subtitle):
        raise FileNotFoundError(f"subtitle file not found: {subtitle}")

    report = CheckReport(
        media=media,
        subtitle=subtitle,
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
    total = report.total_duration
    cues: List[Cue] = read_srt(subtitle)
    if not cues:
        raise ValueError(
            f"subtitle file {subtitle!r} contains no usable cues — cannot compare."
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
    snippets: int = 3,
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
    """Per-window drift suspects + global flags/verdict wording."""
    scored = [o for o in report.windows if o.score and not o.rolled]

    rolled = sum(1 for o in report.windows if o.rolled)
    if rolled:
        report.flags.append(
            f"{rolled} window(s) were re-rolled (subtitles present but no speech "
            "recognised — music interludes or ASR misses)."
        )

    silent = [o for o in scored if o.score.status == "silent-match"]
    if silent:
        report.flags.append(
            f"{len(silent)} window(s) had no speech and no subtitles: counted as "
            "perfect 100 % matches (silent agreement)."
        )

    missing = [o for o in scored if o.score.status == "speech-no-subs"]
    if missing:
        report.flags.append(
            f"{len(missing)} window(s) contain speech that the subtitle file does "
            "not caption at all — subtitle text is missing there."
        )

    no_speech = [o for o in scored if o.score.status == "subs-no-speech"]
    if no_speech:
        report.flags.append(
            f"{len(no_speech)} window(s) kept where the subtitle file shows text "
            "but no speech was recognised."
        )

    drift_suspects = [
        o
        for o in scored
        if o.score.text_accuracy is not None
        and o.score.text_accuracy > DRIFT_TEXT_MIN
        and (o.score.iou or 0.0) * 100.0 < DRIFT_IOU_MAX
    ]
    if drift_suspects:
        where = ", ".join(f"{o.snippet.start:.0f}s" for o in drift_suspects)
        report.flags.append(
            f"SYNC DRIFT SUSPECTED in {len(drift_suspects)} window(s) ({where}s): "
            "text accuracy is high but temporal overlap is low — the subtitle "
            "timing looks shifted by a constant offset."
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
            f"CONSISTENT DRIFT: subtitles run "
            f"{'late' if drift.mean > 0 else 'early'} by {abs(drift.mean):.2f}s "
            f"on average (std {drift.std:.2f}s across {drift.matched_lines} "
            f"matched line(s)) — a constant time shift fixes this:"
            f"\n        {report.fix_command}"
        )
    elif drift.category == CATEGORY_FPS:
        report.fix_command = (
            f'subcheck shift "{report.subtitle}"'
            f" --scale {drift.fix_scale:.6f} --by {drift.fix_by:.3f}"
        )
        report.flags.append(
            f"BAD FPS: subtitle timeline speed does not match the movie "
            f"(drift slope {drift.slope_s_per_s:+.6f} s per s of movie, "
            f"timeline off by ≈{abs(drift.rate_delta_pct):.2f}%) — rescale the "
            "subtitle times instead of shifting them:"
            f"\n        {report.fix_command}"
        )
    elif drift.category == CATEGORY_BAD and drift.matched_lines == 0:
        report.flags.append(
            "Drift could not be measured: no subtitle line could be textually "
            "matched to a speech segment (unrelated text? different language?)."
        )

    summary = report.summary
    if summary.get("composite_mean") is None:
        report.flags.append("No window could be scored — nothing to conclude.")
        return

    mean = summary["composite_mean"]
    n = summary.get("windows_scored", 0)
    if mean >= 95:
        verdict = (
            f"Sampled {n} window(s): the subtitle file looks ACCURATE in the "
            "sampled regions."
        )
    elif mean >= 88:
        verdict = (
            f"Sampled {n} window(s): subtitles look mostly accurate, with minor "
            "text/timing deviations."
        )
    elif mean >= 75:
        verdict = (
            f"Sampled {n} window(s): subtitles deviate noticeably — text and/or "
            "timing mismatches were found."
        )
    else:
        verdict = (
            f"Sampled {n} window(s): subtitles do NOT match the audio well — "
            "verify the file (wrong language, wrong edition, broken sync?)."
        )
    report.flags.append(verdict)
