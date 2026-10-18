"""Command-line interface for subcheck."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional

from . import __version__
from .runner import CheckReport, Options, run_check
from .tools import ToolError


# --------------------------------------------------------------------------- #
# per-snippet drift histograms (shared by check/rescore/compare/autosub)

def drift_samples_of(report: CheckReport) -> List[float]:
    """The per-snippet drift values of a check report (median matched offset
    of each non-rolled window, seconds; + = subtitles run late)."""
    return [
        o.score.drift_seconds
        for o in report.windows
        if o.score and not o.rolled and o.score.drift_seconds is not None
    ]


def drift_histogram_lines(samples: List[float], max_bins: int = 16) -> List[str]:
    """ASCII histogram lines of drift samples, buckets anchored on round
    second multiples so a zero bucket always exists when values cross 0."""
    if not samples:
        return ["(no per-snippet drift samples)"]
    lo, hi = min(samples), max(samples)
    spread = hi - lo
    width = 60.0
    for w in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        if spread <= w * max_bins:
            width = w
            break
    start = math.floor(lo / width) * width
    lines = []
    t = start
    while t < hi + 1e-9:
        if abs(t) < 1e-9:      # keep the zero boundary crisp (+0.00, not -0.00)
            t = 0.0
        count = sum(1 for s in samples if t <= s < t + width)
        lines.append(f"  {t:+7.2f} .. {t + width:+7.2f}s  {'#' * count}")
        t += width
    return lines


def print_drift_histograms(
    label: str,
    samples: List[float],
    total_windows: Optional[int] = None,
    indent: str = "",
) -> None:
    """Console block for one set of per-snippet drift samples."""
    if not samples:
        return
    head = f"{label} — per-snippet drift (seconds, + = subtitles run late), n={len(samples)}"
    if total_windows is not None and total_windows > len(samples):
        head += (f" ({total_windows - len(samples)} window(s) without a "
                 "measurable drift: silent/music or no text match)")
    print(f"{indent}{head}")
    for line in drift_histogram_lines(samples):
        print(f"{indent}  {line.strip()}")



def _fmt_clock(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _fmt_pct(value: Optional[float]) -> str:
    return "    -  " if value is None else f"{value:6.1f}%"


class _RawDefaultsHelpFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Keep multi-line descriptions readable and still show option defaults."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck",
        description=(
            "Check whether a subtitle (SRT) file is accurate for a video:\n"
            "sample random short windows, transcribe them with faster-whisper\n"
            "and compare text (WER) and timing (IoU) against the subtitle file.\n"
            "\n"
            "Other commands:\n"
            "  subcheck autosub movie.mkv\n"
            "      fetch, verify, fix + embed subtitles (see --help)\n"
            "  subcheck quota\n"
            "      show the OpenSubtitles daily download quota\n"
            "  subcheck streams movie.mkv\n"
            "      list the embedded subtitle streams of a video\n"
            "  subcheck compare movie.mp4 a.srt b.srt c.srt\n"
            "      compare subtitle files on the same windows (see --help)\n"
            "  subcheck rescore windows.json other.srt\n"
            "      re-score saved windows against another SRT\n"
            "  subcheck shift movie.srt --by -0.989\n"
            "      fix 'consistent drift' (constant time shift)\n"
            "  subcheck shift movie.srt --scale 0.95904\n"
            "      fix 'bad fps' (timebase/fps correction)\n"
            "  subcheck shift movie.srt --scale F --by B\n"
            "      combine both (see 'subcheck shift --help')\n"
            "  subcheck download-model [--model MODEL]\n"
            "      pre-download Whisper weights ahead of time (see --help)"
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"subcheck {__version__}")
    p.add_argument("media", help="video file path or URL (ffmpeg/ffprobe readable)")
    p.add_argument(
        "subtitle",
        nargs="?",
        help="SRT file to check; omit to auto-detect — a sidecar .srt next to "
             "the media, or an embedded subtitle stream of the video",
    )
    _add_run_options(p, rerolls=True)
    p.add_argument("--embedded", action="store_true",
                   help="use an embedded subtitle stream of the video even when "
                        "a sidecar .srt exists")
    p.add_argument("--sub-stream", type=int, default=None, metavar="INDEX",
                   help="which embedded subtitle stream to use (see "
                        "'subcheck streams <video>')")
    p.add_argument("--sub-language", default=None, metavar="CODE",
                   help="pick the embedded stream by language code (eng, fre, ...)")
    return p
def _add_run_options(p: argparse.ArgumentParser, rerolls: bool = True) -> None:
    """The knobs shared by `check` and `compare`."""
    p.add_argument("--snippets", type=int, default=10,
                   help="number of random windows to check")
    p.add_argument("--duration", type=float, default=10.0,
                   help="window duration in seconds")
    p.add_argument("--min-gap", type=float, default=30.0,
                   help="minimum spacing between windows (seconds)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducible window selection")
    p.add_argument("--model", default="large-v3",
                   help="whisper model (tiny/base/small/medium/large-v3 or an HF repo id)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="compute device; auto = CUDA when available, else CPU")
    p.add_argument("--compute-type", default=None,
                   help="ctranslate2 compute type (default: float16 on CUDA, int8 on CPU)")
    p.add_argument("--model-dir", default=None,
                   help="directory for model weights (default: HF cache / SUBCHECK_MODEL_DIR)")
    p.add_argument("--beam-size", type=int, default=5, help="whisper beam size")
    p.add_argument("--language", default=None,
                   help="spoken language hint (ISO code), auto-detected when omitted")
    if rerolls:
        p.add_argument("--max-rerolls", type=int, default=3,
                       help="re-rolls allowed when a window has subtitles but no speech "
                            "(music interludes)")
    p.add_argument("--no-expand-numbers", action="store_true",
                   help="do not expand numerals to words before comparing")
    p.add_argument("--workdir", default=None,
                   help="keep extracted WAV snippets in this directory (default: temp)")
    p.add_argument("--json", metavar="FILE", default=None,
                   help="also write the full machine-readable report to FILE")


def build_compare_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck compare",
        description=(
            "Compare several subtitle files against the SAME sampled windows.\n"
            "\n"
            "With a video: one ffprobe/ffmpeg/ASR pass samples and transcribes\n"
            "the windows, then every SRT is scored against those identical\n"
            "windows and ranked by composite score:\n"
            "    subcheck compare movie.mp4 a.srt b.srt c.srt [--snippets 4]\n"
            "\n"
            "With --from: windows of a previous 'subcheck check --json' report\n"
            "are re-used, so no media access or ASR is needed (fast, perfect\n"
            "for checking whether a fix improved things):\n"
            "    subcheck compare --from windows.json a.srt a.fixed.srt\n"
            "\n"
            "All files are scored on identical windows, so the ranking is\n"
            "apples-to-apples. Windows where no speech was recognised count as\n"
            "perfect (100 %) for every file that has no text there; a file with\n"
            "text in a silent window is penalised for that window."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument(
        "targets",
        nargs="*",
        metavar="MEDIA_OR_SRT",
        help="without --from: the video followed by the SRT files to compare; "
             "with --from: just the SRT files",
    )
    p.add_argument(
        "--from", dest="from_json", metavar="WINDOWS_JSON", default=None,
        help="reuse windows + ASR segments from this report JSON instead of "
             "probing/transcribing (skips media access)",
    )
    _add_run_options(p, rerolls=False)
    return p


def run_compare_cli(args) -> "CompareReport":
    from .runner import run_compare

    media = None
    if args.from_json:
        subtitles = list(args.targets)
    else:
        if len(args.targets) < 2:
            raise ValueError(
                "compare needs either: a video + subtitle files to compare, "
                "or --from windows.json + subtitle files"
            )
        media = args.targets[0]
        subtitles = args.targets[1:]
    return run_compare(
        subtitles,
        media=media,
        windows_json=args.from_json,
        snippets=args.snippets,
        duration=args.duration,
        min_gap=args.min_gap,
        seed=args.seed,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.model_dir,
        beam_size=args.beam_size,
        language=args.language,
        workdir=args.workdir,
        expand_numbers=not args.no_expand_numbers,
    )


def _drift_raw(entry) -> str:
    """Raw drift facts for a compare row: mean±std in seconds."""
    ds = entry.report.drift_summary
    if ds is None or ds.matched_lines == 0:
        return "no match"
    if ds.mean is None:
        return "no match"
    return f"{ds.mean:+.2f}±{ds.std:.2f}s"


def _mean_std(values: List[float]):
    """(mean, sample std) of drift samples; std 0 for a single value."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) > 1:
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return mean, std


def print_compare_report(report) -> None:
    print(f"subcheck {__version__}  —  subtitle comparison "
          f"({len(report.entries)} file(s), {report.window_count} shared window(s))")
    print("-" * 78)
    if report.media:
        print(f"media          : {report.media}  ({_fmt_clock(report.total_duration)})")
    if report.settings.get("model_repo"):
        print(f"model          : {report.settings.get('model_repo')} "
              f"[{report.settings.get('device')} / {report.settings.get('compute_type')}]")
    print(f"windows        : {report.window_count} window(s) of "
          f"{report.snippet_duration:g}s, seed {report.settings.get('seed')}")
    starts = ", ".join(_fmt_clock(s.start) for s, _, _ in report.records[:6])
    if report.window_count > 6:
        starts += ", ..."
    print(f"at             : {starts}")

    print("-" * 78)
    print(f"{'rank':>4}  {'subtitle':<34} {'text':>8} {'IoU':>8} {'score':>8} "
          f"{'drift mean±std':>15}")
    print("-" * 78)
    for entry in report.entries:
        if entry.report is None:
            print(f"  -   {os.path.basename(entry.subtitle):<34} ERROR: {entry.error}")
            continue
        r = entry.report
        s = r.summary
        name = os.path.basename(entry.subtitle)
        if len(name) > 34:
            name = name[:31] + "..."
        rank = "-" if entry.rank is None else f"#{entry.rank}"
        drift = _drift_raw(entry)
        print(
            f"{rank:>4}  {name:<34}"
            f" {_fmt_pct(s.get('text_accuracy_mean')):>8}"
            f" {_fmt_pct(s.get('iou_mean')):>8}"
            f" {_fmt_pct(s.get('composite_mean')):>8}"
            f" {drift:>15}"
        )
    print("-" * 78)

    fixable = [e for e in report.entries if e.report and e.report.fix_command]
    for e in fixable:
        print(f"fix #{e.rank} ({os.path.basename(e.subtitle)}):")
        print(f"       {e.report.fix_command}")
    errors = [e for e in report.entries if e.error]
    for e in errors:
        print(f"* {e.subtitle}: {e.error}")
    best = next((e for e in report.entries if e.rank == 1), None)
    if best is not None and best.report:
        mean = best.report.summary.get("composite_mean")
        print(f"best match    : {best.subtitle} "
              f"(composite {mean:.1f}%)")
    elif report.error:
        print(f"error: {report.error}")
    # explain what affects the ranking when speech windows are missing
    voiced = sum(1 for _, segs, _ in report.records if segs)
    if voiced < report.window_count:
        print(f"note          : {report.window_count - voiced} window(s) had no "
              "recognised speech — every file with no text there scored 100 % "
              "for them, files with text scored 0 %")

    # per-file drift histograms over the shared snippets
    hists = [
        e
        for e in sorted(
            (e for e in report.entries if e.report and e.rank),
            key=lambda e: e.rank,
        )
        if drift_samples_of(e.report)
    ]
    if hists:
        print("-" * 78)
        print("drift histograms — per-snippet drift per file "
              "(seconds, + = subtitles run late):")
        for e in hists:
            samples = drift_samples_of(e.report)
            name = os.path.basename(e.subtitle)
            if len(name) > 60:
                name = name[:57] + "..."
            mean, std = _mean_std(samples)
            stats = (f"mean {mean:+.2f}s, std {std:.2f}s" if mean is not None
                     else "no data")
            print(f"  #{e.rank} {name} — {stats}, n={len(samples)}:")
            for line in drift_histogram_lines(samples):
                print(f"      {line.strip()}")


def build_shift_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck shift",
        description=(
            "Rewrite every cue time of an SRT file (or of the subtitles embedded\n"
            "in a video) with one transform:\n"
            "    new_t = t * --scale + --by      (defaults: 1.0, 0.0)\n"
            "\n"
            "The two options mirror the two timing defects the check detects:\n"
            "\n"
            "  --by     constant SHIFT: add the same amount to all cue times.\n"
            "           Fix for 'consistent drift' (every line late or early by\n"
            "           the same offset), e.g. --by -0.989 for cues 0.99 s late.\n"
            "\n"
            "  --scale  time-BASE (fps) correction: multiply all cue times by a\n"
            "           factor. Fix for 'bad fps' (the subtitle timeline runs at\n"
            "           the wrong speed, so drift grows over time). Example:\n"
            "           PAL 25 fps subtitles on a 23.976 fps movie need\n"
            "           --scale 0.95904 (= 23.976 / 25).\n"
            "\n"
            "If the input is not an .srt file, it is treated as a video and its\n"
            "embedded subtitle stream is used instead (--sub-stream /\n"
            "--sub-language pick the track; run 'subcheck streams <video>' to\n"
            "list them). The fixed SRT is written next to the input; with\n"
            "--remux OUT.mkv the corrected track is also muxed back into a copy\n"
            "of the video.\n"
            "\n"
            "The options combine (scale first, then shift); each defaults to a\n"
            "no-op, so either can be used alone. Negative times are clamped to\n"
            "0. Output defaults to <name>.fixed.srt (use --out to override)."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument(
        "subtitle",
        metavar="SRT_OR_VIDEO",
        help=".srt file to fix, or a video file whose embedded subtitles "
             "should be fixed",
    )
    p.add_argument("--by", type=float, default=0.0,
                   help="constant time shift in seconds, added to every cue time "
                        "(negative = earlier, positive = later; fix for consistent "
                        "drift)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="timebase (fps) factor, multiplied into every cue time "
                        "(1.0 = unchanged; <1 slows the subtitle timeline down, "
                        ">1 speeds it up; fix for bad-fps drift)")
    p.add_argument("--sub-stream", type=int, default=None, metavar="INDEX",
                   help="embedded stream index to fix (default: default/first "
                        "text stream)")
    p.add_argument("--sub-language", default=None, metavar="CODE",
                   help="pick the embedded stream by language code (eng, fre, ...)")
    p.add_argument("--remux", default=None, metavar="OUT.mkv",
                   help="when fixing a video, also mux the corrected subtitles "
                        "into a copy of the video (mkv/webm)")
    p.add_argument("--in-place", "-i", "--overwrite", action="store_true",
                   dest="in_place",
                   help="overwrite the original file in place with the updated subtitles "
                        "(rewrites the .srt file directly, or remuxes the video file in place)")
    p.add_argument("--out", default=None,
                   help="output SRT path (default: <name>.fixed.srt next to the "
                        "input)")
    return p


def _transform_srt(path: str, out: str, by: float, scale: float) -> int:
    """Rewrite every cue of *path* with new_t = t*scale + by; returns the
    number of clamped cues (delegates to srtio.shift_srt)."""
    from .srtio import shift_srt

    return shift_srt(path, out, by=by, scale=scale)


def _default_fixed_path(target: str) -> str:
    stem = os.path.splitext(os.path.basename(target.split("?")[0]))[0]
    base = os.path.dirname(target)
    if not base or base.startswith(("http://", "https://")):
        base = "."
    return os.path.join(base, stem + ".fixed.srt")


def run_shift(args) -> dict:
    """Apply the transform; returns {out, clamped, note} — *note* describes
    the embedded stream when the input was a video."""
    import shutil
    import tempfile

    import pysrt  # noqa: F401  (validates the file below)

    target = args.subtitle
    in_place = getattr(args, "in_place", False)
    out = args.out

    if target.lower().endswith(".srt"):
        if in_place:
            out = target
        elif out is None:
            out = _default_fixed_path(target)
        return {"out": out, "clamped": _transform_srt(target, out, args.by, args.scale),
                "note": ""}

    # not an .srt: treat it as a video with embedded subtitles
    from .embedded import (
        choose_subtitle_stream,
        extract_subtitle_stream,
        probe_subtitle_streams,
        remux_with_subtitles,
    )

    streams = probe_subtitle_streams(target)
    if not streams:
        raise ValueError(
            f"{target!r} is not an .srt file and contains no embedded subtitle "
            "streams — nothing to shift"
        )
    stream = choose_subtitle_stream(
        streams, index=args.sub_stream, language=args.sub_language
    )
    remux_target = args.remux
    if in_place:
        remux_target = target

    tmpdir = tempfile.mkdtemp(prefix="subcheck-embed-")
    try:
        raw = os.path.join(tmpdir, "embedded.srt")
        extract_subtitle_stream(target, stream, raw)

        # When overwriting in place and no explicit --out was requested,
        # write the shifted srt inside tmpdir so no unwanted sidecar is left behind.
        if in_place and out is None:
            shift_out = os.path.join(tmpdir, "fixed.srt")
            clamped = _transform_srt(raw, shift_out, args.by, args.scale)
            reported_out = target
        else:
            if out is None:
                out = _default_fixed_path(target)
            clamped = _transform_srt(raw, out, args.by, args.scale)
            shift_out = out
            reported_out = out

        text_n = sum(1 for s in streams if s.text_capable)
        note = f"using embedded {stream.desc}"
        if text_n > 1:
            note += (f" ({text_n} text streams available — use --sub-stream or "
                     "--sub-language to pick another)")
        if remux_target:
            remux_with_subtitles(
                target,
                shift_out,
                remux_target,
                language=stream.language,
                title=stream.title or (stream.language.upper() if stream.language else "Subtitles"),
                replace_stream_index=stream.index,
            )
            if in_place or os.path.abspath(remux_target) == os.path.abspath(target):
                note += f"; updated {remux_target} in place"
            else:
                note += f"; remuxed into {remux_target}"
        return {"out": reported_out, "clamped": clamped, "note": note}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_rescore_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck rescore",
        description=(
            "Re-score previously transcribed windows (a report from "
            "'subcheck check --json') against a different SRT file, without "
            "re-running ffmpeg or the ASR model. Useful to compare subtitle "
            "candidates for the same video."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument("windows_json", help="report JSON from 'subcheck check --json'")
    p.add_argument("subtitle", help="SRT file to score")
    p.add_argument("--no-expand-numbers", action="store_true",
                   help="do not expand numerals to words before comparing")
    p.add_argument("--json", metavar="FILE", default=None,
                   help="also write the full machine-readable report to FILE")
    return p


def run_rescore(args) -> CheckReport:
    from .runner import rescore_windows

    return rescore_windows(
        args.windows_json,
        args.subtitle,
        expand_numbers=not args.no_expand_numbers,
    )


def build_autosub_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck autosub",
        description=(
            "End-to-end: fetch subtitle candidates for a video from "
            "OpenSubtitles (cached), verify each against the video's own audio "
            "with the compare engine, pick the best fit — trying its diagnosed "
            "drift/timebase fix first — and embed it into a copy of the video.\n"
            "\n"
            "Usage: subcheck autosub movie.mkv [--limit 3] [--no-embed]\n"
            "       subcheck autosub show.mkv --language en --refresh\n"
            "\n"
            "Credentials come from .env (OPENSUBTITLES_API_KEY/_USERNAME/"
            "_PASSWORD) or ~/.config/opensubtitlescom/config.json. Searches "
            "and downloads are cached under .cache/opensubtitles (0 quota on "
            "hits); --refresh ignores both caches, and a corrupt cached file "
            "is purged and re-downloaded automatically."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument("video", help="video file to find subtitles for")
    p.add_argument("--limit", type=int, default=3,
                   help="how many candidates to fetch and compare (downloads "
                        "only count against the quota when not cached)")
    p.add_argument("--language", default="en", metavar="CODE",
                   help="subtitle language code to search for")
    p.add_argument("--refresh", action="store_true",
                   help="ignore cached search results and cached files "
                        "(re-query/re-download; use when a previous pull "
                        "didn't work)")
    p.add_argument("--skip-embedded", action="store_true",
                   help="do not check embedded subtitle streams; always search OpenSubtitles")
    p.add_argument("--title", default=None,
                   help="override the title derived from the file name")
    p.add_argument("--season", type=int, default=None, metavar="N",
                   help="override the season (episode search)")
    p.add_argument("--episode", type=int, default=None, metavar="N",
                   help="override the episode number")
    p.add_argument("--no-embed", action="store_true",
                   help="do not remux the winning subtitles into the video")
    p.add_argument("--in-place", "-i", "--overwrite", action="store_true",
                   dest="in_place",
                   help="overwrite the original video file in place with the winning subtitles")
    p.add_argument("--embed-out", default=None, metavar="PATH",
                   help="output video path for the embedded copy "
                        "(default: <video>.autosub.<ext> next to the source)")
    p.add_argument("--out-srt", default=None, metavar="PATH",
                   help="where to write the winning .srt "
                        "(default: <video>.autosub.srt next to the source)")
    p.add_argument("--snippets", type=int, default=10,
                   help="number of random windows to verify with")
    p.add_argument("--duration", type=float, default=10.0,
                   help="window duration in seconds")
    p.add_argument("--min-gap", type=float, default=30.0,
                   help="minimum spacing between windows (seconds)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducible window selection")
    p.add_argument("--model", default="large-v3",
                   help="whisper model (tiny/base/small/medium/large-v3 or an HF repo id)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="compute device; auto = CUDA when available, else CPU")
    p.add_argument("--compute-type", default=None,
                   help="ctranslate2 compute type (default: float16 on CUDA, int8 on CPU)")
    p.add_argument("--model-dir", default=None,
                   help="directory for model weights (default: HF cache / SUBCHECK_MODEL_DIR)")
    p.add_argument("--beam-size", type=int, default=5, help="whisper beam size")
    p.add_argument("--no-expand-numbers", action="store_true",
                   help="do not expand numerals to words before comparing")
    p.add_argument("--workdir", default=None,
                   help="keep fetched candidates / wavs / fixed copies here "
                        "(default: ephemeral temp dir)")
    p.add_argument("--json", metavar="FILE", default=None,
                   help="also write the full machine-readable report to FILE")
    return p


def run_autosub_cli(args) -> "AutoSubResult":
    from .autosub import run_autosub

    return run_autosub(
        args.video,
        language=args.language,
        limit=args.limit,
        refresh=args.refresh,
        skip_embedded=getattr(args, "skip_embedded", False),
        title=args.title,
        season=args.season,
        episode=args.episode,
        snippets=args.snippets,
        duration=args.duration,
        min_gap=args.min_gap,
        seed=args.seed,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.model_dir,
        beam_size=args.beam_size,
        embed=not args.no_embed,
        embed_out=args.embed_out,
        final_srt=args.out_srt,
        workdir=args.workdir,
        expand_numbers=not args.no_expand_numbers,
        overwrite=getattr(args, "in_place", False),
    )


def print_autosub_report(result) -> None:
    print(f"subcheck {__version__}  —  autosub (fetch / verify / fix / embed)")
    print("-" * 74)
    meta = result.meta
    label = f"{meta.get('title')}"
    if meta.get("season") is not None:
        label += f" S{meta['season']:02d}E{meta['episode']:02d}"
    print(f"video          : {result.video}")
    print(f"searching      : {label}  [{result.language}]  "
          f"({len(result.candidates)} candidate(s), "
          f"{result.windows_count} shared window(s))")
    if result.quota or result.downloads_new or result.cache_hits:
        from .autosub import describe_quota

        q = describe_quota(result.quota)
        if result.downloads_new or result.cache_hits:
            q += f"  ({result.downloads_new} downloaded, {result.cache_hits} from cache)"
        print(f"quota          : {q}")
    print("-" * 74)
    if result.error:
        print(f"error          : {result.error}")
        for note in result.notes:
            print(f"* {note}")
        return
    if result.candidates:
        print(f"{'rank':>4}  {'release':<38} {'cached':>6} {'before':>8} "
              f"{'mode':<9} {'after':>8} {'drift mean±std':>15}")
        print("-" * 74)
        ranked = sorted((c for c in result.candidates if not c.error),
                        key=lambda c: (c.rank if c.rank is not None else 99))
        for c in ranked:
            name = (c.release or c.file_name) or f"file {c.file_id}"
            if len(name) > 38:
                name = name[:35] + "..."
            rank = "-" if c.rank is None else f"#{c.rank}"
            cached = "yes" if c.from_cache else "no"
            before = _fmt_pct(c.composite)
            after = _fmt_pct(c.composite_after)
            mode = c.best_mode
            if c.best_mode != "as-is" and c.best_by:
                mode += f" {c.best_by:+.2f}s"
            mean, std = _mean_std(c.drift_samples)
            drift = (f"{mean:+.2f}±{std:.2f}s" if mean is not None else "no match")
            print(f"{rank:>4}  {name:<38} {cached:>6} {before:>8} "
                  f"{mode:<9} {after:>8} {drift:>15}")
        for c in result.candidates:
            if c.error:
                print(f"  -   {c.file_name or c.file_id}: {c.error}")
    print("-" * 74)
    for note in result.notes:
        print(f"* {note}")
    chosen = result.chosen
    if chosen is not None:
        name = (chosen.release or chosen.file_name) or f"file {chosen.file_id}"
        print(f"best match     : {name}  "
              f"(composite {chosen.composite_after:.1f}%)")
        if chosen.best_mode != "as-is":
            verb = "shifted" if chosen.best_mode == "shift" else "rescaled"
            print(f"  (would only score {chosen.composite:.1f}% as-is — "
                  f"{verb} by {chosen.best_by:+.3f}s "
                  f"scale={chosen.best_scale:g}, verified on the same windows)")
        if result.final_srt:
            print(f"wrote         : {result.final_srt}")
        if result.embed_status == "ok":
            print(f"embedded      : {result.embed_out}")
        elif result.embed_status:
            print(f"embedding     : {result.embed_status}")

    # per-candidate drift histograms over the shared snippets
    with_drift = [
        c
        for c in sorted(
            (c for c in result.candidates if c.rank and c.drift_samples),
            key=lambda c: c.rank,
        )
    ]
    if with_drift:
        print("-" * 74)
        print("drift histograms — per-snippet drift per candidate "
              "(seconds, + = subtitles run late):")
        for c in with_drift:
            name = (c.release or c.file_name) or f"file {c.file_id}"
            if len(name) > 56:
                name = name[:53] + "..."
            mean, std = _mean_std(c.drift_samples)
            stats = (f"mean {mean:+.2f}s, std {std:.2f}s" if mean is not None
                     else "no data")
            mode = f" — chosen fix: {c.best_mode}" if c.best_mode != "as-is" else ""
            print(f"  #{c.rank} {name} — {stats}, n={len(c.drift_samples)}{mode}")
            for line in drift_histogram_lines(c.drift_samples):
                print(f"      {line.strip()}")


def build_download_model_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck download-model",
        description=(
            "Download or verify Whisper model weights ahead of time from Hugging Face."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="large-v3",
                   help="whisper model (tiny/base/small/medium/large-v3 or an HF repo id)")
    p.add_argument("--model-dir", default=None,
                   help="directory for model weights (default: HF cache / SUBCHECK_MODEL_DIR)")
    return p


def run_download_model(args) -> int:
    from .transcribe import download_model, resolve_model_repo

    repo = resolve_model_repo(args.model)
    print(f"Downloading model {args.model!r} ({repo})...")
    path = download_model(args.model, download_root=args.model_dir)
    print(f"Model ready at: {path}")
    return 0


def build_quota_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck quota",
        description=(
            "Show the OpenSubtitles.com account status and daily download "
            "quota (the reference check_quota.py output, integrated): level, "
            "allowed/remaining downloads, downloads used today and the reset "
            "time. Credentials come from .env or "
            "~/.config/opensubtitlescom/config.json."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    return p


def run_quota_cli(args) -> int:
    from .autosub import OpenSubtitlesClient, load_credentials

    creds = load_credentials()
    client = OpenSubtitlesClient(creds)
    q = client.quota()
    line = "=" * 49
    print(line)
    print(" OpenSubtitles.com Account Status")
    print(line)
    print(f" Username            : {q.get('username')}")
    print(f" Rank / Level        : {q.get('level', 'User')}")
    print(f" Allowed Daily Limit : {q.get('allowed_downloads')} downloads/day")
    print(f" Downloads Remaining : {q.get('remaining_downloads')}")
    print(f" Downloads Used      : {q.get('downloads_count')}")
    print(f" Quota Resets In     : {q.get('reset_time')}")
    print(line)
    return 0


def build_streams_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck streams",
        description=(
            "List the embedded subtitle streams of a video file, so you can "
            "pick one with --sub-stream / --sub-language for 'subcheck check' "
            "or 'subcheck shift'."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument("media", help="video file path or URL")
    return p


def run_streams(args) -> int:
    from .embedded import probe_subtitle_streams

    streams = probe_subtitle_streams(args.media)
    if not streams:
        print(f"{args.media}: no embedded subtitle streams found.")
        return 0
    print(f"{args.media}: {len(streams)} embedded subtitle stream(s):")
    for st in streams:
        tag = ""
        if not st.text_capable:
            tag = "  [bitmap — cannot be checked textually]"
        print(f"  #{st.index:<3} {st.codec:<22} lang={st.language or '-':<4}"
              f" default={'yes' if st.default else 'no':<3}"
              f" forced={'yes' if st.forced else 'no'}{tag}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "check":  # `subcheck check VIDEO [SRT]` is the default
        argv = argv[1:]
    if argv and argv[0] == "autosub":
        args = build_autosub_parser().parse_args(argv[1:])
        try:
            result = run_autosub_cli(args)
        except Exception as exc:  # noqa: BLE001 - creds/network/etc.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print_autosub_report(result)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
            print(f"\nJSON report written to {args.json}")
        return 1 if result.error else 0
    if argv and argv[0] in ("download-model", "pull-model"):
        args = build_download_model_parser().parse_args(argv[1:])
        try:
            return run_download_model(args)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if argv and argv[0] == "quota":
        args = build_quota_parser().parse_args(argv[1:])
        try:
            return run_quota_cli(args)
        except Exception as exc:  # noqa: BLE001 - credentials/network
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if argv and argv[0] == "streams":
        args = build_streams_parser().parse_args(argv[1:])
        try:
            return run_streams(args)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if argv and argv[0] == "compare":
        args = build_compare_parser().parse_args(argv[1:])
        try:
            report = run_compare_cli(args)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print_compare_report(report)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
            print(f"\nJSON report written to {args.json}")
        return 1 if report.error else 0
    if argv and argv[0] == "shift":
        args = build_shift_parser().parse_args(argv[1:])
        try:
            result = run_shift(args)
        except Exception as exc:  # noqa: BLE001 - surface pysrt/IO errors
            print(f"error: {exc}", file=sys.stderr)
            return 1
        target = args.subtitle
        print(f"shifted {target} by {args.by:+.3f}s "
              f"(scale {args.scale:g}) "
              f"({result['clamped']} cue time(s) clamped to stay valid)")
        if result["note"]:
            print(f"({result['note']})")
        print(f"wrote {result['out']}")
        return 0
    if argv and argv[0] == "rescore":
        args = build_rescore_parser().parse_args(argv[1:])
        try:
            report = run_rescore(args)
        except (ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print_report(report)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
            print(f"\nJSON report written to {args.json}")
        return 0

    args = build_parser().parse_args(argv)

    opts = Options(
        media=args.media,
        subtitle=args.subtitle,  # None -> sidecar .srt or embedded stream
        snippets=args.snippets,
        duration=args.duration,
        min_gap=args.min_gap,
        seed=args.seed,
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.model_dir,
        beam_size=args.beam_size,
        language=args.language,
        max_rerolls=args.max_rerolls,
        workdir=args.workdir,
        expand_numbers=not args.no_expand_numbers,
        use_embedded=args.embedded,
        sub_stream=args.sub_stream,
        sub_language=args.sub_language,
    )

    try:
        report = run_check(opts)
    except (ToolError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        print(f"\nJSON report written to {args.json}")
    return 0


def print_report(report: CheckReport) -> None:
    print(f"subcheck {__version__}  —  subtitle accuracy check")
    print("-" * 74)
    print(f"media          : {report.media}")
    sub_label = report.settings.get("subtitle_label") or report.subtitle
    print(f"subtitles      : {sub_label}")
    if report.settings.get("source") == "embedded":
        f = report.settings.get("subtitle_file")
        if f and os.path.exists(f):
            print(f"                (extracted from the video: {f})")
        total = report.settings.get("embedded_streams_total")
        text_n = report.settings.get("embedded_text_streams")
        used = report.settings.get("embedded_stream")
        if text_n and text_n > 1:
            print(f"note           : the video carries {total} subtitle "
                  f"stream(s), {text_n} textual — using stream #{used}; "
                  "rerun with --sub-stream INDEX or --sub-language CODE to "
                  "choose another")
    print(f"duration       : {_fmt_clock(report.total_duration)} "
          f"({report.total_duration:.1f}s)")
    print(f"model          : {report.settings.get('model_repo')} "
          f"[{report.settings.get('device')} / {report.settings.get('compute_type')}]")
    s = report.settings
    print(f"sampled        : {s.get('snippets')} window(s) of "
          f"{report.snippet_duration:g}s, min gap {s.get('min_gap'):g}s, "
          f"{s.get('cues_total', '?')} cues in file"
          + (f", seed {s.get('seed')}" if s.get("seed") is not None else ""))
    print("-" * 74)

    if not report.windows:
        print("no windows could be checked.")
        return

    for o in report.windows:
        sc = o.score
        pos = f"{_fmt_clock(o.snippet.start)} -> {_fmt_clock(o.snippet.end)}"
        if o.rolled:
            print(f"  [{pos}]  rolled away (subtitles present, no speech recognised)")
            continue
        if sc is None:
            print(f"  [{pos}]  status={o.status}")
            continue
        if sc.status == "silent-match":
            print(f"  [{pos}]  silent  (no speech, no subtitles) -> 100% match")
            continue
        drift = "" if sc.drift_seconds is None else f"  drift {sc.drift_seconds:+.2f}s"
        print(
            f"  [{pos}]  {sc.status:16s}"
            f" text {_fmt_pct(sc.text_accuracy)}  iou {_fmt_pct(None if sc.iou is None else sc.iou * 100)}"
            f"  score {_fmt_pct(sc.composite)}{drift}"
        )
        if sc.text_accuracy is not None and sc.text_accuracy < 100:
            print(f"           asr: {_trunc(' '.join(x.text for x in sc.segments))}")
            print(f"           srt: {_trunc(' '.join(x.text for x in sc.cues))}")

    print("-" * 74)
    ds = report.drift_summary
    if ds is not None:
        if ds.matched_lines > 0:
            print(
                "drift stats   : "
                f"mean {ds.mean:+.3f}s   std {ds.std:.3f}s   "
                f"(+ = subtitles late)  over {ds.matched_lines} matched "
                f"line(s) in {ds.matched_windows} window(s)"
            )
        if ds.slope_s_per_s is not None:
            print(
                f"drift slope   : {ds.slope_s_per_s:+.6f} s of drift per "
                f"second of movie  (timeline "
                f"{'runs fast' if ds.slope_s_per_s > 0 else 'runs slow'}, "
                f"off by ≈{abs(ds.rate_delta_pct or 0.0):.2f}%)"
            )
        print(f"classification: {ds.label}")
        if ds.note:
            print(f"               ({ds.note})")
        if report.fix_command:
            print(f"fix command   : {report.fix_command}")
    samples = drift_samples_of(report)
    scored_total = sum(
        1 for o in report.windows if o.score and not o.rolled
    )
    if samples:
        print_drift_histograms("drift histogram", samples, scored_total)
    summary = report.summary
    mean = summary.get("composite_mean")
    if mean is not None:
        n = summary.get("windows_scored", 0)
        print(f"aggregate  ({n} window(s), silent matches counted as perfect):")
        print(f"  text accuracy  : {_fmt_pct(summary.get('text_accuracy_mean'))}")
        print(f"  temporal IoU   : {_fmt_pct(summary.get('iou_mean'))}")
        print(f"  composite      : {_fmt_pct(mean)}"
              "   (= 0.75 * text + 0.25 * IoU)")
    for flag in report.flags:
        print(f"* {flag}")


def _trunc(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
