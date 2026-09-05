"""Command-line interface for subcheck."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .runner import CheckReport, Options, run_check
from .tools import ToolError


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
            "  subcheck rescore windows.json other.srt\n"
            "      re-score saved windows against another SRT\n"
            "  subcheck shift movie.srt --by -0.989\n"
            "      fix 'consistent drift' (constant time shift)\n"
            "  subcheck shift movie.srt --scale 0.95904\n"
            "      fix 'bad fps' (timebase/fps correction)\n"
            "  subcheck shift movie.srt --scale F --by B\n"
            "      combine both (see 'subcheck shift --help')"
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"subcheck {__version__}")
    p.add_argument("media", help="video file path or URL (ffmpeg/ffprobe readable)")
    p.add_argument(
        "subtitle",
        nargs="?",
        help="SRT file to check; omit to let it be auto-detected next to the media",
    )
    p.add_argument("--snippets", type=int, default=3,
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
    p.add_argument("--max-rerolls", type=int, default=3,
                   help="re-rolls allowed when a window has subtitles but no speech "
                        "(music interludes)")
    p.add_argument("--no-expand-numbers", action="store_true",
                   help="do not expand numerals to words before comparing")
    p.add_argument("--workdir", default=None,
                   help="keep extracted WAV snippets in this directory (default: temp)")
    p.add_argument("--json", metavar="FILE", default=None,
                   help="also write the full machine-readable report to FILE")
    return p


def auto_subtitle(media: str) -> Optional[str]:
    """Find an .srt with a matching base name next to the media."""
    import glob
    import os

    base = os.path.splitext(media)[0]
    hits = sorted(glob.glob(base + "*.srt")) + sorted(glob.glob(base + ".*.srt"))
    return hits[0] if hits else None


def build_shift_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subcheck shift",
        description=(
            "Rewrite every cue time of an SRT file with one transform:\n"
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
            "The options combine (scale first, then shift); each defaults to a\n"
            "no-op, so either can be used alone. Negative times are clamped to\n"
            "0. Output defaults to <name>.fixed.srt (use --out to override)."
        ),
        formatter_class=_RawDefaultsHelpFormatter,
    )
    p.add_argument("subtitle", help="SRT file to fix")
    p.add_argument("--by", type=float, default=0.0,
                   help="constant time shift in seconds, added to every cue time "
                        "(negative = earlier, positive = later; fix for consistent "
                        "drift)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="timebase (fps) factor, multiplied into every cue time "
                        "(1.0 = unchanged; <1 slows the subtitle timeline down, "
                        ">1 speeds it up; fix for bad-fps drift)")
    p.add_argument("--out", default=None,
                   help="output path (default: <name>.fixed.srt next to the input)")
    return p


def run_shift(args) -> tuple:
    """Apply a constant time shift (+ optional timebase scale); returns
    (output path, clamped count)."""
    import os

    import pysrt

    path = args.subtitle
    out = args.out
    if out is None:
        root, ext = os.path.splitext(path)
        out = root + ".fixed" + (ext or ".srt")
    subs = pysrt.open(path, encoding="utf-8-sig")
    clamped = 0
    by_ms = args.by * 1000.0
    for item in subs:
        start_ms = int(round(item.start.ordinal * args.scale + by_ms))
        end_ms = int(round(item.end.ordinal * args.scale + by_ms))
        if start_ms < 0:
            start_ms = 0
            clamped += 1
        if end_ms <= start_ms:
            end_ms = start_ms + 1
            clamped += 1
        item.start = pysrt.SubRipTime(milliseconds=start_ms)
        item.end = pysrt.SubRipTime(milliseconds=end_ms)
    subs.save(out, encoding="utf-8")
    return out, clamped


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


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "shift":
        args = build_shift_parser().parse_args(argv[1:])
        try:
            out, clamped = run_shift(args)
        except Exception as exc:  # noqa: BLE001 - surface pysrt/IO errors
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"shifted {args.subtitle} by {args.by:+.3f}s "
              f"(scale {args.scale:g}) "
              f"({clamped} cue time(s) clamped to stay valid)")
        print(f"wrote {out}")
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
    subtitle = args.subtitle
    if subtitle is None:
        subtitle = auto_subtitle(args.media)
        if subtitle is None:
            print(
                "error: no subtitle file given and none found next to the media",
                file=sys.stderr,
            )
            return 2

    opts = Options(
        media=args.media,
        subtitle=subtitle,
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
    print(f"subtitles      : {report.subtitle}")
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
