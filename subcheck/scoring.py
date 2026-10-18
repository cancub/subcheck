"""Stage 6 — per-snippet scoring: text accuracy (WER) + temporal IoU.

WER direction
-------------
We treat the ASR transcript as the *reference* (what is actually said on the
audio, as heard by the ASR) and the subtitle text as the *hypothesis* (what
the subtitle file claims is said). WER = (S + D + I) / N then counts exactly
the subtitle file's deviations from the audio: text it adds, words it drops
and words it rewrites. (A verbatim subtitle file scores 0 WER regardless of
direction; the ASR-as-reference choice mainly avoids penalising subtitles
that omit disfluencies such as "uh" that the ASR happens to transcribe.)

Text Accuracy = max(0, 1 - WER) * 100

IoU (sync quality)
------------------
Speech intervals = union of ASR segments inside the window.
Subtitle intervals = union of subtitle cues inside the window.
IoU = |speech ∩ subtitle| / |speech ∪ subtitle|  (exact 1-D interval algebra)

Composite = 0.75 * text_accuracy + 0.25 * IoU(%)

Straddling cues (edge case)
---------------------------
Cues that start before the window or end after it keep their text — but the
words that fall *outside* the window cannot be matched by ASR, so they would
inflate WER. The candidate text is therefore trimmed to the **active speech
span** (first to last ASR segment inside the window): each subtitle word is
mapped linearly onto its cue's duration and only words overlapping that span
are compared. This keeps straddle penalties bounded and tolerates subtitles
whose timeline is shifted by a fraction of the window without hiding real
text mismatches (words are still compared over the region where speech is
heard).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import jiwer

from .normalize import normalize
from .srtio import Cue
from .transcribe import Segment


# ---------------------------------------------------------------- interval helpers
def _clip(intervals: List[Tuple[float, float]], lo: float, hi: float) -> List[Tuple[float, float]]:
    """Clip intervals to [lo, hi] and drop empties."""
    out = []
    for a, b in intervals:
        a, b = max(a, lo), min(b, hi)
        if b > a:
            out.append((a, b))
    return out


def union_duration(intervals: List[Tuple[float, float]]) -> float:
    """Total covered duration after merging overlapping intervals."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def intersection_duration(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    total = 0.0
    for x in a:
        for y in b:
            lo, hi = max(x[0], y[0]), min(x[1], y[1])
            if hi > lo:
                total += hi - lo
    return total


# ------------------------------------------------------------------ candidate text
def windowed_candidate_text(
    cues: List[Cue],
    wstart: float,
    wend: float,
    speech_span: Optional[Tuple[float, float]] = None,
) -> str:
    """Subtitle text expected inside the window, with boundary trimming.

    *speech_span* (first..last ASR segment inside the window, clipped to it)
    narrows the region that spoken words are actually expected in. Subtitle
    words are mapped linearly onto their cue's duration; only words whose time
    interval overlaps the comparison region are kept, so text that describes
    audio outside the region (music interludes, pre/post-window speech, or
    speech that a shifted timeline moved out of the window) does not distort
    the WER.
    """
    lo, hi = wstart, wend
    if speech_span is not None:
        lo = max(lo, speech_span[0])
        hi = min(hi, speech_span[1])
        if hi <= lo:
            return ""

    words: List[str] = []
    for cue in cues:
        # word-interval model over the cue: word i occupies
        # [cue.start + i*step, cue.start + (i+1)*step)
        n = len(cue.text.split())
        if n == 0 or cue.duration <= 0:
            continue
        cue_words = cue.text.split()
        step = cue.duration / n
        keep_from = 0
        keep_to = n
        # words fully before [lo, hi)
        while keep_from < n and cue.start + (keep_from + 1) * step <= lo:
            keep_from += 1
        # words fully after [lo, hi)
        while keep_to > keep_from and cue.start + (keep_to - 1) * step >= hi:
            keep_to -= 1
        if keep_to > keep_from:
            words.extend(cue_words[keep_from:keep_to])
    return " ".join(words)


# ---------------------------------------------------------------------- scoring
@dataclass
class SnippetScore:
    """Per-snippet comparison result."""

    wstart: float
    wend: float
    segments: List[Segment] = field(default_factory=list)      # global times
    cues: List[Cue] = field(default_factory=list)               # overlapping cues
    status: str = "scored"
    # status values:
    #   "scored"        both sides have text (or the comparison ran normally)
    #   "silent-match"  no speech recognized AND no subtitle text -> 100 %
    #   "speech-no-subs" speech recognized but subtitle file has nothing here
    #   "subs-no-speech" subtitle file has text but no speech was recognized
    #                    (music interlude or ASR failure) -> eligible for re-roll
    text_accuracy: Optional[float] = None   # 0-100
    wer: Optional[float] = None
    iou: Optional[float] = None             # 0-1
    composite: Optional[float] = None       # 0-100
    drift_seconds: Optional[float] = None   # median subtitle-vs-speech offset
    drift_deltas: List[float] = field(default_factory=list)
    # one delta per matched (segment, cue) pair: cue_start - segment_start;
    # positive -> the subtitle cue starts later than its speech (subs late)

    @property
    def speech_duration(self) -> float:
        return union_duration([(s.start, s.end) for s in self.segments])

    @property
    def subtitle_duration(self) -> float:
        return union_duration([(c.start, c.end) for c in self.cues])

    def to_dict(self) -> dict:
        """Plain JSON-friendly view (global timeline)."""
        return {
            "status": self.status,
            "text_accuracy": None if self.text_accuracy is None else round(self.text_accuracy, 2),
            "wer": None if self.wer is None else round(self.wer, 4),
            "iou": None if self.iou is None else round(self.iou, 4),
            "composite": None if self.composite is None else round(self.composite, 2),
            "speech_duration": round(self.speech_duration, 3),
            "subtitle_duration": round(self.subtitle_duration, 3),
            "drift_seconds": None if self.drift_seconds is None else round(self.drift_seconds, 3),
            "drift_pairs": len(self.drift_deltas),
            "drift_deltas": [round(d, 3) for d in self.drift_deltas],
            "asr_segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                for s in self.segments
            ],
            "subtitle_cues": [
                {"index": c.index, "start": round(c.start, 3), "end": round(c.end, 3), "text": c.text}
                for c in self.cues
            ],
        }

    def describe(self) -> str:
        if self.status == "silent-match":
            return "no speech and no subtitles (both silent) -> 100 % match"
        if self.status == "speech-no-subs":
            return "speech detected, but the subtitle file has no text here"
        if self.status == "subs-no-speech":
            return "subtitles present but no speech recognized (music? ASR miss?)"
        return f"text {self.text_accuracy:.1f}%  iou {100 * (self.iou or 0):.1f}%  composite {self.composite:.1f}%"


def score_window(
    segments: List[Segment],
    cues: List[Cue],
    wstart: float,
    wend: float,
    expand_numbers: bool = True,
    cue_pool: Optional[List[Cue]] = None,
) -> SnippetScore:
    """Score one window. *segments* must already carry global (offset) times.

    *cue_pool* (default: *cues*) is the full subtitle file; it is used only
    for drift pairing so that a cue whose start has drifted out of the window
    (but whose text still matches nearby speech) can still contribute an
    offset sample.
    """
    # keep segments/cues only when they actually contribute inside the window
    segs = [s for s in segments if s.end > wstart and s.start < wend]
    cs = [c for c in cues if c.end > wstart and c.start < wend]

    # narrow the comparison region to the span actually covered by recognised
    # speech, so subtitle words timed outside it cannot hurt the text score
    speech_span = None
    if segs:
        speech_span = (
            max(wstart, min(s.start for s in segs)),
            min(wend, max(s.end for s in segs)),
        )

    asr_text = normalize(" ".join(s.text for s in segs), expand_numbers=expand_numbers)
    sub_text = normalize(
        windowed_candidate_text(cs, wstart, wend, speech_span),
        expand_numbers=expand_numbers,
    )

    score = SnippetScore(wstart=wstart, wend=wend, segments=segs, cues=cs)

    if not asr_text and not sub_text:
        score.status = "silent-match"
        score.text_accuracy = 100.0
        score.wer = 0.0
        score.iou = 1.0
        score.composite = 100.0
        return score

    # temporal IoU over the clipped speech/subtitle intervals
    sp = _clip([(s.start, s.end) for s in segs], wstart, wend)
    cu = _clip([(c.start, c.end) for c in cs], wstart, wend)
    inter = intersection_duration(sp, cu)
    union = union_duration(sp) + union_duration(cu) - inter
    score.iou = inter / union if union > 0 else 0.0
    # offset between matched speech segments and cues (timing survives even
    # when the texts do not overlap enough to be compared word for word).
    # The match pool includes cues that no longer overlap the window, so
    # windows keep producing drift samples while |drift| stays below the
    # text-match gate (~12 s).
    if segs:
        pool = cs if cue_pool is None else cue_pool
        score.drift_deltas = pair_deltas(segs, pool)
        if score.drift_deltas:
            score.drift_seconds = _median(score.drift_deltas)

    # subtitle file claims words that were never spoken (music interlude with
    # stale cues, invented dialogue, ...)
    if sub_text and not asr_text:
        score.status = "subs-no-speech"
        score.text_accuracy = 0.0
        score.wer = 1.0
        score.composite = 0.75 * 0.0 + 0.25 * 100.0 * (score.iou or 0)
        return score

    # speech recognised but the subtitle file has no text here (missing cue)
    if asr_text and not sub_text:
        score.status = "speech-no-subs"
        score.text_accuracy = 0.0
        score.wer = 1.0
        score.composite = 0.75 * 0.0 + 0.25 * 100.0 * (score.iou or 0)
        return score

    # both sides have text: reference = what the audio actually says (ASR),
    # hypothesis = what the subtitle file claims (WER counts the subtitle
    # file's substitutions/deletions/insertions relative to the speech).
    wer = float(jiwer.wer(asr_text, sub_text))
    score.status = "scored"
    score.wer = wer
    score.text_accuracy = max(0.0, 1.0 - wer) * 100.0
    score.composite = 0.75 * score.text_accuracy + 0.25 * 100.0 * (score.iou or 0)
    return score


# -------------------------------------------------------------------- drift probe
def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def estimate_drift(
    segments: List[Segment],
    cues: List[Cue],
    wstart: float,
    wend: float,
) -> Optional[float]:
    """Median (subtitle_start - speech_start) over textually matching
    segment/cue pairs inside the window; None when nothing can be paired.

    A positive value means the subtitle cue starts later than the speech it
    describes (subtitles run late); negative means they run early.
    """
    deltas = pair_deltas(segments, cues)
    return _median(deltas) if deltas else None


def pair_deltas(segments: List[Segment], cues: List[Cue]) -> List[float]:
    """One (cue_start - segment_start) offset per textually matched pair."""
    pairs = _match_pairs(segments, cues)
    return [cue_start - seg_start for seg_start, cue_start in pairs]


def _match_pairs(segments: List[Segment], cues: List[Cue]) -> List[Tuple[float, float]]:
    """Pair cues with matching speech segments to estimate timing offsets.

    Matches each cue to its best-matching speech segment. When an ASR segment
    spans multiple subtitle cues (or when a cue appears inside a long segment),
    the speech onset for the cue is interpolated based on word timestamps
    (or linear word pacing), avoiding bogus multi-second offsets from matching
    a later cue directly against the start of a merged segment.

    A pair is only kept when both sides share at least two matching words.
    """
    from difflib import SequenceMatcher

    norm_segs = []
    for s in segments:
        words = normalize(s.text).split()
        if len(words) >= 2:
            norm_segs.append((s, words, set(words)))
    if not norm_segs:
        return []

    norm_cues = []
    for c in cues:
        c_words = normalize(c.text).split()
        if len(c_words) >= 2:
            norm_cues.append((c, c_words, set(c_words)))

    pairs: List[Tuple[float, float]] = []
    used_cues = set()
    for cue, cw, cset in norm_cues:
        if id(cue) in used_cues:
            continue
        best_seg = None
        best_match = None
        best_size = 0

        for seg, sw, sset in norm_segs:
            # temporally near the segment (within 15 s of segment start/end)
            if abs(cue.start - seg.start) > 15.0 and not (
                seg.start - 12.0 <= cue.start <= seg.end + 12.0
            ):
                continue
            common = len(cset & sset)
            if common < 2:
                continue

            sm = SequenceMatcher(None, sw, cw).find_longest_match(
                0, len(sw), 0, len(cw)
            )
            if sm.size >= 2 and sm.size > best_size:
                best_size = sm.size
                best_seg = seg
                best_match = (sm.a, sm.b, sm.size, len(sw), len(cw))

        if best_seg is not None and best_match is not None:
            a, b, size, n_sw, n_cw = best_match
            if getattr(best_seg, "words", None) and a < len(best_seg.words):
                time_at_a = best_seg.words[a][0]
            else:
                seg_step = best_seg.duration / max(1, n_sw)
                time_at_a = best_seg.start + a * seg_step

            cue_step = (
                (cue.duration / max(1, n_cw))
                if (cue.duration > 0 and n_cw > 0)
                else 0.3
            )
            est_cue_start = time_at_a - b * cue_step
            # clamp to segment bounds with 2.0s leeway
            est_cue_start = max(
                best_seg.start - 2.0, min(best_seg.end + 2.0, est_cue_start)
            )
            used_cues.add(id(cue))
            pairs.append((est_cue_start, cue.start))

    return pairs


# ------------------------------------------------------- drift statistics output
# Classification thresholds for the drift offsets measured per window:
#   good:    |average drift| below 100 ms
#   fps:     drift grows (or shrinks) linearly with movie time — the subtitle
#            timeline runs at the wrong speed (frame-rate mismatch)
#   bad:     nothing matched at all, or the spread (std) dwarfs the average
#            (lines off by wildly different amounts / both directions)
#   between the above: consistent drift -> fixable with a constant time shift
GOOD_MAX_ABS_MEAN_S = 0.10   # user requirement: < 100 ms of average drift
GOOD_JITTER_NOTE_S = 0.15    # good results beyond this scatter get a note
CONSISTENT_MAX_CV = 0.5      # std <= 50 % of |mean| -> offsets agree with each other
# ...but only call it erratic once the spread exceeds an absolute floor:
# per-line offset estimates include ASR boundary noise (~0.1-0.3 s), so a
# centred file whose lines scatter by less than ~0.35 s is still a good match.
ERRATIC_MIN_STD_S = 0.35

# fps detection: slope of drift vs movie-time (seconds of drift per second of
# movie). Slope = (subtitle_timebase / movie_timebase) - 1, so e.g. subtitles
# timed for 25 fps PAL played against a 23.976 fps film give ~ +0.041.
FPS_SLOPE_MIN_S_PER_S = 2e-4       # absolute floor (~1.6 s drift per 2 h movie)
FPS_SIGNIFICANCE_SIGMAS = 4.0      # slope must be significant vs fit noise
FPS_SPAN_DRIFT_MIN_S = 1.5         # ... and produce >= this drift across the
                                   # sampled span (windows too close cannot tell
                                   # fps errors apart from constant drift)
# with only two matched windows there are no residuals: require a large slope
FPS_TWO_POINT_SLOPE_MIN_S_PER_S = 2e-3
FPS_TWO_POINT_MIN_SPAN_S = 600.0

CATEGORY_GOOD = "good"
CATEGORY_CONSISTENT = "consistent"
CATEGORY_FPS = "fps"
CATEGORY_BAD = "bad"


def _ols(xs: List[float], ys: List[float]) -> Tuple[float, float, Optional[float], float]:
    """Ordinary least squares y = a + b*x.

    Returns (a, b, sigma_b, s_res). sigma_b is None when there are no
    residuals left (n == 2 or perfectly collinear points).
    """
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    if sxx <= 0:
        return ybar, 0.0, None, 0.0
    b = sxy / sxx
    a = ybar - b * xbar
    if n <= 2:
        return a, b, None, 0.0
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    s_res = (sse / (n - 2)) ** 0.5
    sigma_b = s_res / sxx ** 0.5
    return a, b, sigma_b, s_res


@dataclass(frozen=True)
class DriftSummary:
    """Statistics + classification of the measured drift offsets."""

    mean: Optional[float]          # seconds, + = subtitles late
    std: Optional[float]           # seconds (sample std, 0 for a single delta)
    matched_lines: int             # pooled matched (segment, cue) pairs
    matched_windows: int           # windows that contributed at least one pair
    category: str                  # CATEGORY_GOOD / _CONSISTENT / _FPS / _BAD
    label: str                     # human wording of the category
    note: str = ""                 # e.g. scatter / low-confidence warning
    fix_by: Optional[float] = None  # constant shift (seconds); fps case too
    fix_scale: Optional[float] = None  # time-scale factor (bad fps fix)
    slope_s_per_s: Optional[float] = None    # fitted linear drift slope (fps)
    rate_delta_pct: Optional[float] = None   # implied timeline speed error %

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "label": self.label,
            "mean_s": None if self.mean is None else round(self.mean, 3),
            "std_s": None if self.std is None else round(self.std, 3),
            "matched_lines": self.matched_lines,
            "matched_windows": self.matched_windows,
            "note": self.note,
            "fix_by_s": None if self.fix_by is None else round(self.fix_by, 3),
            "fix_scale": None if self.fix_scale is None else round(self.fix_scale, 6),
            "slope_s_per_s": None if self.slope_s_per_s is None else round(self.slope_s_per_s, 6),
            "rate_delta_pct": None if self.rate_delta_pct is None else round(self.rate_delta_pct, 4),
        }


def summarize_drift(
    deltas: List[float],
    xs: Optional[List[float]] = None,
    matched_windows: int = 0,
    matched_lines: Optional[int] = None,
) -> DriftSummary:
    """Mean/std over the per-window drift samples + classification.

    *deltas* are the *median* matched offsets of each checked window
    (cue_start - speech_start); a positive mean means the subtitles run late
    (cues start after their speech). *xs* holds each window's position on the
    movie timeline (same order as *deltas*): with it we can detect *bad fps*,
    where the drift grows linearly with movie time because the subtitle
    timeline runs at the wrong speed. ``matched_lines`` (default: len(deltas))
    reports how many raw (segment, cue) pairs produced those medians.

    Fixes:
      constant drift -> new_t = t + fix_by            (fix_by = -mean)
      fps mismatch   -> new_t = t * fix_scale + fix_by (fix_scale = 1/(1+slope))
    """
    n = len(deltas)
    if n == 0:
        return DriftSummary(
            mean=None, std=None, matched_lines=matched_lines or 0,
            matched_windows=matched_windows,
            category=CATEGORY_BAD,
            label="no matched text — drift not measurable",
            note="no speech segment could be matched to a subtitle cue",
        )
    if matched_lines is None:
        matched_lines = n
    mean = sum(deltas) / n
    if n > 1:
        variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    abs_mean = abs(mean)

    # --- category "bad fps": drift changes linearly with movie time --------
    # drift(t) ~= a + b*t ; b = (subtitle_timebase / movie_timebase) - 1
    slope: Optional[float] = None
    rate_delta_pct: Optional[float] = None
    fit = None
    if (
        xs is not None
        and len(xs) == n
        and n >= 2
        and max(xs) - min(xs) > 0
    ):
        a, b, sigma_b, _s_res = _ols(xs, deltas)
        slope = b
        span = max(xs) - min(xs)
        drift_change = abs(b) * span
        if n == 2:
            fps_candidate = (
                abs(b) >= FPS_TWO_POINT_SLOPE_MIN_S_PER_S
                and span >= FPS_TWO_POINT_MIN_SPAN_S
                and drift_change >= FPS_SPAN_DRIFT_MIN_S
            )
        else:
            fps_candidate = (
                abs(b) >= FPS_SLOPE_MIN_S_PER_S
                and sigma_b is not None
                and abs(b) >= FPS_SIGNIFICANCE_SIGMAS * sigma_b
                and drift_change >= FPS_SPAN_DRIFT_MIN_S
            )
        if fps_candidate:
            rate = 1.0 / (1.0 + b)          # media time per subtitle second
            rate_delta_pct = (1.0 / (1.0 + b) - 1.0) * 100.0
            scale = 1.0 / (1.0 + b)
            by = -a / (1.0 + b)             # exact: A = (S - a) / (1 + b)
            note = ""
            if n < 3:
                note = "based on only two matched windows — low confidence"
            elif sigma_b is not None and sigma_b > 0:
                note = (
                    f"drift changes by {b * 1000:+.2f} ms per second of movie "
                    f"(slope {b:.6f}, fit noise {sigma_b:.6f})"
                )
            label = (
                f"fps mismatch: drift slope {b:+.6f} s/s — subtitle timeline "
                f"speed off by ≈{abs(rate_delta_pct):.2f}% "
                f"({'runs fast' if b > 0 else 'runs slow'})"
            )
            return DriftSummary(
                mean=mean, std=std, matched_lines=matched_lines,
                matched_windows=matched_windows or n,
                category=CATEGORY_FPS, label=label, note=note,
                fix_by=by, fix_scale=scale,
                slope_s_per_s=b, rate_delta_pct=rate_delta_pct,
            )

    # category: offsets disagree with each other (spread dominates, both in
    # relative and absolute terms)
    if std > max(CONSISTENT_MAX_CV * abs_mean, ERRATIC_MIN_STD_S):
        note = (
            f"per-window offsets vary by more than the usual timing noise "
            f"(std {std:.2f}s vs mean {mean:+.2f}s)"
        )
        return DriftSummary(
            mean=mean, std=std, matched_lines=matched_lines, matched_windows=matched_windows or n,
            category=CATEGORY_BAD,
            label=(
                f"inconsistent drift: mean {mean:+.2f}s, std {std:.2f}s "
                "(windows disagree on the offset)"
            ),
            note=note,
        )

    # category: no systematic offset
    if abs_mean <= GOOD_MAX_ABS_MEAN_S:
        note = ""
        if std > GOOD_JITTER_NOTE_S:
            note = (
                f"average drift is negligible but lines scatter by ±{std:.2f}s "
                "— no single time shift applies"
            )
        return DriftSummary(
            mean=mean, std=std, matched_lines=matched_lines, matched_windows=matched_windows or n,
            category=CATEGORY_GOOD,
            label=(
                "good match: average drift below 100 ms "
                f"(mean {mean * 1000:+.0f} ms)"
            ),
            note=note,
        )

    # category: a consistent, constant time offset
    note = ""
    if n < 3:
        note = f"based on only {n} window(s) — low confidence"
    return DriftSummary(
        mean=mean, std=std, matched_lines=matched_lines, matched_windows=matched_windows or n,
        category=CATEGORY_CONSISTENT,
        label=(
            f"consistent drift: mean {mean:+.2f}s, std {std:.2f}s "
            "(constant offset)"
        ),
        note=note,
        fix_by=-mean,
    )


# ------------------------------------------------------------------- aggregation
def summarize(scores: List[SnippetScore]) -> dict:
    """Aggregate per-window scores into report numbers."""
    counted = [s for s in scores if s.composite is not None]
    if not counted:
        return {
            "windows_scored": 0,
            "text_accuracy_mean": None,
            "iou_mean": None,
            "composite_mean": None,
        }
    texts = [s.text_accuracy for s in counted if s.text_accuracy is not None]
    ious = [s.iou for s in counted if s.iou is not None]
    comps = [s.composite for s in counted if s.composite is not None]
    return {
        "windows_scored": len(counted),
        "text_accuracy_mean": round(sum(texts) / len(texts), 2) if texts else None,
        "iou_mean": round(sum(ious) / len(ious) * 100.0, 2) if ious else None,
        "composite_mean": round(sum(comps) / len(comps), 2) if comps else None,
    }
