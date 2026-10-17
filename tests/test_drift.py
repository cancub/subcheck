import pytest

from subcheck.scoring import (
    CATEGORY_BAD,
    CATEGORY_CONSISTENT,
    CATEGORY_FPS,
    CATEGORY_GOOD,
    _ols,
    pair_deltas,
    summarize_drift,
)
from subcheck.srtio import Cue
from subcheck.transcribe import Segment


def _seg(start, end, text):
    return Segment(start, end, text)


def _cue(idx, start, end, text):
    return Cue(idx, start, end, text)


# ------------------------------------------------------------------ fps detection
def test_growing_drift_is_classified_as_bad_fps():
    # drift ~ 0.08 % per second of movie -> at 60/3500/7300 s the offsets are
    # ~0.05/2.8/5.8 s and clearly growing with movie time
    xs = [60.0, 3500.0, 7300.0]
    deltas = [x * 8e-4 for x in xs]
    ds = summarize_drift(deltas, xs=xs)
    assert ds.category == CATEGORY_FPS
    assert ds.slope_s_per_s == pytest.approx(8e-4, abs=1e-9)
    assert ds.rate_delta_pct is not None
    assert ds.fix_scale == pytest.approx(1 / (1 + 8e-4), rel=1e-6)
    assert "fps" in ds.label


def test_linear_drift_with_constant_offset_is_fps_not_consistent():
    # true subtitle error = +1.1 s constant + 0.04 %/s linear (PAL-like drift)
    xs = [500.0, 3000.0, 6000.0]
    deltas = [1.1 + x * 4e-4 for x in xs]
    ds = summarize_drift(deltas, xs=xs)
    assert ds.category == CATEGORY_FPS
    assert ds.slope_s_per_s == pytest.approx(4e-4, abs=1e-9)
    assert ds.fix_by is not None  # constant part of the fix
    # emulate file times S = A + drift and apply the suggested fix:
    # S' = S*fix_scale + fix_by must land back on the true times A = xs
    for x, d in zip(xs, deltas):
        fixed = (x + d) * ds.fix_scale + ds.fix_by
        assert fixed == pytest.approx(x, abs=1e-6)


def test_constant_drift_is_not_fps():
    xs = [1000.0, 3000.0, 5000.0]
    ds = summarize_drift([1.0, 1.1, 1.05], xs=xs)
    assert ds.category == CATEGORY_CONSISTENT
    assert ds.fix_scale is None


def test_fps_requires_minimum_slope():
    # slope 1.4e-5 (0.1 s over 2 h) is below the floor -> constant/good logic
    xs = [0.0, 3500.0, 7000.0]
    ds = summarize_drift([0.05, 0.10, 0.15], xs=xs)
    assert ds.category != CATEGORY_FPS


def test_fps_two_point_rule_needs_big_slope_and_span():
    # two far-apart windows with a huge slope -> fps
    ds = summarize_drift([0.5, 7.0], xs=[100.0, 3100.0])
    assert ds.category == CATEGORY_FPS
    assert "low confidence" in ds.note
    # same slope over a short span cannot be distinguished from constant drift
    ds2 = summarize_drift([0.5, 0.9], xs=[100.0, 300.0])
    assert ds2.category != CATEGORY_FPS


def test_fps_ignored_without_positions():
    ds = summarize_drift([0.05, 2.8, 5.8])
    assert ds.category != CATEGORY_FPS  # no xs -> constant/erratic logic only


def test_drift_pair_found_beyond_window_edge():
    # the cue has drifted just past the window end: text scoring sees nothing
    # (speech-no-subs) but drift pairing must still find it via the cue pool
    from subcheck.scoring import score_window

    segs = [_seg(100.5, 103.0, "open the door now")]
    far_cue = [_cue(9, 104.0, 107.0, "open the door now")]
    sc = score_window(segs, [], 100.0, 103.5, cue_pool=far_cue)
    assert sc.status == "speech-no-subs"
    assert sc.drift_deltas == [3.5]
    # without a cue pool the pair cannot be formed
    sc2 = score_window(segs, [], 100.0, 103.5)
    assert sc2.drift_deltas == []


def test_ols_roundtrip():
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0 + 0.5 * x for x in xs]
    a, b, sigma, s_res = _ols(xs, ys)
    assert a == pytest.approx(2.0)
    assert b == pytest.approx(0.5)
    assert sigma == 0.0 and s_res == 0.0


# ------------------------------------------------------------------ classification
def test_no_matches_is_bad():
    ds = summarize_drift([], matched_windows=0)
    assert ds.category == CATEGORY_BAD
    assert ds.mean is None and ds.std is None
    assert "no matched" in ds.label


def test_perfect_zero_drift_is_good():
    ds = summarize_drift([0.0, 0.0, 0.0, 0.001], matched_windows=2)
    assert ds.category == CATEGORY_GOOD
    assert abs(ds.mean) <= 0.001


def test_small_average_drift_is_good():
    # average +40 ms with tight spread -> good match (< 100 ms average)
    ds = summarize_drift([0.02, 0.03, 0.06, 0.05], matched_windows=2)
    assert ds.category == CATEGORY_GOOD
    assert "100 ms" in ds.label


def test_centred_file_with_moderate_scatter_is_good_with_note():
    # after applying the fix for two windows off by +0.22/-0.22 s the mean is 0
    ds = summarize_drift([0.22, -0.22], matched_windows=2)
    assert ds.category == CATEGORY_GOOD
    assert "scatter" in ds.note
    assert ds.fix_by is None


def test_consistent_drift_is_fixable():
    # most lines off by the same ~1.0-1.2 s -> consistent drift, mean ~1.1 s
    ds = summarize_drift([1.0, 1.1, 1.2, 1.05, 1.15], matched_windows=3)
    assert ds.category == CATEGORY_CONSISTENT
    assert ds.mean == pytest.approx(1.10, abs=0.01)
    assert ds.std == pytest.approx(0.079, abs=0.01)
    # subtitles run late -> the fix shifts them earlier (negative by)
    assert ds.fix_by == pytest.approx(-ds.mean)
    assert "+1.10" in ds.label and "std" in ds.label  # raw numbers, no drama


def test_consistent_early_drift_fix_sign():
    ds = summarize_drift([-0.9, -1.0, -1.1], matched_windows=1)
    assert ds.category == CATEGORY_CONSISTENT
    assert ds.mean < 0
    assert ds.fix_by > 0  # move the subtitles later


def test_erratic_drift_is_bad():
    # offsets swing in both directions / very different amounts
    ds = summarize_drift([0.3, -0.4, 1.1, -0.9, 0.6], matched_windows=2)
    assert ds.category == CATEGORY_BAD
    assert ds.fix_by is None
    assert "inconsistent" in ds.label


def test_small_mean_but_large_spread_is_bad():
    ds = summarize_drift([0.0, 0.6, -0.5, 0.1], matched_windows=2)
    assert ds.category == CATEGORY_BAD  # std (~0.45s) dwarfs the +0.05 s mean


def test_moderate_scatter_around_zero_is_not_erratic():
    # ±0.22 s residual scatter after centring is inside the ASR-noise floor
    ds = summarize_drift([0.22, -0.22], matched_windows=2)
    assert ds.category == CATEGORY_GOOD


def test_low_confidence_note_for_consistent_with_few_windows():
    ds = summarize_drift([1.2, 1.1], matched_windows=2)
    assert ds.category == CATEGORY_CONSISTENT
    assert "low confidence" in ds.note


def test_single_window_consistent_has_low_confidence_note():
    ds = summarize_drift([1.2], matched_windows=1)
    assert ds.category == CATEGORY_CONSISTENT
    assert "low confidence" in ds.note
    assert ds.std == 0.0


# ------------------------------------------------------------------ pair matching
def test_pair_requires_more_than_one_shared_word():
    segs = [_seg(100.0, 103.0, "open the door")]
    # only "the" is shared -> no pair (accidental filler match)
    assert pair_deltas(segs, [_cue(1, 101.0, 104.0, "the magician opens")]) == []
    # two shared words -> paired, delta = cue_start - seg_start
    assert pair_deltas(segs, [_cue(1, 101.0, 104.0, "open the door quickly")]) == [1.0]


def test_tiny_segment_matches_on_all_its_words():
    segs = [_seg(100.0, 101.5, "give month")]
    assert pair_deltas(segs, [_cue(1, 101.0, 103.0, "give month please")]) == [1.0]
    assert pair_deltas(segs, [_cue(1, 101.0, 103.0, "give nothing")]) == []


def test_single_word_segments_never_pair():
    # one shared interjection ("oh") must not count as a timing match
    segs = [_seg(100.0, 102.0, "oh")]
    assert pair_deltas(segs, [_cue(1, 101.0, 104.0, "oh man what happening")]) == []
    segs2 = [_seg(100.0, 102.0, "right yes")]
    assert pair_deltas(segs2, [_cue(1, 101.0, 104.0, "right yes")]) == [1.0]
