import pytest

from subcheck.scoring import (
    score_window,
    summarize,
    union_duration,
    windowed_candidate_text,
)
from subcheck.srtio import Cue
from subcheck.transcribe import Segment


def _cue(idx, start, end, text):
    return Cue(idx, start, end, text)


def _win(segs, cues, wstart=100.0, wend=110.0):
    return score_window(segs, cues, wstart, wend)


def test_silent_match_is_perfect():
    sc = _win([], [])
    assert sc.status == "silent-match"
    assert sc.text_accuracy == 100.0 and sc.iou == 1.0 and sc.composite == 100.0


def test_perfect_text_and_timing():
    segs = [Segment(100.5, 103.2, "Hello there, my friend.")]
    cues = [_cue(1, 100.5, 103.2, "Hello there my friend")]
    sc = _win(segs, cues)
    assert sc.status == "scored"
    assert sc.text_accuracy == 100.0
    assert sc.iou == 1.0
    assert sc.composite == 100.0


def test_known_wer_value():
    segs = [Segment(100.0, 103.0, "a b c")]          # ASR (reference)
    cues = [_cue(1, 100.0, 103.0, "a c")]            # subtitle (hypothesis)
    sc = _win(segs, cues)
    assert sc.wer == 1 / 3
    assert sc.text_accuracy == pytest.approx(100 * (1 - 1 / 3))


def test_speech_but_no_subtitles():
    segs = [Segment(100.0, 104.0, "important dialogue")]
    sc = _win(segs, [])
    assert sc.status == "speech-no-subs"
    assert sc.text_accuracy == 0.0
    assert sc.iou == 0.0


def test_subtitles_but_no_speech():
    cues = [_cue(1, 100.0, 104.0, "claimed dialogue")]
    sc = _win([], cues)
    assert sc.status == "subs-no-speech"
    assert sc.text_accuracy == 0.0


def test_iou_partial_overlap():
    segs = [Segment(100.0, 105.0, "one two three")]      # speech [100,105]
    cues = [_cue(1, 103.0, 108.0, "one two three")]      # subs  [103,108]
    sc = _win(segs, cues)
    # inter [103,105] = 2s ; union [100,108] = 8s
    assert sc.iou == pytest.approx(2 / 8)


def test_windowed_candidate_trims_straddling_cues():
    # cue starts before the window -> only words overlapping the region remain
    cue = _cue(1, 0.0, 10.0, "one two three four five")  # words at 2s intervals
    text = windowed_candidate_text([cue], 7.0, 17.0)
    assert text == "four five"


def test_windowed_candidate_narrows_to_speech_span():
    cue = _cue(1, 0.0, 10.0, "one two three four five")
    # only the tail of the window contains speech
    text = windowed_candidate_text([cue], 0.0, 10.0, speech_span=(6.0, 10.0))
    assert text == "four five"
    # no recognised speech at all -> nothing is expected
    assert windowed_candidate_text([cue], 0.0, 10.0, speech_span=(3.0, 3.0)) == ""


def test_span_trim_keeps_high_score_for_mild_drift():
    # window 100-110; speech occupies 106-109; subtitle says the same line but
    # its cue starts 1.5 s late; preceding/following cues are outside the span
    segs = [Segment(106.0, 109.0, "so you open the door")]
    cues = [
        _cue(1, 103.0, 105.5, "give me a month come around"),
        _cue(2, 105.5, 109.5, "so you open the door and"),
    ]
    sc = _win(segs, cues)
    assert sc.status == "scored"
    assert sc.text_accuracy > 50.0  # boundary filler no longer wrecks the WER


def test_drift_estimate_positive_when_subs_late():
    segs = [Segment(100.0, 102.0, "line one")]
    cues = [_cue(1, 103.0, 105.0, "line one")]
    sc = _win(segs, cues)
    assert sc.drift_seconds is not None
    assert sc.drift_seconds == pytest.approx(3.0)  # cue_start - seg_start


def test_union_duration_merges():
    assert union_duration([(0, 2), (1, 3), (5, 6)]) == pytest.approx(4.0)


def test_summarize_counts_silent_as_scored():
    s1 = _win([Segment(0.5, 1.5, "hi there")], [_cue(1, 0.5, 1.5, "hi there")])
    s2 = _win([], [])
    agg = summarize([s1, s2])
    assert agg["windows_scored"] == 2
    assert agg["text_accuracy_mean"] == 100.0
    assert agg["composite_mean"] == 100.0
