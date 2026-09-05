import pytest

from subcheck.sampling import SamplingError, SnippetSampler


def _in_bounds(sampler, snippets):
    for sn in snippets:
        assert 0.0 <= sn.start <= sampler.total - sampler.duration + 1e-9
        assert sn.end <= sampler.total + 1e-9


def _well_spaced(snippets, duration, gap):
    spans = sorted((s.start, s.end) for s in snippets)
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 - b1 >= gap - 1e-9, "windows too close together"


def test_seed_reproducible():
    a = SnippetSampler(5000, seed=42).sample_all()
    b = SnippetSampler(5000, seed=42).sample_all()
    c = SnippetSampler(5000, seed=7).sample_all()
    assert a == b
    assert a != c  # astronomically unlikely to collide


def test_non_overlapping_with_min_gap():
    sampler = SnippetSampler(600, duration=10, count=3, min_gap=30, seed=1)
    snips = sampler.sample_all()
    assert len(snips) == 3
    _in_bounds(sampler, snips)
    _well_spaced(snips, 10, 30)


def test_short_media_gap_shrinks():
    # 3x10s in 40s leaves room for two gaps of at most 5s -> no 30s gap possible
    sampler = SnippetSampler(40, duration=10, count=3, min_gap=30, seed=3)
    snips = sampler.sample_all()
    assert len(snips) == 3
    _well_spaced(snips, 10, 0)
    _in_bounds(sampler, snips)


def test_media_too_short_raises():
    with pytest.raises(SamplingError):
        SnippetSampler(5, duration=10, count=1).sample_all()


def test_draw_extra_avoids_reserved():
    sampler = SnippetSampler(1000, duration=10, count=3, min_gap=30, seed=5)
    initial = sampler.sample_all()
    extra = sampler.draw_extra()
    assert extra is not None
    all_snips = list(initial) + [extra]
    _well_spaced(all_snips, 10, 0)
    # reserving everything exhausts the sampler eventually
    for _ in range(200):
        if sampler.draw_extra() is None:
            break
    else:
        raise AssertionError("sampler never exhausted")
