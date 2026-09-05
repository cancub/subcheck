from subcheck import cli


def test_histogram_empty():
    lines = cli.drift_histogram_lines([])
    assert lines == ["(no per-snippet drift samples)"]


def test_histogram_counts_and_bins():
    lines = cli.drift_histogram_lines([0.8, 1.0, 1.2])
    joined = "\n".join(lines)
    # every sample lands in exactly one bucket
    assert sum(line.count("#") for line in lines) == 3
    assert len(lines) <= 16
    # labels carry explicit signs; bins run from +0.80 up past +1.20
    assert "+0.80" in joined and "+1.20" in joined and "+1.25" in joined


def test_histogram_straddles_zero():
    lines = cli.drift_histogram_lines([-1.2, -0.3, 0.4])
    joined = "\n".join(lines)
    # bin grid is anchored at round multiples: a bucket starts at +0.00
    assert any(line.startswith("    +0.00") for line in lines)
    assert sum(line.count("#") for line in lines) == 3
    assert "-1.20" in joined and "+0.40" in joined


def test_histogram_wide_spread_uses_wide_bins():
    lines = cli.drift_histogram_lines([-300.0, 0.0, 200.0])
    assert len(lines) <= 16
    assert sum(line.count("#") for line in lines) == 3


def test_drift_samples_of_report(tmp_path):
    import json

    from subcheck.runner import rescore_windows

    srt = tmp_path / "t.srt"
    srt.write_text(
        "1\n00:01:41,000 --> 00:01:45,000\nHello there, my friend.\n",
        encoding="utf-8",
    )
    wj = tmp_path / "windows.json"
    wj.write_text(
        json.dumps(
            {
                "media": "movie.mp4",
                "total_duration_s": 3600.0,
                "snippet_duration_s": 10.0,
                "settings": {"snippets": 1, "model": "test"},
                "windows": [
                    {"start": 100.0, "end": 110.0, "status": "scored",
                     "wav": "",
                     "asr_segments": [
                         {"start": 101.0, "end": 105.0,
                          "text": "Hello there, my friend."}]},
                    {"start": 2000.0, "end": 2010.0, "status": "scored",
                     "wav": "",
                     "asr_segments": []},  # no speech -> no drift sample
                ],
            }
        ),
        encoding="utf-8",
    )
    report = rescore_windows(str(wj), str(srt))
    samples = cli.drift_samples_of(report)
    assert samples == [0.0]  # second window had no speech/drift
