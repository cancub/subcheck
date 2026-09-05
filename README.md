# subcheck — is this subtitle file accurate for this video?

`subcheck` checks an SRT file against its video by **sampling a few random
10-second windows**, transcribing what is actually spoken in each window with
[**faster-whisper**](https://github.com/SYSTRAN/faster-whisper), and comparing
the transcript against the subtitle file — both **text** (word error rate) and
**timing** (temporal overlap). The output is a per-window and aggregate score
plus flags for the classic failure modes (missing dialogue, music interludes,
constant sync drift).

It is built for remote videos as much as local files: probing and slicing use
`ffprobe`/`ffmpeg` with HTTP range requests, so a multi-gigabyte URL is never
downloaded in full.

## Pipeline

| # | Stage | Tool | What it does |
|---|-------|------|--------------|
| 1 | Probing | `ffprobe` | total duration `T_total` of the media (metadata only) |
| 2 | Sampling | `random` | `N` non-overlapping 10 s windows `[t_i, t_i+10]`, ≥ `min_gap` apart |
| 3 | Slicing | `ffmpeg` | each window → 16 kHz mono PCM WAV (`-ss` before `-i` → keyframe seek over the network, no full download) |
| 4 | Inference | faster-whisper | WAV → local segments with timestamps (Silero VAD on) |
| 5 | Offsetting | python | `T_global = t_i + T_local` (local ASR timeline → media timeline) |
| 6 | Comparison | `pysrt` + `jiwer` | WER text accuracy, temporal IoU, composite score |

### Scoring (per window)

* **Text accuracy** — `WER = (S+D+I)/N` computed with the ASR transcript as the
  reference (what the audio says) and the subtitle text as the hypothesis
  (what the file claims); `accuracy = max(0, 1 − WER) × 100 %`.
  Both sides go through the same normalisation: lower-case, formatting tags
  and acoustic brackets (`[music]`, `(applause)`) removed, punctuation
  dropped, numerals expanded (`10` → `ten`, `2nd` → `second`), whitespace
  collapsed.
* **Temporal IoU** — speech intervals (ASR segments ∩ window) vs subtitle
  intervals (cues ∩ window): `IoU = |speech ∩ subs| / |speech ∪ subs|` using
  exact 1-D interval algebra.
* **Composite** — `0.75 × text accuracy + 0.25 × IoU × 100`.
* **Offset estimate** — median `cue_start − speech_start` over textually
  matched segment/cue pairs (reported even when the texts differ).

Silent windows where both sides are empty count as a 100 % match. Windows
where the subtitle file shows text but no speech is recognised (music
interludes, …) are **re-rolled** up to `--max-rerolls` times instead of being
penalised. Straddling cues (starting before the window / ending after it)
keep their text, but the candidate text is trimmed to the **active speech
span** (first→last ASR segment in the window): subtitle words are mapped
linearly onto their cue's duration and only words overlapping that span are
compared — so boundary words and words describing audio outside the region do
not distort the WER, while genuine text mismatches over the audible span
still count.

### Sync-drift flag

When a window scores > 90 % text accuracy but < 20 % temporal IoU, the file is
flagged for **constant time-offset drift** (accurate text, shifted by ±N s).
A median offset estimate (subtitle-start minus speech-start over matched
segment/cue pairs) is reported too — positive means the subtitles run late.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

`ffmpeg`/`ffprobe` must be on `PATH` (or point at a static build — used in
this repo's tests — via `SUBCHECK_FFMPEG_BIN` / `SUBCHECK_FFPROBE_BIN`).
Model weights download automatically from Hugging Face on first run into the
HF cache (`$HF_HOME`, or `--model-dir`).

### GPU (CUDA) — e.g. a GTX 4070 Ti

`ctranslate2` needs cuBLAS 12 + cuDNN 9 (see the
[faster-whisper README](https://github.com/SYSTRAN/faster-whisper)); the pip
packages work well on Linux:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
export LD_LIBRARY_PATH=$(python3 -c \
  'import os, nvidia.cublas.lib, nvidia.cudnn.lib; \
   print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + \
         os.path.dirname(nvidia.cudnn.lib.__file__))')
```

`subcheck` auto-selects `cuda` + `float16` when a GPU is present
(`--device cuda --compute-type int8_float16` are good knobs; `large-v3`
defaults to the full-precision float16 weights, so ≥ 6 GB VRAM is comfortable,
12 GB on the 4070 Ti is plenty).

## Usage

```bash
# video URL or local file + the SRT to check (auto-detected next to the video
# when omitted)
subcheck "https://example.com/movie.mp4" movie.srt

# a real example from this repo's validation:
subcheck "The.Prestige.2006.1080p.BluRay.x264-[YTS.AM].mp4" \
         "The.Prestige.2006.BluRay.1080p.AC3.x264.dxva-BTT.ENG.srt"

# reproducibility + machine-readable output
subcheck movie.mp4 movie.srt --seed 42 --json report.json

# smaller / faster model on a weak machine
subcheck movie.mp4 movie.srt --model medium --device cpu

# longer check (more windows) with the WAVs kept for inspection
subcheck movie.mp4 movie.srt --snippets 6 --workdir ./snippets
```

Full options (`subcheck --help`): `--snippets` (3), `--duration` (10 s),
`--min-gap` (30 s), `--seed`, `--model` (`large-v3`, also
`tiny|base|small|medium|large-v2|distil-*` or any HF repo id), `--device`
(`auto|cpu|cuda`), `--compute-type`, `--beam-size`, `--language`,
`--max-rerolls` (3), `--no-expand-numbers`, `--workdir`, `--json FILE`.

### Drift statistics and classification

Every transcribed speech segment that can be textually matched to a subtitle
cue contributes one offset sample (`cue_start − segment_start`; positive =
the subtitles run late). The report prints the **mean and standard deviation
of those offsets** across all checked windows and classifies the result:

* **good match** — average drift under 100 ms: nothing to fix.
* **consistent drift** — average offset beyond 100 ms but the lines agree
  with each other (e.g. everything ~1.0–1.2 s late). The tool prints a ready
  fix command:

  ```bash
  subcheck shift "movie.srt" --by -1.051
  ```
* **bad fps** — drift gets *progressively worse*: a least-squares fit of
  drift vs movie-time finds a significant slope (seconds of drift per second
  of movie), i.e. the subtitle timeline runs at the wrong speed (classic
  PAL/NTSC timebase mix-ups). The report shows the slope, the implied speed
  error in % and a timebase fix:

  ```bash
  subcheck shift "movie.srt" --scale 0.999255 --by -1.192
  ```
  (`new_t = t × --scale + --by`; here the timeline was ~0.07 % fast.)
* **bad match** — offsets vary wildly (std large compared to the mean —
  beyond 50 % of it *and* over an absolute floor of 0.35 s, so ordinary ASR
  boundary noise does not trigger it — lines disagree on amount *and*
  direction) or **no line could be matched at all** (unrelated text /
  different language / subtitles shifted so far that no cue text overlaps the
  sampled speech).

Thresholds live at the top of `subcheck/scoring.py`
(`GOOD_MAX_ABS_MEAN_S`, `CONSISTENT_MAX_CV`, `ERRATIC_MIN_STD_S`,
`FPS_SLOPE_MIN_S_PER_S`, `FPS_SIGNIFICANCE_SIGMAS`, `FPS_SPAN_DRIFT_MIN_S`);
each checked window contributes the median offset of its matched lines.

Note: drift matching can follow cues up to ~12 s away from the sampled
speech, which covers timebase errors up to ~0.1–0.15 % across a 2 h movie
(the common 23.976 ↔ 24 fps case). For larger mismatches (e.g. full PAL
25 fps vs 23.976 fps ≈ 4 %) only windows in the first minutes of the film
still match; the rest will be flagged as uncaptioned/erratic, so run such a
file with extra early windows to characterise the slope.

### Other subcommands

* `subcheck rescore windows.json other.srt` — re-score previously
  transcribed windows (from `--json FILE`) against another subtitle file
  without re-running ffmpeg/ASR. Handy for comparing subtitle candidates or
  verifying that a `subcheck shift` really fixed the drift:

  ```bash
  subcheck movie.mp4 movie.srt --seed 42 --json first.json
  subcheck rescore first.json movie.fixed.srt --json fixed.json
  ```


## Interpreting the output

Per window: `text` accuracy, temporal `iou`, composite `score`, optional
`drift ±s`, plus the ASR vs SRT text when they disagree. Aggregate over all
windows, then a verdict flag. The check is **sampling-based**: it certifies
the sampled regions, not the whole file — increase `--snippets` for more
confidence.

## Validation

End-to-end runs on *The Prestige* (1080p rip, 2 h 10 m, ENG SRT, seed 1234,
3 × 10 s windows, faster-whisper large-v3 int8 on CPU):

| Subtitle file under test | text acc. | IoU | composite | detected by the tool |
|---|---|---|---|---|
| SRT built from the ASR output itself (positive control) | 100 % | 100 % | 100 % | "looks ACCURATE" |
| Real `dxva-BTT` ENG SRT | 65.7 % | 53.8 % | 62.7 % | "does NOT match well", consistent drift mean +1.05 s (std 0.31 s) → fix suggested |
| Same SRT shifted +60 s (broken-sync control) | 6.7 % | 19.6 % | 9.9 % | "does NOT match well", missing-caption flags, 2 re-rolls |
| Timeline scaled ×1.0008 (bad-fps control) | — | — | — | "bad fps": slope +0.00075 s/s (~0.07 % fast) → scale fix; after applying it: good match (mean −6 ms) |

The middle row is a real finding, cross-checked by transcribing wider
reference chunks around the sampled windows: that SRT runs ~0.8–1.4 s late
against this particular rip's audio (e.g. "Are you watching closely?" is
audible at 57.3 s but cued at 58.42 s) and carries at least one misplaced cue
("Come around." is claimed at 3452 s where the audio is silent). So a
~63 % composite here genuinely reflects a mis-synced subtitle file, not tool
bias: perfect timing scores 100 %, grossly shifted timing collapses to ~10 %.

## Layout

```
subcheck/            the package
  probe.py           stage 1: ffprobe duration
  sampling.py        stage 2: window sampling + re-rolls
  slicing.py         stage 3: ffmpeg WAV extraction
  transcribe.py      stage 4/5: faster-whisper + offsetting
  srtio.py           SRT parsing/windowing (pysrt)
  normalize.py       shared text normalisation
  scoring.py         WER / IoU / composite / drift estimation
  runner.py          orchestration + report building
  cli.py             CLI
tests/               unit tests (pytest)
```

## Limitations

* Accuracy of the whole check is bounded by the ASR: `large-v3` is very good
  on clear film dialogue, but music, accents and noise degrade it — treat
  borderline scores with caution.
* Reference SRT only — `.vtt`/`.ass` files must be converted first (ass2srt
  etc.).
* Windows shorter than the minimum gap required for `N` windows make the gap
  shrink automatically; media shorter than one window cannot be checked.
