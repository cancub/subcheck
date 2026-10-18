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

### Drift output

Drift is reported as raw numbers — per-window values in the ASCII histogram,
the aggregate **mean and standard deviation** across the sampled windows, and
a one-line classification (good / consistent / fps / no match) — plus a fix
command only when a constant or timebase correction applies. Positive drift
means the subtitles run late.

## System Prerequisites

1. **FFmpeg**: `ffmpeg` and `ffprobe` must be installed and on `PATH`.
   - Ubuntu / Debian: `sudo apt update && sudo apt install -y ffmpeg`
   - Arch Linux: `sudo pacman -S ffmpeg`
   - macOS (Homebrew): `brew install ffmpeg`
   - Custom / static builds: set `SUBCHECK_FFMPEG_BIN` and `SUBCHECK_FFPROBE_BIN` environment variables.

2. **NVIDIA Driver (GPU only)**:
   - For GPU acceleration via CUDA 12, an NVIDIA GPU with driver ≥ 525 (recommended ≥ 550) is required.
   - Verify with: `nvidia-smi`

## Install

```bash
python -m venv .venv && . .venv/bin/activate

# Base install (CPU inference):
pip install -e .

# With GPU (CUDA 12) support:
pip install -e ".[cuda]"

# With OpenSubtitles support for `autosub`:
pip install -e ".[autosub]"

# Or install all features (GPU + autosub):
pip install -e ".[all]"
```

### GPU (CUDA 12) Setup & Verification

`ctranslate2` requires cuBLAS 12 and cuDNN 9 runtimes. Installing `subcheck[cuda]` supplies these packages (`nvidia-cublas-cu12` and `nvidia-cudnn-cu12`).

`subcheck` automatically discovers and preloads these libraries into memory when running inside your virtual environment, so no manual library path configuration is usually necessary.

If you are running external scripts or tools in your shell that directly import `ctranslate2`, you can export their paths to `LD_LIBRARY_PATH` (note: Python 3.12-safe via `__path__[0]`):

```bash
export LD_LIBRARY_PATH=$(python3 -c \
  'import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; \
   print(f"{a.__path__[0]}:{b.__path__[0]}")'):$LD_LIBRARY_PATH
```

**Verify GPU acceleration:**
```bash
python3 -c "import ctranslate2; print('CUDA available:', ctranslate2.get_cuda_device_count() > 0)"
```

`subcheck` auto-selects `cuda` + `float16` when a GPU is present (`--device cuda --compute-type int8_float16` are good knobs; `large-v3` defaults to full-precision float16 weights, so ≥ 6 GB VRAM is comfortable, 12 GB on a 4070 Ti is plenty).

### Model Weights & Caching

Whisper model weights are downloaded from Hugging Face on first use into the standard Hugging Face cache directory (`~/.cache/huggingface/hub/`, or `$SUBCHECK_MODEL_DIR` / `$HF_HOME` / `--model-dir`). Silero VAD weights (~2 MB) are also cached automatically.

To avoid waiting for the initial download on first run (especially for the ~3.1 GB `large-v3` model), you can pre-download weights ahead of time:

```bash
# Pre-download default large-v3 weights:
subcheck download-model

# Or download a specific model:
subcheck download-model --model medium
```

| Model | Weights on Disk | Minimum VRAM (GPU) | Recommended Use Case |
|---|---|---|---|
| `large-v3` *(default)* | ~3.1 GB | ≥ 6 GB (float16) / ≥ 4 GB (int8) | Best accuracy across all accents & background noise |
| `distil-large-v3` | ~1.5 GB | ≥ 4 GB | Faster inference with near-large accuracy |
| `medium` | ~1.5 GB | ≥ 4 GB | Balanced speed and multi-language accuracy |
| `small` | ~485 MB | ≥ 2 GB | Fast checks on low-end GPUs or modern CPUs |
| `base` | ~145 MB | ≥ 1 GB | Very fast verification |
| `tiny` | ~75 MB | < 1 GB | Instant smoke testing and lightweight checks |

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

Full options (`subcheck --help`): `--snippets` (10), `--duration` (10 s),
`--min-gap` (30 s), `--seed`, `--model` (`large-v3`, also
`tiny|base|small|medium|large-v2|distil-*` or any HF repo id), `--device`
(`auto|cpu|cuda`), `--compute-type`, `--beam-size`, `--language`,
`--max-rerolls` (3), `--no-expand-numbers`, `--workdir`, `--json FILE`,
`--embedded`, `--sub-stream INDEX`, `--sub-language CODE`.

### Drift statistics and classification

Every transcribed speech segment that can be textually matched to a subtitle
cue contributes one offset sample (`cue_start − segment_start`; positive =
the subtitles run late). The report prints the **mean and standard deviation
of those offsets** across all checked windows and classifies the result.
Every command that measures drift — `check`, `rescore`, `compare` and
`autosub` — also prints an ASCII **histogram of the per-snippet drift values**
(one bucket row per second range, `#` per window in that bucket; bins are
anchored so `+0.00` marks the zero boundary; windows without a measurable
drift — silent/music or no text match — are excluded and counted in the
header). In `compare`/`autosub` each file/candidate gets its own histogram
over the shared snippets, and the per-candidate values are stored in the
`--json` output (`drift_samples`).

The classification buckets are:

* **good match** — average drift under 100 ms: nothing to fix.
* **consistent drift** — mean offset beyond 100 ms with a small spread
  (constant offset). The tool prints a ready fix command:

  ```bash
  subcheck shift "movie.srt" --by -1.051
  ```
* **fps mismatch** — a least-squares fit of drift vs movie-time finds a
  significant slope (seconds of drift per second of movie): the subtitle
  timeline runs at the wrong speed (classic PAL/NTSC timebase mix-ups). The
  report shows the slope, the implied speed error in % and a timebase fix:

  ```bash
  subcheck shift "movie.srt" --scale 0.999255 --by -1.192
  ```
  (`new_t = t × --scale + --by`; here the timeline was ~0.07 % fast.)
* **bad match** — offsets vary wildly (std large compared to the mean —
  beyond 50 % of it *and* over an absolute floor of 0.35 s, so ordinary ASR
  boundary noise does not trigger it — windows disagree on the offset) or
  **no line could be matched at all** (drift not measurable: no cue text
  overlaps the sampled speech).

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

### Fixing subtitles — `subcheck shift` and its two options

`subcheck shift` rewrites every cue time with one general transform:

```
new_t = t × --scale + --by          (defaults: --scale 1.0, --by 0.0)
```

The two options fix the two *different* timing defects the check can find:

| Option | What it does | Fixes | Example |
|---|---|---|---|
| `--by SECONDS` | **constant shift**: add the same amount to every cue time (negative = earlier, positive = later) | *consistent drift* — every line is off by the same offset, e.g. all cues 0.99 s late | `--by -0.989` |
| `--scale FACTOR` | **timebase (fps) scaling**: multiply every cue time by a factor | *bad fps* — the subtitle timeline runs at the wrong speed, so drift grows over time | `--scale 0.999255` |

Why the scale factor exists: when subtitles are timed for a version of the
movie that runs at a different frame rate, their cue times are wrong by a
*ratio*, not by a constant. For example, subtitles timed for a PAL 25 fps
transfer need `--scale 0.95904` (= 23.976 ÷ 25) to play correctly against a
23.976 fps movie — every cue moves ~4.1 % earlier, so the correction grows
with the cue's position in the film. A `bad fps` classification reports the
fitted values directly and prints the ready-made command:

```bash
subcheck shift "movie.srt" --scale 0.999255 --by -1.192
# new_t = t × 0.999255 + (-1.192)     timeline was ~0.07 % fast and ~1.2 s late
```

The two options combine in one command (scale is applied first, then the
shift) and each defaults to a no-op, so either can be used alone. Output
defaults to `<name>.fixed.srt` next to the input (`--out PATH` to choose);
with `--in-place` (`-i`, `--overwrite`), the file is updated in place
(an `.srt` file is overwritten directly; for a video container like `.mkv`,
the subtitles are remuxed in place using a safe atomic replace). Verify the fix
by re-scoring the result (below).

### Comparing subtitle files — `subcheck compare`

Pick the best of several SRT candidates for the same video. Every file is
scored against the **identical** windows, so the ranking is apples-to-apples:

```bash
# fresh windows: one probe/slice/ASR pass, then every SRT is scored on them
subcheck compare movie.mp4 a.srt b.srt c.srt --snippets 4

# reuse the windows of a previous check (no media access / no ASR)
subcheck check movie.mp4 a.srt --seed 42 --json windows.json
subcheck compare --from windows.json a.srt a.fixed.srt b.srt
```

Output: a ranked table (text accuracy, temporal IoU, composite score, drift
classification per file), the fix command for every fixable file, and the
best match. `--json FILE` writes the full detail (shared windows plus the
per-file, per-window scores).

Notes:

* windows where no speech was recognised count as perfect (100 %) for every
  file that has no text there; a file that does show text in a silent window
  is penalised for that window — identically for all candidates.
* live mode deliberately does **not** re-roll windows per file: candidates
  must all face the same windows. Pick `--snippets` big enough for a
  confident ranking.
* unreadable/empty SRTs are reported as errors and skipped; the remaining
  files are still ranked (exit code 1 when every file failed).

### End-to-end: `subcheck autosub`

One command for the whole pipeline — fetch candidates from OpenSubtitles,
verify them against the video's own audio, fix the best one, embed it:

```bash
subcheck autosub movie.mkv                    # full auto, then muxes the winner
subcheck autosub movie.mkv --limit 3 --no-embed --json report.json
subcheck autosub show.mkv --language en --refresh   # ignore caches (redo pulls)
```

How it works:

1. **Fetch** — the title/season/episode are derived from the file name
   (`Steven.Universe.S01E23....mkv` → query `Steven Universe` S01E23; override
   with `--title/--season/--episode`). Credentials come from `.env`
   (`OPENSUBTITLES_API_KEY/_USERNAME/_PASSWORD`, see `.env.example`) or
   `~/.config/opensubtitlescom/config.json`. Searches and downloads are
   cached under `.cache/opensubtitles/` — a cache hit costs **0** of the
   20-downloads/day quota. `--refresh` bypasses both caches (use it when a
   previous pull gave bad subs), and a corrupt cached file is purged and
   re-downloaded automatically. **Quota visibility:** when an API session is
   opened the account status is printed (level, downloads used/allowed,
   remaining, reset time) and refreshed after every download — e.g.
   `quota: bob (User) — 3 of 20 downloads used today, 17 remaining, quota
   resets at 2026-09-06T00:00:00Z` — plus a summary line in the `--json`
   report (`quota`, `downloads_new`, `cache_hits`). Standalone account
   status: `subcheck quota`.
2. **Verify** — the `--limit` candidates (default 3) are scored against the
   video with one ASR pass over the same sampled windows (the compare
   engine): per-candidate composite, text accuracy and drift diagnosis.
3. **Pick** — when the diagnosis says a candidate is fixable (consistent
   drift / bad fps), `autosub` *applies the fix and re-scores it on the same
   windows* before deciding; on top of that it scans constant shifts from
   −8 s to +8 s and keeps whichever version verifiably scores best (so a
   "common shift" rescue works even when the classifier is unsure). The
   winning candidate is written as `<video>.autosub.srt` with the fix baked
   in.
4. **Embed** — the corrected track is muxed into a *copy* of the video
   (`<video>.autosub.mkv`, lossless `-c copy` video/audio; mkv/webm/mp4/mov
   supported). Use `--in-place` (`-i`, `--overwrite`) to safely overwrite
   the source video in place instead of creating a copy. Skipped with
   `--no-embed`; not possible for remote files or unsupported containers —
   the `.srt` is still written.

Requires the `opensubtitlescom` package for live searches/downloads
(`pip install -e ".[autosub]"`); all cache/verification logic works offline.

Validated end-to-end on a Blu-ray MKV with a cached HDTV-release subtitle
(the two-tier cache means 0 API quota used): the candidate scored 7.5 % as-is,
the shift scan found +4.00 s (matching the manual ~+3.9 s diagnosis), the fix
was verified on the same windows (44.1 % — remaining gap is music/interjection
windows), and the corrected track was muxed into `<video>.autosub.mkv`.

### Embedded subtitles (mkv / mp4 …)

Videos often carry their own subtitle tracks. When no SRT argument is given,
`check` first looks for a sidecar `.srt` next to the media and otherwise uses
an **embedded subtitle stream** — extracting it to an SRT file with ffmpeg and
checking that:

```bash
subcheck check movie.mkv                      # sidecar .srt, or embedded track
subcheck check movie.mkv --embedded           # force the embedded track
subcheck check movie.mkv --sub-stream 3       # pick a specific track
subcheck check movie.mkv --sub-language fre   # ... or by language
subcheck streams movie.mkv                    # list the available tracks
```

`subcheck shift` accepts a video the same way and fixes its embedded
subtitles (optionally muxing the corrected track back into a copy of the
video):

```bash
subcheck shift movie.mkv --by -0.989              # writes movie.fixed.srt
subcheck shift movie.mkv --scale 0.95904 --remux movie.fixed.mkv
```

**Multiple streams.** When the video contains several *text* subtitle tracks
and no `--sub-stream`/`--sub-language` was given, an interactive terminal run
lists them and asks which one to use (Enter = the default-marked track). In
batch mode (no terminal) the default/first text track is used and the report
notes that a choice exists. Bitmap tracks (PGS/VobSub, e.g. Blu-ray
`hdmv_pgs_subtitle`) cannot be compared textually and are rejected with a
clear message — provide an external `.srt` instead.

### Other subcommands

* `subcheck rescore windows.json other.srt` — the single-file variant of
  `compare --from`: re-score previously transcribed windows (from
  `--json FILE`) against one other subtitle file, without re-running
  ffmpeg/ASR. Handy for verifying that a `subcheck shift` really fixed the
  drift:

  ```bash
  subcheck movie.mp4 movie.srt --seed 42 --json first.json
  subcheck rescore first.json movie.fixed.srt --json fixed.json
  ```


## Interpreting the output

Per window: `text` accuracy, temporal `iou`, composite `score`, optional
`drift ±s`, plus the ASR vs SRT text when they disagree. Aggregate over all
windows, then the drift classification and histogram. The check is
**sampling-based**: it certifies the sampled regions, not the whole file —
increase `--snippets` for more confidence.

## Validation

End-to-end runs on *The Prestige* (1080p rip, 2 h 10 m, ENG SRT, seed 1234,
3 × 10 s windows, faster-whisper large-v3 int8 on CPU):

| Subtitle file under test | text acc. | IoU | composite | detected by the tool |
|---|---|---|---|---|
| SRT built from the ASR output itself (positive control) | 100 % | 100 % | 100 % | drift mean +0.00 s (good) |
| Real `dxva-BTT` ENG SRT | 65.7 % | 53.8 % | 62.7 % | consistent drift mean +1.05 s (std 0.31 s) → fix suggested |
| Same SRT shifted +60 s (broken-sync control) | 6.7 % | 19.6 % | 9.9 % | no matched text, missing-caption windows, 2 re-rolls |
| Timeline scaled ×1.0008 (bad-fps control) | — | — | — | fps mismatch: slope +0.00075 s/s (~0.07 % fast) → scale fix; after applying it: drift mean −6 ms (good) |

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
