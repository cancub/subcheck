"""subcheck — verify that a subtitle file is accurate for a given video.

Pipeline:
    1. probe   ffprobe        -> total duration
    2. sample  random         -> non-overlapping 10 s windows
    3. slice   ffmpeg         -> 16 kHz mono PCM WAV snippets
    4. infer   faster-whisper -> local segments
    5. offset  python         -> segments on the global timeline
    6. compare pysrt + jiwer  -> WER text accuracy + temporal IoU
"""

__version__ = "0.1.0"
