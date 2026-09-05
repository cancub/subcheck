"""Stage 3 — slice audio snippets out of the media with ffmpeg.

``-ss`` is placed *before* ``-i`` (input seeking): ffmpeg seeks via keyframes
and demuxes only the needed audio range, so remote videos are fetched with
HTTP range requests instead of a full download.
"""

from __future__ import annotations

import os
from typing import Optional

from .sampling import Snippet
from .tools import ToolError, resolve_binary, run_tool


def _fmt(seconds: float) -> str:
    return f"{seconds:.6f}"


def extract_audio(
    media: str,
    snippet: Snippet,
    out_path: str,
    ffmpeg: Optional[str] = None,
    timeout: float = 300.0,
) -> str:
    """Extract [start, start+duration) of the first audio stream as a
    16 kHz mono PCM WAV file; returns the output path."""
    binary = ffmpeg or resolve_binary("ffmpeg")
    out_path = os.fspath(out_path)
    try:
        run_tool(
            [
                binary,
                "-hide_banner",
                "-loglevel", "error",
                "-nostdin",
                "-y",
                "-ss", _fmt(snippet.start),
                "-t", _fmt(snippet.duration),
                "-i", media,
                "-map", "0:a:0",
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                out_path,
            ],
            timeout=timeout,
        )
    except ToolError as exc:
        raise ToolError(
            f"Failed to slice audio at {snippet.start:.2f}s ({exc}). "
            "Is the media reachable and does it contain an audio stream?"
        ) from None
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise ToolError(f"ffmpeg produced no audio for snippet {snippet}")
    return out_path
