"""Stage 1 — probe the media container for its total duration.

Uses ffprobe, which only reads container metadata (HTTP range requests on
remote URLs) and never downloads or decodes the whole media.
"""

from __future__ import annotations

from typing import Optional

from .tools import resolve_binary, run_tool


def probe_duration(media: str, ffprobe: Optional[str] = None) -> float:
    """Return the total duration of *media* in seconds (float)."""
    binary = ffprobe or resolve_binary("ffprobe")
    out = run_tool(
        [
            binary,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media,
        ]
    ).strip()
    if not out:
        raise ValueError(
            f"ffprobe returned no duration for {media!r}: the file/URL may be "
            "unreachable, not seekable, or not a media container."
        )
    try:
        return float(out.splitlines()[-1])
    except ValueError:
        raise ValueError(f"Unexpected ffprobe duration output: {out!r}") from None
