"""Stage 6 (input side) — SRT parsing, tag stripping and windowing.

pysrt parses the file; we then slice the cues that overlap each sampled
window. Multi-line cues are merged into a single text block. Formatting
(``<i>``, ``{\\an8}`` …) is stripped here; acoustic brackets and punctuation
are handled by ``normalize``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import pysrt

# <i>, <b>, <font color="#...">, and ASS override tags like {\an8}
_TAG_RE = re.compile(r"<[^>]+>|\{[^}]*\}")


class SRTError(ValueError):
    pass


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str  # single text block (lines merged), formatting tags removed

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _merge_text(lines: List[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip())


def read_srt(path: str) -> List[Cue]:
    """Parse an SRT file into cues (absolute times in seconds)."""
    try:
        subs = pysrt.open(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            subs = pysrt.open(path, encoding="cp1252")
        except Exception as exc:  # noqa: BLE001
            raise SRTError(f"Cannot decode subtitle file {path!r}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SRTError(f"Cannot parse subtitle file {path!r}: {exc}") from exc

    cues: List[Cue] = []
    for item in subs:
        start = item.start.ordinal / 1000.0
        end = item.end.ordinal / 1000.0
        text = _TAG_RE.sub("", item.text).strip()
        if not text:
            continue
        if end <= start:
            continue  # malformed/zero-length cue: unusable for timing
        cues.append(
            Cue(
                index=int(item.index) if item.index is not None else len(cues) + 1,
                start=start,
                end=end,
                text=_merge_text(text.splitlines()),
            )
        )
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


def cues_overlapping(cues: List[Cue], wstart: float, wend: float) -> List[Cue]:
    """Cues whose interval overlaps [wstart, wend): max(wstart, Cs) < min(wend, Ce)."""
    out = []
    for cue in cues:
        if cue.end <= wstart:
            continue
        if cue.start >= wend:
            break  # sorted by start: nothing after can overlap either
        out.append(cue)
    return out


def shift_srt(src: str, out: str, by: float = 0.0, scale: float = 1.0) -> int:
    """Rewrite every cue of *src* with ``new_t = t*scale + by`` (pysrt-based,
    preserves text/formatting). Returns how many cue times had to be clamped
    (negative results / zero-length cues)."""
    import pysrt

    subs = pysrt.open(src, encoding="utf-8-sig")
    clamped = 0
    by_ms = by * 1000.0
    for item in subs:
        start_ms = int(round(item.start.ordinal * scale + by_ms))
        end_ms = int(round(item.end.ordinal * scale + by_ms))
        if start_ms < 0:
            start_ms = 0
            clamped += 1
        if end_ms <= start_ms:
            end_ms = start_ms + 1
            clamped += 1
        item.start = pysrt.SubRipTime(milliseconds=start_ms)
        item.end = pysrt.SubRipTime(milliseconds=end_ms)
    subs.save(out, encoding="utf-8")
    return clamped
