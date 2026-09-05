"""Embedded (muxed) subtitle streams — discovery and extraction.

Video containers (mkv, mp4, ...) often carry subtitle tracks. ``subcheck`` can
check or shift *these* instead of an external .srt: ffprobe lists the text
streams, ffmpeg extracts the chosen one to an SRT file.

Only *text* subtitle codecs can be checked (the comparison is textual);
bitmap codecs (hdmv_pgs_subtitle, dvd_subtitle, ...) would need OCR and are
rejected with a clear message.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from .tools import ToolError, resolve_binary, run_tool

# codecs whose content is plain text that can be converted to SRT
TEXT_SUBTITLE_CODECS = {
    "subrip", "ass", "ssa", "webvtt", "mov_text", "text", "sami",
    "subviewer", "subviewer1", "jacosub", "microdvd", "realtext", "mpl2",
    "vplayer", "pjs", "aqtitle", "stl", "eia_608", "c608",
}
# codecs that are raster images (cannot be compared textually without OCR)
BITMAP_SUBTITLE_CODECS = {
    "hdmv_pgs_subtitle", "dvd_subtitle", "xsub", "dvb_subtitle", "pgssub",
}


@dataclass(frozen=True)
class SubtitleStream:
    """One subtitle track of a media file."""

    index: int            # global stream index as reported by ffprobe
    codec: str
    language: str = ""    # ISO code from the container tags, if present
    title: str = ""
    default: bool = False
    forced: bool = False
    text_capable: bool = True   # unknown codecs are tried but may fail to extract
    ordinal: int = 0      # position among the *subtitle* streams (ffmpeg's
                          # -map 0:s:<ordinal> counts per type, not globally)

    @property
    def desc(self) -> str:
        bits = [f"#{self.index}", self.codec]
        if self.language:
            bits.append(self.language)
        if self.default:
            bits.append("default")
        if self.forced:
            bits.append("forced")
        if not self.text_capable:
            bits.append("BITMAP")
        return "stream " + " ".join(bits)


class StreamSelectionError(ValueError):
    """No usable embedded subtitle stream could be picked."""


def probe_subtitle_streams(media: str, ffprobe: Optional[str] = None) -> List[SubtitleStream]:
    """List all subtitle streams of *media* ([] when there are none)."""
    binary = ffprobe or resolve_binary("ffprobe")
    out = run_tool(
        [
            binary, "-v", "error",
            "-select_streams", "s",
            "-show_entries",
            "stream=index,codec_name:stream_disposition=default,forced:stream_tags=language,title",
            "-of", "json",
            media,
        ]
    )
    try:
        payload = json.loads(out)
    except json.JSONDecodeError as exc:
        raise ToolError(f"ffprobe produced no stream list for {media!r}") from exc
    streams = []
    for pos, st in enumerate(payload.get("streams", [])):
        tags = st.get("tags", {}) or {}
        disp = st.get("disposition", {}) or {}
        codec = (st.get("codec_name") or "?").lower()
        streams.append(
            SubtitleStream(
                index=int(st.get("index", -1)),
                codec=codec,
                language=tags.get("language", ""),
                title=tags.get("title", ""),
                default=bool(disp.get("default", 0)),
                forced=bool(disp.get("forced", 0)),
                text_capable=codec in TEXT_SUBTITLE_CODECS
                or codec not in BITMAP_SUBTITLE_CODECS,
                ordinal=pos,
            )
        )
    streams.sort(key=lambda s: s.index)
    for pos, s in enumerate(streams):  # keep ordinals consistent after sorting
        streams[pos] = SubtitleStream(
            index=s.index, codec=s.codec, language=s.language, title=s.title,
            default=s.default, forced=s.forced, text_capable=s.text_capable,
            ordinal=pos,
        )
    return streams


def select_subtitle_stream(
    streams: List[SubtitleStream],
    *,
    index: Optional[int] = None,
    language: Optional[str] = None,
) -> SubtitleStream:
    """Pick the stream to check/shift.

    Priority: explicit *index*, else the first text-capable stream matching
    *language*, else the default-marked text stream, else the first
    text-capable stream. Raises StreamSelectionError with a listing when no
    text stream exists or the selection matches nothing.
    """
    if index is not None:
        for st in streams:
            if st.index == index:
                return st
        raise StreamSelectionError(
            f"no subtitle stream #{index} in the media — found:\n" + _fmt(streams)
        )
    text_streams = [s for s in streams if s.text_capable]
    if language:
        for st in text_streams:
            if st.language.lower() == language.lower():
                return st
        raise StreamSelectionError(
            f"no text subtitle stream with language {language!r} — found:\n"
            + _fmt(streams)
        )
    for st in text_streams:
        if st.default:
            return st
    if text_streams:
        return text_streams[0]
    if streams:
        raise StreamSelectionError(
            "the video has embedded subtitles, but they are bitmap-based "
            f"({', '.join(s.codec for s in streams)}) — text comparison is not "
            "possible without OCR; provide an external .srt instead."
        )
    raise StreamSelectionError("the video has no embedded subtitle streams.")


def choose_subtitle_stream(
    streams: List[SubtitleStream],
    *,
    index: Optional[int] = None,
    language: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> SubtitleStream:
    """Pick the stream to work with — prompting when there is a choice.

    Behaves like :func:`select_subtitle_stream`, but when the video carries
    *several* text streams and no explicit ``--sub-stream``/``--sub-language``
    was given, the user is asked interactively which track to use (when
    ``interactive`` is None this happens only if stdin is a terminal; in
    batch mode the default rule applies silently).
    """
    text_streams = [s for s in streams if s.text_capable]
    if index is None and language is None and len(text_streams) > 1:
        if interactive is None:
            interactive = sys.stdin.isatty()
        if interactive:
            return _prompt_stream(streams)
    return select_subtitle_stream(streams, index=index, language=language)


def _prompt_stream(streams: List[SubtitleStream]) -> SubtitleStream:
    text_streams = [s for s in streams if s.text_capable]
    bitmap = [s for s in streams if not s.text_capable]
    default = next((s for s in text_streams if s.default), text_streams[0])
    print("Multiple embedded subtitle streams found in this video:")
    for s in text_streams:
        marker = "  (default)" if s.index == default.index else ""
        print(f"  [{s.index}]  {s.desc}{marker}")
    if bitmap:
        print(f"  ({len(bitmap)} bitmap track(s) skipped — cannot be checked textually)")
    for attempt in range(3):
        raw = input(
            f"which stream to use? index or Enter for #{default.index}: "
        ).strip()
        if not raw:
            return default
        try:
            wanted = int(raw)
        except ValueError:
            print(f"  '{raw}' is not a number — try again.")
            continue
        for s in streams:
            if s.index == wanted:
                return s
        print(f"  no stream #{wanted} — try again (or press Enter for "
              f"#{default.index}).")
    print(f"Using default {default.desc}.")
    return default


def _fmt(streams: List[SubtitleStream]) -> str:
    return "\n".join("  " + s.desc for s in streams)


def extract_subtitle_stream(
    media: str,
    stream: SubtitleStream,
    out_path: str,
    ffmpeg: Optional[str] = None,
) -> str:
    """Extract one embedded subtitle stream to an SRT file on disk.

    subrip tracks are copied verbatim; other text codecs (ass, mov_text, ...)
    are converted with the SRT encoder.
    """
    binary = ffmpeg or resolve_binary("ffmpeg")
    out_path = os.fspath(out_path)
    cmd = [
        binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", media,
        "-map", f"0:s:{stream.ordinal}",
        "-c:s", "copy" if stream.codec in ("subrip", "text") else "srt",
        out_path,
    ]
    try:
        run_tool(cmd)
    except ToolError as exc:
        raise ToolError(
            f"Could not extract embedded subtitle {stream.desc} from {media!r} "
            f"({exc}). The stream may use a codec that cannot be converted to "
            "SRT."
        ) from None
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise ToolError(
            f"Extraction of embedded subtitle {stream.desc} produced an empty "
            f"file ({out_path})."
        )
    return out_path


def remux_with_subtitles(
    video: str,
    subtitles_srt: str,
    out_path: str,
    ffmpeg: Optional[str] = None,
) -> str:
    """Copy video + audio and attach *subtitles_srt* as the only subtitle
    track (mkv/webm targets; other containers cannot hold SRT)."""
    binary = ffmpeg or resolve_binary("ffmpeg")
    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".mkv", ".webm"):
        raise ToolError(
            f"cannot remux SRT subtitles into a .{ext or '?'} container — "
            "use an .mkv or .webm output"
        )
    run_tool(
        [
            binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", video,
            "-i", subtitles_srt,
            "-map", "0:v:0", "-map", "0:a?", "-map", "1:0",
            "-c:v", "copy", "-c:a", "copy", "-c:s", "srt",
            out_path,
        ]
    )
    return out_path
