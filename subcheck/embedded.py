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
from typing import Dict, List, Optional

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


# ISO 639-1 (2-letter) <-> ISO 639-2 (3-letter) mapping for language matching
_ISO_639_MAP = {
    "en": {"en", "eng"},
    "fr": {"fr", "fra", "fre"},
    "es": {"es", "spa"},
    "de": {"de", "deu", "ger"},
    "it": {"it", "ita"},
    "pt": {"pt", "por"},
    "nl": {"nl", "nld", "dut"},
    "ru": {"ru", "rus"},
    "ja": {"ja", "jpn"},
    "zh": {"zh", "zho", "chi"},
    "ko": {"ko", "kor"},
    "ar": {"ar", "ara"},
    "hi": {"hi", "hin"},
    "pl": {"pl", "pol"},
    "tr": {"tr", "tur"},
    "sv": {"sv", "swe"},
    "da": {"da", "dan"},
    "no": {"no", "nor", "nob", "nno"},
    "fi": {"fi", "fin"},
    "cs": {"cs", "ces", "cze"},
    "el": {"el", "ell", "gre"},
    "he": {"he", "heb"},
    "hu": {"hu", "hun"},
    "id": {"id", "ind"},
    "ro": {"ro", "ron", "rum"},
    "th": {"th", "tha"},
    "uk": {"uk", "ukr"},
    "vi": {"vi", "vie"},
    "fa": {"fa", "fas", "per"},
    "bg": {"bg", "bul"},
    "hr": {"hr", "hrv"},
    "sr": {"sr", "srp"},
    "sk": {"sk", "slk", "slo"},
    "sl": {"sl", "slv"},
    "et": {"et", "est"},
    "lv": {"lv", "lav"},
    "lt": {"lt", "lit"},
}

_LANG_ALIASES: Dict[str, set] = {}
for _code_set in _ISO_639_MAP.values():
    for _c in _code_set:
        _LANG_ALIASES[_c] = _code_set


def matches_language(stream_lang: str, query_lang: str) -> bool:
    """Check if stream_lang matches query_lang, bridging 2-letter and 3-letter codes."""
    sl = (stream_lang or "").strip().lower()
    ql = (query_lang or "").strip().lower()
    if not sl or not ql:
        return False
    if sl == ql:
        return True
    sl_base = sl.split("-")[0].split("_")[0]
    ql_base = ql.split("-")[0].split("_")[0]
    if sl_base == ql_base:
        return True
    if sl in _LANG_ALIASES and ql in _LANG_ALIASES[sl]:
        return True
    if sl_base in _LANG_ALIASES and ql_base in _LANG_ALIASES[sl_base]:
        return True
    if len(ql) == 2 and sl.startswith(ql):
        return True
    if len(sl) == 2 and ql.startswith(sl):
        return True
    return False


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
            if matches_language(st.language, language):
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


# subtitle codec used when remuxing into a given container
_REMUX_CODEC_BY_EXT = {
    ".mkv": "srt",
    ".webm": "webvtt",   # matroska-in-webm only accepts webvtt text subs
    ".mp4": "mov_text",
    ".mov": "mov_text",
    ".m4v": "mov_text",
}


_ISO_639_TO_3 = {
    "en": "eng", "fr": "fre", "es": "spa", "de": "ger", "it": "ita",
    "pt": "por", "nl": "dut", "ru": "rus", "ja": "jpn", "zh": "chi",
    "ko": "kor", "ar": "ara", "hi": "hin", "pl": "pol", "tr": "tur",
    "sv": "swe", "da": "dan", "no": "nor", "fi": "fin", "cs": "cze",
    "el": "gre", "he": "heb", "hu": "hun", "id": "ind", "ro": "rum",
    "th": "tha", "uk": "ukr", "vi": "vie", "fa": "per", "bg": "bul",
    "hr": "hrv", "sr": "srp", "sk": "slo", "sl": "slv", "et": "est",
    "lv": "lav", "lt": "lit",
}


def normalize_container_lang(lang: str, container_ext: str = ".mkv") -> str:
    """Normalize language code to 3-letter ISO 639-2 for Matroska/WebM containers."""
    clean = (lang or "").strip().lower()
    if not clean:
        return ""
    base = clean.split("-")[0].split("_")[0]
    if container_ext in (".mkv", ".webm"):
        if len(clean) == 3:
            return clean
        if base in _ISO_639_TO_3:
            return _ISO_639_TO_3[base]
    return clean


def remux_with_subtitles(
    video: str,
    subtitles_srt: str,
    out_path: str,
    ffmpeg: Optional[str] = None,
    language: str = "",
    title: str = "",
    replace_stream_index: Optional[int] = None,
) -> str:
    """Copy video + audio and update/attach *subtitles_srt* into the container (mkv/webm/mp4/mov...).

    If the source video has existing subtitle streams, other subtitle tracks
    are preserved rather than dropped:
    - If *replace_stream_index* matches an existing subtitle track, that stream is
      replaced in-place with *subtitles_srt*, retaining the other tracks.
    - If *replace_stream_index* is omitted but an existing track matches *language*,
      that matching track is replaced.
    - Otherwise, existing tracks are preserved and the new subtitles are appended.

    No video/audio re-encoding happens (-c copy); the subtitle text is
    encoded to what the container supports (srt/webvtt/mov_text).
    Supports in-place overwrite when out_path == video (writes to a temporary
    file in the target directory first, then atomically replaces the original).
    """
    import tempfile

    binary = ffmpeg or resolve_binary("ffmpeg")
    ext = os.path.splitext(out_path)[1].lower()
    codec = _REMUX_CODEC_BY_EXT.get(ext)
    if codec is None:
        raise ToolError(
            f"cannot remux subtitles into a .{ext or '?'} container — use "
            ".mkv, .webm, .mp4 or .mov output"
        )

    in_place = os.path.abspath(out_path) == os.path.abspath(video)
    target_out = out_path
    tmp_path = None
    if in_place:
        dir_name = os.path.dirname(os.path.abspath(out_path)) or "."
        fd, tmp_path = tempfile.mkstemp(
            prefix=".subcheck-remux-",
            suffix=ext,
            dir=dir_name,
        )
        os.close(fd)
        target_out = tmp_path

    lang_to_use = normalize_container_lang(language, ext)

    existing_subs: List[SubtitleStream] = []
    if not video.startswith(("http://", "https://")) and os.path.isfile(video):
        try:
            existing_subs = probe_subtitle_streams(video)
        except Exception:
            existing_subs = []

    if existing_subs and replace_stream_index is None and language:
        for s in existing_subs:
            if s.text_capable and matches_language(s.language, language):
                replace_stream_index = s.index
                break

    cmd = [
        binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", video,
        "-i", subtitles_srt,
        "-map", "0:v?", "-map", "0:a?",
    ]
    meta_args: List[str] = []
    codec_args: List[str] = ["-c:v", "copy", "-c:a", "copy"]

    if existing_subs:
        codec_args.extend(["-c:s", "copy"])
        out_sub_idx = 0
        replaced_any = False
        for s in existing_subs:
            if ext not in (".mkv", ".webm") and not s.text_capable:
                continue
            if replace_stream_index is not None and s.index == replace_stream_index:
                cmd.extend(["-map", "1:0"])
                codec_args.extend([f"-c:s:{out_sub_idx}", codec])
                stream_lang = lang_to_use or normalize_container_lang(s.language, ext)
                if stream_lang:
                    meta_args.extend([f"-metadata:s:s:{out_sub_idx}", f"language={stream_lang}"])
                stream_title = title or s.title
                if stream_title:
                    meta_args.extend([f"-metadata:s:s:{out_sub_idx}", f"title={stream_title}"])
                if s.default:
                    meta_args.extend([f"-disposition:s:{out_sub_idx}", "default"])
                if s.forced:
                    meta_args.extend([f"-disposition:s:{out_sub_idx}", "forced"])
                replaced_any = True
            else:
                cmd.extend(["-map", f"0:s:{s.ordinal}"])
            out_sub_idx += 1

        if not replaced_any:
            cmd.extend(["-map", "1:0"])
            codec_args.extend([f"-c:s:{out_sub_idx}", codec])
            if lang_to_use:
                meta_args.extend([f"-metadata:s:s:{out_sub_idx}", f"language={lang_to_use}"])
            if title:
                meta_args.extend([f"-metadata:s:s:{out_sub_idx}", f"title={title}"])
    else:
        cmd.extend(["-map", "1:0"])
        codec_args.extend(["-c:s", codec])
        if lang_to_use:
            meta_args.extend(["-metadata:s:s:0", f"language={lang_to_use}"])
        if title:
            meta_args.extend(["-metadata:s:s:0", f"title={title}"])

    cmd.extend(["-map", "0:t?"])  # attachments (fonts / covers)
    cmd.extend(["-map", "0:d?"])  # data streams
    cmd.extend(codec_args)
    cmd.extend(meta_args)
    cmd.append(target_out)

    try:
        run_tool(cmd)
        if in_place and tmp_path:
            os.replace(tmp_path, out_path)
    except Exception:
        if in_place and tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    return out_path
