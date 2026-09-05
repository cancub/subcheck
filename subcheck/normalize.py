"""Text normalization shared by both sides of the comparison.

Spoken text (ASR) and written subtitles differ systematically in punctuation,
casing, whitespace and number spelling ("10" vs. "ten"). This module maps both
sides into a comparable form:

    lower-case
    drop formatting tags (<i>, {\\an8}) and music-note glyphs
    remove non-speech acoustic brackets such as [music] / (applause)
    remove punctuation (unicode-aware)
    expand numerals to words ("10" -> "ten", "2nd" -> "second")
    collapse whitespace
"""

from __future__ import annotations

import re
from typing import Optional

# --------------------------------------------------------------------------- #
# acoustic brackets: [...] / (...) containing a sound cue keyword
_NOISE_WORDS = (
    "music", "song", "singing", "sing", "sings", "sang", "melody", "tune",
    "instrumental", "play", "plays", "playing", "laugh", "laughs", "laughed",
    "laughter", "chuckle", "giggle", "snicker", "chortle", "applause", "applaud",
    "cheer", "cheers", "cheering", "clap", "clapping", "crowd", "chant",
    "chatter", "shout", "yell", "scream", "cry", "cries", "crying", "weep",
    "weeps", "weeping", "sob", "sobs", "sobbing", "whimper", "wail", "gasp",
    "sigh", "groan", "moan", "sniff", "sniffle", "cough", "coughing", "sneeze",
    "whistle", "whistling", "hum", "hums", "humming", "hiss", "roar", "buzz",
    "beep", "ring", "rings", "ringing", "bell", "knock", "knocking", "phone",
    "telephone", "dial", "alarm", "siren", "horn", "explosion", "explodes",
    "gunshot", "shot", "shooting", "thunder", "crash", "static", "noise",
    "sound", "sfx", "fx", "effect", "whisper", "murmur", "mutter", "grunt",
    "yelp", "squeak", "shriek", "vocalize", "mumbling",
)
_BRACKET_RE = re.compile(
    r"[(\[]([^()\[\]]*\b(?:" + "|".join(_NOISE_WORDS) + r")\b[^()\[\]]*)[)\]]",
    re.IGNORECASE,
)
_MUSIC_GLYPH_RE = re.compile(r"[♪♫♬]")

_NUMBER_TOKEN_RE = re.compile(r"(\d+)(st|nd|rd|th)?")

# --------------------------------------------------------------------------- #
# numerals -> words (cardinal and ordinal), enough for everyday dialogue
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def int_to_words(n: int) -> Optional[str]:
    """Cardinal words for 0 <= n <= 999_999_999; None when out of range."""
    if n == 0:
        return "zero"
    if n < 0 or n > 999_999_999:
        return None

    def below_1000(m: int) -> str:
        parts = []
        h, m = divmod(m, 100)
        if h:
            parts.append(_ONES[h] + " hundred")
        if m < 20:
            if m:
                parts.append(_ONES[m])
        else:
            t, u = divmod(m, 10)
            parts.append(_TENS[t])
            if u:
                parts.append(_ONES[u])
        return " ".join(parts)

    def chunk(m: int) -> str:
        return below_1000(m)

    out = []
    m, n = divmod(n, 1_000_000)
    t, n = divmod(n, 1_000)
    if m:
        out.append(chunk(m) + " million")
    if t:
        out.append(chunk(t) + " thousand")
    if n:
        out.append(chunk(n))
    return " ".join(out)


_ORDINAL_LAST = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def _ordinal_words(n: int) -> Optional[str]:
    words = int_to_words(n)
    if words is None:
        return None
    tokens = words.split()
    last = tokens[-1]
    if last in _ORDINAL_LAST:
        tokens[-1] = _ORDINAL_LAST[last]
    elif last.endswith("y"):
        tokens[-1] = last[:-1] + "ieth"  # twenty -> twentieth
    else:
        tokens[-1] = last + "th"  # four -> fourth
    return " ".join(tokens)


def _expand_token(token: str) -> Optional[str]:
    """Expand a numeric token to words; None when it is not a plain number."""
    m = _NUMBER_TOKEN_RE.fullmatch(token)
    if not m:
        return None
    digits, suffix = m.group(1), m.group(2)
    try:
        n = int(digits)
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None
    if suffix:  # ordinal: 1st, 2nd, 3rd, 21st, 100th ...
        return _ordinal_words(n)
    return int_to_words(n)


# --------------------------------------------------------------------------- #
# main entry point


def normalize(text: str, expand_numbers: bool = True) -> str:
    """Normalize *text* for comparison. Returns a single-line lower-case string."""
    s = text or ""
    s = s.replace("\r", " ")
    s = re.sub(r"<[^>]+>|\{[^}]*\}", " ", s)      # formatting tags
    s = _MUSIC_GLYPH_RE.sub(" ", s)
    s = _BRACKET_RE.sub(" ", s)                    # [music], (applause) ...
    s = s.lower()

    out_tokens: list[str] = []
    for token in s.split():
        # punctuation is dropped inside tokens: "don't" -> "dont", "what?" -> "what"
        clean = "".join(ch for ch in token if ch.isalnum())
        if not clean:
            continue
        if expand_numbers:
            expanded = _expand_token(clean)
            if expanded is not None:
                out_tokens.extend(expanded.split())
                continue
        out_tokens.append(clean)
    return " ".join(out_tokens)
