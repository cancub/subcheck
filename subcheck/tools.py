"""Resolution of the external ffmpeg/ffprobe binaries and shared subprocess helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Sequence


class ToolError(RuntimeError):
    """Raised when an external tool (ffmpeg/ffprobe) fails."""


def resolve_binary(name: str) -> str:
    """Resolve an external binary.

    Priority:
      1. $SUBCHECK_FFMPEG_BIN / $SUBCHECK_FFPROBE_BIN  (explicit override, e.g. a
         static build without a system install)
      2. `name` found on PATH
      3. Fallback to ~/.local/bin or ~/.cache/subcheck/bin

    Raises ToolError when unavailable.
    """
    env = os.environ.get(f"SUBCHECK_{name.upper()}_BIN")
    if env:
        if not os.path.isfile(env):
            raise ToolError(
                f"$SUBCHECK_{name.upper()}_BIN points to a missing file: {env!r}"
            )
        return env
    found = shutil.which(name)
    if found:
        return found

    for fallback_dir in ("~/.local/bin", "~/.cache/subcheck/bin"):
        candidate = os.path.expanduser(os.path.join(fallback_dir, name))
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise ToolError(
        f"Could not find '{name}'. Install ffmpeg via your system package manager "
        "(e.g. 'sudo apt install ffmpeg'), download portable binaries without root using "
        "'subcheck download-ffmpeg', or set SUBCHECK_{name.upper()}_BIN to its path."
    )


def download_ffmpeg(target_dir: Optional[str] = None) -> tuple[str, str]:
    """Download portable static ffmpeg and ffprobe binaries.

    Defaults to ~/.local/bin (or ~/.cache/subcheck/bin if ~/.local/bin is unwritable).
    Returns (ffmpeg_path, ffprobe_path).
    """
    import platform
    import stat
    import tarfile
    import tempfile
    import urllib.request

    if target_dir is None:
        target_dir = os.path.expanduser("~/.local/bin")
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError:
        target_dir = os.path.expanduser("~/.cache/subcheck/bin")
        os.makedirs(target_dir, exist_ok=True)

    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    elif arch in ("aarch64", "arm64"):
        url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
    else:
        raise ToolError(
            f"Automatic static ffmpeg download is not supported for architecture {arch!r}. "
            "Please install ffmpeg with your system package manager."
        )

    req = urllib.request.Request(url, headers={"User-Agent": "subcheck"})
    with tempfile.NamedTemporaryFile(suffix=".tar.xz", delete=False) as tf:
        temp_path = tf.name
    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
        with tarfile.open(temp_path, mode="r:xz") as tar:
            for member in tar.getmembers():
                base = os.path.basename(member.name)
                if base in ("ffmpeg", "ffprobe") and member.isfile():
                    dest = os.path.join(target_dir, base)
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        with open(dest, "wb") as df:
                            shutil.copyfileobj(extracted, df)
                        os.chmod(
                            dest,
                            os.stat(dest).st_mode
                            | stat.S_IXUSR
                            | stat.S_IXGRP
                            | stat.S_IXOTH,
                        )
    except Exception as exc:
        raise ToolError(f"Failed to download/extract ffmpeg from {url}: {exc}") from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    ffmpeg_bin = os.path.join(target_dir, "ffmpeg")
    ffprobe_bin = os.path.join(target_dir, "ffprobe")
    if not (os.path.isfile(ffmpeg_bin) and os.path.isfile(ffprobe_bin)):
        raise ToolError("Failed to extract ffmpeg/ffprobe from static archive")
    return ffmpeg_bin, ffprobe_bin


def run_tool(argv: Sequence[str], timeout: Optional[float] = None) -> str:
    """Run an external command and return stdout.

    Raises ToolError (with captured stderr) on non-zero exit.
    """
    proc = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise ToolError(
            f"Command failed (rc={proc.returncode}): {' '.join(argv)}\n"
            + "\n".join(tail)
        )
    return proc.stdout
