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
    raise ToolError(
        f"Could not find '{name}'. Install ffmpeg or set "
        f"SUBCHECK_{name.upper()}_BIN to its path (e.g. a static build)."
    )


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
