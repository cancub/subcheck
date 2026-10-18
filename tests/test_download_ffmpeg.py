"""Tests for ffmpeg resolution, download_ffmpeg helper, and CLI command."""

import os
from unittest.mock import patch
import pytest

from subcheck.cli import main
from subcheck.tools import ToolError, download_ffmpeg, resolve_binary


def test_resolve_binary_env_override(tmp_path, monkeypatch):
    bin_file = tmp_path / "custom_ffmpeg"
    bin_file.write_text("#!/bin/sh\n")
    monkeypatch.setenv("SUBCHECK_FFMPEG_BIN", str(bin_file))
    assert resolve_binary("ffmpeg") == str(bin_file)


def test_resolve_binary_env_missing(monkeypatch):
    monkeypatch.setenv("SUBCHECK_FFMPEG_BIN", "/nonexistent/ffmpeg")
    with pytest.raises(ToolError, match="points to a missing file"):
        resolve_binary("ffmpeg")


def test_resolve_binary_fallback_local_bin(tmp_path, monkeypatch):
    monkeypatch.delenv("SUBCHECK_FFMPEG_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    ffmpeg_file = local_bin / "ffmpeg"
    ffmpeg_file.write_text("#!/bin/sh\n")
    ffmpeg_file.chmod(0o755)

    with patch("os.path.expanduser", side_effect=lambda p: str(p).replace("~", str(tmp_path))):
        assert resolve_binary("ffmpeg") == str(ffmpeg_file)


def test_resolve_binary_not_found(monkeypatch):
    monkeypatch.delenv("SUBCHECK_FFMPEG_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    with patch("os.path.isfile", return_value=False):
        with pytest.raises(ToolError, match="subcheck download-ffmpeg"):
            resolve_binary("ffmpeg")


def test_download_ffmpeg_cli(capsys, tmp_path):
    with patch("subcheck.tools.download_ffmpeg") as mock_dl:
        mock_dl.return_value = (str(tmp_path / "ffmpeg"), str(tmp_path / "ffprobe"))
        rc = main(["download-ffmpeg", "--bin-dir", str(tmp_path)])
        assert rc == 0
        mock_dl.assert_called_once_with(str(tmp_path))
        out = capsys.readouterr().out
        assert f"ffmpeg  ready at: {tmp_path / 'ffmpeg'}" in out
        assert f"ffprobe ready at: {tmp_path / 'ffprobe'}" in out
