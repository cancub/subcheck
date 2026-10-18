"""Tests for model downloading helper and CLI command."""

from unittest.mock import patch
import pytest

from subcheck.cli import main
from subcheck.transcribe import download_model, ASRError


def test_download_model_invokes_fw(tmp_path):
    with patch("faster_whisper.download_model") as mock_fw:
        mock_fw.return_value = str(tmp_path / "model_dir")
        res = download_model("tiny", download_root=str(tmp_path))
        mock_fw.assert_called_once_with(
            "Systran/faster-whisper-tiny", output_dir=str(tmp_path)
        )
        assert res == str(tmp_path / "model_dir")


def test_download_model_error_wrapped():
    with patch("faster_whisper.download_model", side_effect=RuntimeError("disk full")):
        with pytest.raises(ASRError, match="Failed to download model 'tiny'"):
            download_model("tiny")


def test_download_model_cli(capsys, tmp_path):
    with patch("faster_whisper.download_model") as mock_fw:
        mock_fw.return_value = str(tmp_path / "model_dir")
        rc = main(["download-model", "--model", "small", "--model-dir", str(tmp_path)])
        assert rc == 0
        mock_fw.assert_called_once_with(
            "Systran/faster-whisper-small", output_dir=str(tmp_path)
        )
        out = capsys.readouterr().out
        assert "Downloading model 'small'" in out
        assert f"Model ready at: {tmp_path / 'model_dir'}" in out


def test_pull_model_cli_alias(capsys, tmp_path):
    with patch("faster_whisper.download_model") as mock_fw:
        mock_fw.return_value = str(tmp_path / "model_dir")
        rc = main(["pull-model", "--model", "base", "--model-dir", str(tmp_path)])
        assert rc == 0
        mock_fw.assert_called_once_with(
            "Systran/faster-whisper-base", output_dir=str(tmp_path)
        )
