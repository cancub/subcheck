"""Stage 4 — speech recognition of the 10 s snippets with faster-whisper.

Model weights are CTranslate2 format, downloaded from Hugging Face
(Systran/faster-whisper-*). Inference runs on CUDA when available (float16),
otherwise on CPU (int8). Timestamps returned by the model are *local* to each
snippet (starting at ~0 s) — the caller adds the snippet start time.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

_ALIASES = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


class ASRError(RuntimeError):
    """Raised for any problem in the ASR stage."""


def resolve_model_repo(model: str) -> str:
    """Map a short model name to its Hugging Face repository, or pass through
    a full repository id (anything containing '/' or already cached locally)."""
    if "/" in model:
        return model
    if model in _ALIASES:
        return _ALIASES[model]
    raise ASRError(
        f"Unknown model {model!r}. Choose from {sorted(_ALIASES)} or pass a "
        "Hugging Face repo id such as Systran/faster-whisper-large-v3."
    )


def _preload_nvidia_libs() -> None:
    """Pre-load pip-installed nvidia libraries (cublas, cudnn) into process memory.

    ctranslate2 searches for libcublas.so.12 and libcudnn.so via dlopen(), which
    fails if LD_LIBRARY_PATH isn't set. Preloading via ctypes loads them into
    process memory so subsequent dlopen() calls succeed immediately.
    """
    import ctypes
    import glob
    import site

    dirs = []
    try:
        dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:
        pass

    for d in dirs:
        pattern = os.path.join(d, "nvidia", "*", "lib", "*.so*")
        for so_file in sorted(glob.glob(pattern)):
            try:
                ctypes.CDLL(so_file)
            except Exception:
                pass


def cuda_device_count() -> int:
    _preload_nvidia_libs()
    import ctranslate2

    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


@dataclass(frozen=True)
class Segment:
    """One recognized speech segment; times are local to the snippet."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcriber:
    """Lazily loads a faster-whisper model and transcribes PCM WAV snippets."""

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "auto",
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
        beam_size: int = 5,
        language: Optional[str] = None,
        vad_filter: bool = True,
    ) -> None:
        self.repo = resolve_model_repo(model)
        self.beam_size = beam_size
        self.language = language
        self.vad_filter = vad_filter

        if device == "auto":
            self.device = "cuda" if cuda_device_count() > 0 else "cpu"
        elif device in ("cpu", "cuda"):
            self.device = device
        else:
            raise ASRError(f"Unknown device {device!r} (use auto, cpu or cuda)")

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        # Keep the model weights out of tiny tmpfs workspaces when the user
        # pointed HF_HOME / SUBCHECK_MODEL_DIR at a larger disk.
        self.download_root = (
            download_root
            or os.environ.get("SUBCHECK_MODEL_DIR")
            or os.environ.get("HF_HOME")
        )
        self._model = None

    def _load(self):
        if self._model is None:
            if self.device == "cuda":
                _preload_nvidia_libs()
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - env problem
                raise ASRError(
                    "faster-whisper is not installed (pip install faster-whisper)"
                ) from exc
            kwargs = {}
            if self.download_root:
                kwargs["download_root"] = self.download_root
            self._model = WhisperModel(
                self.repo,
                device=self.device,
                compute_type=self.compute_type,
                **kwargs,
            )
        return self._model

    # ---------------------------------------------------------------- audio I/O
    @staticmethod
    def read_wav(path: str) -> np.ndarray:
        """Read a PCM WAV into a float32 mono array at native sample rate."""
        with wave.open(path, "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sw != 2:
            raise ASRError(f"expected 16-bit PCM WAV, got {sw * 8}-bit")
        data = np.frombuffer(raw, dtype=np.int16)
        if nch > 1:
            data = data.reshape(-1, nch).mean(axis=1)
        audio = data.astype(np.float32) / 32768.0
        if rate != 16000:
            raise ASRError(f"expected 16 kHz WAV, got {rate} Hz")
        return audio

    # -------------------------------------------------------------- inference
    def transcribe_wav(self, wav_path: str) -> List[Segment]:
        """Transcribe one snippet WAV; returns segments with local times."""
        audio = self.read_wav(wav_path)
        model = self._load()
        try:
            segments_iter, _info = model.transcribe(
                audio,
                beam_size=self.beam_size,
                language=self.language,
                vad_filter=self.vad_filter,
                condition_on_previous_text=False,
            )
            segments = list(segments_iter)
        except Exception as exc:  # noqa: BLE001 - surface ctranslate2 failures
            raise ASRError(f"ASR failed on {wav_path}: {exc}") from exc
        out = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            out.append(Segment(float(seg.start), float(seg.end), text))
        return out
