"""Локальный faster-whisper: распознавание без интернета и лимитов.

device="auto": пробуем видеокарту, а если на ней не заводится (часто на Windows не хватает
CUDA-библиотек, и это видно только на первом распознавании) — тихо переходим на процессор.
"""
from __future__ import annotations

import asyncio
import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class LocalWhisperSTT:
    def __init__(self, model_size: str, device: str, language: str):
        self.model_size = model_size
        self.language = language
        self._lock = threading.Lock()  # одна модель — один инференс за раз
        want_cuda = device == "cuda" or (device == "auto" and _cuda_available())
        self._load("cuda" if want_cuda else "cpu")
        self._can_fallback = want_cuda and device == "auto"

    def _load(self, device: str) -> None:
        from faster_whisper import WhisperModel

        compute_type = "int8_float16" if device == "cuda" else "int8"
        self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        self.device = device
        log.info("локальный Whisper %s на %s", self.model_size, "видеокарте" if device == "cuda" else "процессоре")

    async def transcribe(self, samples16k: np.ndarray, prompt: str | None = None) -> str:
        audio = samples16k.astype(np.float32) / 32768.0
        return await asyncio.to_thread(self._run, audio, prompt)

    def _transcribe(self, audio: np.ndarray, prompt: str | None) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=prompt,
            beam_size=1,
            vad_filter=False,  # уже нарезано нашим VAD
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def _run(self, audio: np.ndarray, prompt: str | None) -> str:
        with self._lock:
            try:
                return self._transcribe(audio, prompt)
            except RuntimeError as e:
                if not (self._can_fallback and self.device == "cuda"):
                    raise
                log.warning("Whisper на видеокарте не заработал (%s) — перехожу на процессор", e)
                self._can_fallback = False
                self._load("cpu")
                return self._transcribe(audio, prompt)
