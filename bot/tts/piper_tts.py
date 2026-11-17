"""Локальный TTS Piper: ~0.1 с на фразу на CPU, бесплатно, без лимитов и интернета.

Голос (~60 МБ) скачивается при первом запуске в models/piper/.
Русские голоса: ru_RU-dmitri-medium, ru_RU-denis-medium, ru_RU-ruslan-medium, ru_RU-irina-medium.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import urllib.request
from pathlib import Path
from typing import AsyncIterator

import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path("models/piper")
HF_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{locale}/{speaker}/{quality}/{name}{ext}"
OUT_RATE = 24_000  # формат, который ждёт наш плеер


def _ensure_voice(name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    locale, speaker, quality = name.split("-")
    model = MODELS_DIR / f"{name}.onnx"
    for ext in (".onnx", ".onnx.json"):
        path = MODELS_DIR / f"{name}{ext}"
        if not path.exists():
            url = HF_URL.format(lang=locale.split("_")[0], locale=locale, speaker=speaker,
                                quality=quality, name=name, ext=ext)
            log.info("piper: скачиваю %s", url)
            urllib.request.urlretrieve(url, path)
    return model


def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm
    n = int(len(pcm) * dst_rate / src_rate)
    x = np.linspace(0, len(pcm) - 1, n)
    return np.interp(x, np.arange(len(pcm)), pcm.astype(np.float32)).astype(np.int16)


class PiperTTS:
    def __init__(self, voice: str = "ru_RU-dmitri-medium", speed: float = 1.0, liveliness: float = 0.667):
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        self.voice = PiperVoice.load(str(_ensure_voice(voice)))
        # speed > 1 — быстрее; liveliness — «живость» интонации (0.3 монотонно … 1.0 эмоционально)
        self.cfg = SynthesisConfig(length_scale=1.0 / max(0.3, speed), noise_scale=liveliness)
        self._lock = threading.Lock()  # espeak внутри Piper не потокобезопасен — синтез строго по одному

    def _synth(self, text: str) -> bytes:
        with self._lock:
            chunks = list(self.voice.synthesize(text, syn_config=self.cfg))
        if not chunks:
            return b""
        pcm = np.frombuffer(b"".join(c.audio_int16_bytes for c in chunks), dtype=np.int16)
        return _resample(pcm, chunks[0].sample_rate, OUT_RATE).tobytes()

    async def synth_clip(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synth, text)

    async def warmup(self) -> None:
        await asyncio.to_thread(self._synth, "привет")

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Piper синтезирует по предложениям — отдаём каждое сразу, не дожидаясь остальных."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        cancelled = threading.Event()

        def work() -> None:
            try:
                with self._lock:
                    for c in self.voice.synthesize(text, syn_config=self.cfg):
                        if cancelled.is_set():  # ответ прервали — не тратим время на остальное
                            break
                        pcm = np.frombuffer(c.audio_int16_bytes, dtype=np.int16)
                        loop.call_soon_threadsafe(q.put_nowait, _resample(pcm, c.sample_rate, OUT_RATE).tobytes())
            except Exception:
                log.exception("piper synth failed")
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        loop.run_in_executor(None, work)
        try:
            while (pcm := await q.get()) is not None:
                for i in range(0, len(pcm), 4800):
                    yield pcm[i : i + 4800]
        finally:
            cancelled.set()
