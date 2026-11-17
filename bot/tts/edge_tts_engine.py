"""Бесплатный TTS через голоса Microsoft Edge (без ключа). Отдаёт PCM 24 кГц моно.

Скорость: ответ режется на предложения, все синтезируются параллельно, а играть начинаем
с первого, как только оно готово. Сервис иногда тормозит или отвечает пустотой — тогда
реплику подхватывает локальный fallback-TTS (Piper), чтобы бот не молчал.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import AsyncIterator

import edge_tts
import miniaudio

log = logging.getLogger(__name__)

HEDGE_S = 2.0  # ждём Microsoft не дольше этого, потом говорим запасным голосом


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p for p in parts if p.strip()]


class EdgeTTS:
    SKIP_AFTER_FAIL_S = 60  # после сбоя минуту не ждём Microsoft — сразу запасной голос

    def __init__(self, voice: str = "ru-RU-SvetlanaNeural", fallback=None):
        self.voice = voice
        self.fallback = fallback  # другой TTS с методом stream(text)
        self._skip_until = 0.0

    async def _synth(self, text: str) -> bytes:
        try:
            mp3 = bytearray()
            async for chunk in edge_tts.Communicate(text, self.voice).stream():
                if chunk["type"] == "audio":
                    mp3 += chunk["data"]
            return bytes(mp3)
        except edge_tts.exceptions.NoAudioReceived:
            return b""
        except Exception as e:
            log.warning("edge-tts %s error: %s", self.voice, e)
            return b""

    async def warmup(self) -> None:
        await self._synth("привет")

    async def synth_clip(self, text: str) -> bytes:
        """Только голос Microsoft, без запасного (для кеша звуков)."""
        for _ in range(3):
            if mp3 := await self._synth(text):
                return self._decode(mp3)
        return b""

    @staticmethod
    def _decode(mp3: bytes) -> bytes:
        return miniaudio.decode(
            mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=24_000
        ).samples.tobytes()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        if self.fallback is not None and time.monotonic() < self._skip_until:
            async for chunk in self.fallback.stream(text):
                yield chunk
            return
        sentences = split_sentences(text)
        tasks = [asyncio.create_task(self._synth(s)) for s in sentences]
        try:
            for i, task in enumerate(tasks):
                done, _ = await asyncio.wait({task}, timeout=HEDGE_S if i == 0 else 10)
                mp3 = task.result() if done else b""
                if not mp3:
                    rest = " ".join(sentences[i:])
                    if self.fallback is None:
                        log.error("edge-tts: не удалось озвучить фразу")
                        return
                    if time.monotonic() >= self._skip_until:
                        log.info("edge-tts медлит — минуту говорю запасным голосом")
                    self._skip_until = time.monotonic() + self.SKIP_AFTER_FAIL_S
                    async for chunk in self.fallback.stream(rest):
                        yield chunk
                    return
                pcm = self._decode(mp3)
                for j in range(0, len(pcm), 4800):
                    yield pcm[j : j + 4800]
        finally:
            for t in tasks:
                t.cancel()
