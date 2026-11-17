from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import openai
from openai import AsyncOpenAI

from bot.config import Settings
from bot.stt.base import STTEngine

log = logging.getLogger(__name__)


class FallbackSTT:
    """Groq Whisper → вторая модель Whisper (у неё свой лимит) → локальный faster-whisper.

    Локальная модель грузится только когда реально понадобилась (лимиты кончились).
    """

    def __init__(self, settings: Settings, client: AsyncOpenAI):
        from bot.stt.openai_stt import OpenAISTT

        self.settings = settings
        other = "whisper-large-v3-turbo" if settings.stt_model == "whisper-large-v3" else "whisper-large-v3"
        models = [settings.stt_model, other]  # у каждой модели свой дневной лимит
        self.remote = [OpenAISTT(client, m, settings.language) for i, m in enumerate(models) if m not in models[:i]]
        self._blocked_until: dict[int, float] = {}
        self._local: STTEngine | None = None
        self._local_lock = asyncio.Lock()

    async def _get_local(self) -> STTEngine:
        async with self._local_lock:
            if self._local is None:
                from bot.stt.local_stt import LocalWhisperSTT

                log.warning("лимиты Groq на распознавание кончились — гружу локальный Whisper (%s, %s)",
                            self.settings.local_whisper_model, self.settings.local_whisper_device)
                s = self.settings
                self._local = await asyncio.to_thread(
                    LocalWhisperSTT, s.local_whisper_model, s.local_whisper_device, s.language)
            return self._local

    async def transcribe(self, samples16k: np.ndarray, prompt: str | None = None) -> str:
        return (await self.transcribe_scored(samples16k, prompt))[0]

    async def transcribe_scored(self, samples16k: np.ndarray, prompt: str | None = None) -> tuple[str, float | None]:
        now = time.monotonic()
        for i, engine in enumerate(self.remote):
            if self._blocked_until.get(i, 0) > now:
                continue
            try:
                return await engine.transcribe_scored(samples16k, prompt)
            except openai.RateLimitError as e:
                from bot.brain.llm import _retry_after

                wait = _retry_after(e)
                self._blocked_until[i] = time.monotonic() + wait
                log.warning("лимит STT-модели %s исчерпан (~%d мин) — переключаюсь", engine.model, wait // 60)
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                self._blocked_until[i] = time.monotonic() + 30
                log.warning("STT-модель %s не отвечает (%s) — переключаюсь", engine.model, type(e).__name__)
        return await (await self._get_local()).transcribe_scored(samples16k, prompt)


def make_stt(settings: Settings, client: AsyncOpenAI) -> STTEngine:
    if settings.stt_backend == "local":
        from bot.stt.local_stt import LocalWhisperSTT

        return LocalWhisperSTT(settings.local_whisper_model, settings.local_whisper_device, settings.language)
    if settings.stt_base_url and "groq" in settings.stt_base_url:
        return FallbackSTT(settings, client)
    from bot.stt.openai_stt import OpenAISTT

    return OpenAISTT(client, settings.stt_model, settings.language)
