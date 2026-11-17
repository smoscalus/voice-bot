from __future__ import annotations

import numpy as np
from openai import AsyncOpenAI

from bot.audio.convert import pcm16_to_wav


class OpenAISTT:
    def __init__(self, client: AsyncOpenAI, model: str, language: str):
        self.client = client
        self.model = model
        self.language = language

    async def transcribe(self, samples16k: np.ndarray, prompt: str | None = None) -> str:
        wav = pcm16_to_wav(samples16k, 16_000)
        kwargs = {"prompt": prompt} if prompt else {}
        # без встроенных повторов: при 429 сразу уходим на запасную модель (см. FallbackSTT)
        r = await self.client.with_options(max_retries=0, timeout=10).audio.transcriptions.create(
            model=self.model,
            file=("utterance.wav", wav, "audio/wav"),
            language=self.language,
            **kwargs,
        )
        return r.text.strip()
