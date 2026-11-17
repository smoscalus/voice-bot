from __future__ import annotations

import numpy as np
from openai import AsyncOpenAI

from bot.audio.convert import pcm16_to_wav


def segments_confidence(segments) -> float | None:
    """Средняя «уверенность» Whisper по фразе (avg_logprob, взвешенный по длине сегментов).
    Около 0 — уверен, ниже −0.8 — скорее всего каша/шум."""
    total, weight = 0.0, 0.0
    for seg in segments or []:
        lp = getattr(seg, "avg_logprob", None)
        if lp is None and isinstance(seg, dict):
            lp = seg.get("avg_logprob")
        if lp is None:
            continue
        start = getattr(seg, "start", None) if not isinstance(seg, dict) else seg.get("start")
        end = getattr(seg, "end", None) if not isinstance(seg, dict) else seg.get("end")
        w = max(0.1, (end or 0) - (start or 0))
        total += lp * w
        weight += w
    return total / weight if weight else None


class OpenAISTT:
    def __init__(self, client: AsyncOpenAI, model: str, language: str):
        self.client = client
        self.model = model
        self.language = language

    async def transcribe(self, samples16k: np.ndarray, prompt: str | None = None) -> str:
        return (await self.transcribe_scored(samples16k, prompt))[0]

    async def transcribe_scored(self, samples16k: np.ndarray, prompt: str | None = None) -> tuple[str, float | None]:
        wav = pcm16_to_wav(samples16k, 16_000)
        kwargs = {"prompt": prompt} if prompt else {}
        verbose = "whisper" in self.model  # уверенность по сегментам отдают только whisper-модели
        if verbose:
            kwargs["response_format"] = "verbose_json"
        # без встроенных повторов: при 429 сразу уходим на запасную модель (см. FallbackSTT)
        r = await self.client.with_options(max_retries=0, timeout=10).audio.transcriptions.create(
            model=self.model,
            file=("utterance.wav", wav, "audio/wav"),
            language=self.language,
            **kwargs,
        )
        return r.text.strip(), segments_confidence(getattr(r, "segments", None)) if verbose else None
