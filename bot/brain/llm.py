"""Роутер моделей: при исчерпании лимита одной модели автоматически переходит на следующую.

У Groq суточный лимит токенов считается отдельно для каждой модели, поэтому цепочка
qwen → gpt-oss-120b → gpt-oss-20b даёт в разы больше бесплатной болтовни.
"""
from __future__ import annotations

import logging
import re
import time

import openai
from openai import AsyncOpenAI

log = logging.getLogger(__name__)


def _reasoning_for(model: str, configured: str | None) -> dict | None:
    """«Размышления» выключаем, чтобы было быстро. qwen понимает none, gpt-oss — минимум low."""
    if not configured:
        return None
    if "gpt-oss" in model:
        return {"reasoning_effort": "low"}
    return {"reasoning_effort": configured}


def _retry_after(err: Exception) -> float:
    """Из текста ошибки Groq «try again in 7m22.3s» достаём, на сколько модель недоступна."""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(err))
    if not m or not any(m.groups()):
        return 60.0
    h, mnt, s = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + mnt * 60 + s + 1


class LLMRouter:
    def __init__(self, client: AsyncOpenAI, models: list[str], reasoning_effort: str | None,
                 timeout: float = 12.0):
        self.client = client
        self.models = [m for i, m in enumerate(models) if m and m not in models[:i]]
        self.reasoning_effort = reasoning_effort
        self._blocked_until: dict[str, float] = {}
        self.timeout = timeout

    async def create(self, **kwargs):
        """То же, что client.chat.completions.create, но с перебором моделей при 429."""
        now = time.monotonic()
        available = [m for m in self.models if self._blocked_until.get(m, 0) <= now] or self.models[-1:]
        last_err: Exception | None = None
        for model in available:
            kw = dict(kwargs)
            if "gpt-oss" in model and "max_completion_tokens" in kw:
                kw["max_completion_tokens"] += 250  # gpt-oss часть токенов тратит на «размышления»
            try:
                # max_retries=0: библиотека openai сама ждёт по минуте при 429 — нам это не нужно,
                # мы мгновенно переходим на следующую модель
                return await self.client.with_options(max_retries=0, timeout=self.timeout).chat.completions.create(
                    model=model, extra_body=_reasoning_for(model, self.reasoning_effort), **kw)
            except openai.RateLimitError as e:
                wait = _retry_after(e)
                self._blocked_until[model] = time.monotonic() + wait
                log.warning("лимит модели %s исчерпан (~%d мин) — переключаюсь на следующую",
                            model, wait // 60)
                last_err = e
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                self._blocked_until[model] = time.monotonic() + 30
                log.warning("модель %s не отвечает (%s) — пробую следующую", model, type(e).__name__)
                last_err = e
        raise last_err or RuntimeError("no models available")
