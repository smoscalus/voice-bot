"""Роутер моделей: при исчерпании лимита одной модели автоматически переходит на следующую.

У Groq суточный лимит токенов считается отдельно для каждой модели, поэтому цепочка
qwen → gpt-oss-120b → gpt-oss-20b даёт в разы больше бесплатной болтовни.
"""
from __future__ import annotations

import asyncio
import contextlib
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


# ---------------------------------------------------------------- живой ответ важнее фоновых дел
# Локальная модель (Ollama) делает одно дело за раз: пока летописец пишет досье (10–20 с), ответ человеку
# стоит в очереди. Поэтому фоновые вызовы (память, дневник, книжка, сны) ждут, пока бот не отвечает,
# а если ответ понадобился посреди фонового вызова — фоновый обрывается и повторяется позже.
_fg_count = 0
_fg_idle_since = 0.0
_bg_inflight: set[asyncio.Task] = set()
BG_IDLE_GAP_S = 2.5     # после ответа даём паузу — вдруг человек сразу ответит
BG_MAX_WAIT_S = 120     # дольше не ждём: память тоже нужна
BG_MAX_PREEMPT = 3      # столько раз фоновый вызов уступает, потом доделывается без уступок


@contextlib.contextmanager
def foreground():
    """Бот сейчас отвечает человеку: фоновые вызовы ждут, а начатые — обрываются (только локальная модель)."""
    global _fg_count, _fg_idle_since
    _fg_count += 1
    for t in list(_bg_inflight):
        t.cancel()
    try:
        yield
    finally:
        _fg_count -= 1
        if _fg_count <= 0:
            _fg_count = 0
            _fg_idle_since = time.monotonic()


def foreground_busy() -> bool:
    return _fg_count > 0 or time.monotonic() - _fg_idle_since < BG_IDLE_GAP_S


async def _wait_turn() -> None:
    deadline = time.monotonic() + BG_MAX_WAIT_S
    while foreground_busy() and time.monotonic() < deadline:
        await asyncio.sleep(0.3)


# ---------------------------------------------------------------- сколько влезло в голову
# Локальная модель видит только CONTEXT_LIMIT токенов (OLLAMA_CONTEXT_LENGTH). Если промпт больше — Ollama
# молча отрезает начало (а там характер), и бот ведёт себя странно. Поэтому пишем в лог реальный размер
# (его сообщает сама модель) и предупреждаем, когда запас маленький.
WARN_SHARE = 0.85
# Ollama молча обрезает промпт до своего окна: тогда размер ровно 2048/4096 (+пара служебных токенов)
_CUT_SIZES = {n + d for n in (2048, 4096) for d in range(4)}
# «думающая» локальная модель тратит токены на мысли из того же лимита — добавляем запас
THINK_EXTRA_TOKENS = 800


def context_limit() -> int:
    import os

    try:
        return int(os.getenv("OLLAMA_CONTEXT_LENGTH", "") or 8192)
    except ValueError:
        return 8192


def note_usage(usage, tag: str, limit: int, max_out: int = 0) -> None:
    """usage — из ответа модели (prompt_tokens / completion_tokens). limit=0 — облако, окно большое."""
    prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
    if not prompt:
        return
    if limit and prompt in _CUT_SIZES and prompt + 64 < limit:
        log.warning("🧠 %s: промпт ровно %d токенов — похоже, Ollama ОБРЕЗАЛА его до своего окна, а с начала "
                    "пропали характер и правила (отсюда бред и простыни текста). У самой Ollama окно меньше %d: %s",
                    tag or "запрос", prompt, limit, ollama_ctx_hint(limit))
    elif limit and prompt + max_out > limit * WARN_SHARE:
        log.warning("🧠 %s: промпт %d токенов + ответ до %d — почти край окна %d! Модель может забыть начало "
                    "(характер). Помогут OLLAMA_CONTEXT_LENGTH побольше или меньше людей в войсе.",
                    tag or "запрос", prompt, max_out, limit)
    else:
        log.info("🧠 %s: промпт %d токенов%s", tag or "запрос", prompt,
                 f" (запас {limit - prompt - max_out})" if limit else "")


def ollama_ctx_hint(limit: int) -> str:
    """Как задать окно самой Ollama на этой системе — коротко, для лога."""
    import sys

    if sys.platform == "win32":
        return (f"запусти start.bat (он пропишет OLLAMA_CONTEXT_LENGTH={limit}) или задай эту переменную "
                "в системе, потом закрой Ollama у часов и открой снова.")
    return (f"на Linux: ./start.sh config — он настроит службу Ollama, или вручную: "
            f"sudo systemctl edit ollama → Environment=\"OLLAMA_CONTEXT_LENGTH={limit}\" → sudo systemctl restart ollama.")


def _thought(obj) -> str:
    """Текст «размышлений» модели, если сервер его прислал (Ollama: reasoning, иные: reasoning_content)."""
    for name in ("reasoning", "reasoning_content"):
        v = getattr(obj, name, None) or (getattr(obj, "model_extra", None) or {}).get(name)
        if v:
            return v
    return ""


class LLMRouter:
    def __init__(self, client: AsyncOpenAI, models: list[str], reasoning_effort: str | None,
                 timeout: float = 12.0, background: bool = False, preemptible: bool = False,
                 ctx_limit: int = 0, think: bool | None = None, tag: str = ""):
        self.ctx_limit = ctx_limit        # окно модели (только локальная): следим, чтобы промпт влезал
        # размышления локальной модели: False — выключить (живой ответ: быстро), True — включить (фоновые дела:
        # никто не ждёт, а пишут аккуратнее), None — не трогать (как решит сама модель)
        self.think = think
        self.tag = tag                    # для лога: «ответ», «фон», «встревать ли»
        self._think_ok = True             # сервер понимает выключатель размышлений (старая Ollama — нет)
        self._think_seen = False
        self.background = background      # фоновые дела: ждут, пока бот не отвечает людям
        self.preemptible = preemptible    # и обрываются, если ответ понадобился (имеет смысл для Ollama)
        self.client = client
        self.models = [m for i, m in enumerate(models) if m and m not in models[:i]]
        self.reasoning_effort = reasoning_effort
        self._blocked_until: dict[str, float] = {}
        self.timeout = timeout

    async def create(self, **kwargs):
        """То же, что client.chat.completions.create, но с перебором моделей при 429.
        Фоновый роутер сначала ждёт своей очереди (см. foreground)."""
        if not self.background:
            return await self._create(**kwargs)
        for attempt in range(BG_MAX_PREEMPT + 1):
            await _wait_turn()
            task = asyncio.ensure_future(self._create(**kwargs))
            if self.preemptible and attempt < BG_MAX_PREEMPT:
                _bg_inflight.add(task)
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                task.cancel()
                raise
            finally:
                _bg_inflight.discard(task)
            if not task.cancelled():
                return task.result()
            log.info("фоновое дело уступило живому ответу — доделаю позже")
        raise RuntimeError("фоновое дело так и не дождалось очереди")

    async def _create(self, **kwargs):
        tag = kwargs.pop("tag", "")
        if kwargs.get("stream"):
            kwargs.setdefault("stream_options", {"include_usage": True})  # размер промпта придёт в конце потока
        resp = await self._create_raw(**kwargs)
        if not kwargs.get("stream"):
            note_usage(getattr(resp, "usage", None), tag, self.ctx_limit, kwargs.get("max_completion_tokens", 0))
            if resp.choices:
                self.note_thought(_thought(resp.choices[0].message))
        return resp

    def note_thought(self, text: str) -> None:
        """Один раз в лог: думала ли модель — видно, сработал ли выключатель размышлений."""
        if self._think_seen or not self.ctx_limit:
            return
        if text:
            self._think_seen = True
            if self.think is False:
                log.warning("🐢 модель (%s) всё равно думает перед ответом (%d символов мыслей) — выключатель "
                            "не сработал, ответы медленнее. Обнови Ollama.", self.tag or "запрос", len(text))
            else:
                log.info("💭 модель (%s) думает перед ответом — как и задумано", self.tag or "запрос")
        elif self.think:
            self._think_seen = True
            log.info("модель (%s) не думает, хотя можно (THINK_BACKGROUND=on) — значит, не умеет; это не страшно",
                     self.tag or "запрос")

    def _extra_body(self, model: str) -> dict | None:
        if not self.ctx_limit:  # облако: как раньше (GROQ_REASONING_EFFORT)
            return _reasoning_for(model, self.reasoning_effort)
        if self.think is None or not self._think_ok:
            return None
        # Ollama (OpenAI-совместимый вход): reasoning_effort none/low/medium/high; think — для её родного входа
        return {"reasoning_effort": "medium" if self.think else "none", "think": self.think}

    async def _create_raw(self, **kwargs):
        now = time.monotonic()
        available = [m for m in self.models if self._blocked_until.get(m, 0) <= now] or self.models[-1:]
        last_err: Exception | None = None
        for model in available:
            kw = dict(kwargs)
            if "gpt-oss" in model and "max_completion_tokens" in kw:
                kw["max_completion_tokens"] += 250  # gpt-oss часть токенов тратит на «размышления»
            if self.ctx_limit and "max_completion_tokens" in kw:
                if self.think:
                    kw["max_completion_tokens"] += THINK_EXTRA_TOKENS
                # Ollama знает только старое имя max_tokens, новое молча пропускает — и длина ответа без потолка
                kw["max_tokens"] = kw.pop("max_completion_tokens")
            try:
                # max_retries=0: библиотека openai сама ждёт по минуте при 429 — нам это не нужно,
                # мы мгновенно переходим на следующую модель
                return await self.client.with_options(max_retries=0, timeout=self.timeout).chat.completions.create(
                    model=model, extra_body=self._extra_body(model), **kw)
            except openai.BadRequestError as e:
                if self.ctx_limit and self._think_ok and self.think is not None and "reason" in str(e).lower():
                    # старая Ollama не знает выключателя размышлений — работаем без него
                    log.warning("Ollama не понимает выключатель размышлений (%s) — обнови её; пока без него", e)
                    self._think_ok = False
                    return await self._create_raw(**kwargs)
                raise
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
