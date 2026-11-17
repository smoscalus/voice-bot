from __future__ import annotations

import logging
import re
from typing import AsyncIterator

from bot import persona
from bot.brain.llm import LLMRouter

log = logging.getLogger(__name__)

# Qwen иногда вставляет китайские/корейские слова — голос их не прочтёт, вырезаем
_CJK = re.compile(r"[　-鿿가-힯＀-￯]+")
# конец предложения: знак препинания (+ закрывающая кавычка/скобка) и пробел
_SENT_END = re.compile(r"[.!?…]+[\"»)]?\s+")


# эмодзи и прочие пиктограммы: голос их не прочтёт (вместо них персонаж пишет ремарки в скобках)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")
# звёздочки-ремарки «*ухмыляется*» превращаем в скобки «(ухмыляется)»
_STARS = re.compile(r"\*([^*\n]{1,60})\*")


# служебная пометка распознавания — модель иногда копирует её себе в ответ, вслух её читать нельзя
_UNCLEAR_MARK = re.compile(r"\((?:не|недо)?\s*разборчив\w*\)|\(переспрашивает\)", re.IGNORECASE)


def _clean(text: str) -> str:
    text = _UNCLEAR_MARK.sub("", text)
    text = _STARS.sub(r"(\1)", _EMOJI.sub("", _CJK.sub("", text)))
    return re.sub(r"\s{2,}", " ", text).strip()


_REMARK = re.compile(r"\([^)]{0,60}\)")
_GREETING = re.compile(r"^(привет|здорово|здарова|хай|йо|ну|слушай|эй|о)\b[\s,!.]*", re.IGNORECASE)


def _norm_for_repeat(text: str) -> str:
    """Для сравнения: без ремарок в скобках, приветствий в начале, знаков и регистра."""
    t = _REMARK.sub(" ", text.lower().replace("ё", "е"))
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for _ in range(2):  # «привет, слушай, …»
        t = _GREETING.sub("", t).strip()
    return t


def find_repeat(sentence: str, recent: list[str], threshold: int = 80) -> str | None:
    """Если фраза повторяет (почти дословно или по смыслу набора слов) что-то из недавних реплик — вернёт ту фразу.
    Короткие ответы («нормально», «не знаю») повторами не считаем — это норма."""
    from rapidfuzz import fuzz

    s = _norm_for_repeat(sentence)
    if len(s.split()) < 3:
        return None
    for line in recent:
        for old in re.split(r"(?<=[.!?…])\s+", line):
            o = _norm_for_repeat(old)
            if len(o.split()) < 3:
                continue
            if fuzz.token_set_ratio(s, o) >= threshold or fuzz.ratio(s, o) >= threshold - 5:
                return old.strip()
    return None


class Responder:
    def __init__(self, llm: LLMRouter, system_prompt: str, extra_context=None, recent_lines=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.extra_context = extra_context  # функция → доп. текст в системный промпт (отношения к людям)
        self.recent_lines = recent_lines    # функция → последние реплики бота (для защиты от повторов)

    # ---------------------------------------------------------------- задачи
    @staticmethod
    def _task_direct(speaker: str, request: str) -> str:
        return (f"К тебе обратился {speaker}. Его последняя реплика: «{request}». "
                "Ответь ему, учитывая контекст разговора.")

    @staticmethod
    def _task_batch(new_lines: str) -> str:
        return (
            f"С твоего последнего ответа в чате сказали:\n{new_lines}\n\n"
            f"{persona.DIRECT_TASK} "
            "Реплики, обращённые не к тебе, — просто контекст. "
            "Если твою прошлую реплику перебили — не повторяй её."
        )

    @staticmethod
    def _task_proactive(topic: str, angle: str) -> str:
        return persona.PROACTIVE_TASK.format(topic=topic, angle=angle)

    # ---------------------------------------------------------------- целиком
    async def reply_direct(self, transcript: str, speaker: str, request: str) -> str:
        return await self._generate(transcript, self._task_direct(speaker, request))

    async def reply_batch(self, transcript: str, new_lines: str) -> str:
        return await self._generate(transcript, self._task_batch(new_lines))

    async def reply_proactive(self, transcript: str, topic: str, angle: str) -> str:
        return await self._generate(transcript, self._task_proactive(topic, angle))

    # ---------------------------------------------------------------- потоково, по предложениям
    def stream_batch(self, transcript: str, new_lines: str) -> AsyncIterator[str]:
        # на обращения — мало токенов: обычно ответ в пару слов, при интересе — пара фраз
        return self._guarded_stream(transcript, self._task_batch(new_lines), max_tokens=80)

    async def reply_task(self, transcript: str, task: str, max_tokens: int = 100) -> str:
        """Реплика по заданию целиком (нужно, когда ответ сначала проверяем — например, «ПРОПУСК»)."""
        resp = await self.llm.create(messages=self._messages(transcript, task), max_completion_tokens=max_tokens)
        text = _clean(resp.choices[0].message.content or "")
        recent = self.recent_lines() if self.recent_lines else []
        if text and find_repeat(text, recent):
            log.info("своя тема повторяет старое («%s») — промолчу", text)
            return ""
        return text

    def stream_task(self, transcript: str, task: str, max_tokens: int = 60) -> AsyncIterator[str]:
        """Произвольная короткая реплика по заданию (например, пожаловаться на шум)."""
        return self._generate_stream(transcript, task, max_tokens=max_tokens)

    def stream_farewell(self, transcript: str) -> AsyncIterator[str]:
        return self._generate_stream(transcript, persona.FAREWELL_TASK, max_tokens=40)

    def stream_proactive(self, transcript: str, topic: str, angle: str) -> AsyncIterator[str]:
        return self._guarded_stream(transcript, self._task_proactive(topic, angle), max_tokens=100)

    async def _guarded_stream(self, transcript: str, task: str, max_tokens: int) -> AsyncIterator[str]:
        """Поток ответа с защитой от повторов: если первая фраза повторяет недавнюю реплику бота —
        выбрасываем её и генерируем заново с явным запретом; повторы дальше в ответе просто пропускаем."""
        recent = self.recent_lines() if self.recent_lines else []
        gen = self._generate_stream(transcript, task, max_tokens)
        first = None
        async for sentence in gen:
            first = sentence
            break
        if first is not None and (same := find_repeat(first, recent)):
            await gen.aclose()
            log.info("повтор «%s» (уже говорил «%s») — придумываю заново", first, same)
            task = (f"{task}\nВНИМАНИЕ: ты уже говорил «{same}». НЕ повторяй это и не задавай этот вопрос снова "
                    "(даже другими словами). Скажи что-то другое: ответь по сути, подколи за реакцию, "
                    "смени тему или просто отстань.")
            gen = self._generate_stream(transcript, task, max_tokens)
            first = None
        if first is not None:
            yield first
        async for sentence in gen:
            if find_repeat(sentence, recent):
                log.info("пропускаю повтор в ответе: «%s»", sentence)
                continue
            yield sentence

    def _messages(self, transcript: str, task: str) -> list[dict]:
        system = self.system_prompt
        if self.extra_context and (extra := self.extra_context()):
            system = f"{system}\n{extra}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Расшифровка последних минут разговора:\n{transcript}\n\nЗадача: {task}"},
        ]

    async def _generate(self, transcript: str, task: str) -> str:
        resp = await self.llm.create(
            messages=self._messages(transcript, task),
            max_completion_tokens=100,
        )
        return _clean(resp.choices[0].message.content or "")

    async def _generate_stream(self, transcript: str, task: str, max_tokens: int = 100) -> AsyncIterator[str]:
        """Выдаёт ответ по предложениям по мере генерации — озвучка стартует с первого же."""
        stream = await self.llm.create(
            messages=self._messages(transcript, task),
            max_completion_tokens=max_tokens,
            stream=True,
        )
        buf = ""
        try:
            async for event in stream:
                if not event.choices:
                    continue
                buf += event.choices[0].delta.content or ""
                while (m := _SENT_END.search(buf)) is not None:
                    sentence, buf = _clean(buf[: m.end()]), buf[m.end():]
                    if sentence:
                        yield sentence
            if tail := _clean(buf):
                yield tail
        finally:
            await stream.close()
