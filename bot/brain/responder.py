from __future__ import annotations

import re
from typing import AsyncIterator

from bot import persona
from bot.brain.llm import LLMRouter

# Qwen иногда вставляет китайские/корейские слова — голос их не прочтёт, вырезаем
_CJK = re.compile(r"[　-鿿가-힯＀-￯]+")
# конец предложения: знак препинания (+ закрывающая кавычка/скобка) и пробел
_SENT_END = re.compile(r"[.!?…]+[\"»)]?\s+")


# эмодзи и прочие пиктограммы: голос их не прочтёт (вместо них персонаж пишет ремарки в скобках)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")
# звёздочки-ремарки «*ухмыляется*» превращаем в скобки «(ухмыляется)»
_STARS = re.compile(r"\*([^*\n]{1,60})\*")


def _clean(text: str) -> str:
    text = _STARS.sub(r"(\1)", _EMOJI.sub("", _CJK.sub("", text)))
    return re.sub(r"\s{2,}", " ", text).strip()


class Responder:
    def __init__(self, llm: LLMRouter, system_prompt: str, extra_context=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.extra_context = extra_context  # функция → доп. текст в системный промпт (отношения к людям)

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
        return self._generate_stream(transcript, self._task_batch(new_lines), max_tokens=80)

    def stream_farewell(self, transcript: str) -> AsyncIterator[str]:
        return self._generate_stream(transcript, persona.FAREWELL_TASK, max_tokens=40)

    def stream_proactive(self, transcript: str, topic: str, angle: str) -> AsyncIterator[str]:
        return self._generate_stream(transcript, self._task_proactive(topic, angle), max_tokens=100)

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
