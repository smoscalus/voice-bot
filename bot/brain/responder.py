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


# мусор, который выдают маленькие модели: `код в обратных кавычках`, слипшиеся английские слова (OccupationPlanner)
_JUNK = re.compile(r"`[^`]*`|`")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def _junk_word(m: re.Match) -> str:
    """Латиница, прилипшая к русскому слову, или длинное слипшееся CamelCase — мусор модели.
    Нормальные слова вроде YouTube, TikTok, PlayStation оставляем."""
    w, start = m.group(0), m.start()
    glued = start > 0 and "а" <= m.string[start - 1].lower() <= "я"
    camel = len(w) >= 12 and re.search(r"[a-z][A-Z]", w)
    return " " if glued or camel else w


def _clean(text: str) -> str:
    text = _UNCLEAR_MARK.sub("", text)
    text = _LATIN_WORD.sub(_junk_word, _JUNK.sub(" ", text))
    text = _STARS.sub(r"(\1)", _EMOJI.sub("", _CJK.sub("", text)))
    return re.sub(r"\s{2,}", " ", text).strip()


_REMARK = re.compile(r"\([^)]{0,60}\)")
_GREETING = re.compile(r"^(привет|здорово|здарова|хай|йо|ну|слушай|эй|о)\b[\s,!.]*", re.IGNORECASE)


# служебные слова не несут смысла — при сравнении «по смыслу» их не учитываем (уже обрезаны до 5 букв)
_STOP = {"а", "и", "но", "да", "ты", "я", "мы", "вы", "он", "она", "они", "не", "ну", "что", "как", "это",
         "у", "в", "на", "с", "по", "к", "о", "за", "из", "тебя", "тебе", "мне", "меня", "ли", "же", "то",
         "так", "тут", "там", "вот", "уже", "еще", "сейча", "сегод", "вообщ", "прост", "твой", "твоя",
         "твои", "твое"}


def _norm_for_repeat(text: str) -> str:
    """Для сравнения: без ремарок в скобках, приветствий в начале, знаков и регистра."""
    t = _REMARK.sub(" ", text.lower().replace("ё", "е"))
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for _ in range(2):  # «привет, слушай, …»
        t = _GREETING.sub("", t).strip()
    # грубый стемминг: «занимаешься» / «занимался» / «заниматься» → «заним» — ловим повтор по смыслу
    return " ".join(w[:5] for w in t.split())


def find_repeat(sentence: str, recent: list[str], threshold: int = 80) -> str | None:
    """Если фраза повторяет (почти дословно или по смыслу набора слов) что-то из недавних реплик — вернёт ту фразу.
    Короткие ответы («нормально», «не знаю») повторами не считаем — это норма."""
    from rapidfuzz import fuzz

    s = _norm_for_repeat(sentence)
    if len(s.split()) >= 2:  # короткое, но дословно то же («привет, как дела?» по кругу) — тоже повтор
        for line in recent:
            for old in re.split(r"(?<=[.!?…])\s+", line):
                if s == _norm_for_repeat(old):
                    return old.strip()
    s_words = set(s.split()) - _STOP
    if len(s.split()) < 3 or not s_words:
        return None
    for line in recent:
        for old in re.split(r"(?<=[.!?…])\s+", line):
            o = _norm_for_repeat(old)
            o_words = set(o.split()) - _STOP
            if len(o_words) < 2:  # «не знаю», «нормально» — не считаем
                continue
            same = s_words & o_words
            # тот же вопрос другими словами: почти все смысловые слова старой фразы есть в новой
            by_meaning = len(same) >= 2 and len(same) / min(len(s_words), len(o_words)) >= 0.75
            if by_meaning or fuzz.ratio(s, o) >= threshold - 5:
                return old.strip()
    return None


def find_parrot(sentence: str, heard: list[str]) -> str | None:
    """Бот повторяет за человеком его же слова («отстань уже» → «Вася, отстань уже, а то…»)."""
    s = _norm_for_repeat(sentence)
    bot = _norm_for_repeat(persona.BOT_NAME)
    for line in heard:
        h = " ".join(w for w in _norm_for_repeat(line).split() if w != bot)
        if len(h) >= 8 and len(h.split()) >= 2 and h in s:
            return line.strip()
    return find_repeat(sentence, heard)


_SELF_ADDRESS = re.compile(rf"^\s*{persona.BOT_NAME}\s*[,!.:]\s*", re.IGNORECASE)
_NAME_GREETING_ONLY = re.compile(r"^\s*[^,.!?]{1,30},\s*(привет|здорово|здарова|хай)\s*[.!]*\s*$", re.IGNORECASE)
_LEADING_GREETING = re.compile(r"^\s*(привет|здорово|здарова|хай)\s*[,!.]?\s*", re.IGNORECASE)


def _tidy(sentence: str, greeted: bool) -> str:
    """Убираем обращение бота к самому себе («Петрык, привет») и повторное «привет», если уже здоровался."""
    s = _SELF_ADDRESS.sub("", sentence)
    if greeted:
        if _NAME_GREETING_ONLY.match(s):  # «Вася, привет.» отдельной фразой — уже здоровался, выкидываем
            return ""
        s = _LEADING_GREETING.sub("", s)
    s = s.strip()
    return (s[0].upper() + s[1:]) if s else s


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
        heard = [line.split(": ", 1)[1] for line in new_lines.splitlines() if ": " in line]
        return self._guarded_stream(transcript, self._task_batch(new_lines), max_tokens=80, heard=heard)

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

    async def _guarded_stream(self, transcript: str, task: str, max_tokens: int,
                              heard: list[str] = ()) -> AsyncIterator[str]:
        """Поток ответа с защитой от повторов: если первая фраза повторяет недавнюю реплику бота
        или попугайничает за человеком (heard — что ему только что сказали) — выбрасываем её и генерируем
        заново с явным запретом; повторы дальше в ответе просто пропускаем."""
        recent = self.recent_lines() if self.recent_lines else []
        greeted = any(re.search(r"\b(привет|здорово|здарова)\b", line.lower()) for line in recent)
        if len(recent) >= 2 and all("?" in line for line in recent[-2:]):
            # квота на вопросы: два ответа подряд были вопросами — хватит допрашивать
            task += ("\nВ ЭТОТ РАЗ НЕ ЗАДАВАЙ ВОПРОСОВ: ты и так допрашиваешь людей подряд. "
                     "Ответь утверждением — отреагируй, поделись мыслью, догадкой или тем, что понял.")
        gen = self._generate_stream(transcript, task, max_tokens)
        first = None
        async for sentence in gen:
            first = _tidy(sentence, greeted)
            break
        if first is not None and (same := find_repeat(first, recent)):
            await gen.aclose()
            log.info("повтор «%s» (уже говорил «%s») — придумываю заново", first, same)
            task = (f"{task}\nВНИМАНИЕ: ты уже говорил «{same}». НЕ повторяй это и не задавай этот вопрос снова "
                    "(даже другими словами). Скажи что-то другое: ответь по сути, поделись мыслью, "
                    "смени тему или просто помолчи об этом.")
            gen = self._generate_stream(transcript, task, max_tokens)
            first = None
        elif first is not None and (same := find_parrot(first, list(heard))):
            await gen.aclose()
            log.info("попугайничает «%s» (человек сказал «%s») — придумываю заново", first, same)
            task = (f"{task}\nВНИМАНИЕ: не повторяй за человеком его же слова («{same}») — "
                    "ответь своими словами по сути.")
            gen = self._generate_stream(transcript, task, max_tokens)
            first = None
        if first:
            yield first
        async for sentence in gen:
            sentence = _tidy(sentence, greeted)
            if not sentence or find_repeat(sentence, recent) or find_parrot(sentence, list(heard)):
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
