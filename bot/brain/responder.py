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


def _unwrap_speech(m: re.Match) -> str:
    inner = m.group(1)
    return f" {inner} " if "?" in inner or len(inner.split()) >= 6 else m.group(0)


def _clean(text: str) -> str:
    text = re.sub(r"\(([^)]{1,200})\)", _unwrap_speech, _UNCLEAR_MARK.sub("", text))
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


# слова-«каркас» вопросов и неуверенности: «что такое X», «я не уверен, что значит X», «можно объяснить X» —
# одинаковый каркас ещё не повтор, сравниваем то, О ЧЁМ фраза (иначе «что такое фейерверк?» = «что такое мем?»)
_FRAME = {"такое", "такой", "такая", "значи", "означ", "можно", "может", "объяс", "поним", "понял", "понят",
          "знаю", "знаеш", "увере", "интер", "кажет", "думаю", "хочу", "хотел", "слово", "слова", "совсе",
          "точно", "немно", "очень", "всё", "все", "ещё", "было", "есть", "быть", "будет", "меня", "мой",
          "моё", "мое", "мои", "моя", "это", "этот", "эта", "эти", "кто", "где", "когда", "почем", "зачем",
          "или", "если", "чем", "тоже", "просто", "бы", "нет", "ага", "угу", "вы", "вас", "вам"}


_LOST = re.compile(r"не\s+(?:совсем\s+|очень\s+)?(?:понимаю|понял|пойму|ясно)[^.!?]{0,25}о\s+ч[её]м", re.I)


def _content(norm: str) -> list[str]:
    return [w for w in norm.split() if w not in _STOP and w not in _FRAME]


def find_repeat(sentence: str, recent: list[str], threshold: int = 80) -> str | None:
    """Если фраза повторяет (почти дословно или по смыслу) что-то из недавних реплик — вернёт ту фразу.
    Короткие ответы («нормально», «не знаю») повторами не считаем — это норма. Сравниваем суть
    (смысловые слова), а не каркас: новый вопрос про другое слово — не повтор."""
    from rapidfuzz import fuzz

    s = _norm_for_repeat(sentence)
    if _LOST.search(sentence):  # «не понимаю, о чём вы говорите» — второй раз подряд не надо
        for line in recent:
            if m := _LOST.search(line):
                return m.group(0)
    if len(s.split()) >= 2:  # короткое, но дословно то же («привет, как дела?» по кругу) — тоже повтор
        for line in recent:
            for old in re.split(r"(?<=[.!?…])\s+", line):
                if s == _norm_for_repeat(old):
                    return old.strip()
    s_words = set(_content(s))
    if len(s.split()) < 3 or len(s_words) < 2:
        return None
    s_c = " ".join(_content(s))
    for line in recent:
        for old in re.split(r"(?<=[.!?…])\s+", line):
            o_list = _content(_norm_for_repeat(old))
            o_words = set(o_list)
            if len(o_words) < 2:  # «не знаю», «нормально» — не считаем
                continue
            same = s_words & o_words
            # тот же смысл другими словами: почти все смысловые слова одной фразы есть в другой
            by_meaning = len(same) >= 2 and len(same) / min(len(s_words), len(o_words)) >= 0.75
            if by_meaning or (len(s_c) >= 10 and fuzz.ratio(s_c, " ".join(o_list)) >= threshold + 5):
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


_SELF_ADDRESS = re.compile(r"^\s*(?:п[еэаь]?т(?:р[иыеэ]к|ер|р)\w{0,2})\s*[,!.:]\s*", re.IGNORECASE)
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


_ASK_WORD = re.compile(r"(?:что|чё|че)\s+(?:такое|значит|означает)\s+[«\"'„]?([\w-]+)[»\"'“]?\s*[?.!…]*\s*$",
                       re.IGNORECASE)


def _basic_stems() -> set[str]:
    return {part[:5] for w in getattr(persona, "BASIC_WORDS", [])
            for part in w.lower().replace("ё", "е").replace("-", " ").split()}


def asks_basic_word(sentence: str) -> bool:
    """«А что значит врать?», «Что такое да-да-да?» — обычные слова он знает с рождения."""
    m = _ASK_WORD.search(sentence.strip())
    if not m:
        return False
    word = m.group(1).lower().replace("ё", "е")
    parts = word.split("-")
    return word in {"что", "это", "так", "да", "нет"} or all(p[:5] in _basic_stems() for p in parts if p)


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
    def stream_batch(self, transcript: str, new_lines: str, note: str = "") -> AsyncIterator[str]:
        # на обращения — мало токенов: обычно ответ в пару слов, при интересе — пара фраз
        heard = [line.split(": ", 1)[1] for line in new_lines.splitlines() if ": " in line]
        task = self._task_batch(new_lines) + (f"\n{note}" if note else "")
        return self._guarded_stream(transcript, task, max_tokens=80, heard=heard)

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
        # ремарки в скобках — редко: если в двух последних репликах уже была ремарка, в этой без неё
        no_remarks = any(_REMARK.search(line) for line in recent[-2:])
        heard_l = list(heard)

        def prepare(sentence: str) -> tuple[str, str | None, bool]:
            """→ (фраза, на что похожа, это повтор себя (а не эхо за человеком))."""
            s = _tidy(sentence, greeted)
            if no_remarks:
                s = re.sub(r"\s+([.,!?…])", r"\1", re.sub(r"\s{2,}", " ", _REMARK.sub("", s))).strip()
            if not s:
                return "", None, False
            if asks_basic_word(s):
                log.info("выкидываю вопрос про обычное слово: «%s»", s)
                return "", None, False
            own = find_repeat(s, recent)
            return s, own or find_parrot(s, heard_l), bool(own)

        # Повторную фразу просто выкидываем и продолжаем тот же ответ — без новой генерации (на локальной
        # модели это ещё 10–20 с). Заново генерируем, только если от ответа совсем ничего не осталось.
        for attempt in range(2):
            gen = self._generate_stream(transcript, task, max_tokens)
            said_real, pending, bad, own_bad, parrots = False, [], None, False, []
            async for sentence in gen:
                s, same, own = prepare(sentence)
                if same:
                    bad = bad or same
                    own_bad = own_bad or own
                    if not own:
                        parrots.append(s)
                    log.info("пропускаю повтор в ответе: «%s» (уже было «%s»)", s, same)
                    continue
                if not s:
                    continue
                if not _REMARK.sub("", s).strip(" .,!?…"):
                    pending.append(s)  # одна ремарка без слов — скажем, только если за ней будут слова
                    continue
                for r in pending:
                    yield r
                pending.clear()
                said_real = True
                yield s
            if said_real or bad is None:
                return
            if parrots and not own_bad:
                # отброшено только «эхо» за человеком (переспрос его же слов) — лучше сказать его,
                # чем ждать новую генерацию ещё 10–20 с
                yield parrots[0]
                return
            log.info("весь ответ — повтор «%s», придумываю заново", bad)
            task = (f"{task}\nВНИМАНИЕ: ты уже говорил «{bad}». НЕ повторяй это и не задавай этот вопрос снова "
                    "(даже другими словами), не повторяй за людьми их слова. Скажи что-то другое по сути.")

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
        usage_logged = False
        try:
            async for event in stream:
                if getattr(event, "usage", None) and not usage_logged:
                    from bot.brain.llm import note_usage

                    usage_logged = True  # некоторые серверы присылают размер в двух кусках потока
                    note_usage(event.usage, "ответ", getattr(self.llm, "ctx_limit", 0), max_tokens)
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
