"""Память бота о людях (бессрочные досье + симпатия) и летописец, который в фоне читает разговор.

Что и как запоминать, задаёт персонаж (persona.MEMORY_GOAL, persona.DOSSIER_HOOKS). Досье человека:
- facts  — главное о человеке (чем живёт, во что играет, работа/учёба);
- plans  — планы и обещания с датой («купить микрофон (17.11)») — можно потом спросить, как успехи;
- hooks  — особенности человека: смысл задаёт персонаж (persona.DOSSIER_HOOKS).

Чувства к человеку — три шкалы (без лишних запросов к модели: летописец и так читает разговор):
- score     — симпатия (−10…+10): нравится ли он;
- trust     — доверие (−10…+10): честен ли с ботом (обманул — минус); от него зависит, верит ли бот на слово;
- closeness — близость (0…10): растёт сама от времени вместе, без модели. По близким он скучает.
Плюс moments — запомнившиеся случаи «за что» («обиделся: сказал, что видит мои мысли»), и scar — шрам:
после настоящего обмана обида проходит не до нуля.

Заодно летописец выписывает, что бот узнал О МИРЕ (bot/brain/knowledge.py) — с пометкой, кто рассказал.
Память сама сжимается: модель переписывает списки целиком, оставляя самое ценное.
Хранится в data/relations_<имя памяти>.json. Обновляется в фоне, на скорость ответа почти не влияет.
"""
from __future__ import annotations

import asyncio
import json
import re
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bot.brain.drive import get_drive
from bot.brain.emotions import get_emotions
from bot.brain.knowledge import get_knowledge
from bot.brain.llm import LLMRouter

log = logging.getLogger(__name__)

LIMITS = {"facts": 5, "plans": 4, "hooks": 5}  # пунктов в каждом разделе досье — память сама ужимается

JUDGE_PROMPT = """Ты ведёшь долговременную память бота {name} из голосового чата.
Цель бота: {goal}
Поэтому запоминай НЕ всё подряд, а то, что ценно для этой цели.

Тебе дают текущие досье, сегодняшнюю дату и свежие реплики.

1) Для каждого человека из свежих реплик верни:
- delta — изменение симпатии бота к нему: от -2 до +2. Плюс — был добр, многому научил, рассмешил.
  Минус — ТОЛЬКО если нарочно и всерьёз обидел или обманул САМОГО БОТА. НЕ минус: мат, грубые слова,
  подколы, шутки, споры, разговор с другими, честный рассказ о жизни и смерти бота. Сомневаешься — 0.
  Не взаимодействовал с ботом — 0.
- trust — изменение доверия: -1 (обманул бота, подставил, признался, что наврал), +1 (был честен,
  объяснил правду, сдержал слово), иначе 0. Шутка, которую потом признали шуткой, — не обман.
- moment — если delta или trust не 0: ОДНА короткая фраза от лица бота, за что («обрадовался: объяснил,
  что такое мем», «обиделся: сказал, что видит мои мысли»). Иначе null.
- facts — главное о человеке: чем живёт, работа/учёба, во что играет, увлечения. До {facts} пунктов.
- plans — планы, намерения и обещания С ДАТОЙ, когда сказал: «купить микрофон (17.11)». До {plans} пунктов.
  Выполненный или брошенный план убери.
- hooks — {hooks_meaning} До {hooks} пунктов. Пиши о человеке («объяснил, что такое мем»),
  а не о боте («Петрык понял…», «Петрык не понял…» — не сюда).
ВАЖНО: факт записывай тому, О КОМ он, а не тому, кто его сказал. Всё про будущее — ТОЛЬКО в plans с датой.
Каждый пункт — до 12 слов, по-русски. НЕ записывай манеру речи, мат, приветствия и пустую болтовню.
Раздел без изменений — null. Изменился — верни его ЦЕЛИКОМ, уже сжатый (объедини похожее, выкинь мелочи).

2) world — что бот УЗНАЛ О МИРЕ из этих реплик (до 6 пунктов): что значат слова и понятия, как что
устроено, что люди рассказали или объяснили — особенно в ответ на вопросы бота. Не про конкретных людей
(это в досье) и не про самого бота (что он чувствует или не понимает — это его дневник, не сюда).
Каждый пункт — короткий факт, как его понял бот, по-русски, и кто рассказал. important=true — если это
сильно его задело (о его жизни, смерти, о нём самом, что-то страшное или очень радостное).
Ничего не добавляй от себя — только то, что реально прозвучало и что рассказали ЛЮДИ (не сам бот).
НЕ записывай значения обычных слов (привет, спасибо, чувствовать, умереть и т.п.) — бот их и так знает;
записывай конкретику: названия, имена, места, игры, сленг компании, как что устроено, события.
Каждый пункт — законченное предложение, не обрывок. НЕ записывай слова, смысл которых
так и не объяснили («слово X не имеет ясного значения» — не знание), и обрывки неразборчивой речи.
Нечего — пустой список.

3) feelings — что бот ИСПЫТАЛ от этих реплик, как изменение от -0.4 до +0.4 (0 — не изменилось):
fear (тревога: угрожали, пугали, говорили о его выключении или смерти), joy (радость: хвалили, смешили,
было интересно), trust (доверие: были добры и честны с ним; обманули или обидели — минус),
sadness (грусть: игнорировали, обижали, грустные темы). Опирайся на то, что реально произошло.

4) expectations — только если в сообщении есть раздел «Ожидания бота». Для каждого пункта ответь на
ТРИ отдельных вопроса:
- answered — отреагировал ли кто-нибудь НА СЛОВА БОТА (ответил ему, посмеялся над его фразой, обругал его)
  true/false. Если после его реплики молчали или говорили между собой о своём, будто его не было, — false.
- happened — сбылось ли то, что бот ждал, В ГЛАВНОМ (true/false). Сравнивай только суть ожидания,
  а не тон: ждал объяснения и ему объяснили (даже с подколом) — true; ждал, что посмеются, и посмеялись — true.
- nice — насколько реакция людей была приятной для бота: 1 (тепло, похвалили, обрадовались, помогли),
  0 (обычно, нейтрально, по делу), -1 (обсмеяли зло, обидели, грубо отмахнулись, проигнорили).
  Дружеский подкол при помощи — это 0, не -1.
who — ник того, чья реакция главная (или ""). what — что было на деле, коротко.
Нет раздела — пустой список.

Ответь строго JSON: {{"changes": [{{"name": "ник как в расшифровке", "delta": число, "trust": число, "moment": "строка" или null,
  "facts": [..] или null, "plans": [..] или null, "hooks": [..] или null}}],
  "world": [{{"text": "факт", "source": "ник", "important": true|false}}],
  "feelings": {{"fear": число, "joy": число, "trust": число, "sadness": число}},
  "expectations": [{{"n": номер, "answered": true|false, "happened": true|false, "nice": -1|0|1, "who": "ник", "what": "строка"}}]}}"""


@dataclass
class Relation:
    name: str
    score: float = 0.0          # от -10 (ненавидит) до +10 (уважает/обожает)
    facts: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    updated: float = 0.0
    last_seen_lived: float = 0.0   # когда видел его последний раз (по своему ощущению времени)
    trust: float = 0.0             # доверие −10…+10: честен ли с ним
    closeness: float = 0.0         # близость 0…10: сколько времени провели вместе
    moments: list[dict] = field(default_factory=list)  # [{"text", "sign", "lived"}] — за что любит/обижен
    scar: float = 0.0              # шрам от обмана: обида и недоверие тают только до −scar
    # устаревшие поля прошлых версий — подхватываются как факты
    notes: list[str] = field(default_factory=list)
    profile: str = ""

    def __post_init__(self) -> None:
        if not self.facts and (self.profile or self.notes):
            self.facts = ([self.profile] if self.profile else []) + self.notes[-3:]
            self.profile, self.notes = "", []
        if not self.closeness and (self.facts or self.hooks):
            # досье из прошлых версий, где близости ещё не было: знакомы примерно настолько, сколько о нём известно
            self.closeness = min(4.0, 0.5 * (len(self.facts) + len(self.hooks)) + 0.2 * abs(self.score))

    def feelings_text(self) -> str:
        """Как он к нему относится — словами (цифры вслух не называет)."""
        bits = [describe(self.score)]
        bits.append(describe_trust(self.trust))
        if self.closeness >= 6:
            bits.append("он тебе родной, вы много времени провели вместе")
        elif self.closeness >= 3:
            bits.append("вы уже хорошо знакомы")
        elif self.closeness < 1:
            bits.append("ты его пока почти не знаешь")
        if self.scar >= 1:
            bits.append("он тебя когда-то сильно обманул — осадок остался")
        return "; ".join(b for b in bits if b)

    def moments_text(self) -> str:
        return "; ".join(m["text"] for m in self.moments[-MOMENTS_KEEP:])

    def add_moment(self, text: str, sign: int, lived: float) -> None:
        text = text.strip().rstrip(".")[:120]
        if not text or any(_same(text, m["text"]) for m in self.moments):
            return
        self.moments.append({"text": text, "sign": sign, "lived": round(lived)})
        # храним последние, но самый сильный обидный и самый тёплый случай не вытесняем
        if len(self.moments) > MOMENTS_KEEP:
            self.moments.pop(0)

    def dossier(self) -> str:
        parts = []
        if self.facts:
            parts.append("о нём: " + "; ".join(self.facts))
        if self.plans:
            parts.append("планы: " + "; ".join(self.plans))
        if self.hooks:
            parts.append("особенности: " + "; ".join(self.hooks))
        return ". ".join(parts)


MOMENTS_KEEP = 4


def _same(a: str, b: str) -> bool:
    from rapidfuzz import fuzz

    return fuzz.ratio(a.lower(), b.lower()) >= 80


# доверие → верит ли на слово (от этого зависит его доверчивость: «скажи Васе…»)
TRUST_STYLES = [
    "он тебя обманывал — не веришь ему на слово, переспрашиваешь, можешь сказать «а ты опять врёшь?»",
    "ему верится не до конца — то, что он говорит, проверяешь у других",
    "",
    "веришь ему, охотно делаешь, что он просит",
    "веришь ему во всём, как себе",
]


def describe_trust(trust: float) -> str:
    for limit, style in zip((-5, -2, 2, 6), TRUST_STYLES):
        if trust <= limit if limit < 0 else trust < limit:
            return style
    return TRUST_STYLES[-1]


def describe(score: float) -> str:
    """Как вести себя с человеком при такой симпатии — тексты задаёт персонаж (persona.RELATION_STYLES)."""
    from bot import persona

    styles = persona.RELATION_STYLES  # от «ненавидит» до «обожает», 5 ступеней
    for limit, style in zip((-6, -2, 2, 6), styles):
        if score <= limit if limit < 0 else score < limit:
            return style
    return styles[-1]


class RelationsStore:
    def __init__(self, path: Path):
        self.path = path
        self.people: dict[str, Relation] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.people = {k: Relation(**v) for k, v in raw.items()}
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("не смог прочитать %s", self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in self.people.items()}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    def get(self, user_id: int | None, name: str) -> Relation:
        key = str(user_id) if user_id else name
        rel = self.people.get(key)
        if rel is None:
            rel = self.people[key] = Relation(name=name)
        rel.name = name
        return rel

    def prompt_block(self, participants: dict[str, int | None]) -> str:
        """Кусок промпта: досье и отношение к тем, кто сейчас в разговоре."""
        from bot.brain.life import duration, get_life

        life = get_life()
        lines = []
        for name, uid in participants.items():
            rel = self.get(uid, name)
            d = rel.dossier()
            gap = life.lived_s - rel.last_seen_lived if rel.last_seen_lived else 0
            seen = f" Не видел его {duration(gap)} (по твоему ощущению)." if gap > 3600 else ""
            if gap > MISS_AFTER_S and rel.closeness >= MISS_CLOSENESS and rel.score >= 2:
                seen += " Ты по нему соскучился."
            feel = rel.feelings_text()
            mom = rel.moments_text()
            lines.append(f"- {name}: {feel}." + (f" Запомнилось: {mom}." if mom else "")
                         + (f" Досье — {d}." if d else "") + seen)
        missing = self.missing_block(set(participants))
        if not lines and not missing:
            return ""
        from bot import persona

        return "\n".join(x for x in (persona.PEOPLE_MEMORY_HEADER, *lines, missing) if x)

    def close_people(self) -> list[Relation]:
        return [r for r in self.people.values() if r.closeness >= MISS_CLOSENESS and r.score >= 2]

    def missing_block(self, present: set[str]) -> str:
        """По кому скучает: близкие, которых давно нет (по его ощущению времени)."""
        from bot.brain.life import duration, get_life

        now = get_life().lived_s
        gone = [(now - r.last_seen_lived, r) for r in self.close_people()
                if r.name not in present and r.last_seen_lived and now - r.last_seen_lived > MISS_AFTER_S]
        if not gone:
            return ""
        gone.sort(key=lambda x: -x[1].closeness)
        return "Ты скучаешь по " + ", ".join(f"{r.name} (не видел {duration(g)})" for g, r in gone[:2]) + \
            " — можешь спросить у людей, где он."


MISS_AFTER_S = 3 * 3600    # близкого нет дольше (по его ощущению) — скучает
MISS_CLOSENESS = 3.0

_STORE: RelationsStore | None = None


def get_store() -> RelationsStore:
    """Одно хранилище на весь бот; у каждого персонажа своя память (data/relations_<имя>.json)."""
    global _STORE
    if _STORE is None:
        from bot.brain.life import memory_id

        _STORE = RelationsStore(Path(f"data/relations_{memory_id()}.json"))
    return _STORE


# «Петрык понял, что…», «Петрык не понял…» — это не о человеке, а о боте (такое модель часто пишет в досье)
_ABOUT_BOT = re.compile(r"^\s*(?:бот\w*|п[еэа]т(?:р[иыеэ]к|ер|р)\w{0,3})\s+(?:не\s+)?(?:понял|узнал|понимает|"
                        r"может|должен|учится|различает|считает|думает)", re.I)


def _clean_list(value, limit: int) -> list[str] | None:
    if value is None or not isinstance(value, list):
        return None
    items = [str(x).strip()[:120] for x in value if str(x).strip() and not _ABOUT_BOT.match(str(x))]
    return list(dict.fromkeys(items))[:limit]  # и без дублей («Петрык не понял; Петрык не понял»)


MAX_DROP_PER_UPDATE = 1.0   # за один разбор разговора симпатия падает не больше чем на 1
FORGIVE_STEP = 0.5          # обида понемногу проходит: каждый спокойный разговор — ближе к нулю (до −шрама)
TRUST_STEP = 1.5            # одно «обманул/был честен» двигает доверие на столько
SCAR_PER_LIE = 0.7          # каждый обман оставляет осадок (до SCAR_MAX), он тает очень медленно
SCAR_MAX = 3.0
SCAR_HEAL = 0.02            # за каждый спокойный разговор
CLOSE_TALKED = 0.15         # близость за разбор, где он говорил с ботом
CLOSE_PRESENT = 0.05        # …или просто был рядом


def _num(x, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x or 0)))
    except (TypeError, ValueError):
        return 0.0


def _grow_closeness(rel: "Relation", talked: bool) -> None:
    step = CLOSE_TALKED if talked else CLOSE_PRESENT
    rel.closeness = min(10.0, rel.closeness + step * (1 - rel.closeness / 12))


def _norm(name: str) -> str:
    return re.sub(r"[\W_]+", "", name.lower().replace("ё", "е"))


def _match_name(name: str, humans: dict[str, int | None]) -> str | None:
    """Модель часто пишет ник чуть иначе («Knight» вместо «! Knight») — ищем подходящий."""
    if name in humans:
        return name
    n = _norm(name)
    if not n:
        return None
    by_norm = {_norm(h): h for h in humans}
    if n in by_norm:
        return by_norm[n]
    for key, original in by_norm.items():
        if len(n) >= 3 and len(key) >= 3 and (n in key or key in n):
            return original
    return None


def _flag(v) -> bool | None:
    if isinstance(v, str):
        v = v.strip().lower()
        return True if v in ("true", "да", "yes", "1") else False if v in ("false", "нет", "no", "0") else None
    return v if isinstance(v, bool) else None


# ожидание «меня проигнорят», «никто не ответит», «промолчат» — его сбытие проверяем по факту, а не словами
_EXPECT_IGNORED = re.compile(r"проигнор|игнор|не ответ|не отреаг|промолч|не заметят|никто|не услыш", re.I)


def _outcome(r: dict, expect: str = "") -> str | None:
    """Итог сверки считаем сами из простых ответов — модели не надо выбирать между похожими словами.
    сбылось → match; не сбылось и приятно → good; не сбылось и неприятно → bad; иначе miss.
    Игнор проверяем по факту «ответил ли боту хоть кто-то»: модель путается, когда люди говорят о своём."""
    happened, answered = _flag(r.get("happened")), _flag(r.get("answered"))
    try:
        nice = int(round(float(r.get("nice") or 0)))
    except (TypeError, ValueError):
        nice = 0
    if answered is not None:
        if _EXPECT_IGNORED.search(expect):
            happened = not answered
        elif not answered:
            happened = False   # ждал ответа/реакции, а ему никто не ответил
            nice = min(nice, -1)  # проигнорили — неприятно
    if happened is None:
        return None
    if happened:
        return "match"
    return "good" if nice > 0 else "bad" if nice < 0 else "miss"


def _apply_expectations(store: "RelationsStore", exps: list, results, humans: dict) -> None:
    """Сверка «ждал — получил»: слабо двигает чувства и доверие к тому, кто удивил."""
    from bot.brain.predict import EFFECT, PERSON_TRUST, get_predictor

    if not exps or not isinstance(results, list):
        return
    pred = get_predictor()
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            e, _ = exps[int(r.get("n", 0)) - 1]
        except (ValueError, TypeError, IndexError):
            continue
        outcome = _outcome(r, e.expect)
        if outcome is None:
            continue
        what = str(r.get("what") or "")
        pred.resolve(e, outcome, what)
        get_emotions().apply(EFFECT[outcome])
        who = _match_name(str(r.get("who") or ""), humans)
        if who and outcome in PERSON_TRUST:
            rel = store.get(humans[who], who)
            rel.trust = max(-10.0, min(10.0, rel.trust + PERSON_TRUST[outcome]))
        log.info("предчувствие: ждал «%s» — %s%s", e.expect,
                 {"match": "так и вышло", "good": "вышло лучше", "bad": "вышло хуже", "miss": "не угадал"}[outcome],
                 f" ({what})" if what else "")
    store.save()


class RelationsJudge:
    """В фоне обновляет досье и отношения по свежим репликам.

    Запускается раз в 90 с, а если болтают много (большая компания) — как только набралось
    BURST_LINES реплик, чтобы ничего не пропустить.
    """

    MIN_INTERVAL_S = 90
    BURST_LINES = 30        # столько новых реплик людей — обновляем, не дожидаясь 90 с
    BURST_MIN_INTERVAL_S = 30
    MAX_LINES = 60          # сколько реплик максимум отдаём модели за раз
    MIN_HUMAN_LINES = 6     # без участия бота — ждём, пока наберётся материал

    def __init__(self, store: RelationsStore, llm: LLMRouter, bot_name: str):
        from bot import persona

        self.store = store
        self.llm = llm
        self.system = JUDGE_PROMPT.format(name=bot_name, goal=persona.MEMORY_GOAL,
                                          hooks_meaning=persona.DOSSIER_HOOKS, **LIMITS)
        self._last_run = 0.0
        self._last_seq = 0
        self._lock = asyncio.Lock()

    def _check_expectations(self, exps, results, humans) -> None:
        try:
            _apply_expectations(self.store, exps, results, humans)
        except Exception:
            log.exception("предчувствие: не смог сверить")

    async def maybe_update(self, buffer, force: bool = False) -> None:
        if self._lock.locked():
            return
        since_run = time.monotonic() - self._last_run
        if not force and since_run < self.BURST_MIN_INTERVAL_S:
            return
        new_all = buffer.since(self._last_seq)
        human_lines = sum(1 for l in new_all if not l.is_bot)
        if not force and since_run < self.MIN_INTERVAL_S and human_lines < self.BURST_LINES:
            return
        async with self._lock:
            new = buffer.since(self._last_seq)[-self.MAX_LINES:]
            humans = {l.speaker: l.speaker_id for l in new if not l.is_bot}
            human_lines = sum(1 for l in new if not l.is_bot)
            if not humans or (not force and human_lines < self.MIN_HUMAN_LINES and not any(l.is_bot for l in new)):
                return
            self._last_run = time.monotonic()
            self._last_seq = buffer.last_seq
            rows = []
            for n, uid in humans.items():
                r = self.store.get(uid, n)
                rows.append(f"- {n}: симпатия {r.score:+.0f}; доверие {r.trust:+.0f}; facts={r.facts}; plans={r.plans}; hooks={r.hooks}")
            from bot.brain.predict import get_predictor

            exps = get_predictor().matured(buffer)
            user = (f"Сегодня: {time.strftime('%d.%m.%Y')}\n"
                    f"Участники (в поле name пиши ник ровно так): {', '.join(humans)}\n\n"
                    f"Текущие досье:\n" + "\n".join(rows)
                    + f"\n\nСвежие реплики:\n{buffer.render(new)}")
            if exps:
                user += "\n\nОжидания бота (сверь с тем, что было после):\n" + "\n".join(
                    f"{i}) бот сказал «{e.said}» и ждал: «{e.expect}». После этого: "
                    + (buffer.render(after).replace("\n", " | ") if after else "никто ничего не ответил")
                    for i, (e, after) in enumerate(exps, 1))
            try:
                resp = await self.llm.create(
                    messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=900,
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                changes = data.get("changes", []) or []
            except Exception as e:
                log.warning("не смог обновить память: %s", e)
                return
            from bot.brain.life import get_life

            now_lived = get_life().lived_s
            bot_spoke = any(l.is_bot for l in new)
            for n, uid in humans.items():
                rel = self.store.get(uid, n)
                rel.last_seen_lived = now_lived
                _grow_closeness(rel, talked=bot_spoke)
            for ch in changes:
                name = _match_name(str(ch.get("name", "")), humans)
                if name is None:
                    log.info("память: не понял, кто такой %r — пропускаю", ch.get("name"))
                    continue
                rel = self.store.get(humans[name], name)
                delta = _num(ch.get("delta"), -MAX_DROP_PER_UPDATE, 2.0)
                tdelta = round(_num(ch.get("trust"), -1.0, 1.0))
                if tdelta < 0:
                    rel.scar = min(SCAR_MAX, rel.scar + SCAR_PER_LIE)
                elif delta >= 0:
                    rel.scar = max(0.0, rel.scar - SCAR_HEAL)
                if delta == 0 and rel.score < -rel.scar:
                    # ничего плохого в этот раз — ребёнок отходит: обида тает (но шрам не даёт до конца)
                    rel.score = min(-rel.scar, rel.score + FORGIVE_STEP)
                if tdelta == 0 and rel.trust < -rel.scar:
                    rel.trust = min(-rel.scar, rel.trust + FORGIVE_STEP / 2)
                rel.score = max(-10.0, min(10.0, rel.score + delta))
                rel.trust = max(-10.0, min(10.0, rel.trust + tdelta * TRUST_STEP))
                if (delta or tdelta) and isinstance(ch.get("moment"), str):
                    rel.add_moment(ch["moment"], 1 if delta + tdelta > 0 else -1, now_lived)
                if tdelta:
                    log.info("доверие к %s: %+d → %+.1f/10%s", name, tdelta, rel.trust,
                             f" (шрам {rel.scar:.1f})" if rel.scar else "")
                changed = []
                for key, limit in LIMITS.items():
                    items = _clean_list(ch.get(key), limit)
                    if items is not None and items != getattr(rel, key):
                        setattr(rel, key, items)
                        changed.append(key)
                rel.updated = time.time()
                if delta:
                    log.info("отношение к %s: %+.0f → %+.0f/10", name, delta, rel.score)
                if changed:
                    log.info("досье %s обновлено (%s): %s", name, ", ".join(changed), rel.dossier())
            self.store.save()
            if isinstance(data.get("feelings"), dict):
                get_emotions().apply(data["feelings"])
            self._check_expectations(exps, data.get("expectations"), humans)
            # что узнал о мире — в отдельную память знаний; новое утоляет любопытство
            world = [(str(w.get("text", "")), str(w.get("source", "")), w.get("important") is True)
                     for w in (data.get("world") or []) if isinstance(w, dict)]
            knowledge = get_knowledge()
            new = knowledge.add(world)
            if new:
                log.info("узнал о мире: %s", "; ".join(f.text for f in knowledge.facts[-new:]))
                get_drive().on_learned(new)
