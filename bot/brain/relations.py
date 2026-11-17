"""Память бота о людях: бессрочные досье + отношение (симпатия).

Досье подчинено цели бота (persona.MEMORY_GOAL) — у Петрыка это провокации: он запоминает
не всё подряд, а то, что пригодится, чтобы потом зацепить человека.

Досье каждого человека:
- facts  — главное о человеке (чем живёт, во что играет, работа/учёба);
- plans  — планы и обещания с датой («бросить универ (17.11)») — можно потом спросить, выполнил ли;
- hooks  — зацепки для провокаций: хвастовство, слабости, противоречия, больные места, конфликты.

Память сама сжимается: при каждом обновлении модель переписывает списки целиком, оставляя
только самое ценное (лимиты на длину ниже). Хранится в data/relations_<имя бота>.json.
На скорость ответа почти не влияет: обновляется в фоне, а в ответ добавляется ~60 токенов на человека.
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
from bot.brain.llm import LLMRouter

log = logging.getLogger(__name__)

LIMITS = {"facts": 5, "plans": 4, "hooks": 5}  # пунктов в каждом разделе досье — память сама ужимается

JUDGE_PROMPT = """Ты ведёшь долговременную память бота {name} из голосового чата: досье на людей и его отношение к ним.
Цель бота: {goal}
Поэтому запоминай НЕ всё подряд, а то, что ценно для этой цели.

Тебе дают текущие досье, сегодняшнюю дату и свежие реплики. Для каждого человека из свежих реплик верни:
- delta — изменение симпатии бота к нему: от -2 (всерьёз обидел, унизил бота) до +2 (поддержал, рассмешил,
  дал боту классный материал). Не взаимодействовал с ботом — 0. Дружеские подколы и мат — не минус.
- provoked — сколько раз в этих репликах БОТ провоцировал, подкалывал или троллил этого человека (0, 1, 2…).
- facts — главное о человеке: чем живёт, работа/учёба, во что играет, увлечения. До {facts} пунктов.
- plans — планы, намерения и обещания С ДАТОЙ, когда сказал: «купить микрофон (17.11)». До {plans} пунктов.
  Выполненный или брошенный план убери из plans; если он сорвался или человек слился — перенеси в hooks
  («обещал бросить универ — так и не бросил»).
- hooks — зацепки для провокаций: хвастовство, слабости, противоречия (говорил одно — делает другое),
  неудачи, больные места, с кем конфликтует. До {hooks} пунктов.
ВАЖНО:
- Факт записывай тому, О КОМ он, а не тому, кто его сказал. «Ты 60 еле поднял», сказанное Петей
  Васе, — это зацепка Васи, а не Пети.
- Всё, что человек собирается сделать в будущем («завтра экзамен», «в пятницу пересдаю», «на выходных
  рыбалка», «хочу переехать») — ТОЛЬКО в plans с датой, а не в facts.
Каждый пункт — до 12 слов, по-русски. НЕ записывай манеру речи, мат, приветствия и пустую болтовню.
Если в разделе ничего не изменилось — верни null (раздел останется прежним). Если изменилось — верни
раздел ЦЕЛИКОМ, уже сжатый: объедини похожие пункты, выкинь самое неважное, если не влезает в лимит.

Ответь строго JSON: {{"changes": [{{"name": "ник как в расшифровке", "delta": число, "provoked": число,
  "facts": [..] или null, "plans": [..] или null, "hooks": [..] или null}}]}}"""


@dataclass
class Relation:
    name: str
    score: float = 0.0          # от -10 (ненавидит) до +10 (уважает/обожает)
    facts: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    updated: float = 0.0
    # устаревшие поля прошлых версий — подхватываются как факты
    notes: list[str] = field(default_factory=list)
    profile: str = ""

    def __post_init__(self) -> None:
        if not self.facts and (self.profile or self.notes):
            self.facts = ([self.profile] if self.profile else []) + self.notes[-3:]
            self.profile, self.notes = "", []

    def dossier(self) -> str:
        parts = []
        if self.facts:
            parts.append("о нём: " + "; ".join(self.facts))
        if self.plans:
            parts.append("планы: " + "; ".join(self.plans))
        if self.hooks:
            parts.append("зацепки: " + "; ".join(self.hooks))
        return ". ".join(parts)


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
        lines = []
        for name, uid in participants.items():
            rel = self.get(uid, name)
            d = rel.dossier()
            lines.append(f"- {name}: симпатия {rel.score:+.0f}/10 — {describe(rel.score)}." + (f" Досье — {d}." if d else ""))
        if not lines:
            return ""
        return ("Твоя память об участниках (цифры вслух не называй; зацепки и планы — твои патроны для провокаций, "
                "используй к месту, а не всё сразу):\n" + "\n".join(lines))


_STORE: RelationsStore | None = None


def get_store() -> RelationsStore:
    """Одно хранилище на весь бот; у каждого персонажа своя память (data/relations_<имя>.json)."""
    global _STORE
    if _STORE is None:
        from bot import persona

        _STORE = RelationsStore(Path(f"data/relations_{persona.BOT_NAME}.json"))
    return _STORE


def _clean_list(value, limit: int) -> list[str] | None:
    if value is None or not isinstance(value, list):
        return None
    items = [str(x).strip()[:120] for x in value if str(x).strip()]
    return items[:limit]


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
        self.system = JUDGE_PROMPT.format(name=bot_name, goal=persona.MEMORY_GOAL, **LIMITS)
        self._last_run = 0.0
        self._last_seq = 0
        self._lock = asyncio.Lock()

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
                rows.append(f"- {n}: симпатия {r.score:+.0f}; facts={r.facts}; plans={r.plans}; hooks={r.hooks}")
            user = (f"Сегодня: {time.strftime('%d.%m.%Y')}\n"
                    f"Участники (в поле name пиши ник ровно так): {', '.join(humans)}\n\n"
                    f"Текущие досье:\n" + "\n".join(rows)
                    + f"\n\nСвежие реплики:\n{buffer.render(new)}")
            try:
                resp = await self.llm.create(
                    messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=900,
                )
                changes = json.loads(resp.choices[0].message.content or "{}").get("changes", [])
            except Exception as e:
                log.warning("не смог обновить память: %s", e)
                return
            for ch in changes:
                name = _match_name(str(ch.get("name", "")), humans)
                if name is None:
                    log.info("память: не понял, кто такой %r — пропускаю", ch.get("name"))
                    continue
                rel = self.store.get(humans[name], name)
                try:
                    delta = max(-2.0, min(2.0, float(ch.get("delta") or 0)))
                except (TypeError, ValueError):
                    delta = 0.0
                rel.score = max(-10.0, min(10.0, rel.score + delta))
                try:  # сколько раз бот его провоцировал — утоляет голод и «насыщает» им
                    get_drive().on_provoked(name, max(0, min(5, int(ch.get("provoked") or 0))))
                except (TypeError, ValueError):
                    pass
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
