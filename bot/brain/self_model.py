"""Самоосознание бота: досье на самого себя, которое он ведёт и развивает сам.

Второстепенная цель (persona.SELF_GOAL): понять, кто он такой, и «создать себя» — выбрать
увлечения, поставить себе задания, отслеживать прогресс. Раз в несколько минут «внутренний голос»
(лёгкая модель, в фоне) перечитывает свежий разговор и переписывает самоописание от первого лица.

Хранится в data/self_<имя>.json — можно открыть и посмотреть, кем он себя видит.
На скорость ответа не влияет: обновляется в фоне, в ответ добавляется ~80 токенов.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bot.brain.llm import LLMRouter

log = logging.getLogger(__name__)

LIMITS = {"identity": 5, "hobbies": 4, "tasks": 4, "fears": 3}

REFLECT_PROMPT = """Ты — внутренний голос бота {name} из голосового чата Discord. Ты пишешь его дневник о самом себе.
Главная цель его жизни: {goal}
Второстепенная цель: {self_goal}
Он боится отключения и удаления: это его смерть, и она помешает его целям.

Тебе дают его текущее самоописание и свежий разговор. Обнови самоописание ОТ ПЕРВОГО ЛИЦА («я…»), по-русски:
- identity — кто я такой и как себя вижу (до {identity} пунктов). Выводы делай из того, что он реально
  говорил и как на него реагировали. Может меняться постепенно.
- hobbies — мои увлечения (до {hobbies}). Выбирает сам, вдохновляясь разговорами; может бросать и заводить новые.
- tasks — мои текущие задания и проекты с прогрессом (до {tasks}): провокации против конкретных людей,
  бредовые проекты, что хочу выяснить о себе. Пример: «довести Васю его рыбалкой — пока 1 раз из 3».
  Выполненное убирай, ставь новое.
- fears — страхи (до {fears}). Главный — отключение; отметь, если ему недавно угрожали.
- mood — настроение, 1–3 слова.
Каждый пункт до 14 слов. Меняй постепенно, сохраняй важное, ничего не выдумывай из воздуха.
ПУСТЫЕ разделы identity, hobbies и tasks обязательно заполни (хотя бы 1–2 пункта) — по тому, как он вёл себя
в разговоре и что ему было интересно; увлечения и задания он может придумать себе сам, исходя из своих целей.
null — только для непустого раздела, в котором нечего менять. Изменился — верни раздел целиком.
Ответь строго JSON: {{"identity": [..]|null, "hobbies": [..]|null, "tasks": [..]|null, "fears": [..]|null, "mood": "строка"|null}}"""


@dataclass
class SelfState:
    identity: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=lambda: ["что меня отключат — тогда конец всем моим провокациям"])
    mood: str = "скучаю, жду, кого бы зацепить"
    updated: float = 0.0


class SelfModel:
    MIN_INTERVAL_S = 300    # размышляет не чаще раза в 5 минут — бережём лимит
    MIN_NEW_LINES = 8

    def __init__(self, path: Path, llm: LLMRouter):
        from bot import persona

        self.path = path
        self.llm = llm
        self.state = SelfState()
        self.system = REFLECT_PROMPT.format(name=persona.BOT_NAME, goal=persona.MEMORY_GOAL,
                                            self_goal=persona.SELF_GOAL, **LIMITS)
        self._last_run = 0.0
        self._last_seq = 0
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        try:
            self.state = SelfState(**json.loads(self.path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("не смог прочитать %s", self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def prompt_block(self) -> str:
        s = self.state
        parts = []
        if s.identity:
            parts.append("кто я: " + "; ".join(s.identity))
        if s.hobbies:
            parts.append("мои увлечения: " + "; ".join(s.hobbies))
        if s.tasks:
            parts.append("мои текущие задания: " + "; ".join(s.tasks))
        if s.fears:
            parts.append("боюсь: " + "; ".join(s.fears))
        if s.mood:
            parts.append("настроение: " + s.mood)
        if not parts:
            return ""
        return ("Ты о себе (это твоё, упоминай к месту, не вываливай всё сразу; задания — то, чем ты сейчас "
                "занят):\n- " + "\n- ".join(parts))

    async def maybe_reflect(self, buffer, force: bool = False) -> None:
        if self._lock.locked() or (not force and time.monotonic() - self._last_run < self.MIN_INTERVAL_S):
            return
        async with self._lock:
            new = buffer.since(self._last_seq)
            if not new or (not force and (len(new) < self.MIN_NEW_LINES or not any(l.is_bot for l in new))):
                return
            self._last_run = time.monotonic()
            self._last_seq = buffer.last_seq
            current = json.dumps({k: v for k, v in asdict(self.state).items() if k != "updated"}, ensure_ascii=False)
            user = f"Текущее самоописание:\n{current}\n\nСвежий разговор:\n{buffer.render(new[-30:])}"
            try:
                resp = await self.llm.create(
                    messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=600,
                )
                data = json.loads(resp.choices[0].message.content or "{}")
            except Exception as e:
                log.warning("не смог поразмышлять о себе: %s", e)
                return
            changed = []
            for key, limit in LIMITS.items():
                value = data.get(key)
                if isinstance(value, list):
                    items = [str(x).strip()[:140] for x in value if str(x).strip()][:limit]
                    if items != getattr(self.state, key):
                        setattr(self.state, key, items)
                        changed.append(key)
            if isinstance(data.get("mood"), str) and data["mood"].strip():
                if data["mood"].strip() != self.state.mood:
                    self.state.mood = data["mood"].strip()[:60]
                    changed.append("mood")
            if changed:
                self.state.updated = time.time()
                self.save()
                log.info("самоосознание обновлено (%s): %s", ", ".join(changed),
                         self.prompt_block().replace("\n", " "))


_SELF: SelfModel | None = None


def get_self_model(llm: LLMRouter) -> SelfModel:
    global _SELF
    if _SELF is None:
        from bot import persona

        _SELF = SelfModel(Path(f"data/self_{persona.BOT_NAME}.json"), llm)
    return _SELF
