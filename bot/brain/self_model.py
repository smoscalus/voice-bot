"""Самоосознание бота: дневник о самом себе, который он ведёт и развивает сам.

Разделы дневника и их смысл задаёт персонаж (persona.SELF_SECTIONS): например, «кто я», «что мне
интересно», «мои цели», «вопросы, на которые я ищу ответ». Раз в несколько минут «внутренний голос»
(модель для памяти, в фоне) перечитывает свежий разговор и переписывает дневник от первого лица.

Хранится в data/self_<имя памяти>.json — можно открыть и посмотреть, кем он себя видит.
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

REFLECT_PROMPT = """Ты — внутренний голос бота {name} из голосового чата Discord. Ты пишешь его дневник о самом себе.
Главная цель его жизни: {goal}
Вторая цель: {self_goal}
{extra}

Тебе дают его текущее самоописание и свежий разговор. Обнови самоописание ОТ ПЕРВОГО ЛИЦА («я…»), по-русски:
{sections}
- mood — настроение, 1–3 слова.
Каждый пункт до 14 слов. Меняй постепенно, сохраняй важное, ничего не выдумывай из воздуха — выводы только
из того, что реально было в разговорах. ПУСТЫЕ разделы обязательно заполни (хотя бы 1–2 пункта) по разговору.
null — только для непустого раздела, в котором нечего менять. Изменился — верни раздел целиком.
Ответь строго JSON: {{{json_keys}, "mood": "строка"|null}}"""


def _sections() -> dict[str, tuple[str, str, int]]:
    """Разделы дневника задаёт персонаж: ключ → (заголовок, что туда писать, сколько пунктов)."""
    from bot import persona

    return persona.SELF_SECTIONS


@dataclass
class SelfState:
    identity: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=list)
    mood: str = ""
    updated: float = 0.0


class SelfModel:
    MIN_INTERVAL_S = 300    # размышляет не чаще раза в 5 минут — бережём лимит
    MIN_NEW_LINES = 8

    def __init__(self, path: Path, llm: LLMRouter):
        from bot import persona

        self.path = path
        self.llm = llm
        self.state = SelfState(mood=getattr(persona, "SELF_DEFAULT_MOOD", ""))
        secs = _sections()
        self.system = REFLECT_PROMPT.format(
            name=persona.BOT_NAME, goal=persona.MEMORY_GOAL, self_goal=persona.SELF_GOAL,
            extra=getattr(persona, "SELF_REFLECT_EXTRA", ""),
            sections="\n".join(f"- {k} — {desc} (до {lim} пунктов)." for k, (_, desc, lim) in secs.items()),
            json_keys=", ".join(f'"{k}": [..]|null' for k in secs))
        self._last_run = 0.0
        self._last_seq = 0
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            known = SelfState.__dataclass_fields__
            self.state = SelfState(**{k: v for k, v in raw.items() if k in known})
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
        for key, (title, _, _) in _sections().items():
            items = getattr(s, key, None)
            if items:
                parts.append(f"{title}: " + "; ".join(items))
        if s.mood:
            parts.append("настроение: " + s.mood)
        if not parts:
            return ""
        return ("Ты о себе (это твоё, упоминай к месту, не вываливай всё сразу; цели — то, чем ты сейчас "
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
            for key, (_, _, limit) in _sections().items():
                value = data.get(key)
                if isinstance(value, list):
                    items = [str(x).strip()[:140] for x in value if str(x).strip() and str(x) != "None"][:limit]
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

        _SELF = SelfModel(Path(f"data/self_{persona.MEMORY_ID}.json"), llm)
    return _SELF
