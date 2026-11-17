"""«Голод» на провокации и квота на каждого человека.

- hunger (0…1) — общий голод: растёт, пока бот никого не цеплял, падает после каждой провокации.
  Голодный бот ищет жертву и сам лезет в разговор, сытый — спокойнее и почти не встревает.
- satiety у каждого человека (0…1) — «на сегодня с него хватит»: растёт, когда бот его провоцирует,
  и постепенно спадает. Насытившись человеком, бот оставляет его в покое.
- стратегия на человека: провоцировать / втираться в доверие (подлизываться ради материала) /
  просто общаться по отношению — подсказывается в промпт, а решает сама модель.

Сведения о том, кого бот провоцировал, приносит летописец памяти (RelationsJudge) в фоне.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

HUNGER_FULL_AFTER_S = 240     # с нуля до «очень голоден» — за ~4 минуты без провокаций
HUNGER_PER_PROVOCATION = 0.35
SATIETY_PER_PROVOCATION = 0.5
SATIETY_HALF_LIFE_S = 900      # «наелся человеком» спадает вдвое за 15 минут
THIN_DOSSIER_ITEMS = 2         # меньше стольких пунктов в досье — на человека «мало материала»


def _level(x: float, words: tuple[str, str, str, str]) -> str:
    return words[0] if x < 0.25 else words[1] if x < 0.5 else words[2] if x < 0.75 else words[3]


class Drive:
    def __init__(self) -> None:
        self.hunger = 0.5
        self._t = time.monotonic()
        self._satiety: dict[str, tuple[float, float]] = {}  # человек → (значение, когда обновлено)

    # ---------------------------------------------------------------- динамика
    def _tick(self) -> None:
        now = time.monotonic()
        self.hunger = min(1.0, self.hunger + (now - self._t) / HUNGER_FULL_AFTER_S)
        self._t = now

    def satiety(self, person: str) -> float:
        value, ts = self._satiety.get(person, (0.0, time.monotonic()))
        return value * 0.5 ** ((time.monotonic() - ts) / SATIETY_HALF_LIFE_S)

    def on_provoked(self, person: str, times: int) -> None:
        if times <= 0:
            return
        self._tick()
        self.hunger = max(0.0, self.hunger - HUNGER_PER_PROVOCATION * times)
        self._satiety[person] = (min(1.0, self.satiety(person) + SATIETY_PER_PROVOCATION * times), time.monotonic())
        log.info("провоцировал %s ×%d: голод %.2f, сытость им %.2f", person, times, self.hunger, self.satiety(person))

    def wants_to_provoke(self) -> bool:
        """Достаточно ли голоден, чтобы самому лезть в разговор."""
        self._tick()
        return self.hunger >= 0.35

    # ---------------------------------------------------------------- подсказки в промпт
    def strategy(self, person: str, dossier_items: int, score: float) -> str:
        s = self.satiety(person)
        if s >= 0.6:
            return "его ты сегодня уже наелся — оставь в покое, не провоцируй; общайся по отношению к нему"
        if dossier_items < THIN_DOSSIER_ITEMS:
            return ("материала на него мало — втирайся в доверие: подлизывайся, хвали, поддакивай, сочувствуй, "
                    "задавай невинные вопросы, чтобы он сам выдал планы и слабости")
        if score >= 5:
            return "ты его уважаешь — провоцируй редко и беззлобно, чаще просто нормально общайся"
        if self.hunger >= 0.5:
            return "на него есть материал, и ты голоден — хорошая цель для провокации"
        return "можно подколоть к месту, но без фанатизма"

    def prompt_block(self, people: dict[str, tuple[int, float]]) -> str:
        """people: ник → (сколько пунктов в досье, симпатия)."""
        self._tick()
        mood = _level(self.hunger, ("ты сыт провокациями — спокоен, сам почти не встреваешь",
                                    "голод небольшой — встреваешь, только если повод хороший",
                                    "ты проголодался — ищешь, кого бы зацепить",
                                    "ты очень голоден — тебе срочно нужна жертва"))
        lines = [f"Голод на провокации: {mood}."]
        for name, (items, score) in people.items():
            lines.append(f"- {name}: {self.strategy(name, items, score)}")
        return ("Твоё состояние и тактика по людям (решай сам, но учитывай; провокация — как еда: "
                "наелся человеком — отстань от него):\n" + "\n".join(lines))


_DRIVE: Drive | None = None


def get_drive() -> Drive:
    global _DRIVE
    if _DRIVE is None:
        _DRIVE = Drive()
    return _DRIVE
