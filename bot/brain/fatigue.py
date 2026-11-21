"""Бодрость бота: устаёт от настоящей нагрузки, а не по расписанию.

level (0…1): 1 — бодрый, 0 — без сил. Тратится, когда он медленно отвечает (видеокарта не тянет, модель
думает подолгу), когда его дёргают без передышки и когда налетают все разом. Восстанавливается в тишине.
Если железо быстрое и его не мучают — он не устаёт: усталость только от настоящих причин.

Проявляется так (см. session.py): подустал — отвечает короче и вместо «погоди…» говорит «соображаю…»;
вымотан — сам реже лезет в разговор; без сил — сам просит пару минут передохнуть и правда молчит
(очередь пустеет, видеокарта отдыхает — после перерыва снова отвечает быстро).
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

SLOW_S = 6.0            # ответ дольше этого (от конца фразы до первых слов) — утомляет
FAST_S = 3.0            # быстрее — бодрит
SLOW_COST = 0.05        # столько бодрости за каждые лишние SLOW_S секунд ожидания (не больше MAX_SLOW_COST)
MAX_SLOW_COST = 0.12
CALL_COST = 0.012       # каждое обращение — чуть-чуть (полчаса без передышки — заметно)
OVERLOAD_COST = 0.1     # налетели все разом
REST_AFTER_S = 20.0     # с последнего обращения/реплики столько тишины — начинает отдыхать
REST_PER_S = 1 / 300    # с нуля до бодрого — ~5 минут тишины

TIRED, WORN, SPENT = 0.65, 0.4, 0.15   # пороги: подустал / вымотан / без сил
BREAK_S = 150.0         # перерыв, который он сам себе берёт
BREAK_GAP_S = 15 * 60   # и не чаще раза в столько
AFTER_BREAK = 0.7       # после перерыва бодрость не меньше этого

STATE_TEXT = {
    "tired": ("Ты подустал: соображаешь медленнее обычного. Отвечай короче, без длинных рассуждений. "
              "Можешь иногда, к месту, сказать, что голова тугая, — но не в каждой реплике."),
    "worn": ("Ты вымотан: тебя давно дёргают, и ты еле соображаешь. Отвечай коротко, в пару слов. "
             "Можешь сказать, что устал или что вы много болтаете, — по-своему и не в каждой реплике."),
}


class Fatigue:
    def __init__(self) -> None:
        self.level = 1.0
        self._busy_at = time.monotonic()   # последний раз дёргали/говорил — отдыхать начинает после
        self._t = time.monotonic()
        self._state = "fresh"
        self.resting_until = 0.0
        self._last_break = 0.0

    # ---------------------------------------------------------------- время
    def _tick(self) -> None:
        now = time.monotonic()
        quiet_from = max(self._t, self._busy_at + REST_AFTER_S)
        if now > quiet_from:
            self.level = min(1.0, self.level + (now - quiet_from) * REST_PER_S)
        self._t = now
        self._log_state()

    def _spend(self, amount: float, why: str) -> None:
        self._tick()
        self.level = max(0.0, self.level - amount)
        self._busy_at = time.monotonic()
        if amount >= 0.04:
            log.info("😮‍💨 бодрость −%.2f (%s) → %.2f", amount, why, self.level)
        self._log_state()

    def _log_state(self) -> None:
        state = self.state()
        if state != self._state:
            names = {"fresh": "бодрый", "tired": "подустал", "worn": "вымотан", "spent": "без сил"}
            log.info("😮‍💨 усталость: %s → %s (бодрость %.2f)", names[self._state], names[state], self.level)
            self._state = state

    # ---------------------------------------------------------------- события
    def on_reply_latency(self, seconds: float) -> None:
        """Сколько прошло от конца фразы человека до первых слов ответа (без рефлекса «погоди…»)."""
        if seconds > SLOW_S:
            self._spend(min(MAX_SLOW_COST, SLOW_COST * (seconds - SLOW_S) / SLOW_S + 0.01),
                        f"долго соображал, {seconds:.0f} с")
        elif seconds < FAST_S:
            self._tick()
            self.level = min(1.0, self.level + 0.01)
            self._busy_at = time.monotonic()

    def on_call(self) -> None:
        self._spend(CALL_COST, "обратились")

    def on_overload(self) -> None:
        self._spend(OVERLOAD_COST, "налетели все разом")

    def on_spoke(self) -> None:
        self._tick()
        self._busy_at = time.monotonic()

    # ---------------------------------------------------------------- состояние
    def state(self) -> str:
        if self.level < SPENT:
            return "spent"
        if self.level < WORN:
            return "worn"
        if self.level < TIRED:
            return "tired"
        return "fresh"

    def current(self) -> str:
        self._tick()
        return self.state()

    def resting(self) -> bool:
        return time.monotonic() < self.resting_until

    def wants_break(self) -> bool:
        """Без сил и давно не отдыхал — сам попросит передышку."""
        now = time.monotonic()
        return self.current() == "spent" and not self.resting() and now - self._last_break > BREAK_GAP_S

    def start_break(self) -> None:
        now = time.monotonic()
        self.resting_until = now + BREAK_S
        self._last_break = now
        log.info("😮‍💨 выдохся — беру перерыв на %d с", BREAK_S)

    def end_break(self) -> None:
        self.resting_until = 0.0
        self._tick()
        self.level = max(self.level, AFTER_BREAK)
        self._busy_at = time.monotonic() - REST_AFTER_S  # дальше отдыхает как обычно
        self._log_state()

    def max_tokens(self, normal: int) -> int:
        """Подустал — говорит короче."""
        return {"fresh": normal, "tired": int(normal * 0.75), "worn": int(normal * 0.55)}.get(self.current(),
                                                                                              int(normal * 0.5))

    def prompt_block(self) -> str:
        state = self.current()
        return STATE_TEXT.get("worn" if state == "spent" else state, "")


_fatigue: Fatigue | None = None


def get_fatigue() -> Fatigue:
    global _fatigue
    if _fatigue is None:
        _fatigue = Fatigue()
    return _fatigue
