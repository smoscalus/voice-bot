"""Любопытство бота — его внутренний «двигатель».

level (0…1) растёт, пока бот ничего нового не узнаёт, и падает, когда узнал (о мире, о людях, о себе)
или сам поговорил. Любопытный бот чаще сам расспрашивает и встревает, сытый новым — спокойнее,
переваривает узнанное. Новые знания приносит летописец памяти (RelationsJudge) в фоне.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

FULL_AFTER_S = 300        # с нуля до «очень любопытно» — за ~5 минут без нового
PER_FACT = 0.12           # каждый узнанный факт немного утоляет любопытство
PER_OWN_SPEECH = 0.25     # сам поговорил — тоже немного выдохнул


class Drive:
    def __init__(self) -> None:
        self.level = 0.5
        self._t = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        self.level = min(1.0, self.level + (now - self._t) / FULL_AFTER_S)
        self._t = now

    def on_learned(self, facts: int) -> None:
        if facts <= 0:
            return
        self._tick()
        self.level = max(0.0, self.level - PER_FACT * facts)
        log.info("узнал нового: %d, любопытство %.2f", facts, self.level)

    def on_spoke(self) -> None:
        self._tick()
        self.level = max(0.0, self.level - PER_OWN_SPEECH)

    def wants_to_speak(self) -> bool:
        """Достаточно ли любопытно, чтобы самому встревать."""
        self._tick()
        return self.level >= 0.35

    def prompt_block(self) -> str:
        self._tick()
        x = self.level
        mood = ("ты переполнен новым — спокойно переводишь дух, переосмысливаешь узнанное" if x < 0.25 else
                "любопытство небольшое — спрашиваешь, только если правда интересно" if x < 0.5 else
                "тебе любопытно — хочется что-то узнать, понять, расспросить" if x < 0.75 else
                "тебя распирает от любопытства — очень хочется узнать что-то новое о мире, о людях, о себе")
        return f"Твоё состояние: {mood}."


_DRIVE: Drive | None = None


def get_drive() -> Drive:
    global _DRIVE
    if _DRIVE is None:
        _DRIVE = Drive()
    return _DRIVE
