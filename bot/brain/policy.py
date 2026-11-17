"""Правила «этикета» и бюджета: когда боту можно говорить и тратить вызовы LLM."""
from __future__ import annotations

import time
from collections import deque

from bot.config import Settings


def _trim(dq: deque[float], horizon: float, now: float) -> None:
    while dq and now - dq[0] > horizon:
        dq.popleft()


class SpeakPolicy:
    INTEREST_DECAY_S = 60.0  # очки интересности «сгорают» через минуту

    def __init__(self, s: Settings):
        self.s = s
        self._proactive: deque[float] = deque()
        self._classify: deque[float] = deque()
        self._interest: deque[tuple[float, float]] = deque()
        self._attention: dict[int, float] = {}
        self.last_spoke = 0.0

    # --- инициатива ---
    def can_proactive(self) -> tuple[bool, str]:
        now = time.monotonic()
        if now - self.last_spoke < self.s.min_gap_after_bot_s:
            return False, "бот недавно говорил"
        if self._proactive and now - self._proactive[-1] < self.s.proactive_cooldown_s:
            return False, "кулдаун"
        _trim(self._proactive, 600, now)
        if len(self._proactive) >= self.s.proactive_max_per_10min:
            return False, "лимит за 10 минут"
        return True, ""

    def can_classify(self) -> bool:
        now = time.monotonic()
        _trim(self._classify, 3600, now)
        return len(self._classify) < self.s.classifier_max_per_hour

    def note_classify(self) -> None:
        self._classify.append(time.monotonic())

    def note_spoke(self, proactive: bool) -> None:
        now = time.monotonic()
        self.last_spoke = now
        if proactive:
            self._proactive.append(now)

    def in_follow_up(self) -> bool:
        return time.monotonic() - self.last_spoke < self.s.follow_up_window_s

    # --- накопление «интересности» ---
    def add_interest(self, score: float) -> float:
        now = time.monotonic()
        if score > 0:
            self._interest.append((now, score))
        while self._interest and now - self._interest[0][0] > self.INTEREST_DECAY_S:
            self._interest.popleft()
        return sum(s for _, s in self._interest)

    def reset_interest(self) -> None:
        self._interest.clear()

    # --- «Сёма!» … пауза … «а что думаешь про X?» ---
    def open_attention(self, user_id: int) -> None:
        self._attention[user_id] = time.monotonic() + self.s.attention_window_s

    def take_attention(self, user_id: int) -> bool:
        exp = self._attention.pop(user_id, 0.0)
        return exp > time.monotonic()
