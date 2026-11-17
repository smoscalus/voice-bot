"""Дешёвая локальная логика без LLM: wake word, стоп-слова и префильтр «интересности»."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz


def normalize(text: str) -> str:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class TriggerResult:
    wake: bool = False
    stop: bool = False
    remainder: str = ""          # текст без слова-обращения
    score: float = 0.0           # вклад в «интересность»
    topics: set[str] = field(default_factory=set)


class TriggerDetector:
    def __init__(self, wake_words: list[str], wake_phrases: list[str], stop_words: list[str],
                 topics: dict[str, list[str]], opinion_cues: list[str], dispute_cues: list[str],
                 fuzzy_threshold: int = 85):
        self.wake_words = [normalize(w) for w in wake_words]
        self.wake_phrases = [normalize(p) for p in wake_phrases]
        self.stop_words = {normalize(w) for w in stop_words}
        self.topics = {name: [normalize(k) for k in kws] for name, kws in topics.items()}
        self.opinion_cues = [normalize(c) for c in opinion_cues]
        self.dispute_cues = [normalize(c) for c in dispute_cues]
        self.fuzzy_threshold = fuzzy_threshold

    def _is_wake_token(self, tok: str) -> bool:
        for w in self.wake_words:
            if tok == w:
                return True
            # короткие слова («бот», «гпт») — только точное совпадение, иначе ложные срабатывания
            if len(w) >= 4 and len(tok) >= 4 and fuzz.ratio(tok, w) >= self.fuzzy_threshold:
                return True
        return False

    def analyze(self, text: str) -> TriggerResult:
        res = TriggerResult()
        norm = normalize(text)
        tokens = norm.split()

        # --- стоп ---
        res.stop = len(tokens) <= 3 and any(t in self.stop_words for t in tokens)

        # --- wake word ---
        remainder = norm
        for p in self.wake_phrases:
            if p in remainder:
                res.wake = True
                remainder = remainder.replace(p, " ")
        kept = []
        for t in remainder.split():
            if self._is_wake_token(t):
                res.wake = True
            else:
                kept.append(t)
        res.remainder = " ".join(kept)

        # --- префильтр интересности ---
        for topic, kws in self.topics.items():
            hits = sum(
                1 for t in tokens for k in kws
                if (t.startswith(k) if len(k) >= 4 else t == k)
            )
            if hits:
                res.topics.add(topic)
                res.score += min(hits, 2)
        if any(c in norm for c in self.opinion_cues):
            res.score += 3.0
        if any(c in norm for c in self.dispute_cues):
            res.score += 1.5
        if "?" in text:
            res.score += 0.5
        return res
