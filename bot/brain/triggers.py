"""Дешёвая локальная логика без LLM: wake word, стоп-слова и префильтр «интересности»."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz


def normalize(text: str) -> str:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_SECOND_PERSON = {"ты", "тебя", "тебе", "тобой", "твой", "твоя", "твое", "твои", "твоего", "вы", "скажи",
                  "расскажи", "ответь", "слышишь", "знаешь", "думаешь", "хочешь", "можешь", "понял", "понимаешь"}
_LEAD_WORDS = {"эй", "ну", "слушай", "а", "о", "так", "вот", "алло", "привет", "здарова", "здорово", "йо", "хей"}


def voice_command(text: str, mute: list[str], unmute: list[str]) -> str | None:
    """«помолчи» → "mute", «можешь говорить» → "unmute". «не молчи» — разрешение, а не просьба замолчать."""
    # команда — в начале фразы после имени («Петрык, ну помолчи уже»), а не где-то в середине
    #  («ты тихо говоришь» — не просьба замолчать)
    t = " " + " ".join(normalize(text).split()[:4]) + " "
    if t.split()[:1] in (["ты"], ["вы"], ["он"], ["я"]):
        return None
    for p in unmute:
        ph = f" {normalize(p)} "
        if ph in t and (ph.startswith(" не ") or f" не{ph}" not in t):  # «не говори» — это не разрешение
            return "unmute"
    for p in mute:
        ph = f" {normalize(p)} "
        if ph in t and f" не{ph}" not in t:
            return "mute"
    return None


@dataclass
class TriggerResult:
    wake: bool = False
    vocative: bool = False       # имя звучит как обращение к нему («Петрык, …»), а не разговор о нём
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

    def _is_vocative(self, text: str) -> bool:
        """«Петрык, …», «…, Петрык?», «эй, Петрык» — к нему. «Петрык думает», «спроси Петрыка» — о нём."""
        words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w-]+", text.lower().replace("ё", "е"))]
        if len(words) <= 2:
            return True  # «Петрык, привет», «Петрык?» — коротко, значит к нему
        toks = [w for w, _, _ in words]
        if _SECOND_PERSON & set(toks):
            return True  # «Петрык, ты…», «тебе», «твой» — говорят с ним
        for i, (w, a, b) in enumerate(words):
            if not self._is_wake_token(w):
                continue
            before = text[:a].rstrip()
            after = text[b:].lstrip()
            lead = all(t in _LEAD_WORDS for t in toks[:i])  # в начале (можно после «эй», «ну», «слушай»)
            plain = not w.endswith(("ка", "ку", "ком", "ке", "ом", "ым", "ем"))  # «Петрика», «Петрику» — о нём
            if lead or (i == len(words) - 1 and plain):
                return True
            if plain and after[:1] in (",", "!", "?", ":"):
                return True  # «…, Петрык, …» — выделено запятыми, обращение
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
        if res.wake:
            res.vocative = self._is_vocative(text)

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
