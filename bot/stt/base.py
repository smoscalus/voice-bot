from __future__ import annotations

import re
from typing import Protocol

import numpy as np


class STTEngine(Protocol):
    async def transcribe(self, samples16k: np.ndarray, prompt: str | None = None) -> str: ...


# Whisper на тишине/шуме «галлюцинирует» фразами из субтитров YouTube.
_HALLUCINATIONS = (
    "субтитр", "продолжение следует", "спасибо за просмотр", "подписывайтесь",
    "dimatorzok", "редактор субтитров", "благодарю за внимание",
    # эхо подсказки распознавателю (Whisper на шуме выдаёт сам prompt)
    "разговор друзей", "расговор друзей", "программирование, игры, музыка",
)
_PROMPT_ECHOES = {"вася василиса", "вася василиса бот"}


def _norm(text: str) -> str:
    return re.sub(r"[^\w ]", "", text.lower().replace("ё", "е")).strip()


def clean_transcript(text: str, prompt: str | None = None) -> str | None:
    t = text.strip()
    if prompt and t:
        # Whisper на шуме иногда «пересказывает» подсказку целиком — это не речь человека
        from rapidfuzz import fuzz

        if len(t) >= 0.5 * len(prompt) and fuzz.ratio(_norm(t), _norm(prompt)) >= 70:
            return None
        # или «Петрык. Дота, катка, мид, мид, мид, кринж, кринж…» — почти все слова из подсказки
        vocab = {w[:5] for w in _norm(prompt).split()}
        words = [w[:5] for w in _norm(t).split()]
        if len(words) >= 4 and sum(w in vocab for w in words) / len(words) >= 0.7:
            return None
    if not t:
        return None
    low = t.lower()
    if any(h in low for h in _HALLUCINATIONS):
        return None
    if re.sub(r"[^\w ]", "", low.replace("ё", "е")).strip() in _PROMPT_ECHOES:
        return None
    if not re.search(r"\w", t):  # только пунктуация
        return None
    # «СМЕХ», «СПОКОЙНАЯ МУЗЫКА», «[музыка]» — Whisper подписывает звуки как в субтитрах
    if t.isupper() and len(t.split()) <= 3:
        return None
    if re.fullmatch(r"[\[\(].*[\]\)]\.?", t):
        return None
    return t
