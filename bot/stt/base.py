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


def clean_transcript(text: str) -> str | None:
    t = text.strip()
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
