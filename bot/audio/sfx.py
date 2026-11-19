"""Короткие звуки голосом бота: «угу/ага» пока человек рассказывает и «хм…» пока думает.

Генерируются один раз тем же TTS (голос Svetlana) и кешируются на диск — дальше играют мгновенно.
"""
from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BACKCHANNEL = ["Угу.", "Ага.", "Мгм.", "Ну-ну.", "Ага, ага."]
FILLERS = ["Хм…", "Так…", "Ммм…", "Ну…", "Эээ…", "Пф…"]
CALLS = ["М?", "Да?", "А?", "Слушаю."]   # позвали по имени — отозвался, пока думает
AGAIN = ["Ага…", "А, так…", "Угу, щас…", "А, ну…"]  # человек договорил ещё — «услышал, думаю заново»
CACHE_DIR = Path("models/sfx")


def _trim(pcm: bytes, rate: int = 24_000, threshold: int = 300, keep_ms: int = 30) -> bytes:
    """Срезаем тишину по краям — чтобы звук звучал сразу, без задержки."""
    a = np.frombuffer(pcm, dtype=np.int16)
    loud = np.nonzero(np.abs(a) > threshold)[0]
    if loud.size == 0:
        return b""
    pad = rate * keep_ms // 1000
    return a[max(0, loud[0] - pad): loud[-1] + pad].tobytes()


def _scale(pcm: bytes, k: float) -> bytes:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * k
    return np.clip(a, -32768, 32767).astype(np.int16).tobytes()


class SoundBank:
    def __init__(self, tts, voice_key: str):
        self.tts = tts
        self.dir = CACHE_DIR / voice_key.replace("/", "_")
        self.clips: dict[str, list[bytes]] = {"back": [], "filler": [], "call": [], "again": []}
        self._last: bytes | None = None

    async def _clip(self, text: str) -> bytes:
        path = self.dir / (hashlib.md5(text.encode()).hexdigest() + ".pcm")
        if path.exists():
            return path.read_bytes()
        if hasattr(self.tts, "synth_clip"):  # только «родной» голос, без запасного
            pcm = await self.tts.synth_clip(text)
        else:
            pcm = b"".join([c async for c in self.tts.stream(text)])
        pcm = _trim(pcm)
        if pcm:
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pcm)
        return pcm

    async def load(self) -> None:
        for kind, texts, volume in (("back", BACKCHANNEL, 0.7), ("filler", FILLERS, 1.0), ("call", CALLS, 1.0),
                                     ("again", AGAIN, 1.0)):
            for t in texts:
                try:
                    if pcm := await self._clip(t):
                        self.clips[kind].append(_scale(pcm, volume))
                except Exception as e:
                    log.warning("не смогла подготовить звук %r: %s", t, e)
        log.info("звуки готовы: угу ×%d, хм ×%d, м? ×%d", len(self.clips["back"]), len(self.clips["filler"]),
                 len(self.clips["call"]))

    def random(self, kind: str) -> bytes | None:
        """Случайный звук, но не тот же, что прозвучал только что («угу… угу» подряд режет слух)."""
        pool = self.clips.get(kind) or []
        if not pool:
            return None
        clip = random.choice([c for c in pool if c is not self._last] or pool)
        self._last = clip
        return clip
