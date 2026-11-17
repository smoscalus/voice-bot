"""Скользящий текстовый буфер разговора.

Держит реплики за последние window_s секунд (и не больше max_lines).
Реплики приходят из параллельных STT-запросов не по порядку, поэтому вставка
сортированная по времени *начала* фразы. seq — монотонный номер вставки, по нему
удобно брать «что нового с прошлой оценки».
"""
from __future__ import annotations

import bisect
import time
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(order=True)
class Line:
    ts: float
    seq: int = field(compare=False)
    speaker: str = field(compare=False)
    text: str = field(compare=False)
    speaker_id: int | None = field(default=None, compare=False)
    is_bot: bool = field(default=False, compare=False)
    end_ts: float | None = field(default=None, compare=False)  # когда фраза закончилась


class TranscriptBuffer:
    def __init__(self, window_s: float = 180, max_lines: int = 40):
        self.window_s = window_s
        self.max_lines = max_lines
        self._lines: list[Line] = []
        self._seq = 0

    def add(self, speaker: str, text: str, *, ts: float | None = None,
            speaker_id: int | None = None, is_bot: bool = False, end_ts: float | None = None) -> Line:
        self._seq += 1
        line = Line(ts=ts or time.time(), seq=self._seq, speaker=speaker, text=text,
                    speaker_id=speaker_id, is_bot=is_bot, end_ts=end_ts)
        bisect.insort(self._lines, line)
        self._prune()
        return line

    def remove(self, line: Line) -> None:
        try:
            self._lines.remove(line)
        except ValueError:
            pass

    @property
    def last_seq(self) -> int:
        return self._seq

    def recent(self, n: int | None = None) -> list[Line]:
        self._prune()
        return self._lines[-n:] if n else list(self._lines)

    def since(self, seq: int) -> list[Line]:
        return [l for l in self._lines if l.seq > seq]

    @staticmethod
    def render(lines: Iterable[Line]) -> str:
        return "\n".join(
            f"[{time.strftime('%H:%M:%S', time.localtime(l.ts))}] {l.speaker}{' (бот)' if l.is_bot else ''}: {l.text}"
            for l in lines
        )

    def _prune(self) -> None:
        cutoff = time.time() - self.window_s
        i = 0
        while i < len(self._lines) and self._lines[i].ts < cutoff:
            i += 1
        if i:
            del self._lines[:i]
        if len(self._lines) > self.max_lines:
            del self._lines[: len(self._lines) - self.max_lines]
