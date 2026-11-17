"""Нарезка непрерывного аудио каждого участника на фразы (utterances) с помощью VAD.

Важные особенности Discord:
- аудио приходит отдельно на каждого пользователя (по SSRC) — микшировать не нужно,
  и мы сразу знаем, кто говорит;
- когда человек молчит, пакеты просто перестают приходить. Поэтому конец фразы
  определяется не только по «тихим» кадрам, но и по таймеру (flush_stale),
  который дёргается из asyncio-цикла.

feed() вызывается из потока приёма voice-recv, flush_stale() — из event loop,
поэтому всё состояние под одним lock.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable

import webrtcvad

from bot.audio.convert import FRAME_MS, MONO48_FRAME_BYTES, DISCORD_RATE, stereo48_to_mono48


@dataclass
class Utterance:
    user_id: int
    user_name: str
    pcm_mono48: bytes
    started_at: float  # time.time()
    duration_s: float


class _UserState:
    def __init__(self, user_id: int, user_name: str, aggressiveness: int, preroll_frames: int):
        self.user_id = user_id
        self.user_name = user_name
        self.vad = webrtcvad.Vad(aggressiveness)
        self.preroll: collections.deque[bytes] = collections.deque(maxlen=preroll_frames)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.frames = bytearray()
        self.voiced_ms = 0
        self.started_mono = 0.0
        self.started_wall = 0.0
        self.last_voice = 0.0


class UtteranceSegmenter:
    def __init__(
        self,
        on_utterance: Callable[[Utterance], None],
        aggressiveness: int = 2,
        silence_ms: int = 700,
        min_utterance_ms: int = 400,
        max_utterance_s: float = 15.0,
        preroll_ms: int = 300,
    ):
        self._on_utterance = on_utterance
        self._aggr = aggressiveness
        self._silence_s = silence_ms / 1000
        self._min_voiced_ms = min_utterance_ms
        self._max_s = max_utterance_s
        self._preroll_frames = max(1, preroll_ms // FRAME_MS)
        self._users: dict[int, _UserState] = {}
        self._lock = threading.Lock()
        self._last_voice_any = 0.0

    # --- вызывается из потока voice-recv ---------------------------------
    def feed(self, user_id: int, user_name: str, pcm_stereo48: bytes) -> None:
        mono = stereo48_to_mono48(pcm_stereo48)
        vad = self._vad_for(user_id, user_name)  # создаёт запись о человеке при первом же кадре любой длины
        try:
            is_speech = len(mono) == MONO48_FRAME_BYTES and vad.is_speech(mono, DISCORD_RATE)
        except Exception:
            is_speech = False

        ready: list[Utterance] = []
        now = time.monotonic()
        with self._lock:
            st = self._users.get(user_id)
            if st is None:  # человека успели удалить (вышел из канала) — просто пропускаем кадр
                return
            st.user_name = user_name
            if not st.active:
                st.preroll.append(mono)
                if is_speech:
                    st.active = True
                    st.started_mono = now - len(st.preroll) * FRAME_MS / 1000
                    st.started_wall = time.time() - len(st.preroll) * FRAME_MS / 1000
                    st.frames = bytearray(b"".join(st.preroll))
                    st.preroll.clear()
                    st.voiced_ms = FRAME_MS
                    st.last_voice = now
                    self._last_voice_any = now
            else:
                st.frames += mono
                if is_speech:
                    st.voiced_ms += FRAME_MS
                    st.last_voice = now
                    self._last_voice_any = now
                if now - st.last_voice >= self._silence_s or now - st.started_mono >= self._max_s:
                    self._finalize(st, ready)
        self._emit(ready)

    # --- вызывается из event loop ---------------------------------------
    def flush_stale(self) -> None:
        """Закрыть фразы тех, от кого давно нет речи (в т.ч. пакеты перестали приходить)."""
        ready: list[Utterance] = []
        now = time.monotonic()
        with self._lock:
            for st in self._users.values():
                if st.active and now - st.last_voice >= self._silence_s:
                    self._finalize(st, ready)
        self._emit(ready)

    def seconds_since_last_voice(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_voice_any

    def someone_speaking(self, min_voiced_ms: int = 300, started_after: float = 0.0) -> bool:
        """Кто-то сейчас говорит (набрал min_voiced_ms речи); started_after — только начавшие после этого момента."""
        now = time.monotonic()
        with self._lock:
            return any(
                st.active and st.voiced_ms >= min_voiced_ms and now - st.last_voice < 0.3
                and st.started_mono >= started_after
                for st in self._users.values()
            )

    def long_speech_pause(self, min_voiced_ms: int = 3000, pause_s: tuple[float, float] = (0.25, 0.5)) -> bool:
        """Кто-то уже долго рассказывает и сейчас сделал короткую паузу — момент для «угу»."""
        now = time.monotonic()
        with self._lock:
            return any(
                st.active and st.voiced_ms >= min_voiced_ms and pause_s[0] <= now - st.last_voice <= pause_s[1]
                for st in self._users.values()
            )

    def drop_user(self, user_id: int) -> None:
        with self._lock:
            self._users.pop(user_id, None)

    # --- внутреннее -----------------------------------------------------
    def _vad_for(self, user_id: int, user_name: str) -> webrtcvad.Vad:
        with self._lock:
            st = self._users.get(user_id)
            if st is None:
                st = self._users[user_id] = _UserState(user_id, user_name, self._aggr, self._preroll_frames)
            return st.vad

    def _finalize(self, st: _UserState, out: list[Utterance]) -> None:
        if st.voiced_ms >= self._min_voiced_ms:
            out.append(
                Utterance(
                    user_id=st.user_id,
                    user_name=st.user_name,
                    pcm_mono48=bytes(st.frames),
                    started_at=st.started_wall,
                    duration_s=len(st.frames) / 2 / DISCORD_RATE,
                )
            )
        st.reset()

    def _emit(self, ready: list[Utterance]) -> None:
        for utt in ready:
            self._on_utterance(utt)
