"""Форматы аудио.

Discord (приём и отправка): PCM s16le, 48 кГц, стерео, кадры по 20 мс = 3840 байт.
STT:                         PCM s16le, 16 кГц, моно (WAV-контейнер для OpenAI).
OpenAI TTS (format=pcm):     PCM s16le, 24 кГц, моно.
"""
from __future__ import annotations

import io
import wave

import numpy as np

DISCORD_RATE = 48_000
FRAME_MS = 20
DISCORD_FRAME_BYTES = DISCORD_RATE * FRAME_MS // 1000 * 2 * 2  # 3840
MONO48_FRAME_BYTES = DISCORD_FRAME_BYTES // 2  # 1920
SILENCE_FRAME = b"\x00" * DISCORD_FRAME_BYTES


def stereo48_to_mono48(pcm: bytes) -> bytes:
    a = np.frombuffer(pcm, dtype=np.int16)
    a = a[: a.size - a.size % 2]
    mono = a.reshape(-1, 2).astype(np.int32).sum(axis=1) // 2
    return mono.astype(np.int16).tobytes()


# ФНЧ для 48→16 кГц: срез ~7 кГц (речь целиком), окно Хэмминга. Без него высокие частоты
# «заворачиваются» в слышимый диапазон и Whisper хуже разбирает слова.
_LOWPASS = np.sinc(np.arange(-48, 49) * 7000 / 24000) * np.hamming(97)
_LOWPASS /= _LOWPASS.sum()


def mono48_to_mono16(pcm: bytes) -> np.ndarray:
    """48 кГц → 16 кГц для распознавания: нормальный фильтр + выравнивание громкости (тихие микрофоны)."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if a.size < 3:
        return a.astype(np.int16)
    x = np.convolve(a, _LOWPASS, mode="same")[::3]
    peak = float(np.percentile(np.abs(x), 99.5))
    if peak > 50:  # совсем тишину не раздуваем
        x *= min(8.0, 0.5 * 32767 / peak)  # тихий голос — громче, громкий — не трогаем дальше половины шкалы
    return np.clip(x, -32768, 32767).astype(np.int16)


def pcm16_to_wav(samples: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


class Upsampler24kMonoTo48kStereo:
    """Потоковый апсемплинг ×2 линейной интерполяцией + дублирование в стерео.

    Хранит последний сэмпл предыдущего чанка, поэтому на стыках чанков нет щелчков.
    """

    def __init__(self) -> None:
        self._prev = 0.0
        self._rem = b""  # нечётный хвостовой байт, если чанк порезан посреди сэмпла

    def process(self, pcm: bytes) -> bytes:
        pcm = self._rem + pcm
        cut = len(pcm) - len(pcm) % 2
        pcm, self._rem = pcm[:cut], pcm[cut:]
        if not pcm:
            return b""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        prev = np.concatenate(([self._prev], x[:-1]))
        out = np.empty(x.size * 2, dtype=np.float32)
        out[0::2] = (prev + x) / 2
        out[1::2] = x
        self._prev = float(x[-1])
        stereo = np.repeat(out, 2)  # L=R, интерливинг
        return np.clip(stereo, -32768, 32767).astype(np.int16).tobytes()
