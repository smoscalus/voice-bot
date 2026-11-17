"""Отправка голоса бота в канал.

StreamingPCMSource — discord.AudioSource, который читает из буфера, наполняемого
по мере прихода чанков TTS. Плеер discord.py вызывает read() из своего потока
каждые 20 мс и ждёт ровно 3840 байт PCM 48к стерео; b"" означает конец потока.
Благодаря этому воспроизведение начинается с первого чанка TTS, не дожидаясь
генерации всей фразы. FFmpeg не нужен.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import AsyncIterator

import discord

from bot.audio.convert import DISCORD_FRAME_BYTES, SILENCE_FRAME, Upsampler24kMonoTo48kStereo

log = logging.getLogger(__name__)
STALL_TIMEOUT_S = 25.0  # если TTS/LLM не прислали ни кусочка столько времени — считаем, что зависли


class StreamingPCMSource(discord.AudioSource):
    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._finished = False
        self._up = Upsampler24kMonoTo48kStereo()
        self.fed_bytes = 0      # сколько звука TTS уже выдал
        self.played_bytes = 0   # сколько из него реально прозвучало

    def feed_pcm24_mono(self, chunk: bytes) -> None:
        data = self._up.process(chunk)
        with self._lock:
            self._buf += data
            self.fed_bytes += len(data)

    def finish(self) -> None:
        with self._lock:
            self._finished = True

    def read(self) -> bytes:
        with self._lock:
            if len(self._buf) >= DISCORD_FRAME_BYTES:
                frame = bytes(self._buf[:DISCORD_FRAME_BYTES])
                del self._buf[:DISCORD_FRAME_BYTES]
                self.played_bytes += DISCORD_FRAME_BYTES
                return frame
            if self._finished:
                if self._buf:
                    frame = bytes(self._buf).ljust(DISCORD_FRAME_BYTES, b"\x00")
                    self._buf.clear()
                    return frame
                return b""
        return SILENCE_FRAME  # TTS ещё не догнал — отдаём тишину, а не обрываем поток

    def is_opus(self) -> bool:
        return False


class VoicePlayer:
    def __init__(self, vc: discord.VoiceClient):
        self.vc = vc
        self.interrupted = False   # последнюю реплику перебили
        self.played_ratio = 1.0    # какая доля сгенерированного звука успела прозвучать
        self.started_at = 0.0      # time.monotonic() начала текущей реплики

    @property
    def is_playing(self) -> bool:
        return self.vc.is_playing()

    def _stop_playing(self) -> None:
        # ВАЖНО: у VoiceRecvClient метод stop() останавливает и воспроизведение, И ПРИЁМ звука —
        # бот «глохнет» без всяких ошибок. Поэтому глушим только воспроизведение.
        stop_playing = getattr(self.vc, "stop_playing", None)
        if stop_playing is not None:
            stop_playing()
        else:
            self.vc.stop()

    def stop(self) -> None:
        if self.vc.is_playing():
            self._stop_playing()

    def interrupt(self) -> None:
        """Перебили: замолкаем сразу (как голосовой режим ChatGPT)."""
        if self.vc.is_playing():
            self.interrupted = True
            self._stop_playing()

    async def play_stream(self, chunks: AsyncIterator[bytes]) -> None:
        """Проиграть поток PCM 24к моно; вернуться, когда звук закончится или его остановят."""
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        source = StreamingPCMSource()
        self.interrupted = False
        self.started_at = time.monotonic()

        def _after(err: Exception | None) -> None:
            if err:
                log.warning("player error: %s", err)
            loop.call_soon_threadsafe(done.set)

        self.stop()
        self.vc.play(source, after=_after)
        it = chunks.__aiter__()
        done_wait = asyncio.ensure_future(done.wait())
        try:
            while True:
                nxt = asyncio.ensure_future(it.__anext__())
                finished, _ = await asyncio.wait({nxt, done_wait}, timeout=STALL_TIMEOUT_S,
                                                 return_when=asyncio.FIRST_COMPLETED)
                if nxt not in finished:  # перебили или TTS завис — бросаем генерацию
                    nxt.cancel()
                    await asyncio.wait({nxt})  # дождаться отмены, иначе aclose() ругнётся
                    if not finished:
                        log.warning("TTS завис дольше %s с — обрываю реплику", STALL_TIMEOUT_S)
                    break
                try:
                    chunk = nxt.result()
                except StopAsyncIteration:
                    break
                source.feed_pcm24_mono(chunk)
        finally:
            source.finish()
            if hasattr(it, "aclose"):
                try:
                    await it.aclose()  # даём TTS/LLM понять, что дальше не нужно
                except Exception:
                    pass
        try:
            await asyncio.wait_for(done_wait, timeout=60)
        except asyncio.TimeoutError:
            log.warning("плеер не завершился — останавливаю принудительно")
            self.stop()
        if source.fed_bytes:
            self.played_ratio = source.played_bytes / source.fed_bytes
        else:
            self.played_ratio = 0.0 if self.interrupted else 1.0
