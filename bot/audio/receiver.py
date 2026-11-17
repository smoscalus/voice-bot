"""Приём голоса из Discord.

Сам discord.py умеет только *отправлять* звук. Приём даёт расширение
discord-ext-voice-recv: подключаемся с cls=VoiceRecvClient и вешаем AudioSink.
write() вызывается из отдельного потока, по одному разу на 20-мс пакет каждого
говорящего, уже декодированный из Opus в PCM 48 кГц стерео.
"""
from __future__ import annotations

import logging

import discord
from discord.ext import voice_recv

from bot.audio import dave_patch
from bot.audio.segmenter import UtteranceSegmenter

log = logging.getLogger(__name__)
dave_patch.apply()


class SegmentingSink(voice_recv.AudioSink):
    def __init__(self, segmenter: UtteranceSegmenter):
        super().__init__()
        self.segmenter = segmenter

    def wants_opus(self) -> bool:
        return False  # пусть библиотека декодирует Opus → PCM

    def write(self, user: discord.Member | discord.User | None, data: voice_recv.VoiceData) -> None:
        # user = None, если SSRC ещё не сопоставлен с пользователем (первые пакеты)
        if user is None or user.bot or not data.pcm:
            return
        try:
            self.segmenter.feed(user.id, user.display_name, data.pcm)
        except Exception:
            # исключение здесь убивает весь поток приёма voice-recv — бот глохнет насовсем.
            # Поэтому один кривой кадр просто пропускаем.
            log.exception("кадр от %s не обработан — пропускаю", user)

    @voice_recv.AudioSink.listener()
    def on_voice_member_disconnect(self, member: discord.Member, ssrc: int | None) -> None:
        self.segmenter.drop_user(member.id)

    def cleanup(self) -> None:
        pass
