"""Заплатка: расшифровка E2EE (DAVE) для входящего голоса.

Discord шифрует голос сквозным шифрованием (DAVE). discord.py 2.7 умеет только
*отправлять* в такой канал: он не вызывает DaveSession.prepare_transition /
execute_transition, поэтому расшифровщики для других участников не создаются
(ошибка NoDecryptorForUser). А discord-ext-voice-recv снимает лишь транспортное
шифрование, и Opus получает мусор («corrupted stream»).

Что делаем:
1. После MLS commit/welcome вызываем prepare_transition (создаёт ключи участников),
   а на execute — execute_transition.
2. В AudioReader после транспортной расшифровки прогоняем кадр через
   dave_session.decrypt(user_id, audio, data).
Когда библиотеки научатся этому сами, модуль можно удалить.
"""
from __future__ import annotations

import logging
import struct
import time

from discord import gateway as _gateway
from discord import voice_state as _voice_state
from discord.ext.voice_recv import reader as _reader
from discord.ext.voice_recv.rtp import OPUS_SILENCE

try:
    import davey
except ImportError:  # без davey discord.py и не включит DAVE
    davey = None

log = logging.getLogger(__name__)
_PATCHED = False
DAVE_MAGIC = b"\xfa\xfa"  # маркер в конце каждого кадра со сквозным шифрованием (протокол DAVE)

_WS = _gateway.DiscordVoiceWebSocket


# --------------------------------------------------------------- 1. ключи участников
def _patch_transitions() -> None:
    orig_binary = _WS.received_binary_message
    orig_execute = _voice_state.VoiceConnectionState._execute_transition

    async def received_binary_message(self, msg: bytes) -> None:
        await orig_binary(self, msg)
        state = self._connection
        session = state.dave_session
        if session is None or len(msg) < 5 or msg[2] not in (_WS.MLS_ANNOUNCE_COMMIT_TRANSITION, _WS.MLS_WELCOME):
            return
        transition_id = struct.unpack_from(">H", msg, 3)[0]
        try:
            session.prepare_transition(transition_id, state.dave_protocol_version)
            if transition_id == 0:  # нулевой переход применяется сразу, без execute от сервера
                session.execute_transition(0)
            log.info("DAVE: смена ключей, переход %s подготовлен", transition_id)
        except Exception as e:
            log.warning("DAVE prepare_transition(%s) failed: %s", transition_id, e)

    async def _execute_transition(self, transition_id: int) -> None:
        session = self.dave_session
        if session is not None and transition_id in self.dave_pending_transitions:
            try:
                session.execute_transition(transition_id)
                log.info("DAVE: переход %s применён", transition_id)
            except Exception as e:
                log.warning("DAVE execute_transition(%s) failed: %s", transition_id, e)
        await orig_execute(self, transition_id)

    _WS.received_binary_message = received_binary_message
    _voice_state.VoiceConnectionState._execute_transition = _execute_transition


# --------------------------------------------------------------- 2. расшифровка кадров
class DaveHealth:
    """Здоровье приёма звука — по нему сессия понимает, что бот «оглох», и перезаходит в канал.

    «Успех» — это кадр, который реально декодировался в звук (а не просто расшифровался):
    после смены ключей расшифровка может «пройти», а на выходе — мусор, который Opus не примет.
    """

    def __init__(self) -> None:
        self.last_ok = 0.0
        self.last_fail = 0.0
        self.fail_since_ok = 0
        self._last_warn: dict[str, float] = {}

    def note_ok(self) -> None:
        self.last_ok = time.monotonic()
        self.fail_since_ok = 0

    def note_fail(self, reason: str, who) -> None:
        now = time.monotonic()
        self.last_fail = now
        self.fail_since_ok += 1
        key = f"{reason}:{who}"
        if now - self._last_warn.get(key, 0) > 10:  # не чаще раза в 10 с на причину и человека
            self._last_warn[key] = now
            log.warning("звук от %s не разобрать: %s", who, reason)

    def is_deaf(self, window_s: float = 5.0) -> bool:
        now = time.monotonic()
        # ошибки идут прямо сейчас, а нормального звука не было window_s секунд
        return self.fail_since_ok > 50 and now - self.last_fail < 1.0 and now - self.last_ok > window_s


def get_health(vc) -> DaveHealth:
    h = getattr(vc, "_dave_health", None)
    if h is None:
        h = vc._dave_health = DaveHealth()
    return h


def _wrap_reader(rd: "_reader.AudioReader") -> None:
    dec = rd.decryptor
    transport_decrypt = dec.decrypt_rtp
    vc = rd.voice_client
    health = get_health(vc)
    ok: set[int] = set()

    def decrypt_rtp(packet) -> bytes:
        data = transport_decrypt(packet)
        if data == OPUS_SILENCE:
            return data
        marked = data[-2:] == DAVE_MAGIC  # у кадров со сквозным шифрованием в конце метка 0xFAFA
        conn = getattr(vc, "_connection", None)
        session = getattr(conn, "dave_session", None)
        if session is None or (not getattr(conn, "dave_protocol_version", 0) and not marked):
            if marked:
                health.note_fail("кадр зашифрован, а сессии шифрования нет", packet.ssrc)
                return OPUS_SILENCE
            health.note_ok()  # шифрования нет — обычный Opus
            return data
        user_id = vc._ssrc_to_id.get(packet.ssrc)
        if user_id is None:
            health.note_fail("неизвестный поток (ssrc ещё не сопоставлен с человеком)", packet.ssrc)
            return OPUS_SILENCE
        try:
            out = session.decrypt(user_id, davey.MediaType.audio, data)
            health.note_ok()
            if user_id not in ok:
                ok.add(user_id)
                log.info("DAVE: голос пользователя %s расшифровывается", user_id)
            return out
        except Exception as e:
            # незашифрованный кадр в переходный период — это обычный Opus, пропускаем.
            # Но если на нём метка шифрования — это НЕ открытый звук, а мусор (раньше он уходил в декодер молча)
            if not marked and (session.can_passthrough(user_id) or "Unencrypted" in str(e)):
                health.note_ok()
                return data
            health.note_fail(f"не расшифровался ({e})", user_id)
            return OPUS_SILENCE

    dec.decrypt_rtp = decrypt_rtp


def _patch_reader() -> None:
    original_init = _reader.AudioReader.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _wrap_reader(self)

    _reader.AudioReader.__init__ = __init__


def _patch_decoder() -> None:
    """Один битый кадр (смена ключей, потеря пакета) не должен убивать весь поток приёма."""
    from discord.opus import OpusError
    from discord.ext.voice_recv import opus as _opus

    from bot.audio.convert import SILENCE_FRAME

    orig_decode = _opus.PacketDecoder._decode_packet

    def _decode_packet(self, packet):
        try:
            return orig_decode(self, packet)
        except OpusError:
            vc = getattr(self.sink, "voice_client", None)
            if vc is not None:  # битый звук — тоже симптом проблем с приёмом
                get_health(vc).note_fail("битый звук (Opus не декодировал)", self.ssrc)
            return packet, SILENCE_FRAME

    _opus.PacketDecoder._decode_packet = _decode_packet


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_decoder()
    if davey is not None:
        _patch_transitions()
        _patch_reader()
    _PATCHED = True
