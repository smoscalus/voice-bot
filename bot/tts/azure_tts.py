"""Microsoft Azure Speech — те же голоса, что в Edge (Svetlana и др.), но через официальный API:
первый звук за ~0.2–0.3 с, звук приходит потоком, без перебоев.

Нужны AZURE_SPEECH_KEY и AZURE_SPEECH_REGION (portal.azure.com → «Speech service», тариф Free F0).
"""
from __future__ import annotations

import logging
from typing import AsyncIterator
from xml.sax.saxutils import escape

import aiohttp

log = logging.getLogger(__name__)


class AzureTTS:
    def __init__(self, key: str, region: str, voice: str = "ru-RU-SvetlanaNeural", fallback=None):
        self.url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        self.voice = voice
        self.fallback = fallback
        self.headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",  # ровно то, что ждёт плеер
            "User-Agent": "discord-voice-bot",
        }
        self._session: aiohttp.ClientSession | None = None

    def _http(self) -> aiohttp.ClientSession:
        # одно соединение на всё время — экономит ~100 мс TLS-рукопожатия на каждой фразе
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15, sock_connect=5))
        return self._session

    def _ssml(self, text: str) -> str:
        lang = "-".join(self.voice.split("-")[:2])
        return (f"<speak version='1.0' xml:lang='{lang}'>"
                f"<voice name='{self.voice}'>{escape(text)}</voice></speak>")

    async def warmup(self) -> None:
        async for _ in self.stream("привет"):
            pass

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        got_audio = False
        try:
            async with self._http().post(self.url, headers=self.headers,
                                         data=self._ssml(text).encode("utf-8")) as resp:
                if resp.status != 200:
                    body = (await resp.read())[:200]
                    raise RuntimeError(f"Azure TTS HTTP {resp.status}: {body!r}")
                async for chunk in resp.content.iter_chunked(4800):
                    got_audio = True
                    yield chunk
        except Exception as e:
            if got_audio or self.fallback is None:
                raise
            log.warning("Azure TTS не ответил (%s) — говорю запасным голосом", e)
            async for chunk in self.fallback.stream(text):
                yield chunk
