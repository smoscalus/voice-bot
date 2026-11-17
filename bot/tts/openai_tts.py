from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

TTS_INSTRUCTIONS = "Говори по-русски, живо и дружелюбно, разговорным тоном, не торопясь."


class OpenAITTS:
    """Стримит сырой PCM s16le 24 кГц моно (response_format='pcm')."""

    def __init__(self, client: AsyncOpenAI, model: str, voice: str, instructions: str | None = TTS_INSTRUCTIONS):
        self.client = client
        self.model = model
        self.voice = voice
        self.instructions = instructions

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        kwargs = {"instructions": self.instructions} if self.instructions and "gpt-4o" in self.model else {}
        async with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",
            **kwargs,
        ) as resp:
            async for chunk in resp.iter_bytes(4800):  # 100 мс аудио
                yield chunk
