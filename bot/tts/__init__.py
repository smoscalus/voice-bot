from __future__ import annotations

from openai import AsyncOpenAI

from bot.config import Settings


def _piper_fallback(settings: Settings):
    """Локальный Piper подстрахует облачный голос, если тот тормозит или недоступен."""
    try:
        from bot.tts.piper_tts import PiperTTS

        return PiperTTS(settings.piper_voice)
    except Exception:
        return None


def make_tts(settings: Settings, client: AsyncOpenAI):
    if settings.tts_backend == "piper":
        from bot.tts.piper_tts import PiperTTS

        return PiperTTS(settings.piper_voice, settings.piper_speed, settings.piper_liveliness)
    if settings.tts_backend == "azure":
        from bot.tts.azure_tts import AzureTTS

        return AzureTTS(settings.azure_speech_key, settings.azure_speech_region,
                        settings.edge_voice, fallback=_piper_fallback(settings))
    if settings.tts_backend == "edge":
        from bot.tts.edge_tts_engine import EdgeTTS

        return EdgeTTS(settings.edge_voice, fallback=_piper_fallback(settings))
    from bot.tts.openai_tts import OpenAITTS

    return OpenAITTS(client, settings.tts_model, settings.tts_voice)
