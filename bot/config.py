from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _s(name: str, default: str) -> str:
    return os.getenv(name, default)


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _list(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


PROVIDERS = ("local", "groq")


@dataclass(frozen=True)
class Settings:
    discord_token: str
    provider: str                # «мозги»: local (Ollama на этом компе) или groq (облако)
    ollama_url: str
    openai_api_key: str
    openai_base_url: str | None  # любой OpenAI-совместимый сервер: Groq, OpenRouter, Ollama…
    llm_timeout_s: float         # облако отвечает за доли секунды, локальной модели нужно больше
    tts_backend: str
    edge_voice: str  # голос Microsoft — для edge и для azure
    piper_voice: str
    piper_speed: float       # темп голоса Piper: 1.0 обычный, 1.2 быстрее
    piper_liveliness: float  # живость интонации Piper: 0.3 монотонно … 1.0 эмоционально
    azure_speech_key: str
    azure_speech_region: str

    stt_backend: str
    stt_model: str
    stt_base_url: str | None  # куда слать распознавание (может отличаться от мозгов: Ollama + Groq Whisper)
    stt_api_key: str
    local_whisper_model: str
    local_whisper_device: str
    language: str

    llm_model: str
    classifier_model: str
    memory_model: str  # модель для досье и самоосознания (в фоне; нужна поумнее)
    reasoning_effort: str | None  # для «думающих» моделей (qwen3, gpt-oss): none/low — быстрее
    llm_fallback_models: list[str]  # куда переключаться, когда у модели кончился лимит
    tts_model: str
    tts_voice: str

    vad_aggressiveness: int
    silence_ms: int
    min_utterance_ms: int
    max_utterance_s: float

    buffer_window_s: float
    buffer_max_lines: int

    prefilter_threshold: float
    interest_confidence: float
    proactive_cooldown_s: float
    proactive_max_per_10min: int
    min_gap_after_bot_s: float
    follow_up_window_s: float
    attention_window_s: float
    quiet_gap_s: float
    classifier_max_per_hour: int
    barge_in_ms: int  # сколько мс надо проговорить поверх бота, чтобы он замолчал (0 = выкл)
    interest_per_line: float  # «интерес» от любой реплики людей (любопытный лезет чаще)
    sfx_enabled: bool  # «угу»/«хм…» голосом бота
    own_initiative_gap_s: float  # «свои дела»: сам начинает тему не чаще, чем раз в столько секунд
    persona_prompt: str  # auto / full / lite — полный или облегчённый характер (lite — для локальных моделей)


def _brain(provider: str) -> dict:
    """Всё, что зависит от выбора «мозгов», — в одном месте. Переключение = смена PROVIDER."""
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        return dict(
            openai_api_key=key or "none",
            openai_base_url="https://api.groq.com/openai/v1",
            llm_model=_s("GROQ_MODEL", "qwen/qwen3.8-27b"),
            classifier_model=_s("GROQ_LIGHT_MODEL", "openai/gpt-oss-20b"),
            memory_model=_s("GROQ_MEMORY_MODEL", "openai/gpt-oss-120b"),
            llm_fallback_models=_list("GROQ_FALLBACK_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b"),
            reasoning_effort=_s("GROQ_REASONING_EFFORT", "none") or None,
            stt_backend="openai",
            stt_model=_s("GROQ_STT_MODEL", "whisper-large-v3"),  # полная — точнее turbo, на Groq не медленнее
            stt_base_url="https://api.groq.com/openai/v1",
            stt_api_key=key or "none",
            llm_timeout_s=12.0,
        )
    # local: Ollama на этом компьютере (или в соседнем docker-контейнере) + локальный Whisper
    model = _s("OLLAMA_MODEL", "qwen2.5:7b")
    return dict(
        openai_api_key="ollama",
        openai_base_url=_s("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/v1",
        llm_model=model,
        classifier_model=_s("OLLAMA_LIGHT_MODEL", model),
        memory_model=_s("OLLAMA_MEMORY_MODEL", model),
        llm_fallback_models=[],
        reasoning_effort=None,
        # распознавание: STT_PROVIDER=auto — через Groq, если есть ключ (быстро и точно, а мозги всё равно локальные),
        # иначе локальный Whisper; local / groq — выбрать вручную
        **_stt_for_local(),
        llm_timeout_s=_f("OLLAMA_TIMEOUT_S", 90),
    )


def _stt_for_local() -> dict:
    key = os.getenv("GROQ_API_KEY") or (os.getenv("OPENAI_API_KEY", "") if os.getenv("OPENAI_API_KEY", "").startswith("gsk_") else "")
    choice = _s("STT_PROVIDER", "auto").lower()
    if choice == "groq" or (choice == "auto" and key):
        return dict(stt_backend="openai", stt_model=_s("GROQ_STT_MODEL", "whisper-large-v3"),
                    stt_base_url="https://api.groq.com/openai/v1", stt_api_key=key or "none")
    return dict(stt_backend="local", stt_model="", stt_base_url=None, stt_api_key="none")


def has_groq_key() -> bool:
    load_dotenv()
    return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("gsk_"))


def load_settings(provider: str | None = None) -> Settings:
    """provider — принудительно (для переключения командой /brain), иначе из PROVIDER в .env."""
    load_dotenv()
    provider = (provider or os.getenv("PROVIDER") or ("groq" if has_groq_key() else "local")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"PROVIDER должен быть одним из {PROVIDERS}, а не {provider!r}")
    return Settings(
        discord_token=os.environ["DISCORD_TOKEN"],
        provider=provider,
        ollama_url=_s("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        **_brain(provider),
        tts_backend=_s("TTS_BACKEND", "piper"),
        edge_voice=_s("EDGE_VOICE", "ru-RU-DmitryNeural"),
        piper_voice=_s("PIPER_VOICE", "ru_RU-dmitri-medium"),
        piper_speed=_f("PIPER_SPEED", 1.0),
        piper_liveliness=_f("PIPER_LIVELINESS", 0.667),
        azure_speech_key=_s("AZURE_SPEECH_KEY", ""),
        azure_speech_region=_s("AZURE_SPEECH_REGION", "westeurope"),
        local_whisper_model=_s("LOCAL_WHISPER_MODEL", "auto"),  # auto: small на видеокарте, base на процессоре
        local_whisper_device=_s("LOCAL_WHISPER_DEVICE", "auto"),
        language=_s("LANGUAGE", "ru"),
        tts_model=_s("TTS_MODEL", "gpt-4o-mini-tts"),
        tts_voice=_s("TTS_VOICE", "alloy"),
        vad_aggressiveness=_i("VAD_AGGRESSIVENESS", 2),
        silence_ms=_i("SILENCE_MS", 500),
        min_utterance_ms=_i("MIN_UTTERANCE_MS", 400),
        max_utterance_s=_f("MAX_UTTERANCE_S", 15),
        buffer_window_s=_f("BUFFER_WINDOW_S", 180),
        buffer_max_lines=_i("BUFFER_MAX_LINES", 40),
        prefilter_threshold=_f("PREFILTER_THRESHOLD", 2.5),
        interest_confidence=_f("INTEREST_CONFIDENCE", 0.6),
        proactive_cooldown_s=_f("PROACTIVE_COOLDOWN_S", 90),
        proactive_max_per_10min=_i("PROACTIVE_MAX_PER_10MIN", 5),
        min_gap_after_bot_s=_f("MIN_GAP_AFTER_BOT_S", 12),
        follow_up_window_s=_f("FOLLOW_UP_WINDOW_S", 20),
        attention_window_s=_f("ATTENTION_WINDOW_S", 8),
        quiet_gap_s=_f("QUIET_GAP_S", 0.8),
        classifier_max_per_hour=_i("CLASSIFIER_MAX_PER_HOUR", 60),
        barge_in_ms=_i("BARGE_IN_MS", 0),
        interest_per_line=_f("INTEREST_PER_LINE", 0.7),
        sfx_enabled=_s("SFX", "on").lower() in ("on", "1", "true"),
        own_initiative_gap_s=_f("OWN_INITIATIVE_GAP_S", 600),
        persona_prompt=_s("PERSONA_PROMPT", "auto").lower(),
    )
