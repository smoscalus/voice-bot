from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands, voice_recv
from openai import AsyncOpenAI

from bot.brain.ollama import ensure_ollama
from bot.config import PROVIDERS, Settings, has_groq_key, load_settings
from bot import persona
from bot.audio.sfx import SoundBank
from bot.session import VoiceSession
from bot.stt import make_stt
from bot.tts import make_tts

log = logging.getLogger("bot")


class VoiceBot(commands.Bot):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.members = True  # привилегированный: включить в Developer Portal (нужен для display_name)
        # только слеш-команды; when_mentioned убирает варнинг про message content intent
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.settings = settings
        self.openai: AsyncOpenAI | None = None
        self.sessions: dict[int, VoiceSession] = {}
        self.stt = None
        self.tts = None
        self.sfx = None
        self._stt_cache: dict[str, object] = {}  # локальный Whisper тяжёлый — грузим один раз
        self._brain_lock = asyncio.Lock()

    async def _activate_brain(self, settings: Settings, wait_ollama_s: float = 120) -> None:
        """Подготовить «мозги»: для local — поднять/докачать Ollama, для groq — проверить ключ."""
        if settings.provider == "local":
            await ensure_ollama(settings.ollama_url,
                                [settings.llm_model, settings.classifier_model, settings.memory_model],
                                wait_s=wait_ollama_s)
        elif settings.openai_api_key in ("", "none"):
            raise RuntimeError("для PROVIDER=groq нужен GROQ_API_KEY в .env (бесплатно на console.groq.com)")
        client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        # распознавание может идти не туда же, куда мозги (локальная Ollama + Whisper в облаке Groq)
        stt_client = (AsyncOpenAI(api_key=settings.stt_api_key, base_url=settings.stt_base_url)
                      if settings.stt_backend == "openai" and settings.stt_base_url != settings.openai_base_url
                      else client)
        stt_key = f"{settings.stt_backend}:{settings.stt_base_url}"
        if stt_key not in self._stt_cache:
            log.info("готовлю распознавание речи (%s)… при первом запуске модель скачивается", stt_key)
            self._stt_cache[stt_key] = await asyncio.to_thread(make_stt, settings, stt_client)
        self.settings, self.openai, self.stt = settings, client, self._stt_cache[stt_key]
        await self._warmup_llm()
        log.info("мозги: %s (%s)", settings.provider, settings.llm_model)

    async def switch_provider(self, provider: str) -> str:
        """Переключить мозги на ходу (команда /brain). Возвращает текст для ответа в чат."""
        async with self._brain_lock:
            if provider == self.settings.provider:
                return f"Уже работаю на **{provider}** ({self.settings.llm_model})."
            new = load_settings(provider)
            await self._activate_brain(new, wait_ollama_s=6)  # на ходу долго не ждём
            for session in self.sessions.values():
                session.set_brain(self.settings, self.openai, self.stt)
            return f"Переключился на **{provider}** ({new.llm_model}). Память и досье на месте."

    async def setup_hook(self) -> None:
        try:
            await self._activate_brain(self.settings)
        except Exception as e:
            # local не поднялся, но есть ключ облака — не падаем, а работаем на облаке
            if self.settings.provider == "local" and has_groq_key():
                log.error("%s — запускаюсь на облаке (groq)", e)
                await self._activate_brain(load_settings("groq"))
            else:
                raise
        self.tts = await asyncio.to_thread(make_tts, self.settings, self.openai)
        if hasattr(self.tts, "warmup"):
            await self.tts.warmup()
        log.info("TTS ready (%s)", self.settings.tts_backend)
        voice_key = self.settings.edge_voice if self.settings.tts_backend in ("edge", "azure") else self.settings.piper_voice
        if self.settings.sfx_enabled:
            self.sfx = SoundBank(self.tts, voice_key)
            await self.sfx.load()
        await self._warmup_llm()
        register_commands(self)
        synced = await self.tree.sync()
        log.info("slash commands synced: %s", [c.name for c in synced])

    async def _warmup_llm(self) -> None:
        """Ollama грузит модель в видеопамять на первом запросе (+2–10 с) — делаем это заранее."""
        try:
            # с настоящим системным промптом: локальная Ollama закеширует его, и первый ответ будет быстрым
            await self.openai.with_options(timeout=max(30.0, self.settings.llm_timeout_s)).chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "system", "content": persona.SYSTEM_PROMPT},
                          {"role": "user", "content": "привет"}],
                max_completion_tokens=1,
            )
            log.info("LLM warmed up")
        except Exception as e:
            log.warning("LLM warmup failed: %s", e)

    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        # остались одни в канале — выходим, чтобы не жечь деньги на тишине
        session = self.sessions.get(member.guild.id)
        if session and not [m for m in session.vc.channel.members if not m.bot]:
            self.sessions.pop(member.guild.id, None)
            await session.stop()


def register_commands(bot: VoiceBot) -> None:
    # Запасной способ без слеш-команд: «@Бот join» / «@Бот leave» в любом текстовом канале.
    @bot.command(name="join")
    async def join_text(ctx: commands.Context):
        member = ctx.author
        if not isinstance(member, discord.Member) or not member.voice or not member.voice.channel:
            await ctx.reply("Сначала зайди в голосовой канал.")
            return
        if ctx.guild.id in bot.sessions:
            await bot.sessions.pop(ctx.guild.id).stop()
        vc = await member.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        session = VoiceSession(vc, bot.settings, bot.openai, bot.stt, bot.tts, bot.sfx)
        session.start()
        bot.sessions[ctx.guild.id] = session
        await ctx.reply(f"Зашёл в **{member.voice.channel.name}** 🙂")

    @bot.command(name="leave")
    async def leave_text(ctx: commands.Context):
        session = bot.sessions.pop(ctx.guild.id, None)
        if session:
            await session.stop(farewell=True)
        await ctx.reply("Пока!")

    @bot.tree.command(name="join", description="Позвать бота в твой голосовой канал")
    async def join(interaction: discord.Interaction):
        member = interaction.user
        if not isinstance(member, discord.Member) or not member.voice or not member.voice.channel:
            await interaction.response.send_message("Сначала зайди в голосовой канал.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        guild_id = interaction.guild_id
        if guild_id in bot.sessions:
            await bot.sessions.pop(guild_id).stop()
        vc = await member.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        session = VoiceSession(vc, bot.settings, bot.openai, bot.stt, bot.tts, bot.sfx)
        session.start()
        bot.sessions[guild_id] = session
        await interaction.followup.send(f"Зашёл в **{member.voice.channel.name}**. Зовите меня по имени 🙂")

    @bot.tree.command(name="leave", description="Выгнать бота из голосового канала")
    async def leave(interaction: discord.Interaction):
        session = bot.sessions.pop(interaction.guild_id, None)
        if not session:
            await interaction.response.send_message("Я и так не в канале.", ephemeral=True)
            return
        await interaction.response.defer()
        await session.stop(farewell=True)
        await interaction.followup.send("Пока!")

    @bot.tree.command(name="brain", description="Переключить мозги бота: local (Ollama на компе) или groq (облако)")
    @app_commands.describe(provider="local — без лимитов, на этом компе; groq — умнее и быстрее, но с лимитами")
    @app_commands.choices(provider=[app_commands.Choice(name=p, value=p) for p in PROVIDERS])
    @app_commands.default_permissions(manage_guild=True)
    async def brain(interaction: discord.Interaction, provider: app_commands.Choice[str] | None = None):
        if provider is None:
            await interaction.response.send_message(
                f"Сейчас мозги: **{bot.settings.provider}** ({bot.settings.llm_model}). "
                f"Переключить: `/brain provider:local` или `/brain provider:groq`.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        try:
            msg = await bot.switch_provider(provider.value)
        except Exception as e:
            msg = f"Не получилось переключиться на {provider.value}: {e}. Остаюсь на {bot.settings.provider}."
        await interaction.followup.send(msg)

    @bot.command(name="brain")
    async def brain_text(ctx: commands.Context, provider: str = ""):
        if provider not in PROVIDERS:
            await ctx.reply(f"Сейчас: {bot.settings.provider}. Варианты: {', '.join(PROVIDERS)}")
            return
        try:
            await ctx.reply(await bot.switch_provider(provider))
        except Exception as e:
            await ctx.reply(f"Не получилось: {e}")

    @bot.tree.command(name="proactive", description="Включить/выключить инициативные реплики")
    @app_commands.describe(enabled="true — бот может сам вклиниваться в разговор")
    async def proactive(interaction: discord.Interaction, enabled: bool):
        session = bot.sessions.get(interaction.guild_id)
        if not session:
            await interaction.response.send_message("Я не в канале.", ephemeral=True)
            return
        session.proactive_enabled = enabled
        await interaction.response.send_message(f"Инициатива: {'вкл' if enabled else 'выкл'}.")


def run() -> None:
    import os

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/bot.log", encoding="utf-8")],
    )
    # voice-recv очень болтлив на INFO («unexpected rtcp packet», «extra keys») — глушим
    logging.getLogger("discord.ext.voice_recv").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = load_settings()
    VoiceBot(settings).run(settings.discord_token, log_handler=None)
