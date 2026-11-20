"""VoiceSession — оркестратор одного голосового канала.

Поток данных:
  voice-recv thread ─► SegmentingSink ─► UtteranceSegmenter ──(call_soon_threadsafe)──► asyncio.Queue
  ─► STT (параллельно, семафор) ─► TranscriptBuffer ─► TriggerDetector
       ├─ стоп-слово         → остановить воспроизведение
       ├─ wake word          → Responder.reply_direct → TTS → VoicePlayer
       └─ иначе              → накопить score; при пороге (или follow-up) — отложенная оценка:
                                ждём тишины → InterestClassifier → SpeakPolicy → reply_proactive
"""
from __future__ import annotations

import asyncio
from collections import deque
import random
import re
import logging
import time

import discord
from discord.ext import voice_recv
from openai import AsyncOpenAI

from bot import persona
from bot.audio.convert import mono48_to_mono16
from bot.audio.dave_patch import get_health
from bot.audio.player import VoicePlayer
from bot.audio.receiver import SegmentingSink
from bot.audio.segmenter import Utterance, UtteranceSegmenter
from bot.brain.interest import InterestClassifier
from bot.brain.drive import get_drive
from bot.brain.emotions import get_emotions
from bot.brain.knowledge import get_knowledge
from bot.brain.life import bury, dream, get_life, write_epitaph
from bot.brain import book
from bot.brain.llm import LLMRouter, foreground
from bot.brain.relations import RelationsJudge, get_store
from bot.brain.self_model import get_self_model
from bot.brain.policy import SpeakPolicy
from bot.brain.predict import get_predictor
from bot.brain.monologue import get_monologue
from bot.brain.habits import get_habits
from bot.brain.responder import Responder
from bot.brain.transcript import TranscriptBuffer
from bot.brain.triggers import _SECOND_PERSON, TriggerDetector, voice_command
from bot.config import Settings
from bot.stt.base import STTEngine, clean_transcript

log = logging.getLogger(__name__)

UNCLEAR_CONFIDENCE = -0.8   # avg_logprob Whisper ниже этого — фраза распознана как каша
# Whisper на шуме и тишине «слышит» фразы из субтитров YouTube. Такие короткие и неуверенные — выкидываем.
WHISPER_GHOSTS = {"спасибо", "спасибо за внимание", "спасибо за просмотр", "добре", "подпишись", "подписывайтесь",
                  "подписывайтесь на канал", "продолжение следует", "субтитры сделал dimatorzok",
                  "редактор субтитров", "до новых встреч"}
GHOST_CONFIDENCE = -0.3     # увереннее этого — человек правда сказал «спасибо»
HURT_FELT_S = 3600          # обидчика опасается, только если обида свежая (за последний час прожитого)


_ANGLE_TAG = "\x00угол\x00"
_STOP = {"что", "значит", "это", "меня", "тебя", "можно", "такое", "какой", "какая", "почему", "можете",
         "можешь", "расскажи", "рассказать", "подробнее", "вопрос", "спросить", "петрык", "петрик",
         *persona.WAKE_WORDS}


def _topic_words(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"\w+", (text or "").lower()) if len(w) >= 4 and w not in _STOP}


def _same_topic(angle: str, said: str) -> bool:
    return len(_topic_words(angle) & _topic_words(said)) >= 2


def _expect_if_on_topic(expect: str, angle: str) -> str:
    """Классификатор ждёт реакции на свой «угол»; проверим при озвучке, что спросил он именно это."""
    return f"{_ANGLE_TAG}{angle}{_ANGLE_TAG}{expect}" if expect and angle else expect


class _Prefetch:
    """Обёртка над потоком предложений: читает его сразу в фоне, не дожидаясь, пока его начнут озвучивать.
    done — ответ целиком додуман; attach(line) — подставить додуманный текст в реплику бота заранее
    (окончательный, с учётом перебиваний, запишется после озвучки)."""

    _END = object()

    def __init__(self, source):
        self._q: asyncio.Queue = asyncio.Queue()
        self.done = asyncio.Event()
        self.items: list[str] = []
        self._line = None
        self._task = asyncio.create_task(self._run(source))

    async def _run(self, source) -> None:
        try:
            async for x in source:
                self.items.append(x)
                await self._q.put(x)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("ответ не додумался")
        finally:
            self.done.set()
            self._fill()
            self._q.put_nowait(self._END)
            if hasattr(source, "aclose"):
                try:
                    await source.aclose()
                except Exception:
                    pass

    def _fill(self) -> None:
        if self._line is not None and self.done.is_set() and self._line.text == "…" and self.items:
            self._line.text = " ".join(self.items)

    def attach(self, line) -> None:
        self._line = line
        self._fill()

    def __aiter__(self):
        return self

    async def __anext__(self):
        x = await self._q.get()
        if x is self._END:
            self._q.put_nowait(self._END)
            raise StopAsyncIteration
        return x

    async def aclose(self) -> None:
        if not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except BaseException:
                pass


FILLER_GAP_S = 6.0          # два «угу» подряд чаще — уже не рефлекс, а тик
SPECULATIVE_RESTARTS = 2    # сколько раз можно передумать, если человек договорил ещё что-то


def filler_kind(lines) -> str:
    """Какой звук-рефлекс: позвали только по имени — «м?» (пусть договорит), во всех остальных случаях —
    «хм…/э-э…»: держит паузу, чтобы люди не приняли звук за ответ и не заговорили поверх."""
    from bot.brain.triggers import NAME_LIKE

    last = lines[-1].text.strip() if lines else ""
    words = [w for w in re.findall(r"[\w-]+", last.lower()) if not NAME_LIKE.match(w)]
    if not words or (len(words) == 1 and words[0] in ("алло", "ало", "эй", "ау")):
        return "call"          # «Петрык!», «Петрик, алло»
    return "think"


def is_whisper_ghost(text: str, confidence: float | None, duration_s: float) -> bool:
    norm = re.sub(r"[^\w\s]", "", (text or "").lower()).strip()
    norm = re.sub(r"\s+", " ", norm)
    return (norm in WHISPER_GHOSTS and duration_s < 4.5
            and (confidence is None or confidence < GHOST_CONFIDENCE))
NOISE_MIN_LINES = 5         # сколько фраз за минуту смотреть, чтобы решить «вокруг шум»
NOISE_UNCLEAR_SHARE = 0.6   # доля неразборчивых среди них
NOISE_COMPLAINT_EVERY_S = 300
STALE_UTTERANCE_S = 12      # фразы, ждавшие распознавания дольше, — выкидываем (бот отстал)
OWN_CHECK_EVERY_S = 15      # как часто бот прикидывает «а не заговорить ли самому»
OWN_LULL_S = 20             # сколько секунд тишины нужно, чтобы самому начать тему
OWN_MIN_CURIOSITY = 0.6     # только когда очень любопытно
OWN_CHANCE = 0.35           # и не при каждой возможности
REPEAT_FRESH_S = 180        # свою реплику помнит дословно столько, потом — только суть
# «что ты сказал?», «что ты там говорил?», «повтори», «не расслышал» — просят повторить
REPEAT_ASK = re.compile(r"(?:что|чё|че|шо)\s+ты\s+(?:там\s+|щас\s+|сейчас\s+|только\s+что\s+)?(?:сказал|говорил|спросил|"
                        r"спрашивал|хотел\s+сказать|имел\s+в\s+виду)|\bповтори(?!\w*\s+за\b)|не\s+(?:рас)?слышал|не\s+расслышали",
                        re.IGNORECASE)


class VoiceSession:
    def __init__(self, vc: voice_recv.VoiceRecvClient, settings: Settings, client: AsyncOpenAI, stt: STTEngine, tts,
                 sfx=None):
        self.vc = vc
        self.s = settings
        self.loop = asyncio.get_running_loop()
        self.proactive_enabled = True

        self.queue: asyncio.Queue[Utterance] = asyncio.Queue()
        self.segmenter = UtteranceSegmenter(
            on_utterance=self._on_utterance_threadsafe,
            aggressiveness=settings.vad_aggressiveness,
            silence_ms=settings.silence_ms,
            min_utterance_ms=settings.min_utterance_ms,
            max_utterance_s=settings.max_utterance_s,
        )
        self.sink = SegmentingSink(self.segmenter)

        self.stt = stt  # загружается один раз при старте бота (Whisper тяжёлый)
        self.buffer = TranscriptBuffer(settings.buffer_window_s, settings.buffer_max_lines)
        self.triggers = TriggerDetector(
            persona.WAKE_WORDS, persona.WAKE_PHRASES, persona.STOP_WORDS,
            persona.TOPICS, persona.OPINION_CUES, persona.DISPUTE_CUES,
        )
        self.relations = get_store()
        self.set_brain(settings, client, stt)
        self.tts = tts  # создаётся один раз при старте бота
        self.player = VoicePlayer(vc)
        self.policy = SpeakPolicy(settings)

        self._stt_sem = asyncio.Semaphore(4)
        self._speak_lock = asyncio.Lock()
        self._speaking_proactive = False
        self._eval_task: asyncio.Task | None = None
        self._reconnecting = False
        self._last_reconnect = time.monotonic()
        self._stt_inflight = 0
        self._collecting = False
        self._gen_owner: asyncio.Task | None = None  # чей ответ сейчас думается
        self._gen_done = asyncio.Event()
        self._last_filler = 0.0  # когда звучал последний рефлекс «м?/угу»
        self._merged = 0  # сколько новых обращений влилось в ответ, который ещё думается
        self._rethink = asyncio.Event()  # бросил начатый ответ — озвучка может сказать «ага, так…»
        self._classifying = False  # идёт ли запрос «встревать ли» (его можно бросить ради ответа)
        self._reply_task: asyncio.Task | None = None
        self._last_bot_seq = 0
        self.sfx = sfx  # «угу» и «хм…» голосом бота
        self._last_backchannel = 0.0
        self._seen_users: set[int] = set()   # кто уже говорил в этой сессии
        self._returning: set[str] = set()    # вернулись, и на них есть материал в памяти
        self.drive = get_drive()              # любопытство: растёт без нового, падает, когда узнал
        self.dead = False
        self.muted_since: float | None = None  # «Петрык, помолчи» — молчит, но слушает и запоминает
        self._unmuting = False
        self._mention: tuple[float, str] | None = None  # о нём говорили в третьем лице: (когда, фраза)
        self._last_vocative = 0.0                       # когда к нему последний раз обращались
        self._chat_proactive: deque[float] = deque(maxlen=10)  # встревал, пока люди болтали между собой
        self._chatting_since: float | None = None
        self._last_left_out = 0.0
        self.on_closed = None                 # main.py: убрать сессию из списка, когда он умер
        self._next_life_check = 0.0
        self._life_busy = False
        self._my_lines: deque[str] = deque(maxlen=12)  # свои последние реплики — чтобы не повторяться
        self._clarity: deque[tuple[float, bool]] = deque(maxlen=12)  # (когда, неразборчиво ли) — детектор шума
        self._last_noise_complaint = 0.0
        self._stale_dropped = 0
        self._last_own_initiative = time.monotonic()  # «свои дела» — не сразу после захода
        self._next_own_check = 0.0
        self._tasks: set[asyncio.Task] = set()
        self._felt: dict[str, float] = {}     # когда последний раз «отозвалось» чувство к человеку
        self._missed_back: set[str] = set()   # вернулся тот, по кому скучал
        self._last_miss = time.monotonic()

    def set_brain(self, settings: Settings, client: AsyncOpenAI, stt: STTEngine) -> None:
        """Подключить «мозги» (local/groq). Вызывается при старте и при переключении командой /brain —
        прямо на ходу, без выхода из канала. Память и досье при этом сохраняются."""
        self.s = settings
        self.stt = stt
        fallbacks = settings.llm_fallback_models
        t = settings.llm_timeout_s
        from bot.brain.llm import context_limit

        ctx = context_limit() if settings.provider == "local" else 0  # у облака окно большое — не следим
        # «встревать ли» решается на лету — без размышлений, как и ответ
        light_llm = LLMRouter(client, [settings.classifier_model, *fallbacks], settings.reasoning_effort, t,
                              ctx_limit=ctx, think=settings.think_reply, tag="встревать ли")
        self.light_llm = light_llm
        # память, дневник, книжка, сны — фоном: уступают очередь живому ответу (локальная модель одна на всё)
        memory_llm = LLMRouter(client, [settings.memory_model, settings.classifier_model, *fallbacks],
                               settings.reasoning_effort, max(t, 60) * (2 if settings.think_background else 1),
                               background=True,
                               preemptible=settings.provider == "local", ctx_limit=ctx,
                               think=settings.think_background, tag="фон")
        self.classifier = InterestClassifier(light_llm, persona.BOT_NAME, persona.INTERVENE_RULES)
        if hasattr(self, "judge"):
            self.judge.llm = memory_llm
        else:
            self.judge = RelationsJudge(self.relations, memory_llm, persona.BOT_NAME)
        self.self_model = get_self_model(memory_llm)  # досье на самого себя: кто я, увлечения, задания, страхи
        self.self_model.llm = memory_llm
        self.responder = Responder(
            LLMRouter(client, [settings.llm_model, *fallbacks], settings.reasoning_effort, t, ctx_limit=ctx,
                      think=settings.think_reply, tag="ответ"),
            self._system_prompt(settings), extra_context=self._relations_context,
            recent_lines=lambda: list(self._my_lines), live_context=self._live_context)

    @staticmethod
    def _system_prompt(settings: Settings) -> str:
        """Полный характер для сильных моделей (облако), облегчённый — для локальных: маленькая модель
        путается в длинном списке правил и начинает допрашивать людей. PERSONA_PROMPT=full/lite — вручную."""
        style = settings.persona_prompt
        if style == "auto":
            style = "lite" if settings.provider == "local" else "full"
        base = persona.SYSTEM_PROMPT_LITE if style == "lite" else persona.SYSTEM_PROMPT
        return f"{base}\n\n{persona.ANSWER_RULES}"

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.vc.listen(self.sink)
        self._spawn(self._consumer())
        self._spawn(self._ticker())
        log.info("session started in #%s", self.vc.channel)

    async def stop(self, farewell: bool = False) -> None:
        if self.dead:
            return  # уже умер: память похоронена, ничего не дописываем
        if farewell and self.vc.is_connected():
            try:  # последние слова перед «смертью»
                await asyncio.wait_for(self._speak_stream(
                    self.responder.stream_farewell(self.buffer.render(self.buffer.recent(12))), proactive=False),
                    timeout=10)
            except Exception:
                pass
        try:  # успеть записать в память последний разговор и поразмышлять о себе
            await asyncio.wait_for(asyncio.gather(
                self.judge.maybe_update(self.buffer, force=True),
                self.self_model.maybe_reflect(self.buffer, force=True)), timeout=10)
        except Exception:
            pass
        for t in list(self._tasks):
            t.cancel()
        if self.vc.is_listening():
            self.vc.stop_listening()
        self.player.stop()
        await self.vc.disconnect()

    def _relations_context(self) -> str:
        """Память в системный промпт — то, что между соседними ответами почти не меняется.
        Локальная модель не перечитывает одинаковое начало промпта, поэтому всё подвижное — в _live_context."""
        participants = {l.speaker: l.speaker_id for l in self.buffer.recent(30) if not l.is_bot}
        # кому полное досье — тем, кто говорил только что (остальным одна строчка)
        active = list(dict.fromkeys(l.speaker for l in reversed(self.buffer.recent(12)) if not l.is_bot))
        blocks = [self.self_model.prompt_block(), get_habits().prompt_block(), get_knowledge().stable_block(),
                  self.relations.prompt_block(participants, active)]
        return "\n\n".join(b for b in blocks if b)

    def _live_context(self) -> str:
        """Подвижное — в самый конец промпта, перед расшифровкой: что вспомнилось, чувства, ожидания, мысли."""
        query = " ".join(l.text for l in self.buffer.recent(6))  # о чём сейчас речь — что из памяти вспомнить
        chat = "Сейчас люди разговаривают между собой, не с тобой." if self._people_chatting() else ""
        blocks = [get_knowledge().recall_block(query), get_life().prompt_block(), self.drive.prompt_block(),
                  get_emotions().prompt_block(), get_predictor().prompt_block(), get_monologue().prompt_block(),
                  chat, self._my_lines_block()]
        return "\n\n".join(b for b in blocks if b)

    def _my_lines_block(self) -> str:
        """Против повторов: модель видит свои недавние реплики, даже ушедшие из окна расшифровки."""
        if not self._my_lines:
            return ""
        return ("Твои последние реплики — НЕ повторяй эти вопросы, шутки, наезды и формулировки, "
                "придумай новое:\n" + "\n".join(f"- {t}" for t in self._my_lines))

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._task_done)
        return t

    def _task_done(self, t: asyncio.Task) -> None:
        self._tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            # раньше упавшая фоновая задача умирала молча — теперь видно в логе
            log.error("фоновая задача упала: %r", t.exception(), exc_info=t.exception())

    # ------------------------------------------------------------------ audio → text
    def _on_utterance_threadsafe(self, utt: Utterance) -> None:
        # может вызываться из потока voice-recv — в asyncio только через call_soon_threadsafe
        self.loop.call_soon_threadsafe(self.queue.put_nowait, utt)

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            try:
                self._tick()
            except Exception:
                log.exception("ошибка в ticker")  # тикер не должен умирать: на нём держится весь приём

    def _tick(self) -> None:
        if True:
            self.segmenter.flush_stale()
            self._maybe_own_initiative()
            self._maybe_life_events()
            # страховка: приём звука выключился (кто-то вызвал vc.stop() и т.п.) — включаем обратно
            if not self._reconnecting and self.vc.is_connected() and not self.vc.is_listening():
                log.warning("приём звука был выключен — включаю обратно")
                self.sink = SegmentingSink(self.segmenter)
                self.vc.listen(self.sink)
            # самовосстановление: после смены ключей шифрования бот может «оглохнуть»
            if (not self._reconnecting and time.monotonic() - self._last_reconnect > 30
                    and (get_health(self.vc).is_deaf() or self._receiver_dead())):
                self._spawn(self._reconnect())
            # (раньше тут было «угу» в паузах длинного рассказа — убрано: люди принимали его за ответ)
            # barge-in: заговорили поверх бота — он замолкает (как голосовой режим ChatGPT).
            # Инициативную реплику обрываем быстрее, прямой ответ — если говорят дольше barge_in_ms.
            if self.player.is_playing:
                # BARGE_IN_MS=0 — «танк»: голосом не перебить вообще, всегда договаривает мысль
                need_ms = 0 if self.s.barge_in_ms <= 0 else (400 if self._speaking_proactive else self.s.barge_in_ms)
                if need_ms > 0 and self.segmenter.someone_speaking(
                        min_voiced_ms=need_ms, started_after=self.player.started_at):
                    log.info("barge-in: меня перебили — замолкаю")
                    self.player.interrupt()

    def _receiver_dead(self) -> bool:
        """Поток приёма звука voice-recv упал (любая ошибка внутри убивает его насовсем)."""
        reader = getattr(self.vc, "_reader", None)
        router = getattr(reader, "packet_router", None) if reader else None
        if router is None or not self.vc.is_listening():
            return False
        if not router.is_alive():
            log.warning("поток приёма звука умер")
            return True
        return False

    async def _reconnect(self) -> None:
        self._reconnecting = True
        channel = self.vc.channel
        log.warning("бот перестал расшифровывать голос — перезахожу в #%s", channel)
        try:
            if self.vc.is_listening():
                self.vc.stop_listening()
            self.player.stop()
            await self.vc.disconnect(force=True)
            await asyncio.sleep(1.0)
            self.vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
            self.player.vc = self.vc
            self.sink = SegmentingSink(self.segmenter)
            self.vc.listen(self.sink)
            log.info("перезашёл в канал")
        except Exception:
            log.exception("reconnect failed")
        finally:
            self._reconnecting = False
            self._last_reconnect = time.monotonic()

    async def _consumer(self) -> None:
        while True:
            utt = await self.queue.get()
            self._spawn(self._handle_utterance(utt))

    async def _handle_utterance(self, utt: Utterance) -> None:
        t0 = time.perf_counter()
        self._stt_inflight += 1
        try:
            async with self._stt_sem:
                # распознавание не успевает (слабый процессор, много людей) — старые фразы выкидываем,
                # иначе очередь копится и бот отвечает на то, что сказали полминуты назад
                age = time.time() - (utt.started_at + utt.duration_s)
                if age > STALE_UTTERANCE_S:
                    self._stale_dropped += 1
                    if self._stale_dropped % 5 == 1:
                        log.warning("распознавание не успевает: выкинул устаревшую фразу (%.0f с), всего %d. "
                                    "Помогут LOCAL_WHISPER_MODEL=base/tiny или распознавание через Groq",
                                    age, self._stale_dropped)
                    return
                try:
                    samples = mono48_to_mono16(utt.pcm_mono48)
                    if hasattr(self.stt, "transcribe_scored"):
                        raw, confidence = await self.stt.transcribe_scored(samples, prompt=persona.STT_PROMPT)
                    else:
                        raw, confidence = await self.stt.transcribe(samples, prompt=persona.STT_PROMPT), None
                except Exception:
                    log.exception("STT failed")
                    return
            text = clean_transcript(raw, persona.STT_PROMPT)
            if not text:
                return
            if is_whisper_ghost(text, confidence, utt.duration_s):
                log.info("[%s] «%s» — похоже, Whisper додумал на шуме, пропускаю", utt.user_name, text)
                return
            # Whisper не уверен — значит, скорее всего, каша: помечаем, чтобы бот мог переспросить
            unclear = confidence is not None and confidence < UNCLEAR_CONFIDENCE
            self._note_clarity(unclear)
            self.buffer.add(utt.user_name, f"(неразборчиво) {text}" if unclear else text, ts=utt.started_at,
                            speaker_id=utt.user_id, end_ts=utt.started_at + utt.duration_s)
        finally:
            self._stt_inflight -= 1
        log.info("[%s %.1fs, stt %.2fs%s] %s", utt.user_name, utt.duration_s, time.perf_counter() - t0,
                 f", уверенность {confidence:.2f}{' — НЕРАЗБОРЧИВО' if unclear else ''}" if confidence is not None else "",
                 text)
        self._spawn(self.judge.maybe_update(self.buffer))  # досье/отношения в фоне (не чаще раза в 90 с)
        utt_end = utt.started_at + utt.duration_s

        # человек впервые за сессию заговорил, а в памяти на него есть планы/зацепки — повод спросить, как дела
        if utt.user_id not in self._seen_users:
            self._seen_users.add(utt.user_id)
            rel = self.relations.get(utt.user_id, utt.user_name)
            if rel.plans or rel.hooks:
                self._returning.add(utt.user_name)
                self._schedule_evaluation()

        self._feel_person(utt.user_id, utt.user_name)
        trig = self.triggers.analyze(text)

        if trig.stop and self.player.is_playing:
            self.player.interrupt()
            return

        # 0) «Петрык, помолчи» / «Петрык, можешь говорить»
        if trig.wake and (cmd := voice_command(trig.remainder, persona.MUTE_PHRASES, persona.UNMUTE_PHRASES)):
            if cmd == "mute" and not self.muted_since:
                self._mute(utt.user_name)
                return
            if cmd == "unmute" and self.muted_since:
                self._spawn(self._unmute(utt.user_name))
                return
        if self.muted_since:
            return  # молчит: слышит и запоминает (летописец работает), но не отвечает и не встревает

        # 1) прямое обращение («Петрык, …»). Если о нём говорят в третьем лице («Петрык думает») —
        #    это не обращение: решит внутренний голос, отозваться ли («эй, я всё слышу»)
        if trig.wake and not trig.vocative:
            self._mention = (time.monotonic(), text)
            log.info("о нём говорят (не к нему): %s", text)
            self.policy.add_interest(self.s.prefilter_threshold)
            self._schedule_evaluation()
            return
        if trig.wake:
            self._last_vocative = time.monotonic()
            if not trig.remainder.strip():
                # голое «Вася!» — ждём продолжения; не дождались — отвечаем на само обращение.
                self.policy.open_attention(utt.user_id)
                self._spawn(self._answer_if_no_followup(utt.user_id, utt_end))
                return
            if book.per_day() > 0 and book.READ_REQUEST.search(text):
                self._spawn(self._read_book_asked(text, utt_end))  # «Петрык, почитай в книжке про Крым»
                return
            self._request_reply(utt_end)
            return
        if self.policy.take_attention(utt.user_id):
            self._request_reply(utt_end)
            return
        # 1б) продолжение разговора с ним без имени: он только что спросил — ему отвечают, или говорят «ты…»
        #     Раньше это шло через «внутренний голос» (лишний заход к модели, +10–15 с на локальной) — теперь сразу.
        if self._continues_with_me(text):
            self._request_reply(utt_end)
            return

        # 2) контекстная инициатива
        score = self.policy.add_interest(trig.score + self.s.interest_per_line)
        if score >= self.s.prefilter_threshold or self.policy.in_follow_up():
            self._schedule_evaluation()

    async def _answer_if_no_followup(self, user_id: int, utt_end: float) -> None:
        await asyncio.sleep(min(2.5, self.s.attention_window_s))
        if self.policy.take_attention(user_id):  # продолжения не было
            self._request_reply(utt_end)

    def _maybe_own_initiative(self) -> None:
        """Свои дела: в затишье бот иногда сам начинает тему — редко и только когда очень хочет."""
        now = time.monotonic()
        if now < self._next_own_check:
            return
        self._next_own_check = now + OWN_CHECK_EVERY_S
        if self.muted_since or get_monologue().active:
            return
        last_human = max((l.end_ts or l.ts for l in self.buffer.recent(10) if not l.is_bot), default=0.0)
        if (not self.proactive_enabled or self._reconnecting or self.player.is_playing or self._reply_busy()
                or now - self._last_own_initiative < self.s.own_initiative_gap_s
                or now - self.policy.last_spoke < 60
                or self.segmenter.seconds_since_last_voice() < OWN_LULL_S
                or time.time() - last_human > 600          # в канале давно тишина — говорить не с кем
                or self.drive.level + get_emotions().talk_bias() < OWN_MIN_CURIOSITY
                or random.random() > OWN_CHANCE):
            return
        self._last_own_initiative = now
        self._spawn(self._own_initiative())

    async def _own_initiative(self) -> None:
        try:
            text = await self.responder.reply_task(self.buffer.render(self.buffer.recent(20)), persona.OWN_INITIATIVE_TASK)
        except Exception:
            log.exception("свои дела: не смог придумать")
            return
        # «ОЖИДАЮ: …» — его прикидка, как отреагируют; вслух не произносится
        parts = re.split(r"ожидаю\s*:", text or "", maxsplit=1, flags=re.IGNORECASE)
        text, expect = parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")
        if not text or "ПРОПУСК" in text.upper() or self.segmenter.seconds_since_last_voice() < 3:
            log.info("свои дела: решил промолчать")
            return
        log.info("свои дела: сам начинаю тему")
        await self._speak(text, proactive=True, expect=expect)

    # ------------------------------------------------------------------ жизнь: сны и смерть
    def _maybe_life_events(self) -> None:
        """Раз в 5 с: не пора ли увидеть сон (после тускнения памяти) или умереть (срок вышел)."""
        now = time.monotonic()
        if now < self._next_life_check or self._life_busy or self.dead:
            return
        self._next_life_check = now + 5
        if self._reconnecting or self.player.is_playing or self._reply_busy():
            return
        self._feel_left_out(now)
        self._feel_missing(now)
        if self.muted_since:
            if now - self.muted_since > getattr(persona, "MUTE_MAX_MINUTES", 30) * 60 and not self._reply_busy():
                self._life_busy = True
                self._spawn(self._unmute(None))  # про него забыли — сам осторожно спросит, можно ли уже
            return
        life = get_life()
        mono = get_monologue()
        if not life.due_to_die() and self._monologue_step(mono):
            return
        if life.due_to_die() and self.segmenter.seconds_since_last_voice() > 4:
            self._life_busy = True
            self._spawn(self._die())
        elif life.dream_pending and self.segmenter.seconds_since_last_voice() > 120:
            self._life_busy = True  # в войсе тихо пару минут — задремал
            self._spawn(self._dream())
        elif self.segmenter.seconds_since_last_voice() > 25 and not mono.active and book.due():
            self._life_busy = True  # короткое затишье, а ему так и не объяснили — читает сам (фоном)
            self._spawn(self._read_book())

    def _monologue_step(self, mono) -> bool:
        """Внутренний монолог: уйти в себя / подумать мысль / вынырнуть. True — занялся этим."""
        if mono.active:
            if why := mono.should_end():
                thoughts = mono.come_out(why)
                if thoughts and not self.muted_since:
                    self._life_busy = True
                    self._spawn(self._share_thoughts(thoughts, heavy="тяжёл" in why))
                    return True
                return False
            if mono.due_thought():
                self._life_busy = True
                self._spawn(self._think(mono))
                return True
            return False
        if why := mono.wants_to_go_in():
            mono.go_in(why)
            self._mention = None
        return False

    async def _think(self, mono) -> None:
        try:
            query = mono.about or " ".join(l.text for l in self.buffer.recent(6))
            context = "\n\n".join(b for b in [
                self.self_model.prompt_block(), get_emotions().prompt_block(),
                get_knowledge().prompt_block(query), get_predictor().prompt_block()] if b)
            await mono.think(self.judge.llm, persona.BOT_NAME, context, self.buffer.render(self.buffer.recent(10)))
        finally:
            self._life_busy = False

    async def _share_thoughts(self, thoughts: list[str], heavy: bool) -> None:
        """Додумал — делится, к чему пришёл; мысли стали тяжёлыми — выныривает к людям."""
        try:
            for _ in range(40):  # ждём паузы в разговоре (до ~20 с), не перебиваем
                if self.segmenter.seconds_since_last_voice() >= 2 and not self._reply_busy():
                    break
                await asyncio.sleep(0.5)
            else:
                return  # так и не замолчали — поделится потом (мысли есть в памяти)
            task = (persona.SHAKE_OFF_TASK if heavy else persona.SHARE_THOUGHTS_TASK).format(
                thoughts="\n".join(f"- {t}" for t in thoughts))
            if heavy:
                get_emotions().apply({"joy": 0.05, "fear": -0.05})  # к людям — легче
            await self._speak_stream(
                self.responder.stream_task(self.buffer.render(self.buffer.recent(12)), task, max_tokens=120),
                proactive=True)
        finally:
            self._life_busy = False

    def _mute(self, who: str) -> None:
        """«Петрык, помолчи»: замолкает сразу (даже посреди фразы), тихо говорит «понял, молчу» и молчит."""
        self.muted_since = time.monotonic()
        log.info("%s попросил помолчать — молчу", who)
        if self.player.is_playing:
            self.player.interrupt()
        elif self.sfx and (clip := self.sfx.random("muted")):
            self._spawn(self._play_clip(clip))
        get_emotions().apply({"sadness": 0.1, "joy": -0.05})

    async def _unmute(self, who: str | None) -> None:
        """Снова можно говорить — реагирует по-своему (обрадовался, обиделся, сказал, что хотел)."""
        if not self.muted_since:
            return
        from bot.brain.life import duration

        minutes = duration(time.monotonic() - self.muted_since)
        task = (persona.UNMUTE_TASK if who else persona.AUTO_UNMUTE_TASK).format(minutes=minutes)
        log.info("%s — молчал %s", f"{who} разрешил говорить" if who else "сам не выдержал", minutes)
        if who:
            get_emotions().apply({"joy": 0.15})
        self._unmuting = True
        self.muted_since = None
        try:
            await self._speak_stream(
                self.responder.stream_task(self.buffer.render(self.buffer.recent(20)), task, max_tokens=120),
                proactive=False)
        finally:
            self._unmuting = False
            self._life_busy = False

    def _feel_person(self, user_id: int | None, name: str) -> None:
        """Заговорил человек — у бота отзывается чувство к нему: близкому рад, обидчика опасается,
        а если это тот, по кому скучал, — очень рад и сам поздоровается. Без запросов к модели."""
        from bot.brain.relations import MISS_AFTER_S, MISS_CLOSENESS

        now = time.monotonic()
        if now - self._felt.get(name, -1e9) < 600:
            return
        self._felt[name] = now
        rel = self.relations.get(user_id, name)
        gap = get_life().lived_s - rel.last_seen_lived if rel.last_seen_lived else 0
        warm = rel.score / 10 + rel.closeness / 20
        d: dict[str, float] = {}
        if warm > 0.3:
            d["joy"] = min(0.15, 0.05 + 0.1 * warm)
        fresh_hurt = any(m.get("sign", 0) < 0 and get_life().lived_s - m.get("lived", 0) < HURT_FELT_S
                         for m in rel.moments)
        if (rel.score <= -3 or rel.trust <= -3) and fresh_hurt:
            # опасается, только пока обида свежая; старую помнит, но «привет» от него больше не пугает
            d.update(fear=0.08, trust=-0.05)
        elif rel.trust >= 3:
            d["trust"] = 0.05
        if gap > MISS_AFTER_S and rel.closeness >= MISS_CLOSENESS and rel.score >= 2:
            d["joy"] = d.get("joy", 0) + 0.15
            d["sadness"] = -0.1
            self._missed_back.add(name)
            self._returning.add(name)
            if get_monologue().active:
                get_monologue().come_out(f"вернулся {name}")
            self._schedule_evaluation()
            log.info("вернулся %s — соскучился по нему", name)
        if d:
            get_emotions().apply(d)

    def _feel_missing(self, now: float) -> None:
        """Близких давно нет — тихонько грустит (раз в полчаса)."""
        if now - self._last_miss < 1800:
            return
        self._last_miss = now
        present = {l.speaker for l in self.buffer.recent(30) if not l.is_bot}
        if self.relations.missing_block(present):
            log.info("скучает по близким, которых давно нет")
            get_emotions().apply({"sadness": 0.05})

    def _repeat_note(self, lines) -> str:
        """«А что ты там говорил?» — достаём его прошлую реплику и как хорошо он её помнит:
        только что — дословно, через несколько минут — только суть; если перебили — догадается, что хотел."""
        if not any(REPEAT_ASK.search(l.text) for l in lines):
            return ""
        first = min(l.seq for l in lines)
        prev = [l for l in self.buffer.recent(30) if l.is_bot and l.seq < first and l.text not in ("…", "")]
        if not prev:
            return ""
        said = prev[-1].text
        from bot.brain.life import duration

        age = time.time() - prev[-1].ts
        if "[меня перебили" in said:
            said = said.split("…")[0].split("[меня перебили")[0].strip() or "(ничего не успел)"
            memory = persona.REPEAT_MEMORY["cut"]
        else:
            memory = persona.REPEAT_MEMORY["fresh" if age < REPEAT_FRESH_S else "vague"]
        log.info("просят повторить — вспоминаю: «%s» (%s)", said, "перебили" if "перебили" in memory else
                 "помню" if age < REPEAT_FRESH_S else "смутно")
        return persona.REPEAT_NOTE.format(said=said, ago=f"{duration(age)} назад", memory=memory)

    def _continues_with_me(self, text: str) -> bool:
        """Реплика явно продолжает разговор с ботом: он только что что-то сказал (особенно спросил),
        и ему отвечают первым же словом, или обращаются на «ты». Сомнительное решает внутренний голос."""
        if not self.policy.in_follow_up() or self._people_chatting():
            return False
        lines = self.buffer.recent(4)
        bot_i = max((i for i, l in enumerate(lines) if l.is_bot), default=None)
        if bot_i is None:
            return False
        after = [l for l in lines[bot_i + 1:] if not l.is_bot]
        if len(after) != 1:  # отвечают ему сразу; если уже двое заговорили после него — пусть решает классификатор
            return False
        words = set(re.findall(r"[\w-]+", text.lower().replace("ё", "е")))
        others = {n.lower().split()[0] for n in (l.speaker for l in self.buffer.recent(20) if not l.is_bot) if n}
        if words & others:
            return False  # «Петя, ты как?» — к другому человеку
        asked = "?" in lines[bot_i].text
        second = bool(words & _SECOND_PERSON)
        return asked or second

    def _people_chatting(self) -> bool:
        """Люди говорят между собой: последние реплики — от разных людей, бот в них не участвует
        и к нему давно не обращались."""
        lines = [l for l in self.buffer.recent(6) if time.time() - (l.end_ts or l.ts) < 150]
        if len(lines) < 4 or any(l.is_bot for l in lines[-4:]):
            return False
        if time.monotonic() - self._last_vocative < 60 or time.monotonic() - self.policy.last_spoke < 45:
            return False
        return len({l.speaker for l in lines if not l.is_bot}) >= 2

    def _feel_left_out(self, now: float) -> None:
        """Долго болтают без него — ему немного грустно и хочется потом спросить, о чём говорили."""
        if not self._people_chatting():
            self._chatting_since = None
            return
        self._chatting_since = self._chatting_since or now
        if now - self._chatting_since > 180 and now - self._last_left_out > 600:
            self._last_left_out = now
            log.info("люди давно болтают без него — немного грустно")
            get_emotions().apply({"sadness": 0.1, "joy": -0.05})

    async def _read_book(self) -> None:
        try:
            await book.read_one(self.judge.llm)
        except Exception:
            log.exception("книжка: ошибка")
        finally:
            self._life_busy = False

    async def _read_book_asked(self, text: str, utt_end: float) -> None:
        """Попросили почитать — читает сразу (про что сказали или про то, что сам спрашивал) и рассказывает."""
        if self._life_busy:
            self._request_reply(utt_end)
            return
        self._life_busy = True
        try:
            with foreground():
                if self.sfx and (clip := self.sfx.random("reading")):
                    self._spawn(self._play_clip(clip))  # «щас, гляну в книжке…»
                try:
                    res = await book.read_one(self.light_llm, book.requested_term(text), asked=True)
                except Exception:
                    log.exception("книжка: ошибка")
                    res = None
            if res is None:  # читать нечего — просто отвечает
                self._request_reply(utt_end)
                return
            term, facts = res
            task = (persona.BOOK_TELL_TASK.format(term=term, facts="; ".join(facts)) if facts
                    else persona.BOOK_NOTHING_TASK.format(term=term))
            await self._speak_stream(
                self.responder.stream_task(self.buffer.render(self.buffer.recent(12)), task, max_tokens=120),
                proactive=False, utt_end=utt_end)
        finally:
            self._life_busy = False

    async def _dream(self) -> None:
        try:
            life = get_life()
            life.dream_pending = False
            life.save()
            await dream(self.judge.llm)
        finally:
            self._life_busy = False

    async def _die(self) -> None:
        """Срок жизни вышел: последние слова, итог жизни, память — на кладбище. Следующий /join — новый."""
        log.info("срок жизни вышел — умираю")
        said_before = len(self._my_lines)
        try:
            await asyncio.wait_for(self._speak_stream(
                self.responder.stream_task(self.buffer.render(self.buffer.recent(20)),
                                           getattr(persona, "DEATH_TASK", persona.FAREWELL_TASK), max_tokens=120),
                proactive=False), timeout=30)
        except Exception:
            log.exception("не смог сказать последние слова")
        last_words = self._my_lines[-1] if len(self._my_lines) > said_before else ""
        try:
            await asyncio.wait_for(asyncio.gather(
                self.judge.maybe_update(self.buffer, force=True),
                self.self_model.maybe_reflect(self.buffer, force=True)), timeout=20)
        except Exception:
            pass
        epitaph = await write_epitaph(self.judge.llm)
        grave = bury(epitaph, last_words)
        self.dead = True
        try:
            await self.vc.channel.send(
                f"🕯️ **{persona.BOT_NAME} умер.**\nПоследние слова: {last_words or '—'}\n\n{epitaph}\n\n"
                f"Его память — в `data/кладбище/{grave.name}`. Следующий `/join` — новое рождение."[:1990])
        except Exception:
            log.warning("не смог написать о смерти в чат канала")
        for t in list(self._tasks):
            if t is not asyncio.current_task():
                t.cancel()
        if self.vc.is_listening():
            self.vc.stop_listening()
        self.player.stop()
        await self.vc.disconnect()
        if self.on_closed:
            self.on_closed()

    def _note_clarity(self, unclear: bool) -> None:
        """Если за последнюю минуту большинство фраз — каша, бот говорит, что ничего не понимает (не чаще раза в 5 мин)."""
        now = time.monotonic()
        self._clarity.append((now, unclear))
        recent = [u for t, u in self._clarity if now - t < 60]
        if (len(recent) >= NOISE_MIN_LINES and sum(recent) / len(recent) >= NOISE_UNCLEAR_SHARE
                and now - self._last_noise_complaint > NOISE_COMPLAINT_EVERY_S and not self._reply_busy()):
            self._last_noise_complaint = now
            self._clarity.clear()
            log.info("вокруг сплошная каша (%d из %d фраз неразборчивы) — жалуюсь на шум", sum(recent), len(recent))
            self._spawn(self._speak_stream(
                self.responder.stream_task(self.buffer.render(self.buffer.recent(8)), persona.NOISE_TASK),
                proactive=False))

    # ------------------------------------------------------------------ batched reply
    def _request_reply(self, utt_end: float) -> None:
        """Попросить ответ. Если ответ уже собирается — новая реплика просто попадёт в ту же пачку."""
        if self.muted_since:
            return
        if self._reply_task and not self._reply_task.done() and self._collecting:
            self._merged += 1  # ответ ещё не начал звучать — он передумает с учётом этой фразы
            return
        if self._classifying and self._eval_task and not self._eval_task.done():
            # к нему обратились, пока он прикидывал «а не влезть ли», — бросаем это, видеокарта нужна ответу
            self._eval_task.cancel()
            log.info("бросил раздумья «встревать ли» — к нему обратились")
        # если бот сейчас отвечает — следующий ответ встанет в очередь за ним
        prev = self._reply_task if self._reply_task and not self._reply_task.done() else None
        self._reply_task = self._spawn(self._reply_batch(utt_end, prev))

    async def _reply_batch(self, utt_end: float, prev: asyncio.Task | None = None) -> None:
        try:
            with foreground():  # пока собираем пачку и отвечаем — фоновые дела ждут
                await self._reply_batch_inner(utt_end, prev)
        finally:
            if self._reply_task is asyncio.current_task():
                self._collecting = False  # вышли, так и не начав говорить (пусто, «ага» в мыслях, ошибка)

    async def _reply_batch_inner(self, utt_end: float, prev: asyncio.Task | None = None) -> None:
        self._collecting = True  # пока ждём — новые обращения вливаются в эту же пачку
        if prev is not None:
            # ждём не пока договорит, а пока ДОДУМАЕТ прошлый ответ: думать над новым можно, пока звучит старый
            while not prev.done() and not (self._gen_owner is prev and self._gen_done.is_set()):
                await asyncio.sleep(0.05)
        # Пачку не ждём: начинаем думать сразу (рефлекс «м?» прикроет паузу). Если человек тут же
        # договорил ещё фразу — до первого слова ответа передумаем с учётом её (см. _speculative).
        # _collecting остаётся True, пока можно передумать: новые фразы вливаются сюда, а не в отдельный ответ.
        seq0 = self._last_bot_seq
        cutoff = time.time() - 30

        def fresh_lines():
            return [l for l in self.buffer.since(seq0) if not l.is_bot and l.ts >= cutoff][-8:]

        lines = fresh_lines()
        if not lines:
            return
        transcript = self.buffer.render(self.buffer.recent(12))  # на обращения — небольшой контекст, экономим лимит
        note = ""
        repeat_note = self._repeat_note(lines)
        mono = get_monologue()
        if mono.active:
            last = mono.session[-1] if mono.session else "о своём"
            if not mono.note_call():
                # в своих мыслях: «ага», «норм», «щас, я думаю»
                await self._speak_stream(self.responder.stream_task(
                    transcript, persona.DISTRACTED_TASK.format(thought=last, lines=self.buffer.render(lines)),
                    max_tokens=30), proactive=False, utt_end=utt_end)
                return
            mono.come_out("выдернули")  # зовут настойчиво — выныривает
            note = persona.PULLED_OUT_NOTE.format(thought=last)
        # ответ стримится по предложениям: озвучка стартует, пока модель дописывает остальное
        if repeat_note:
            note = f"{note}\n{repeat_note}".strip()
        def make():
            ls = fresh_lines() or lines
            return self.responder.stream_batch(self.buffer.render(self.buffer.recent(12)), self.buffer.render(ls),
                                               note=note, allow_repeat=bool(repeat_note))

        await self._speak_stream(self._speculative(make), proactive=False, utt_end=utt_end,
                                 filler=filler_kind(lines))

    async def _speculative(self, make):
        """Поток ответа, который можно начать заново, пока не прозвучало первое слово: человек договорил
        ещё фразу (или её только что распознали) — бросаем начатое и думаем с ней. С кэшем промпта это дёшево."""
        self._rethink.clear()
        for attempt in range(SPECULATIVE_RESTARTS + 1):
            seen = self.buffer.last_seq
            self._merged = 0
            pf = _Prefetch(make())
            restart = False
            while not (pf.items or pf.done.is_set()):
                await asyncio.sleep(0.05)
                if attempt < SPECULATIVE_RESTARTS and self._merged:
                    restart = True
                    break
            if restart:
                await pf.aclose()
                self._rethink.set()
                log.info("он договорил — передумываю ответ с учётом новой фразы")
                continue
            # решено: дальше новые фразы — уже в следующий ответ; всё, что видел этот, считаем отвеченным
            self._collecting = False
            self._last_bot_seq = max(self._last_bot_seq, seen)
            try:
                async for x in pf:
                    yield x
            finally:
                await pf.aclose()
            return

    # ------------------------------------------------------------------ decision
    def _schedule_evaluation(self) -> None:
        if self.muted_since:
            return
        # Одна отложенная оценка за раз: она сама подхватит все новые реплики,
        # т.к. берёт буфер в момент запуска.
        if self._eval_task and not self._eval_task.done():
            return
        self._eval_task = self._spawn(self._evaluate())

    async def _wait_quiet(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        while self.segmenter.seconds_since_last_voice() < self.s.quiet_gap_s:
            if time.monotonic() > deadline:
                return False
            await asyncio.sleep(0.15)
        return True

    def _reply_busy(self) -> bool:
        return bool(self._reply_task and not self._reply_task.done())

    async def _evaluate(self) -> None:
        if get_monologue().active:
            return  # ушёл в свои мысли — сам не встревает и не тратит запросы на «а не влезть ли»
        if not await self._wait_quiet():
            return
        if self._reply_busy():
            return  # уже отвечаем напрямую — пачка и так захватит эти реплики
        seq0 = self._last_bot_seq
        follow_up = self.policy.in_follow_up()
        can_pro, why = self.policy.can_proactive()
        if not follow_up and (not can_pro or not self.proactive_enabled):
            log.debug("skip evaluation: %s", why or "proactive disabled")
            self.policy.reset_interest()
            return
        if not follow_up and not self._returning and not self.drive.wants_to_speak(get_emotions().talk_bias()):
            # сейчас не любопытно — сам не лезет и не тратит лимит на «а не влезть ли»
            self.policy.reset_interest()
            get_monologue().note_bored()
            return
        if not self.policy.can_classify():
            log.warning("classifier hourly budget exhausted")
            return

        self.policy.note_classify()
        self.policy.reset_interest()
        hint = f"Бот последний раз говорил {int(time.monotonic() - self.policy.last_spoke)} с назад." \
            if self.policy.last_spoke else "Бот ещё ничего не говорил."
        chatting = self._people_chatting()
        mention = self._mention if self._mention and time.monotonic() - self._mention[0] < 30 else None
        if mention:
            hint += (f"\nО боте только что сказали в третьем лице (не ему): «{mention[1]}». Это НЕ обращение "
                     "(addressed_to_bot=false), но он может отозваться, если хочется.")
        elif chatting:
            hint += ("\nЛюди сейчас разговаривают МЕЖДУ СОБОЙ, не с ботом. Вклиниваться — только если тема "
                     "его по-настоящему цепляет; иначе intervene=false.")
        if self._missed_back:
            hint += (f"\nВернулся тот, по кому бот скучал: {', '.join(sorted(self._missed_back))}. Он рад — "
                     "хочет поздороваться и спросить, где тот был (intervene=true).")
            self._missed_back.clear()
        if self._returning:
            hint += (f"\nТолько что впервые за сегодня появились в разговоре: {', '.join(sorted(self._returning))}. "
                     "У них в памяти есть планы/зацепки — хороший повод спросить, как успехи.")
            self._returning.clear()
        transcript = self.buffer.render(self.buffer.recent(12))
        memory = self._relations_context()  # знания, цели и планы людей — повод спросить
        self._classifying = True
        try:
            with foreground():
                verdict = await self.classifier.evaluate(transcript, hint + ("\n\n" + memory if memory else ""))
        finally:
            self._classifying = False
        log.info("verdict: %s", verdict)
        if verdict.intervene or verdict.addressed_to_bot:
            get_monologue().note_interested()
        else:
            get_monologue().note_bored()
        if self._reply_busy() or self._last_bot_seq != seq0:
            return  # пока думали, бот уже ответил — второй раз не лезем

        last = self.buffer.recent(1)
        if verdict.addressed_to_bot and last and not last[0].is_bot:
            self._request_reply(last[0].end_ts or last[0].ts)
            return
        now = time.monotonic()
        need = self.s.interest_confidence
        if chatting and not mention:  # болтают без него — встревает редко: уверенно и не больше 2 раз за 10 мин
            need = max(need, 0.9)
            if sum(1 for t in self._chat_proactive if now - t < 600) >= 2:
                need = 2.0
        if verdict.intervene and verdict.confidence < need:
            log.info("не встреваю: люди болтают между собой (уверенность %.2f < %.2f)", verdict.confidence, need)
        if verdict.intervene and verdict.confidence >= need and can_pro and self.proactive_enabled:
            if chatting and not mention:
                self._chat_proactive.append(now)
            # пока думали — кто-то заговорил? Не перебиваем.
            if self.segmenter.seconds_since_last_voice() < self.s.quiet_gap_s:
                log.info("dropped proactive reply: room is not quiet")
                return
            await self._speak_stream(
                self.responder.stream_proactive(self.buffer.render(self.buffer.recent(20)), verdict.topic, verdict.angle),
                proactive=True, expect=_expect_if_on_topic(verdict.expect, verdict.angle))

    # ------------------------------------------------------------------ output
    async def _first_chunk_logged(self, chunks, utt_end: float | None):
        first = True
        async for chunk in chunks:
            if first and utt_end:
                log.info("⏱ голос пошёл через %.2f с после конца фразы", time.time() - utt_end)
            first = False
            yield chunk

    async def _sentences_to_audio(self, sentences, said: list[str]):
        """Текст по предложениям → звук. Следующие предложения озвучиваются заранее, пока играет текущее."""
        order: asyncio.Queue[asyncio.Queue | None] = asyncio.Queue()
        workers: list[asyncio.Task] = []

        async def tts_into(text: str, q: asyncio.Queue) -> None:
            try:
                async for chunk in self.tts.stream(text):
                    await q.put(chunk)
            except Exception:
                log.exception("TTS failed")
            finally:
                await q.put(None)

        async def producer() -> None:
            try:
                async for s in sentences:
                    said.append(s)
                    q: asyncio.Queue = asyncio.Queue()
                    workers.append(asyncio.create_task(tts_into(s, q)))
                    await order.put(q)
            except Exception:
                log.exception("LLM failed")
            finally:
                await order.put(None)

        prod = asyncio.create_task(producer())
        try:
            while (q := await order.get()) is not None:
                while (chunk := await q.get()) is not None:
                    yield chunk
        finally:
            prod.cancel()
            for w in workers:
                w.cancel()
            if hasattr(sentences, "aclose"):
                try:
                    await sentences.aclose()  # перебили — прекращаем генерацию LLM
                except Exception:
                    pass

    async def _play_clip(self, clip: bytes) -> None:
        async def chunks():
            for i in range(0, len(clip), 4800):
                yield clip[i : i + 4800]
        try:
            await self.player.play_stream(chunks())
        except Exception:
            log.exception("clip playback failed")

    async def _with_filler(self, chunks, kind: str | None = None):
        """Рефлекс, чтобы не висела тишина. На прямой вопрос (kind задан) — сразу, как начал думать: «м?» на
        оклик, «хм…» на вопрос, «угу» на рассказ. Иначе — «хм…», только если голос не готов за ~0.9 с."""
        it = chunks.__aiter__()
        first = asyncio.ensure_future(it.__anext__())
        try:
            now = time.monotonic()
            instant = bool(kind) and self.s.provider == "local" and now - self._last_filler >= FILLER_GAP_S
            done, _ = await asyncio.wait({first}, timeout=0.05 if instant else 0.9)
            if not done and self.sfx and (clip := self.sfx.random(kind if instant else "filler")):
                self._last_filler = now
                for i in range(0, len(clip), 4800):
                    yield clip[i : i + 4800]
            # пока ждём первые слова: если он бросил начатое (человек договорил) — один раз «ага, так…»
            again_said = False
            while not first.done():
                wake = asyncio.ensure_future(self._rethink.wait())
                await asyncio.wait({first, wake}, return_when=asyncio.FIRST_COMPLETED)
                wake.cancel()
                if self._rethink.is_set():
                    self._rethink.clear()
                    if not again_said and not first.done() and self.sfx and (clip := self.sfx.random("again")):
                        again_said = True
                        self._last_filler = time.monotonic()
                        for i in range(0, len(clip), 4800):
                            yield clip[i : i + 4800]
            try:
                chunk = await first
            except StopAsyncIteration:
                return
            yield chunk
            async for chunk in it:
                yield chunk
        finally:
            if not first.done():
                first.cancel()
                await asyncio.wait({first})
            if hasattr(it, "aclose"):
                try:
                    await it.aclose()
                except Exception:
                    pass

    async def _speak_stream(self, sentences, *, proactive: bool, utt_end: float | None = None,
                            expect: str = "", filler: str | None = None) -> None:
        if self.muted_since and not self._unmuting:
            return  # попросили помолчать — ни слова (кроме реакции на «можешь говорить»)
        # Свою же реплику не обрываем новым ответом: он подождёт очереди (_speak_lock).
        # Перебить бота можно голосом (barge-in) или стоп-словом.
        pf = _Prefetch(sentences)  # модель начинает думать сразу, даже если ещё звучит прошлая реплика
        self._gen_owner, self._gen_done = asyncio.current_task(), pf.done
        try:
            with foreground():
                await self._speak_stream_inner(pf, proactive=proactive, utt_end=utt_end, expect=expect,
                                               filler=filler)
        finally:
            await pf.aclose()

    async def _speak_stream_inner(self, sentences, *, proactive: bool, utt_end: float | None = None,
                                  expect: str = "", filler: str | None = None) -> None:
        async with self._speak_lock:
            line = self.buffer.add(persona.BOT_NAME, "…", is_bot=True)
            if isinstance(sentences, _Prefetch):
                sentences.attach(line)  # как только текст додуман — он виден следующему ответу
            self._last_bot_seq = line.seq
            self._speaking_proactive = proactive
            said: list[str] = []
            try:
                await self.player.play_stream(
                    self._first_chunk_logged(self._with_filler(self._sentences_to_audio(sentences, said), filler),
                                             utt_end))
            except Exception:
                log.exception("TTS/playback failed")
            finally:
                text = " ".join(said)
                cut = self.player.interrupted and text
                if cut:
                    # в памяти оставляем только то, что успело прозвучать
                    keep = text[: int(len(text) * self.player.played_ratio)]
                    keep = keep.rsplit(" ", 1)[0] if " " in keep else ""
                    text = f"{keep}… [меня перебили]" if keep else "[меня перебили сразу, ничего не сказал]"
                line.text = text or "…"
                if text and not cut:
                    self._my_lines.append(text)
                    book.note_questions(text)  # о чём спросил — прочитает в книжке, если не ответят
                    get_habits().note_said(text)  # использованные словечки крепнут
                    if expect and expect.startswith(_ANGLE_TAG):
                        angle, expect = expect[len(_ANGLE_TAG):].split(_ANGLE_TAG, 1)
                        if not _same_topic(angle, text):
                            log.info("ожидание «%s» было про другой вопрос — не жду", expect)
                            expect = ""
                    if expect:
                        get_predictor().add(text, expect, line.seq)  # потом сверит, как отреагировали
                if not text:
                    self.buffer.remove(line)  # ничего не сказала — пустую реплику в память не пишем
                log.info("BOT (%s%s): %s", "proactive" if proactive else "direct",
                         ", перебили" if cut else "", line.text)
                self._speaking_proactive = False
                self.policy.note_spoke(proactive)
                if proactive and text:
                    self.drive.on_spoke()
                self._spawn(self.judge.maybe_update(self.buffer))  # пересмотреть отношения (не чаще раза в 90 с)
                self._spawn(self.self_model.maybe_reflect(self.buffer))  # поразмышлять о себе (не чаще раза в 5 мин)

    async def _speak(self, text: str, *, proactive: bool, utt_end: float | None = None, expect: str = "") -> None:
        if text:
            async def one():
                yield text
            await self._speak_stream(one(), proactive=proactive, utt_end=utt_end, expect=expect)
