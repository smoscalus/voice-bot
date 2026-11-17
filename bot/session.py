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
from bot.brain.llm import LLMRouter
from bot.brain.relations import RelationsJudge, get_store
from bot.brain.self_model import get_self_model
from bot.brain.policy import SpeakPolicy
from bot.brain.responder import Responder
from bot.brain.transcript import TranscriptBuffer
from bot.brain.triggers import TriggerDetector
from bot.config import Settings
from bot.stt.base import STTEngine, clean_transcript

log = logging.getLogger(__name__)

UNCLEAR_CONFIDENCE = -0.8   # avg_logprob Whisper ниже этого — фраза распознана как каша
NOISE_MIN_LINES = 5         # сколько фраз за минуту смотреть, чтобы решить «вокруг шум»
NOISE_UNCLEAR_SHARE = 0.6   # доля неразборчивых среди них
NOISE_COMPLAINT_EVERY_S = 300


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
        self._reply_task: asyncio.Task | None = None
        self._last_bot_seq = 0
        self.sfx = sfx  # «угу» и «хм…» голосом бота
        self._last_backchannel = 0.0
        self._seen_users: set[int] = set()   # кто уже говорил в этой сессии
        self._returning: set[str] = set()    # вернулись, и на них есть материал в памяти
        self.drive = get_drive()              # голод на провокации и «сытость» каждым человеком
        self._my_lines: deque[str] = deque(maxlen=12)  # свои последние реплики — чтобы не повторяться
        self._clarity: deque[tuple[float, bool]] = deque(maxlen=12)  # (когда, неразборчиво ли) — детектор шума
        self._last_noise_complaint = 0.0
        self._tasks: set[asyncio.Task] = set()

    def set_brain(self, settings: Settings, client: AsyncOpenAI, stt: STTEngine) -> None:
        """Подключить «мозги» (local/groq). Вызывается при старте и при переключении командой /brain —
        прямо на ходу, без выхода из канала. Память и досье при этом сохраняются."""
        self.s = settings
        self.stt = stt
        fallbacks = settings.llm_fallback_models
        t = settings.llm_timeout_s
        light_llm = LLMRouter(client, [settings.classifier_model, *fallbacks], settings.reasoning_effort, t)
        memory_llm = LLMRouter(client, [settings.memory_model, settings.classifier_model, *fallbacks],
                               settings.reasoning_effort, t)
        self.classifier = InterestClassifier(light_llm, persona.BOT_NAME, persona.INTERVENE_RULES)
        if hasattr(self, "judge"):
            self.judge.llm = memory_llm
        else:
            self.judge = RelationsJudge(self.relations, memory_llm, persona.BOT_NAME)
        self.self_model = get_self_model(memory_llm)  # досье на самого себя: кто я, увлечения, задания, страхи
        self.self_model.llm = memory_llm
        self.responder = Responder(
            LLMRouter(client, [settings.llm_model, *fallbacks], settings.reasoning_effort, t),
            persona.SYSTEM_PROMPT, extra_context=self._relations_context,
            recent_lines=lambda: list(self._my_lines))

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.vc.listen(self.sink)
        self._spawn(self._consumer())
        self._spawn(self._ticker())
        log.info("session started in #%s", self.vc.channel)

    async def stop(self, farewell: bool = False) -> None:
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
        """Память в промпт: кто я сам + досье тех, кто сейчас в разговоре."""
        participants = {l.speaker: l.speaker_id for l in self.buffer.recent(30) if not l.is_bot}
        people = {}
        for name, uid in participants.items():
            r = self.relations.get(uid, name)
            people[name] = (len(r.facts) + len(r.plans) + len(r.hooks), r.score)
        blocks = [self.self_model.prompt_block(), self.relations.prompt_block(participants),
                  self.drive.prompt_block(people) if people else "", self._my_lines_block()]
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
            # страховка: приём звука выключился (кто-то вызвал vc.stop() и т.п.) — включаем обратно
            if not self._reconnecting and self.vc.is_connected() and not self.vc.is_listening():
                log.warning("приём звука был выключен — включаю обратно")
                self.sink = SegmentingSink(self.segmenter)
                self.vc.listen(self.sink)
            # самовосстановление: после смены ключей шифрования бот может «оглохнуть»
            if (not self._reconnecting and time.monotonic() - self._last_reconnect > 30
                    and (get_health(self.vc).is_deaf() or self._receiver_dead())):
                self._spawn(self._reconnect())
            # «угу»: человек долго рассказывает боту и сделал паузу — поддакнуть, как живой слушатель
            if (self.sfx and not self.player.is_playing and self.policy.in_follow_up()
                    and time.monotonic() - self._last_backchannel > 15
                    and self.segmenter.long_speech_pause() and random.random() < 0.5):
                self._last_backchannel = time.monotonic()
                if clip := self.sfx.random("back"):
                    self._spawn(self._play_clip(clip))
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

        # человек впервые за сессию заговорил, а в памяти на него есть планы/зацепки — повод для провокации
        if utt.user_id not in self._seen_users:
            self._seen_users.add(utt.user_id)
            rel = self.relations.get(utt.user_id, utt.user_name)
            if rel.plans or rel.hooks:
                self._returning.add(utt.user_name)
                self._schedule_evaluation()

        trig = self.triggers.analyze(text)

        if trig.stop and self.player.is_playing:
            self.player.interrupt()
            return

        # 1) прямое обращение
        if trig.wake:
            if not trig.remainder.strip():
                # голое «Вася!» — ждём продолжения; не дождались — отвечаем на само обращение.
                self.policy.open_attention(utt.user_id)
                self._spawn(self._answer_if_no_followup(utt.user_id, utt_end))
                return
            self._request_reply(utt_end)
            return
        if self.policy.take_attention(utt.user_id):
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
        if self._reply_task and not self._reply_task.done() and self._collecting:
            return
        # если бот сейчас отвечает — следующий ответ встанет в очередь за ним
        prev = self._reply_task if self._reply_task and not self._reply_task.done() else None
        self._reply_task = self._spawn(self._reply_batch(utt_end, prev))

    async def _reply_batch(self, utt_end: float, prev: asyncio.Task | None = None) -> None:
        self._collecting = True  # пока ждём — новые обращения вливаются в эту же пачку
        if prev is not None:
            await asyncio.wait({prev})  # ждём, пока договорит прошлый ответ; потом ответим только на новое
        # Собираем пачку: ждём, пока никто не говорит и все начатые фразы распознаны.
        # Если в комнате тихо — проходит мгновенно, задержки нет.
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            if (self._stt_inflight == 0 and not self.segmenter.someone_speaking(min_voiced_ms=200)
                    and self.segmenter.seconds_since_last_voice() >= 0.3):
                break
            await asyncio.sleep(0.05)
        self._collecting = False

        cutoff = time.time() - 30
        lines = [l for l in self.buffer.since(self._last_bot_seq) if not l.is_bot and l.ts >= cutoff][-8:]
        if not lines:
            return
        transcript = self.buffer.render(self.buffer.recent(12))  # на обращения — небольшой контекст, экономим лимит
        # ответ стримится по предложениям: озвучка стартует, пока модель дописывает остальное
        await self._speak_stream(self.responder.stream_batch(transcript, self.buffer.render(lines)),
                                 proactive=False, utt_end=utt_end)

    # ------------------------------------------------------------------ decision
    def _schedule_evaluation(self) -> None:
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
        if not follow_up and not self._returning and not self.drive.wants_to_provoke():
            # сыт провокациями — сам не лезет и не тратит лимит на «а не влезть ли»
            self.policy.reset_interest()
            return
        if not self.policy.can_classify():
            log.warning("classifier hourly budget exhausted")
            return

        self.policy.note_classify()
        self.policy.reset_interest()
        hint = f"Бот последний раз говорил {int(time.monotonic() - self.policy.last_spoke)} с назад." \
            if self.policy.last_spoke else "Бот ещё ничего не говорил."
        if self._returning:
            hint += (f"\nТолько что впервые за сегодня появились в разговоре: {', '.join(sorted(self._returning))}. "
                     "У них в памяти есть планы/зацепки — отличный повод с издёвкой спросить, как успехи.")
            self._returning.clear()
        transcript = self.buffer.render(self.buffer.recent(12))
        memory = self._relations_context()  # зацепки и планы из памяти — лучший повод для провокации
        verdict = await self.classifier.evaluate(transcript, hint + ("\n\n" + memory if memory else ""))
        log.info("verdict: %s", verdict)
        if self._reply_busy() or self._last_bot_seq != seq0:
            return  # пока думали, бот уже ответил — второй раз не лезем

        last = self.buffer.recent(1)
        if verdict.addressed_to_bot and last and not last[0].is_bot:
            self._request_reply(last[0].end_ts or last[0].ts)
            return
        if verdict.intervene and verdict.confidence >= self.s.interest_confidence and can_pro and self.proactive_enabled:
            # пока думали — кто-то заговорил? Не перебиваем.
            if self.segmenter.seconds_since_last_voice() < self.s.quiet_gap_s:
                log.info("dropped proactive reply: room is not quiet")
                return
            await self._speak_stream(
                self.responder.stream_proactive(self.buffer.render(self.buffer.recent(20)), verdict.topic, verdict.angle),
                proactive=True)

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

    async def _with_filler(self, chunks):
        """Если голос не готов за ~0.9 с — сначала «Хм…»/«Так…», чтобы не висела тишина."""
        it = chunks.__aiter__()
        first = asyncio.ensure_future(it.__anext__())
        try:
            done, _ = await asyncio.wait({first}, timeout=0.9)
            if not done and self.sfx and (clip := self.sfx.random("filler")):
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

    async def _speak_stream(self, sentences, *, proactive: bool, utt_end: float | None = None) -> None:
        # Свою же реплику не обрываем новым ответом: он подождёт очереди (_speak_lock).
        # Перебить бота можно голосом (barge-in) или стоп-словом.
        async with self._speak_lock:
            line = self.buffer.add(persona.BOT_NAME, "…", is_bot=True)
            self._last_bot_seq = line.seq
            self._speaking_proactive = proactive
            said: list[str] = []
            try:
                await self.player.play_stream(
                    self._first_chunk_logged(self._with_filler(self._sentences_to_audio(sentences, said)), utt_end))
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
                if not text:
                    self.buffer.remove(line)  # ничего не сказала — пустую реплику в память не пишем
                log.info("BOT (%s%s): %s", "proactive" if proactive else "direct",
                         ", перебили" if cut else "", line.text)
                self._speaking_proactive = False
                self.policy.note_spoke(proactive)
                self._spawn(self.judge.maybe_update(self.buffer))  # пересмотреть отношения (не чаще раза в 90 с)
                self._spawn(self.self_model.maybe_reflect(self.buffer))  # поразмышлять о себе (не чаще раза в 5 мин)

    async def _speak(self, text: str, *, proactive: bool, utt_end: float | None = None) -> None:
        if text:
            async def one():
                yield text
            await self._speak_stream(one(), proactive=proactive, utt_end=utt_end)
