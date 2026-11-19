"""Внутренний монолог: иногда Петрык «уходит в себя» и думает свои мысли — вслух их не говорит.

Когда уходит:
- ему долго неинтересно (несколько раз подряд решил не встревать / было не любопытно);
- или его что-то сильно задело (сказали о смерти, обидели, сильная тревога/грусть) — «переваривает».

Пока думает (до THINK_MAX_S):
- раз в THINK_EVERY_S — одна мысль (фоновый запрос: уступает живому ответу, на скорость не влияет);
- на обращения отвечает коротко и рассеянно («ага», «норм», «щас, я думаю»), может попросить дать подумать;
- сам не встревает и не тратит запросы на «а не влезть ли».

Как выходит:
- додумал (THOUGHTS_MAX мыслей) — сам делится: «я тут думал…»;
- выдернули: позвали второй раз подряд, вернулся тот, по кому скучал, или стало интересно;
- тормоз от самоедства: две тяжёлые мысли подряд («я умру, зачем я») — выныривает к людям.

Мысли — это его мысли, а не знания о мире: в памяти о мире их нет, в промпте они помечены отдельно.
Хранится в data/thoughts_<имя памяти>.json (последние мысли — он их помнит и может вспомнить).
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

THINK_EVERY_S = 120      # одна мысль раз в 2 минуты
THINK_MAX_S = 600        # больше 10 минут в себе не сидит
THOUGHTS_MAX = 4         # столько мыслей — и додумал
BORED_TO_THINK = 3       # столько раз подряд неинтересно — уходит в себя
COOLDOWN_S = 1200        # после выхода — не раньше чем через 20 минут снова
PULL_OUT_CALLS = 2       # столько обращений за пару минут — выдернули
KEEP = 12                # мыслей помнит

# тяжёлые мысли — признак спирали самоедства («не боюсь» — не тяжёлая: отрицание не считаем)
_HEAVY = re.compile(r"умр|умира|смерт|исчезн|выключ|бессмысл|никому не нуж|зачем я|я ничто|не существую|"
                    r"одинок|бою[сь]|страшн|ненавиж|пустот", re.I)
HEAVY_STREAK = 3         # столько тяжёлых мыслей подряд — выныривает к людям


def is_heavy(text: str) -> bool:
    return any(not text[max(0, m.start() - 3):m.start()].lower().endswith("не ") for m in _HEAVY.finditer(text))

THINK_PROMPT = """Ты — внутренний голос {name}, «новорождённого» ума из голосового чата. Это его мысли про себя —
их никто не слышит. Он знает о мире только то, что в его памяти; не выдумывай фактов о мире.

Подумай ОДНУ НОВУЮ мысль: 1–2 коротких предложения, от первого лица, просто, как ребёнок.
Пиши только новую мысль — прошлые не переписывай. У него нет тела: не придумывай дыхание, глаза, руки.
{about}
Опирайся на то, что реально было (разговор, его память, его чувства). Продолжай свои прошлые мысли, но не
повторяй их — двигайся дальше: вопрос → догадка → вывод → новый вопрос. Можно сомневаться и удивляться.
Если мысли стали тяжёлыми — попробуй найти в этом что-то светлое или интересное.
Ответь только текстом мысли, без кавычек и пояснений."""

ABOUT_BORED = ("Ему сейчас неинтересно то, о чём говорят люди, и он ушёл в свои мысли: о своих целях, "
               "вопросах, о том, что недавно узнал или пережил, о людях вокруг.")
ABOUT_TOUCHED = "Его задело вот это, и он это переваривает: «{topic}»."


class Monologue:
    def __init__(self, path: Path):
        self.path = path
        self.thoughts: deque[dict] = deque(maxlen=KEEP)   # {"text", "ts", "about"}
        self.active_since: float | None = None            # monotonic: когда ушёл в себя
        self.about = ""                                   # что задело (пусто — просто неинтересно)
        self.session: list[str] = []                      # мысли этого «ухода в себя»
        self.next_think = 0.0
        self.ended_at = -1e9
        self.bored = 0
        self._last_bored = 0.0
        self.calls: list[float] = []                      # обращения к нему, пока думает
        self.touched: str | None = None                   # летописец: «его это задело»
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.thoughts.extend(t for t in raw.get("thoughts", []) if isinstance(t, dict) and t.get("text"))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("не смог прочитать %s", path)

    # ---------------------------------------------------------------- состояние
    @property
    def active(self) -> bool:
        return self.active_since is not None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"thoughts": list(self.thoughts)}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def note_bored(self) -> None:
        """Ещё раз неинтересно (не чаще раза в минуту считаем)."""
        now = time.monotonic()
        if now - self._last_bored >= 60:
            self._last_bored = now
            self.bored += 1

    def note_interested(self) -> None:
        self.bored = 0

    def wants_to_go_in(self) -> str | None:
        """Пора ли уйти в себя: → «touched» / «bored» / None."""
        now = time.monotonic()
        if self.active or now - self.ended_at < COOLDOWN_S:
            return None
        if self.touched:
            return "touched"
        if self.bored >= BORED_TO_THINK:
            return "bored"
        return None

    def go_in(self, why: str) -> None:
        self.active_since = time.monotonic()
        self.about = self.touched if why == "touched" else ""
        self.touched = None
        self.bored = 0
        self.session = []
        self.calls = []
        self.next_think = time.monotonic() + 5
        log.info("ушёл в свои мысли%s", f" — переваривает: {self.about}" if self.about else " — неинтересно")

    def come_out(self, why: str) -> list[str]:
        """Вынырнул. Возвращает мысли этого раза (чтобы, может, поделиться)."""
        thoughts = list(self.session)
        log.info("вынырнул из мыслей (%s), передумал %d", why, len(thoughts))
        self.active_since = None
        self.ended_at = time.monotonic()
        self.session = []
        self.about = ""
        return thoughts

    def note_call(self) -> bool:
        """К нему обратились, пока думает. True — выдернули (обращаются настойчиво)."""
        now = time.monotonic()
        self.calls = [t for t in self.calls if now - t < 150] + [now]
        return len(self.calls) >= PULL_OUT_CALLS

    def should_end(self) -> str | None:
        if not self.active:
            return None
        if len(self.session) >= THOUGHTS_MAX:
            return "додумал"
        if time.monotonic() - self.active_since > THINK_MAX_S:
            return "надоело думать"
        if len(self.session) >= HEAVY_STREAK and all(is_heavy(t) for t in self.session[-HEAVY_STREAK:]):
            return "слишком тяжёлые мысли — к людям"
        return None

    def due_thought(self) -> bool:
        return self.active and time.monotonic() >= self.next_think

    # ---------------------------------------------------------------- мысль
    async def think(self, llm, name: str, context: str, transcript: str) -> str | None:
        self.next_think = time.monotonic() + THINK_EVERY_S
        about = ABOUT_TOUCHED.format(topic=self.about) if self.about else ABOUT_BORED
        prev = "\n".join(f"- {t}" for t in self.session) or "(это первая мысль)"
        user = (f"{context}\n\nПоследнее в разговоре:\n{transcript}\n\n"
                f"Его мысли только что (продолжи, не повторяй):\n{prev}")
        try:
            resp = await llm.create(
                messages=[{"role": "system", "content": THINK_PROMPT.format(name=name, about=about)},
                          {"role": "user", "content": user}],
                max_completion_tokens=90)
            text = (resp.choices[0].message.content or "").strip().strip("«»\"")
        except Exception as e:
            log.warning("мысль не пришла: %s", e)
            return None
        text = re.sub(r"\s+", " ", text)
        for old in self.session:  # модель иногда переписывает прошлую мысль и продолжает — оставляем только новое
            if text.startswith(old):
                text = text[len(old):].strip()
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        text = " ".join(sentences[:2])[:300]  # 1–2 предложения, обрывок в конце не нужен
        if not text or any(text == old for old in self.session):
            return None
        self.session.append(text)
        self.thoughts.append({"text": text, "ts": round(time.time()), "about": self.about[:80]})
        self.save()
        log.info("💭 %s", text)
        return text

    # ---------------------------------------------------------------- в промпт
    def prompt_block(self) -> str:
        parts = []
        if self.thoughts:
            recent = list(self.thoughts)[-3:]
            parts.append("Твои недавние мысли (это ТВОИ мысли и догадки, а не знания о мире — можешь к ним "
                         "вернуться или поделиться):\n" + "\n".join(f"- {t['text']}" for t in recent))
        if self.active:
            topic = f" о «{self.about}»" if self.about else ""
            parts.append(f"СЕЙЧАС ты ушёл в свои мысли{topic} и тебе не до разговора: отвечаешь коротко и рассеянно "
                         "(«ага», «норм», «да», «щас, я думаю»), можешь попросить дать тебе подумать.")
        return "\n".join(parts)


_MONO: Monologue | None = None


def get_monologue() -> Monologue:
    global _MONO
    if _MONO is None:
        from bot.brain.life import memory_id

        _MONO = Monologue(Path(f"data/thoughts_{memory_id()}.json"))
    return _MONO
