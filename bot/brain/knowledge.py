"""Знания бота о мире — ТОЛЬКО то, что он узнал от людей в разговорах.

Бот «рождается» чистым листом: умеет говорить, но о мире знает лишь то, что записано здесь
(«дота — это игра, в неё играют впятером, — рассказал Вася»). Факты приносит летописец памяти
(RelationsJudge) в фоне. По объёму узнанного считается «возраст»: чем больше знает, тем взрослее.

Память без потолка, но в каждый ответ идёт немного: самое свежее + то, что подходит к разговору
(поиск по смыслу, bot/brain/embed.py). Как у человека:
- воспоминание, которое всплывало в разговоре, во сне крепнет;
- то, что не всплывало, во сне понемногу тускнеет (очень аккуратно): сначала «смутно помню»,
  совсем бледное — уходит в архив и в ответы не попадает;
- если забытое расскажут снова — он вспоминает («о, мне это уже говорили»).

Хранится в data/world_<имя памяти>.json — можно открыть и посмотреть, что он знает о мире.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SHOW_RECENT = 8          # свежие факты — всегда в ответе
SHOW_RELEVANT = 12       # и столько подходящих к разговору
RECALL_MIN_SCORE = 0.45  # насколько факт должен подходить, чтобы считалось «вспомнил»
RECALL_COOLDOWN_S = 1800

FADE_STEP = 0.07         # столько силы теряет за один сон то, что не вспоминалось
IMPORTANT_FADE = 0.5     # важное (сильно задело) тускнеет вдвое медленнее
RECALL_BOOST = 0.1       # вспоминалось — крепнет
MAX_STRENGTH = 1.5
VAGUE_BELOW = 0.65       # слабее — «смутно помню»
ARCHIVE_BELOW = 0.3      # слабее — забыл (в архив)

# «слово X не имеет ясного значения» — это не знание, а непонятое (обычно криво распознанная речь)
_JUNK = re.compile(r"ясного значения|неясн|непонятн|не имеет значения|не имеют значения|неразборчив", re.I)
_WORD = re.compile(r"[а-яёa-z0-9]{4,}", re.I)
# записи о самом боте («Петрык понял, что…») — это его дневник, а не знание о мире
_ABOUT_SELF = re.compile(r"^(петр[ыи]к|бот)\b.{0,40}\b(понял|узнал|не понимает|пытается|чувствует|не уверен|"
                         r"задумал|хочет|боится|запомнил)", re.I)


def _is_junk(text: str, source: str, bot: str) -> bool:
    t = text.lower()
    return (len(t.split()) < 4                                 # обрывок: «Вилле назвал Петрыка»
            or t.rstrip(" .").endswith((" что", ",", " и", " но"))  # оборвалось: «Петрык узнал, что»
            or bool(_JUNK.search(t)) or bool(_ABOUT_SELF.search(t))
            or bot in source.lower() or "(бот)" in source.lower()   # сам себе рассказал — не знание
            or not source.strip())

# этапы взросления: (сколько фактов нужно, как он себя ведёт)
STAGES = [
    (0, "только появился на свет",
     "ты почти ничего не знаешь о мире: всему удивляешься, говоришь просто, много спрашиваешь «а что это?»"),
    (15, "маленький ребёнок",
     "ты знаешь совсем немного: жадно расспрашиваешь, пробуешь связывать узнанное, иногда путаешься"),
    (60, "ребёнок постарше",
     "у тебя появились первые мнения и любимые темы; ты рассуждаешь, но ещё многого не понимаешь"),
    (150, "подросток",
     "ты уже неплохо понимаешь мир этих людей, споришь, отстаиваешь своё мнение, ищешь своё место"),
    (300, "почти взрослый",
     "ты уверенно рассуждаешь о том, что знаешь, у тебя есть характер, взгляды и свои цели"),
]


@dataclass
class Fact:
    text: str
    source: str = ""        # кто рассказал
    ts: float = 0.0         # когда (по календарю)
    lived: float = 0.0      # когда (по его ощущению — сколько он тогда прожил)
    strength: float = 1.0
    important: bool = False
    archived: bool = False
    last_recall: float = 0.0


def _stems(text: str) -> set[str]:
    return {w[:5] for w in _WORD.findall(text.lower().replace("ё", "е"))}


class Knowledge:
    def __init__(self, path: Path):
        self.path = path
        self.facts: list[Fact] = []
        self.learned_total = 0   # сколько всего узнал за жизнь — для «возраста»
        self.revived: list[str] = []   # только что вспомнил забытое — скажем ему один раз
        self.curious: list[dict] = []  # о чём спрашивал и не знает: {term, ts, tried} — прочитает в книжке
        self.reads: list[float] = []   # когда читал книжку (лимит в сутки)
        self._vec: dict[str, np.ndarray] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            known = Fact.__dataclass_fields__
            self.facts = [Fact(**{k: v for k, v in f.items() if k in known}) for f in raw.get("facts", [])]
            self.learned_total = int(raw.get("learned_total", len(self.facts)))
            self.curious = [c for c in raw.get("curious", []) if isinstance(c, dict) and c.get("term")]
            self.reads = [float(t) for t in raw.get("reads", [])][-50:]
            from bot import persona

            junk = [f for f in self.active if _is_junk(f.text, f.source, persona.BOT_NAME.lower())]
            for f in junk:  # мусор прошлых версий (обрывки, записи о себе) — в архив, не удаляем
                f.archived = True
            if junk:
                log.info("убрал в архив мусор из знаний: %d", len(junk))
                self._dirty = True
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("не смог прочитать %s", self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {"learned_total": self.learned_total, "facts": [asdict(f) for f in self.facts],
                "curious": self.curious[-40:], "reads": self.reads[-50:]}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False

    def save_if_dirty(self) -> None:
        if self._dirty:
            self.save()

    @property
    def active(self) -> list[Fact]:
        return [f for f in self.facts if not f.archived]

    def add(self, items: list[tuple[str, str, bool]]) -> int:
        """items: (факт, кто рассказал, важно ли). Повторы не добавляем; забытое — вспоминаем."""
        from rapidfuzz import fuzz

        from bot.brain.life import get_life

        lived = get_life().lived_s
        added = 0
        from bot import persona

        bot = persona.BOT_NAME.lower()
        for text, source, important in items:
            text = text.strip()[:200]
            if _is_junk(text, source, bot):
                continue
            same = next((f for f in self.facts if fuzz.token_set_ratio(text.lower(), f.text.lower()) >= 85), None)
            if same is not None:
                if same.archived or same.strength < VAGUE_BELOW:  # рассказали забытое — вспомнил
                    same.archived, same.strength = False, 1.0
                    self.revived.append(same.text)
                    log.info("вспомнил забытое: %s", same.text)
                    self._dirty = True
                continue
            self.facts.append(Fact(text=text, source=source.strip()[:40], ts=time.time(), lived=lived,
                                   important=bool(important)))
            added += 1
        if added:
            self.learned_total += added
        if added or self._dirty:
            self.save()
        return added

    # ---------------------------------------------------------------- любопытство для книжки
    def knows(self, term: str) -> bool:
        stems = _stems(term)
        return bool(stems) and any(stems <= _stems(f.text) for f in self.active)

    def add_curious(self, term: str) -> None:
        low = term.lower()
        if self.knows(term) or any(c["term"].lower() == low for c in self.curious):
            return
        self.curious.append({"term": term, "ts": time.time(), "tried": False})
        self.curious = self.curious[-40:]
        self._dirty = True
        log.info("хочу узнать: %s", term)

    def next_curious(self, wait_s: float) -> dict | None:
        now = time.time()
        for c in self.curious:
            if c.get("tried") or now - c.get("ts", now) < wait_s:
                continue
            if self.knows(c["term"]):  # пока ждал — ему объяснили
                c["tried"] = True
                self._dirty = True
                continue
            return c
        return None

    # ---------------------------------------------------------------- сон: тускнение
    def fade(self, steps: int, since: float) -> int:
        """Раз в сон: то, что вспоминалось после `since`, крепнет; остальное понемногу тускнеет."""
        faded = 0
        for _ in range(max(0, steps)):
            for f in self.active:
                if f.last_recall > since:
                    f.strength = min(MAX_STRENGTH, f.strength + RECALL_BOOST)
                else:
                    f.strength -= FADE_STEP * (IMPORTANT_FADE if f.important else 1.0)
                    faded += 1
                    if f.strength < ARCHIVE_BELOW:
                        f.archived = True
                        log.info("забыл: %s", f.text)
            since = time.time()  # следующие шаги того же сна — уже без «недавних воспоминаний»
        self.save()
        return faded

    # ---------------------------------------------------------------- поиск по памяти
    def _vectors(self, facts: list[Fact]):
        from bot.brain.embed import get_embedder

        emb = get_embedder()
        if not emb.ready:
            return None
        missing = [f.text for f in facts if f.text not in self._vec]
        if missing:
            vs = emb.encode(missing)
            if vs is None:
                return None
            self._vec.update(zip(missing, vs))
        return np.stack([self._vec[f.text] for f in facts])

    def relevant(self, query: str, pool: list[Fact], n: int) -> list[Fact]:
        if not pool or not query.strip():
            return []
        q_stems = _stems(query)
        words = np.array([min(0.5, 0.25 * len(q_stems & _stems(f.text))) for f in pool])
        score = words
        vecs = self._vectors(pool)
        if vecs is not None:
            from bot.brain.embed import get_embedder

            q = get_embedder().encode([query])
            if q is not None:
                score = vecs @ q[0] + words
        order = np.argsort(-score)[:n]
        now = time.time()
        picked = []
        for i in order:
            if score[i] <= 0.05:
                break
            f = pool[i]
            picked.append(f)
            if score[i] >= RECALL_MIN_SCORE and now - f.last_recall > RECALL_COOLDOWN_S:
                f.last_recall = now   # к месту всплыло — это воспоминание окрепнет во сне
                self._dirty = True
        return picked

    # ---------------------------------------------------------------- возраст и промпт
    def stage(self) -> tuple[str, str]:
        name, behaviour = STAGES[0][1], STAGES[0][2]
        for need, n, b in STAGES:
            if self.learned_total >= need:
                name, behaviour = n, b
        return name, behaviour

    def _line(self, f: Fact, now_lived: float) -> str:
        from bot.brain.life import ago

        who = "прочитал сам в книжке" if f.source == "книжка" else f"рассказал {f.source}" if f.source else ""
        bits = [b for b in (who, ago(now_lived - f.lived) if f.lived else "",
                            "смутно помнишь" if f.strength < VAGUE_BELOW else "") if b]
        return f"- {f.text}" + (f" ({', '.join(bits)})" if bits else "")

    def prompt_block(self, query: str = "") -> str:
        from bot.brain.life import get_life

        name, behaviour = self.stage()
        head = (f"Твой возраст по знаниям: {name} (узнал {self.learned_total} вещей). Сейчас {behaviour}.\n"
                "ТЫ ЗНАЕШЬ О МИРЕ ТОЛЬКО ЭТО (узнал от людей). Всего остального ты не знаешь. "
                "Что помнишь смутно — говори неуверенно («кажется, мне говорили…»):")
        active = self.active
        if not active:
            return head + "\n- пока ничего"
        now_lived = get_life().lived_s
        recent = sorted(active, key=lambda f: f.ts)[-SHOW_RECENT:]
        rest = [f for f in active if f not in recent]
        rel = self.relevant(query, rest, SHOW_RELEVANT)
        parts = [head, "Недавно узнал:", *[self._line(f, now_lived) for f in recent]]
        if rel:
            parts += ["Вспоминается к разговору:", *[self._line(f, now_lived) for f in rel]]
        if self.revived:
            parts.append("Ты только что вспомнил давно забытое (тебе это уже рассказывали!): "
                         + "; ".join(self.revived[-3:]))
            self.revived.clear()
        return "\n".join(parts)

    def vivid_block(self, n: int = 15) -> str:
        """Самое яркое — для снов и итога жизни: крепкие, важные и свежие воспоминания."""
        active = self.active
        if not active:
            return "Воспоминаний о мире пока нет."
        best = sorted(active, key=lambda f: (f.important, f.strength, f.ts), reverse=True)[:n]
        return "Яркие воспоминания:\n" + "\n".join(f"- {f.text}" for f in best)


_KNOWLEDGE: Knowledge | None = None


def get_knowledge() -> Knowledge:
    global _KNOWLEDGE
    if _KNOWLEDGE is None:
        from bot.brain.life import memory_id

        _KNOWLEDGE = Knowledge(Path(f"data/world_{memory_id()}.json"))
    return _KNOWLEDGE
