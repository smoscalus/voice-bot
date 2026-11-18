"""Знания бота о мире — ТОЛЬКО то, что он узнал от людей в разговорах.

Бот «рождается» чистым листом: умеет говорить, но о мире знает лишь то, что записано здесь
(«дота — это игра, в неё играют впятером, — рассказал Вася»). Факты приносит летописец памяти
(RelationsJudge) в фоне. Когда фактов становится слишком много — модель сама сжимает их,
объединяя похожие и выкидывая мелочи. По объёму знаний считается «возраст» бота: чем больше
он узнал, тем взрослее рассуждает.

Хранится в data/world_<имя памяти>.json — можно открыть и посмотреть, что он знает о мире.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# «слово X не имеет ясного значения» — это не знание, а непонятое (обычно криво распознанная речь)
_JUNK = re.compile(r"ясного значения|неясн|непонятн|не имеет значения|не имеют значения|неразборчив", re.I)

MAX_FACTS = 150      # больше — сжимаем
COMPRESS_TO = 90
SHOW_IN_PROMPT = 45  # столько свежих/важных фактов видит модель в каждом ответе

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
    source: str = ""   # кто рассказал
    ts: float = 0.0


class Knowledge:
    def __init__(self, path: Path):
        self.path = path
        self.facts: list[Fact] = []
        self.learned_total = 0   # сколько всего узнал за жизнь (не уменьшается при сжатии) — для «возраста»
        self._compressing = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.facts = [Fact(**f) for f in raw.get("facts", [])]
            self.learned_total = int(raw.get("learned_total", len(self.facts)))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("не смог прочитать %s", self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {"learned_total": self.learned_total, "facts": [asdict(f) for f in self.facts]}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, items: list[tuple[str, str]]) -> int:
        """items: (факт, кто рассказал). Повторы и почти-повторы не добавляем. Возвращает, сколько нового."""
        from rapidfuzz import fuzz

        added = 0
        for text, source in items:
            text = text.strip()[:200]
            if len(text) < 4 or _JUNK.search(text):
                continue
            if any(fuzz.token_set_ratio(text.lower(), f.text.lower()) >= 85 for f in self.facts):
                continue
            self.facts.append(Fact(text=text, source=source.strip()[:40], ts=time.time()))
            added += 1
        if added:
            self.learned_total += added
            self.save()
        return added

    # ---------------------------------------------------------------- возраст
    def stage(self) -> tuple[str, str]:
        name, behaviour = STAGES[0][1], STAGES[0][2]
        for need, n, b in STAGES:
            if self.learned_total >= need:
                name, behaviour = n, b
        return name, behaviour

    def prompt_block(self) -> str:
        name, behaviour = self.stage()
        head = (f"Твой возраст по знаниям: {name} (узнал {self.learned_total} вещей). Сейчас {behaviour}.\n"
                "ТЫ ЗНАЕШЬ О МИРЕ ТОЛЬКО ЭТО (узнал от людей). Всего остального ты не знаешь:")
        if not self.facts:
            return head + "\n- пока ничего"
        shown = self.facts[-SHOW_IN_PROMPT:]
        lines = [f"- {f.text}" + (f" (рассказал {f.source})" if f.source else "") for f in shown]
        return head + "\n" + "\n".join(lines)

    # ---------------------------------------------------------------- сжатие
    async def maybe_compress(self, llm) -> None:
        """Когда фактов много — модель объединяет похожие и выкидывает мелочи (как память у человека)."""
        if len(self.facts) <= MAX_FACTS or self._compressing.locked():
            return
        async with self._compressing:
            listing = "\n".join(f"- {f.text} | {f.source}" for f in self.facts)
            prompt = (f"Это знания о мире, которые бот узнал от людей (факт | кто рассказал). Сожми до {COMPRESS_TO} "
                      "пунктов: объедини похожие, выкинь мелочи и устаревшее, сохрани самое важное и интересное. "
                      "Ничего не добавляй от себя. Ответь строго JSON: "
                      '{"facts": [{"text": "факт", "source": "кто рассказал"}]}')
            try:
                resp = await llm.create(
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": listing}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=3000,
                )
                new = json.loads(resp.choices[0].message.content or "{}").get("facts", [])
            except Exception as e:
                log.warning("не смог сжать знания: %s", e)
                return
            facts = [Fact(text=str(f.get("text", ""))[:200], source=str(f.get("source", ""))[:40], ts=time.time())
                     for f in new if isinstance(f, dict) and str(f.get("text", "")).strip()]
            if len(facts) >= COMPRESS_TO // 3:  # защита от «модель вернула пустоту»
                log.info("знания о мире сжаты: %d → %d", len(self.facts), len(facts))
                self.facts = facts
                self.save()


_KNOWLEDGE: Knowledge | None = None


def get_knowledge() -> Knowledge:
    global _KNOWLEDGE
    if _KNOWLEDGE is None:
        from bot import persona

        _KNOWLEDGE = Knowledge(Path(f"data/world_{persona.MEMORY_ID}.json"))
    return _KNOWLEDGE
