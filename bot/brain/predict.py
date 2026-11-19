"""Предчувствие: перед важной репликой бот прикидывает, как люди отреагируют, а потом сверяет с тем, что было.

Без лишних запросов к модели:
- ожидание пишет «внутренний голос», когда решает встрять (поле expect), или сам бот, когда начинает
  тему в затишье (строка «ОЖИДАЮ: …» в конце, вслух не звучит);
- сверку делает летописец памяти (RelationsJudge), который и так читает, что было после.

Итог сверки слабо двигает чувства: угадал — увереннее и спокойнее; приятно ошибся — радость; ждал
хорошего, а его проигнорили или обсмеяли — тревога и недоверие к тому, кто так ответил. Насколько хорошо
он угадывает, бот знает сам (блок в промпте) — так он учится предугадывать людей.

Хранится в data/predict_<имя памяти>.json (только итоги; несверенные ожидания живут до перезапуска).
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MATURE_S = 25          # раньше не сверяем — люди ещё не успели отреагировать
MIN_REACTIONS = 2      # сверяем, когда после реплики сказали хотя бы столько…
SILENCE_S = 60         # …или прошло столько (молчание — тоже реакция: проигнорили)
EXPIRE_S = 600         # не сверили за 10 минут — забываем
MAX_PENDING = 3
HISTORY = 20

# как итог сверки двигает общие чувства — слабо (14b иногда оценивает криво); меньше 0.05 чувства не замечают
EFFECT = {
    "match": {"trust": 0.05, "fear": -0.05},
    "good": {"joy": 0.1, "trust": 0.05},
    "bad": {"fear": 0.08, "sadness": 0.05, "trust": -0.05},
    "miss": {"fear": 0.05},
}
PERSON_TRUST = {"good": 0.3, "bad": -0.2}   # доверие к тому, кто удивил (не угадал сам — человек не виноват)


@dataclass
class Expectation:
    said: str
    expect: str
    seq: int                                  # номер реплики бота в расшифровке
    ts: float = field(default_factory=time.time)


class Predictor:
    def __init__(self, path: Path):
        self.path = path
        self.pending: list[Expectation] = []
        self.history: deque[dict] = deque(maxlen=HISTORY)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.history.extend(h for h in raw.get("history", []) if isinstance(h, dict))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("не смог прочитать %s", path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"history": list(self.history)}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, said: str, expect: str, seq: int) -> None:
        expect = (expect or "").strip().strip("«»\"").rstrip(".")[:150]
        if not expect or not said:
            return
        self.pending.append(Expectation(said[:200], expect, seq))
        self.pending = self.pending[-MAX_PENDING:]
        log.info("ожидаю: %s", expect)

    def matured(self, buffer) -> list[tuple[Expectation, list]]:
        """Ожидания, которые уже можно сверить, — с репликами людей после них."""
        now = time.time()
        self.pending = [e for e in self.pending if now - e.ts < EXPIRE_S]
        out = []
        for e in self.pending:
            after = [l for l in buffer.since(e.seq) if not l.is_bot][:8]
            if now - e.ts >= MATURE_S and (len(after) >= MIN_REACTIONS or now - e.ts >= SILENCE_S):
                out.append((e, after))
        return out

    def resolve(self, e: Expectation, outcome: str, what: str) -> str:
        """outcome: match | good | bad | miss. Возвращает его же (для лога)."""
        if e in self.pending:
            self.pending.remove(e)
        self.history.append({"said": e.said, "expect": e.expect, "outcome": outcome, "what": what[:150],
                             "ts": round(time.time())})
        self.save()
        return outcome

    def accuracy(self) -> tuple[int, int]:
        hits = sum(1 for h in self.history if h.get("outcome") in ("match", "good"))
        return hits, len(self.history)

    def prompt_block(self) -> str:
        if not self.history:
            return ""
        _, n = self.accuracy()
        exact = sum(1 for h in self.history if h.get("outcome") == "match")
        if n < 3:
            sense = "ты только начинаешь угадывать, как люди отреагируют"
        elif exact / n >= 0.6:
            sense = "ты уже неплохо чувствуешь, как люди отреагируют"
        elif exact / n >= 0.3:
            sense = "иногда угадываешь, как люди отреагируют, иногда нет"
        else:
            sense = "люди часто реагируют не так, как ты ждёшь — ты их пока плохо понимаешь"
        last = list(self.history)[-2:]
        recent = "; ".join(f"ждал «{h['expect']}» — вышло: {h['what'] or h['outcome']}" for h in last)
        return f"Твоё чутьё на людей: {sense} (угадал {exact} из {n}). Недавно: {recent}."


_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        from bot.brain.life import memory_id

        _PREDICTOR = Predictor(Path(f"data/predict_{memory_id()}.json"))
    return _PREDICTOR
