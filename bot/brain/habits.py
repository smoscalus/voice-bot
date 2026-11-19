"""Словечки и привычки: «как я сам говорю и веду себя» — отдельно от знаний о мире.

Знание — «алло говорят в начале разговора». Привычка — «когда начинаю разговор, я сам говорю „алло“».
Знания подаются в промпт поиском (всплывают, только если похожи на тему), а привычки — всегда:
иначе «ты тоже можешь так говорить» теряется среди фактов про Крым и токены.

Откуда берутся: летописец замечает, когда люди учат бота говорить или вести себя («можешь говорить алло»,
«называй меня Вася», «не говори так часто „не понял“») или запрещают («хватит говорить алло»).
Как живут: использовал словечко — оно крепнет; во сне неиспользованные тускнеют и со временем уходят.
Как у ребёнка: что заходит — остаётся.

Хранится в data/habits_<имя памяти>.json.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)

MAX_KEEP = 12        # всего привычек помнит
MAX_PROMPT = 8       # в голове одновременно
USE_BOOST = 0.15     # использовал — крепнет
FADE_STEP = 0.2      # за сон, если не использовал
FORGET_BELOW = 0.3


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip(" .,!?«»\"'")


class Habits:
    def __init__(self, path: Path):
        self.path = path
        self.items: list[dict] = []   # {"text", "word", "source", "ts", "strength", "uses", "last_use"}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.items = [h for h in raw.get("habits", []) if isinstance(h, dict) and h.get("text")]
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("не смог прочитать %s", path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"habits": self.items}, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def _find(self, text: str, word: str) -> dict | None:
        from rapidfuzz import fuzz

        for h in self.items:
            if word and h.get("word") and _norm(h["word"]) == _norm(word):
                return h
            if fuzz.ratio(_norm(h["text"]), _norm(text)) >= 80:
                return h
        return None

    def learn(self, items, stops, bot_name: str = "") -> None:
        """От летописца: новые привычки (items) и запреты (stops — словечки, которые просили не говорить)."""
        changed = False
        for it in items or []:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()[:140]
            word = str(it.get("word") or "").strip()[:30]
            source = str(it.get("source") or "").strip()[:40]
            if len(text.split()) < 3 or (bot_name and _norm(source) == _norm(bot_name)):
                continue
            old = self._find(text, word)
            if old:  # повторили — значит, важно: крепнет
                old["strength"] = min(1.5, old.get("strength", 1.0) + 0.3)
                continue
            self.items.append({"text": text, "word": word, "source": source, "ts": round(time.time()),
                               "strength": 1.0, "uses": 0, "last_use": 0})
            log.info("новая привычка: %s%s", text, f" (от {source})" if source else "")
            changed = True
        for w in stops or []:
            w = _norm(str(w))
            if not w:
                continue
            before = len(self.items)
            self.items = [h for h in self.items if w != _norm(h.get("word", "")) and w not in _norm(h["text"])]
            if len(self.items) < before:
                log.info("отучили от привычки: %s", w)
                changed = True
        if len(self.items) > MAX_KEEP:
            self.items.sort(key=lambda h: -h.get("strength", 1.0))
            self.items = self.items[:MAX_KEEP]
        if changed:
            self.save()

    def note_said(self, text: str) -> None:
        """Бот что-то сказал — использованные словечки крепнут."""
        low = " " + _norm(text) + " "
        used = False
        for h in self.items:
            w = _norm(h.get("word", ""))
            if w and re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low):
                h["uses"] = h.get("uses", 0) + 1
                h["last_use"] = round(time.time())
                h["strength"] = min(1.5, h.get("strength", 1.0) + USE_BOOST)
                log.info("использовал словечко «%s» (%d-й раз)", h["word"], h["uses"])
                used = True
        if used:
            self.save()

    def fade(self, since: float, steps: int = 1) -> int:
        """Во сне: что не использовал с прошлого сна — тускнеет; совсем бледное — забывает."""
        for h in self.items:
            if max(h.get("last_use", 0), h.get("ts", 0)) < since:
                h["strength"] = h.get("strength", 1.0) - FADE_STEP * max(1, steps)
        keep = [h for h in self.items if h.get("strength", 1.0) >= FORGET_BELOW]
        forgotten = len(self.items) - len(keep)
        if forgotten:
            log.info("забыл привычки: %s", "; ".join(h["text"] for h in self.items if h not in keep))
        self.items = keep
        self.save()
        return forgotten

    def prompt_block(self) -> str:
        if not self.items:
            return ""
        top = sorted(self.items, key=lambda h: -h.get("strength", 1.0))[:MAX_PROMPT]
        lines = [f"- {h['text']}" for h in top]
        return ("Твои словечки и привычки — так говоришь ТЫ САМ, тебя этому научили. Используй к месту, "
                "в подходящей ситуации, но не в каждой фразе:\n" + "\n".join(lines))


_HABITS: Habits | None = None


def get_habits() -> Habits:
    global _HABITS
    if _HABITS is None:
        from bot.brain.life import memory_id

        _HABITS = Habits(Path(f"data/habits_{memory_id()}.json"))
    return _HABITS
