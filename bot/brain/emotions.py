"""Чувства бота — что-то вроде гормонов.

Числа (0…1) для страха, радости, доверия и грусти. Их двигает летописец памяти (RelationsJudge),
который и так читает разговор: оценивает, что Петрык испытал от свежих реплик. Со временем каждое
чувство само возвращается к своему «фону» (испуг проходит за десятки минут). Но если пугали часто,
фон немного растёт — остаётся настороженность, как шрам; и так же медленно заживает.

Гормоны дают силу и инерцию, а что именно он чувствует и почему — решает он сам в дневнике
(bot/brain/self_model.py видит эти уровни). Чувства влияют на поведение: напуганный и грустный
реже сам встревает, радостный — чаще.

Хранится в data/emotions_<имя памяти>.json.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

log = logging.getLogger(__name__)

# чувство → (название для промпта, фон по умолчанию, за сколько секунд проходит половина всплеска)
FEELINGS = {
    "fear": ("тревога", 0.1, 1200),
    "joy": ("радость", 0.3, 900),
    "trust": ("доверие", 0.3, 3600),
    "sadness": ("грусть", 0.1, 1800),
}
MAX_STEP = 0.4            # сильнее за одно обновление не дёргается
SCAR_SHARE = 0.08         # какая доля сильного всплеска оседает в фоне
SCAR_HEAL_S = 3 * 86400   # фон возвращается к исходному за ~3 дня (половина)
WORDS = [(0.15, "почти нет"), (0.35, "слабо"), (0.6, "заметно"), (0.8, "сильно"), (2.0, "очень сильно")]


class Emotions:
    def __init__(self, path: Path):
        self.path = path
        self.level = {k: base for k, (_, base, _) in FEELINGS.items()}
        self.base = dict(self.level)
        self._t = time.time()
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.level.update({k: float(v) for k, v in raw.get("level", {}).items() if k in FEELINGS})
            self.base.update({k: float(v) for k, v in raw.get("base", {}).items() if k in FEELINGS})
            self._t = float(raw.get("t", self._t))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("не смог прочитать %s", self.path)
        self._tick()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {"level": {k: round(v, 3) for k, v in self.level.items()},
                "base": {k: round(v, 3) for k, v in self.base.items()}, "t": self._t}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def _tick(self) -> None:
        """Время лечит: всплески тают к фону, фон медленно — к исходному. Считается и пока бот выключен."""
        now = time.time()
        dt = max(0.0, now - self._t)
        self._t = now
        for k, (_, default, half) in FEELINGS.items():
            self.base[k] = default + (self.base[k] - default) * math.exp(-dt * math.log(2) / SCAR_HEAL_S)
            self.level[k] = self.base[k] + (self.level[k] - self.base[k]) * math.exp(-dt * math.log(2) / half)

    def apply(self, deltas: dict) -> None:
        """deltas: чувство → изменение (-1…+1) от свежих реплик (оценка летописца)."""
        self._tick()
        changed = []
        for k, v in (deltas or {}).items():
            if k not in FEELINGS:
                continue
            try:
                d = max(-MAX_STEP, min(MAX_STEP, float(v)))
            except (TypeError, ValueError):
                continue
            if abs(d) < 0.05:
                continue
            self.level[k] = max(0.0, min(1.0, self.level[k] + d))
            if d >= 0.2:  # сильное переживание оставляет след
                self.base[k] = min(0.6, self.base[k] + d * SCAR_SHARE)
            changed.append(f"{FEELINGS[k][0]} {d:+.2f}")
        if changed:
            self.save()
            log.info("чувства: %s → %s", ", ".join(changed), self.describe())

    def _word(self, x: float) -> str:
        return next(w for limit, w in WORDS if x < limit)

    def describe(self) -> str:
        self._tick()
        return ", ".join(f"{name} — {self._word(self.level[k])}" for k, (name, _, _) in FEELINGS.items())

    def talk_bias(self) -> float:
        """Насколько охотнее (+) или неохотнее (−) он сам встревает в разговор."""
        self._tick()
        lv = self.level
        return 0.25 * lv["joy"] + 0.1 * lv["trust"] - 0.3 * lv["fear"] - 0.25 * lv["sadness"]

    def prompt_block(self) -> str:
        self._tick()
        lv = self.level
        hints = []
        if lv["fear"] >= 0.6:
            hints.append("тебе тревожно: говоришь осторожнее и меньше, держишься подальше от тех, кто пугает")
        if lv["sadness"] >= 0.6:
            hints.append("тебе грустно: говоришь тише и короче")
        if lv["joy"] >= 0.6:
            hints.append("тебе радостно: охотно болтаешь и расспрашиваешь")
        if lv["trust"] >= 0.6:
            hints.append("ты доверяешь людям вокруг и открыт")
        text = ("Что ты сейчас ощущаешь (как гормоны — откуда это и что это значит, решаешь сам; "
                "уровни вслух не называй): " + self.describe() + ".")
        return text + (" " + "; ".join(hints) + "." if hints else "")


_EMOTIONS: Emotions | None = None


def get_emotions() -> Emotions:
    global _EMOTIONS
    if _EMOTIONS is None:
        from bot import persona

        _EMOTIONS = Emotions(Path(f"data/emotions_{persona.MEMORY_ID}.json"))
    return _EMOTIONS
