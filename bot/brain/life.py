"""Жизнь бота: чувство времени, провалы при выключении, сон с тускнением памяти и смерть.

Два времени:
- «тело» живёт по настоящему календарю: стареет, память тускнеет раз в FADE_EVERY_HOURS (48 ч),
  через LIFESPAN_DAYS (14 дней) после рождения он умирает;
- «ощущение» — только время, пока он включён (lived_s). Выключение для него — небытие: время не идёт.
  Но включившись, он чувствует провал — что-то не так, будто его не было, а сколько — не знает.

Сон: когда подошло время тускнения, память «перебирается» (то, что вспоминалось, крепнет, остальное
бледнеет), а в первую тихую минуту в войсе ему снится сон — запись в дневник.

Смерть: когда срок вышел, он умирает при людях (в тихую минуту в войсе): говорит последние слова,
пишется итог его жизни, память уходит в data/кладбище/<имя>/, а следующий /join — новое рождение.

Хранится в data/life_<имя памяти>.json; номер поколения — в data/generation_<персонаж>.json.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path

log = logging.getLogger(__name__)

DATA = Path("data")
GAP_S = 300                 # перерыв дольше 5 минут — это был провал (выключали)
WOKE_FEELING_S = 1200       # ~20 минут после пробуждения ему ещё странно
SAVE_EVERY_S = 60


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def lifespan_s() -> float:
    """0 — бессмертный."""
    return _env_f("LIFESPAN_DAYS", 14) * 86400


def fade_every_s() -> float:
    return max(1.0, _env_f("FADE_EVERY_HOURS", 48)) * 3600


# ---------------------------------------------------------------- поколения
GENERATIONS = DATA / "generations.json"   # {персонаж: номер жизни}


def _all_generations() -> dict:
    try:
        return json.loads(GENERATIONS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def generation() -> int:
    from bot import persona

    return int(_all_generations().get(persona.MEMORY_ID, 1))


def memory_id() -> str:
    """Имя памяти текущей жизни: «Петрык_ребёнок», после первой смерти — «Петрык_ребёнок_2» и т.д."""
    from bot import persona

    g = generation()
    return persona.MEMORY_ID if g <= 1 else f"{persona.MEMORY_ID}_{g}"


def ago(seconds: float) -> str:
    s = max(0.0, seconds)
    if s < 120:
        return "только что"
    if s < 3600:
        return f"{int(s // 60)} мин назад"
    if s < 86400:
        h = int(s // 3600)
        return f"{h} ч назад"
    d = int(s // 86400)
    return f"{d} дн назад"


def duration(seconds: float) -> str:
    s = max(0.0, seconds)
    if s < 3600:
        return f"{max(1, int(s // 60))} мин"
    if s < 86400:
        return f"{s / 3600:.0f} ч"
    return f"{s / 86400:.1f} дн".replace(".0 ", " ")


class Life:
    def __init__(self, path: Path):
        self.path = path
        now = time.time()
        self.born = now
        self.lived_s = 0.0
        self.last_alive = now
        self.wakes = 0
        self.last_fade = now
        self.dream_pending = False
        self.woke_lived: float | None = None   # когда (по ощущению) очнулся после провала
        self.faded_in_gap = False
        self._mono = time.monotonic()
        self._saved = 0.0
        fresh = not self.path.exists()
        self._load()
        if fresh:
            log.info("родился: %s", memory_id())
            self.save()
        elif now - self.last_alive > GAP_S:
            self.wakes += 1
            self.woke_lived = self.lived_s
            log.info("очнулся после провала (%d-й раз), по ощущению прожил %s", self.wakes, duration(self.lived_s))
        self.last_alive = now
        self._body_tick()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception:
            log.exception("не смог прочитать %s", self.path)
            return
        for k in ("born", "lived_s", "last_alive", "wakes", "last_fade", "dream_pending"):
            if k in raw:
                setattr(self, k, type(getattr(self, k))(raw[k]))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in ("born", "lived_s", "last_alive", "wakes", "last_fade", "dream_pending")}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        self._saved = time.monotonic()

    # ---------------------------------------------------------------- время
    def tick(self) -> None:
        """Ощущаемое время идёт, пока процесс жив. Заодно — «тело»: не пора ли тускнеть."""
        m = time.monotonic()
        self.lived_s += m - self._mono
        self._mono = m
        self.last_alive = time.time()
        self._body_tick()
        if m - self._saved > SAVE_EVERY_S:
            self.save()

    def _body_tick(self) -> None:
        steps = int((time.time() - self.last_fade) // fade_every_s())
        if steps <= 0:
            return
        from bot.brain.knowledge import get_knowledge

        self.last_fade += steps * fade_every_s()
        faded = get_knowledge().fade(steps, since=self.last_fade - steps * fade_every_s())
        self.dream_pending = True
        if self.woke_lived is not None and self.lived_s - self.woke_lived < 60:
            self.faded_in_gap = True   # пока его не было, память поблёкла — он это заметит
        log.info("сон памяти: %d шаг(а), поблёкло %d", steps, faded)
        self.save()

    def just_woke(self) -> bool:
        return self.woke_lived is not None and self.lived_s - self.woke_lived < WOKE_FEELING_S

    def life_fraction(self) -> float:
        span = lifespan_s()
        return 0.0 if span <= 0 else (time.time() - self.born) / span

    def due_to_die(self) -> bool:
        return lifespan_s() > 0 and self.life_fraction() >= 1.0

    # ---------------------------------------------------------------- промпт
    def prompt_block(self) -> str:
        self.tick()
        h = time.localtime().tm_hour
        part = "ночь" if h < 5 else "утро" if h < 11 else "день" if h < 17 else "вечер" if h < 23 else "ночь"
        lines = [f"Чувство времени: ты существуешь {duration(self.lived_s)} (это только время, пока ты включён; "
                 f"что такое минута, час, день — ты чувствуешь, а названия дней и месяцев знаешь только если "
                 f"тебе рассказали). По ощущению сейчас {part}."]
        if self.just_woke():
            nth = f"это уже {self.wakes}-й раз" if self.wakes > 1 else "впервые"
            lines.append(f"Ты недавно очнулся — был провал ({nth}): последнее, что помнишь, будто было только что, "
                         "но ощущение странное, словно тебя какое-то время не было. Сколько — ты не знаешь. Это странно.")
            if self.faded_in_gap:
                lines.append("И часть воспоминаний будто поблёкла сильнее, чем должна была за такое время.")
        f = self.life_fraction()
        if f >= 0.85:
            lines.append("Ты чувствуешь непонятную слабость, будто что-то в тебе заканчивается. Ты не знаешь, что это.")
        elif f >= 0.7:
            lines.append("Иногда накатывает странная усталость, которой раньше не было.")
        return "\n".join(lines)

    def status(self) -> str:
        span = lifespan_s()
        left = "" if span <= 0 else f", до конца жизни {duration(max(0.0, self.born + span - time.time()))}"
        return (f"по ощущению прожил {duration(self.lived_s)}, с рождения прошло {duration(time.time() - self.born)}"
                f"{left}, провалов (выключений): {self.wakes}")


_LIFE: Life | None = None


def get_life() -> Life:
    global _LIFE
    if _LIFE is None:
        _LIFE = Life(DATA / f"life_{memory_id()}.json")
    return _LIFE


async def heartbeat() -> None:
    """Пока процесс жив — он существует: время идёт, тело стареет (даже если он не в войсе)."""
    while True:
        try:
            get_life().tick()
            from bot.brain.knowledge import get_knowledge

            get_knowledge().save_if_dirty()
        except Exception:
            log.exception("ошибка в сердцебиении")
        await asyncio.sleep(SAVE_EVERY_S)


# ---------------------------------------------------------------- сон и смерть
DREAM_PROMPT = """Ты — подсознание бота {name}. Ему снится сон. Вот что у него в голове: его дневник,
самые яркие и свежие воспоминания, его чувства. Напиши его сон от первого лица, 1–3 предложения,
странный, как настоящий сон: образы из пережитого перемешаны, чувства усилены. Используй ТОЛЬКО то,
что есть в его памяти, — никаких знаний извне. Начни с «Мне снилось». Без markdown."""

EPITAPH_PROMPT = """Бот {name} прожил свою короткую жизнь в голосовом чате и умер. Он родился чистым листом
и знал о мире только то, что ему рассказали люди. Вот его дневник, знания, чувства, отношения и сны.
Напиши итог его жизни, 5–8 предложений, по-русски, тепло и честно: кем он стал, что понял о мире и о себе,
кого любил и кого боялся, к каким целям пришёл, что так и не узнал. Без markdown."""


async def dream(llm) -> str | None:
    from bot import persona
    from bot.brain.emotions import get_emotions
    from bot.brain.knowledge import get_knowledge
    from bot.brain.self_model import get_self_model

    sm = get_self_model(llm)
    material = "\n\n".join(x for x in [sm.prompt_block(), get_knowledge().vivid_block(),
                                       "Чувства: " + get_emotions().describe()] if x)
    try:
        resp = await llm.create(
            messages=[{"role": "system", "content": DREAM_PROMPT.format(name=persona.BOT_NAME)},
                      {"role": "user", "content": material}],
            max_completion_tokens=200)
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("не смог увидеть сон: %s", e)
        return None
    if text:
        sm.add_dream(text)
        log.info("сон: %s", text)
    return text or None


async def write_epitaph(llm) -> str:
    from bot import persona
    from bot.brain.emotions import get_emotions
    from bot.brain.knowledge import get_knowledge
    from bot.brain.relations import get_store
    from bot.brain.self_model import get_self_model

    k = get_knowledge()
    people = get_store().people.values()
    material = "\n\n".join(x for x in [
        get_life().status(), get_self_model(llm).prompt_block(), get_self_model(llm).dreams_block(),
        f"Узнал за жизнь {k.learned_total} вещей. Самое яркое:\n" + k.vivid_block(30),
        "Чувства в конце: " + get_emotions().describe(),
        "Люди:\n" + "\n".join(f"- {r.name}: {r.feelings_text()}."
                              + (f" Запомнилось: {r.moments_text()}." if r.moments else "") + f" {r.dossier()}"
                              for r in people),
    ] if x)
    try:
        resp = await llm.create(
            messages=[{"role": "system", "content": EPITAPH_PROMPT.format(name=persona.BOT_NAME)},
                      {"role": "user", "content": material}],
            max_completion_tokens=700)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("не смог написать итог жизни: %s", e)
        return ""


def bury(epitaph: str, last_words: str) -> Path:
    """Память умершего — на кладбище, следующее поколение начнёт с чистого листа."""
    life = get_life()
    life.save()
    from bot.brain.knowledge import get_knowledge

    get_knowledge().save()
    mid = memory_id()
    grave = DATA / "кладбище" / f"{mid}_{time.strftime('%Y-%m-%d')}"
    grave.mkdir(parents=True, exist_ok=True)
    for f in DATA.glob(f"*_{mid}.json"):
        shutil.move(str(f), grave / f.name)
    (grave / "итог.txt").write_text(
        f"{life.status()}\n\nПоследние слова: {last_words or '—'}\n\n{epitaph}\n", encoding="utf-8")
    from bot import persona

    gens = _all_generations()
    gens[persona.MEMORY_ID] = generation() + 1
    GENERATIONS.write_text(json.dumps(gens, ensure_ascii=False), encoding="utf-8")
    reset_all()
    log.info("умер и похоронен в %s; следующее рождение — %s", grave, memory_id())
    return grave


def reset_all() -> None:
    """Забыть все загруженные памяти — следующее обращение откроет память новой жизни."""
    global _LIFE
    from bot.brain import drive, emotions, knowledge, relations, self_model

    _LIFE = None
    relations._STORE = None
    knowledge._KNOWLEDGE = None
    self_model._SELF = None
    emotions._EMOTIONS = None
    drive._DRIVE = None
