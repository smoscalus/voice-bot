"""«Книжка»: когда ему не объяснили, что такое X, он в тихую минуту сам читает об этом.

Источник — короткое описание статьи из Википедии. Читает он ТОЛЬКО то, о чём сам спрашивал и на что
ему не ответили, — так он растёт по своим интересам, а не знает всё подряд. Из прочитанного модель
выписывает 1–3 простых факта, как понял бы ребёнок, с пометкой «книжка».

Не чаще BOOK_PER_DAY раз в сутки (по умолчанию 10, 0 — книжки нет). Один запрос к модели на чтение.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

log = logging.getLogger(__name__)

SOURCE = "книжка"
WAIT_ANSWER_S = 180         # сначала даём людям ответить самим
MIN_GAP_S = 600             # читает не чаще раза в 10 минут
_HEADERS = {"User-Agent": "PetrykVoiceBot/1.0 (Discord voice bot; reads short summaries)"}

# «что такое дота», «что значит кикнуть», «кто такой Наполеон», «что за майнкрафт»
_QUESTION = re.compile(r"(?:что\s+такое|что\s+значит|что\s+означает|кто\s+так(?:ой|ая|ие)|что\s+за)\s+"
                       r"[\"«'„]?([^?\"»'.,!()]{2,40})", re.I)

PROMPT = """Бот-ребёнок {name} сам прочитал в детской книжке про «{term}». Вот что там написано.
Выпиши 1–3 самых простых и понятных факта, как понял бы ребёнок: коротко, по-русски, без дат и цифр,
если они не главное. Каждый факт — законченное предложение с названием («Крым — …», не «он …»).
Только из этого текста, ничего от себя. Если текст не про «{term}» в том смысле,
в каком спрашивали в разговоре, или ничего не понятно — пустой список.
Ответь строго JSON: {{"facts": ["факт", ...]}}"""


def per_day() -> int:
    try:
        return int(os.getenv("BOOK_PER_DAY", "") or 10)
    except ValueError:
        return 10


def note_questions(text: str) -> None:
    """Бот что-то спросил — запоминаем, что ему интересно (прочитает, если не ответят)."""
    from bot.brain.knowledge import get_knowledge

    for m in _QUESTION.finditer(text or ""):
        term = re.sub(r"\s+", " ", m.group(1)).strip(" -—")
        term = re.sub(r"^(это|эти|этот|эта)\s+", "", term, flags=re.I)
        if 2 <= len(term) <= 40 and len(term.split()) <= 4:
            get_knowledge().add_curious(term)


def due() -> bool:
    from bot.brain.knowledge import get_knowledge

    k = get_knowledge()
    now = time.time()
    if per_day() <= 0 or len([t for t in k.reads if now - t < 86400]) >= per_day():
        return False
    if k.reads and now - k.reads[-1] < MIN_GAP_S:
        return False
    return k.next_curious(WAIT_ANSWER_S) is not None


def _get_json(url: str, params: dict | None = None) -> dict | None:
    import urllib.parse
    import urllib.request

    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        if getattr(e, "code", None) == 404:
            return None
        raise


def _fetch_sync(term: str) -> tuple[str, str] | None:
    data = _get_json("https://ru.wikipedia.org/w/api.php", {
        "action": "query", "list": "search", "srsearch": term, "srlimit": 1, "format": "json"}) or {}
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    import urllib.parse

    title = hits[0]["title"]
    summary = _get_json("https://ru.wikipedia.org/api/rest_v1/page/summary/"
                        + urllib.parse.quote(title.replace(" ", "_"))) or {}
    extract = (summary.get("extract") or "").strip()
    return (title, extract[:1200]) if extract else None


async def _fetch(term: str) -> tuple[str, str] | None:
    import asyncio

    return await asyncio.to_thread(_fetch_sync, term)


async def read_one(llm) -> list[str]:
    """Прочитать про одну непонятую вещь. Возвращает, что он вынес из книжки."""
    from bot import persona
    from bot.brain.knowledge import get_knowledge

    k = get_knowledge()
    item = k.next_curious(WAIT_ANSWER_S)
    if item is None:
        return []
    item["tried"] = True
    k.reads.append(time.time())
    k.save()
    term = item["term"]
    try:
        found = await _fetch(term)
    except Exception as e:
        log.warning("книжка: не смог прочитать про «%s»: %s", term, e)
        return []
    if not found:
        log.info("книжка: про «%s» ничего не нашёл", term)
        return []
    title, text = found
    try:
        resp = await llm.create(
            messages=[{"role": "system", "content": PROMPT.format(name=persona.BOT_NAME, term=term)},
                      {"role": "user", "content": f"Статья «{title}»:\n{text}"}],
            response_format={"type": "json_object"},
            max_completion_tokens=300)
        facts = [str(f).strip() for f in json.loads(resp.choices[0].message.content or "{}").get("facts", [])
                 if str(f).strip()][:3]
    except Exception as e:
        log.warning("книжка: не понял прочитанное про «%s»: %s", term, e)
        return []
    added = k.add([(f, SOURCE, False) for f in facts])
    log.info("книжка: прочитал про «%s» (%s): %s", term, title, "; ".join(facts) if facts else "ничего не понял")
    if added:
        from bot.brain.drive import get_drive

        get_drive().on_learned(added)
    return facts
