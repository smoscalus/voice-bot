"""Подготовка локальной Ollama: дождаться, пока она поднимется, и докачать нужные модели.

Благодаря этому «одна кнопка» работает и без ручного `ollama pull`: бот сам всё скачает при первом запуске.
"""
from __future__ import annotations

import asyncio
import json
import logging

import aiohttp

log = logging.getLogger(__name__)


async def ollama_alive(url: str) -> bool:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as http:
            async with http.get(f"{url}/api/tags") as r:
                return r.status == 200
    except Exception:
        return False


async def report_gpu(url: str) -> None:
    """После прогрева: сколько модели в видеопамяти. Не 100% — часть считает процессор, и ответы в разы медленнее."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as http:
            async with http.get(f"{url}/api/ps") as r:
                models = (await r.json()).get("models", [])
    except Exception as e:
        log.info("не смог спросить Ollama, где модель: %s", e)
        return
    for m in models:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        if not size:
            continue
        pct = round(100 * vram / size)
        name = m.get("name", "?")
        from bot.brain.llm import context_limit, ollama_ctx_hint

        want = context_limit()
        if (ctx := m.get("context_length")) and ctx < want:
            log.warning("🧠 %s загружена с окном %d токенов, а бот рассчитывает на %d — длинные промпты Ollama "
                        "будет обрезать (теряется характер, бот несёт бред). Как поправить: %s",
                        name, ctx, want, ollama_ctx_hint(want))
        elif ctx:
            log.info("🧠 %s: окно %d токенов — совпадает с настройкой бота", name, ctx)
        if pct >= 99:
            log.info("🎮 %s целиком на видеокарте (%.1f ГБ)", name, vram / 2**30)
        elif pct == 0:
            log.warning("🐢 %s работает на ПРОЦЕССОРЕ — видеокарту Ollama не видит, ответы будут очень медленными. "
                        "Обнови Ollama; для AMD на Windows попробуй переменную OLLAMA_VULKAN=1 и перезапусти её.",
                        name)
        else:
            log.warning("🐢 %s на видеокарте только на %d%% — остальное считает процессор, ответы медленные. "
                        "Не хватает видеопамяти: закрой игры/браузер, возьми модель поменьше "
                        "или уменьши OLLAMA_CONTEXT_LENGTH / OLLAMA_NUM_PARALLEL.", name, pct)


async def ensure_ollama(url: str, models: list[str], wait_s: float = 120) -> None:
    """Ждёт Ollama до wait_s секунд и скачивает недостающие модели. Бросает RuntimeError, если Ollama нет."""
    for i in range(max(1, int(wait_s / 2))):
        if await ollama_alive(url):
            break
        if i % 10 == 0:  # не спамим в лог каждые 2 секунды
            log.info("жду Ollama на %s …", url)
        await asyncio.sleep(2)
    else:
        raise RuntimeError(
            f"Ollama не отвечает на {url}. Установи её с ollama.com и запусти, "
            "или переключись на облако: PROVIDER=groq")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as http:
        async with http.get(f"{url}/api/tags") as r:
            have = {m["name"] for m in (await r.json()).get("models", [])}
        for model in dict.fromkeys(models):  # без повторов, порядок сохраняем
            if model in have or f"{model}:latest" in have:
                continue
            log.warning("модели %s нет в Ollama — скачиваю (один раз, может занять несколько минут)…", model)
            last_pct = -10
            async with http.post(f"{url}/api/pull", json={"model": model, "stream": True}) as r:
                async for raw in r.content:
                    try:
                        ev = json.loads(raw)
                    except ValueError:
                        continue
                    if ev.get("error"):
                        raise RuntimeError(f"Ollama не смогла скачать {model}: {ev['error']}")
                    if ev.get("total") and ev.get("completed"):
                        pct = int(ev["completed"] * 100 / ev["total"])
                        if pct >= last_pct + 10:
                            last_pct = pct
                            log.info("  %s: %d%%", model, pct)
            log.info("модель %s готова", model)
