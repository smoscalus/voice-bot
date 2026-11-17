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
