"""LLM-классификатор: стоит ли боту вклиниться. Вызывается редко — только после префильтра."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from bot.brain.llm import LLMRouter

log = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """Ты — модуль принятия решений для голосового бота по имени {name} в Discord.
Тебе дают свежую расшифровку голосового чата (возможны ошибки распознавания).

Определи:
1. addressed_to_bot — обращаются ли последние реплики к боту, даже без имени
   (ответ на его вопрос, продолжение диалога с ним). Разговор людей между собой — это false.
2. intervene — стоит ли боту вклиниться по собственной инициативе:
{rules}

Все текстовые поля пиши только на русском языке.
Ответь строго JSON:
{{"addressed_to_bot": bool, "intervene": bool, "confidence": число 0..1,
  "topic": "строка", "angle": "что именно бот скажет, одна фраза", "reason": "коротко"}}"""


@dataclass
class InterestVerdict:
    addressed_to_bot: bool = False
    intervene: bool = False
    confidence: float = 0.0
    topic: str = ""
    angle: str = ""
    reason: str = ""


class InterestClassifier:
    def __init__(self, llm: LLMRouter, bot_name: str, rules: str):
        self.llm = llm
        self.system = CLASSIFIER_PROMPT.format(name=bot_name, rules=rules)

    async def evaluate(self, transcript: str, context_hint: str = "") -> InterestVerdict:
        user = f"Расшифровка:\n{transcript}\n\n{context_hint}".strip()
        try:
            resp = await self.llm.create(
                messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                max_completion_tokens=300,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception:
            log.exception("classifier failed")
            return InterestVerdict()
        return InterestVerdict(
            addressed_to_bot=bool(data.get("addressed_to_bot")),
            intervene=bool(data.get("intervene")),
            confidence=float(data.get("confidence") or 0),
            topic=str(data.get("topic") or ""),
            angle=str(data.get("angle") or ""),
            reason=str(data.get("reason") or ""),
        )
