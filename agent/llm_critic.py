"""LLM-критик (A-L3-2): одна проверка пачки гипотез от LLM до запуска. Владелец — LLM/agent-инженер.

Роли агента: генератор (llm_hypotheses) предлагает, критик по правилам (critic.py) отсекает пустое,
недопустимое и не успевающее, а этот критик проверяет смысл: утечку, подгонку под данные,
несоответствие метрике, повтор провалившегося направления, цену.

Критик — фильтр, а не опора: при любом сбое (нет LLM, мусор в ответе, мало времени) гипотезы проходят.
Включается переменной AGENT_LLM_CRITIC=1; по умолчанию выключен — на харнессе не дал выигрыша (ADR 0012).
"""

from __future__ import annotations

import json
import os

from agent.llm_client import LLMClient
from agent.llm_hypotheses import _items
from agent.memory import ResearchMemory
from agent.prompting import build_critic_messages
from interfaces.schemas import DataProfile, ExperimentResult, Hypothesis, RunBudget, TaskSpec
from runtime.log import get_logger

log = get_logger("llm_critic")

CRITIC_ENV = "AGENT_LLM_CRITIC"
MAX_TOKENS = 400  # ответ — короткий список вердиктов; рассуждения модели сверху (LLM_REASONING_TOKENS)
MIN_SECONDS_LEFT = 300  # запрос стоит 10–20 с: под конец бюджета время нужнее экспериментам
# сколько ждать ответа, решает LLMClient по seconds_left — то же правило, что у генератора (D3)
MAX_REASON_CHARS = 200


def enabled() -> bool:
    return os.getenv(CRITIC_ENV) == "1"


def candidates_json(hyps: list[Hypothesis]) -> str:
    items = [
        {"id": h.id, "kind": h.kind.value, "title": h.title, "rationale": h.rationale, "params": h.params} for h in hyps
    ]
    return "[\n" + ",\n".join("  " + json.dumps(item, ensure_ascii=False) for item in items) + "\n]"


def critique(
    spec: TaskSpec,
    profile: DataProfile,
    history: list[tuple[Hypothesis, ExperimentResult]],
    batch: list[Hypothesis],
    llm: LLMClient,
    budget: RunBudget,
    memory: ResearchMemory | None = None,
) -> dict[str, str]:
    """id → причина отклонения. Пусто, если критик выключен, недоступен или ничего не отклонил."""
    candidates = [h for h in batch if h.source == "llm"]  # playbook уже проверен людьми — не тратим запрос
    if not candidates or not enabled() or not llm.available() or budget.seconds_left() < MIN_SECONDS_LEFT:
        return {}
    messages = build_critic_messages(spec, profile, history, candidates_json(candidates), memory)
    if messages is None:
        return {}
    try:
        raw = llm.chat_json(messages, temperature=0.1, max_tokens=MAX_TOKENS, seconds_left=budget.seconds_left())
    except Exception:  # noqa: BLE001 — сбой критика не должен останавливать цикл
        log.exception("Критик LLM не ответил — гипотезы проходят без проверки")
        return {}
    ids = {h.id for h in candidates}
    rejected = {}
    for item in _items(raw):
        if not isinstance(item, dict) or item.get("id") not in ids:
            continue
        if str(item.get("verdict", "")).strip().lower() == "reject":
            rejected[item["id"]] = " ".join(str(item.get("reason") or "без объяснения").split())[:MAX_REASON_CHARS]
    log.info("Критик LLM: отклонено %d из %d гипотез", len(rejected), len(candidates))
    return rejected
