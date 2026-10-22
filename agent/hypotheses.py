"""Откуда берутся гипотезы. Владелец — LLM/agent-инженер (playbook — Research/DS).

Если LLM доступна — она смотрит на DataProfile и историю экспериментов и предлагает новые гипотезы
(agent/llm_hypotheses.py). Недостающие и все гипотезы без LLM — из research/playbook.yaml по порядку.
Повторы (тот же id или те же params, что уже проверялись) отсекаются.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent.llm_client import LLMClient
from agent.memory import ResearchMemory
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    Hypothesis,
    HypothesisKind,
    RunBudget,
    TaskSpec,
    TaskType,
)

PLAYBOOK_PATH = Path(__file__).resolve().parent.parent / "research" / "playbook.yaml"


def baseline_hypothesis() -> Hypothesis:
    return Hypothesis(
        id="baseline",
        kind=HypothesisKind.BASELINE,
        title="LightGBM с параметрами по умолчанию",
        params={"model": "lightgbm"},
        source="baseline",
    )


def load_playbook(path: Path = PLAYBOOK_PATH) -> list[dict]:
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("hypotheses", [])


def _applies(item: dict, spec: TaskSpec, profile: DataProfile) -> bool:
    task_types = item.get("task_types")
    if task_types and spec.task_type.value not in task_types:
        return False
    if item.get("requires") == "categorical" and not profile.categorical_features:
        return False
    if item.get("requires") == "imbalance" and spec.task_type != TaskType.REGRESSION:
        return profile.target_summary.get("imbalance_ratio", 1) > 3
    return item.get("requires") != "imbalance"


def params_key(params: dict) -> str:
    """Канонический вид params: одинаковые гипотезы с разными id считаем повтором.

    validation не входит в ключ: схема одна на весь прогон (ADR 0008, D1) и есть у всех гипотез из истории,
    а у новых кандидатов её ещё нет — иначе повтор бейзлайна или прошлой гипотезы не находился бы.
    """
    params = {k: v for k, v in params.items() if k != "validation"}
    return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)


def propose_hypotheses(
    spec: TaskSpec,
    profile: DataProfile,
    history: list[tuple[Hypothesis, ExperimentResult]],
    k: int = 3,
    budget: RunBudget | None = None,
    llm: LLMClient | None = None,
    memory: ResearchMemory | None = None,
) -> list[Hypothesis]:
    """Сначала гипотезы от LLM (если она доступна), недостающие — из playbook. Повторы истории отсекаются.

    budget (контракт 1.1, D3): по остатку времени LLM-клиент решает, звать ли модель и сколько ждать;
    playbook бюджет не требует.
    """
    out: list[Hypothesis] = []
    if llm is not None:
        from agent.llm_hypotheses import llm_hypotheses  # здесь: llm_hypotheses сам импортирует этот модуль

        out = llm_hypotheses(spec, profile, history, k, llm, memory, budget)
    tried = {h.id for h, _ in history} | {h.id for h in out}
    seen_params = {params_key(h.params) for h, _ in history} | {params_key(h.params) for h in out}
    for item in load_playbook():
        if len(out) >= k:
            break
        params = item.get("params", {})
        if item["id"] in tried or params_key(params) in seen_params or not _applies(item, spec, profile):
            continue
        out.append(
            Hypothesis(
                id=item["id"],
                kind=HypothesisKind(item["kind"]),
                title=item["title"],
                rationale=item.get("rationale", ""),
                params=params,
                source="playbook",
            )
        )
        seen_params.add(params_key(params))
    return out
