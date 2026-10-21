"""Цикл AutoResearch: гипотеза → эксперимент → оценка → решение. Владелец — LLM/agent-инженер.

Правило принятия: новая гипотеза принимается, только если она лучше текущей лучшей
с запасом min_delta (часть разброса по фолдам). Это защита от «улучшений», которые являются шумом.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from agent.critic import estimated_seconds, review
from agent.hypotheses import propose_hypotheses
from agent.llm_client import LLMClient
from agent.llm_hypotheses import reflect
from agent.memory import ResearchMemory
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, Hypothesis, RunBudget, TaskSpec
from ml.experiment import run_experiment
from runtime.log import get_logger

log = get_logger("loop")

NOISE_FACTOR = 0.25  # какую долю std лучшего результата считаем шумом
BATCH = 3  # гипотез за один запрос к LLM: дешевле, чем спрашивать на каждой итерации


def min_delta(best: ExperimentResult | None) -> float:
    return NOISE_FACTOR * (best.std_score or 0.0) if best else 0.0


def run_autoresearch(
    spec: TaskSpec,
    profile: DataProfile,
    train: pd.DataFrame,
    budget: RunBudget,
    history: list[tuple[Hypothesis, ExperimentResult]],
    best: tuple[Hypothesis, ExperimentResult],
    on_new_best: Callable[[Hypothesis, ExperimentResult], None],
    max_iterations: int = 50,
    llm: LLMClient | None = None,
    memory: ResearchMemory | None = None,
) -> tuple[Hypothesis, ExperimentResult]:
    gib = spec.metric.greater_is_better
    if llm is None:
        llm = LLMClient(rub_limit=budget.llm_rub_limit)  # без ключа или с LLM_MOCK=1 — сразу недоступна
    memory = memory if memory is not None else ResearchMemory()
    queue: list[Hypothesis] = []
    batch_start = len(history)  # с какого места истории начинается текущая пачка гипотез
    baseline_seconds = best[1].fit_seconds  # на входе best — бейзлайн: от него считаем цену гипотез
    for iteration in range(1, max_iterations + 1):
        # Не успеваем даже эксперимент ценой в бейзлайн — дальше пробовать нечего
        if not budget.has_time(needed_seconds=baseline_seconds * 1.2):
            log.info("Время на исходе — завершаю цикл после %d итераций", iteration - 1)
            break
        if not queue:
            batch = history[batch_start:]
            if batch and not any(memory.verdicts.get(h.id, "").startswith("принята") for h, _ in batch):
                reflect(spec, best, batch, llm, memory)  # вся пачка мимо — спросим LLM, почему
            queue = propose_hypotheses(spec, profile, history, k=BATCH, llm=llm, memory=memory)
            batch_start = len(history)
        if not queue:
            log.info("Гипотезы закончились — завершаю цикл")
            break
        hyp = queue.pop(0)
        log.info("Итерация %d: %s — %s", iteration, hyp.id, hyp.title)
        reason = review(hyp, spec)
        needed = estimated_seconds(hyp, baseline_seconds)
        if reason is None and not budget.has_time(needed_seconds=needed):
            reason = f"не успеет: оценка {needed:.0f} с, осталось {budget.seconds_left():.0f} с"
        if reason is not None:
            # в историю со статусом SKIPPED: гипотезу не предложат снова, а LLM увидит причину
            log.info("  ⊘ пропуск критиком: %s", reason)
            history.append((hyp, _skipped(hyp, spec, reason)))
            memory.record(hyp.id, False, reason)
            continue
        result = run_experiment(spec, profile, hyp, train, budget)
        history.append((hyp, result))
        if result.status != ExperimentStatus.OK:
            log.info("  ✗ %s: %s", result.status.value, result.error)
            memory.record(hyp.id, False, result.status.value)
            continue
        if result.beats(best[1], gib, min_delta(best[1])):
            log.info("  ✓ лучше: %.5f → %.5f", best[1].mean_score, result.mean_score)
            memory.record(hyp.id, True)
            best = (hyp, result)
            on_new_best(hyp, result)
        else:
            log.info("  · не лучше: %.5f (лучший %.5f)", result.mean_score, best[1].mean_score)
            memory.record(hyp.id, False, "не лучше лучшего с учётом шума")
    return best


def _skipped(hyp: Hypothesis, spec: TaskSpec, reason: str) -> ExperimentResult:
    return ExperimentResult(
        hypothesis_id=hyp.id, status=ExperimentStatus.SKIPPED, metric_name=spec.metric.name, error=reason
    )
