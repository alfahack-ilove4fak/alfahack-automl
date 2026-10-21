"""Цикл AutoResearch: гипотеза → эксперимент → оценка → решение. Владелец — LLM/agent-инженер.

Правило принятия — agent/acceptance.py: попарное сравнение по фолдам со скорректированным t-тестом
(ADR 0008). Это защита от «улучшений», которые являются шумом.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from agent.acceptance import accept
from agent.hypotheses import propose_hypotheses
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, Hypothesis, RunBudget, TaskSpec
from ml.experiment import run_experiment
from runtime.log import get_logger

log = get_logger("loop")


def run_autoresearch(
    spec: TaskSpec,
    profile: DataProfile,
    train: pd.DataFrame,
    budget: RunBudget,
    history: list[tuple[Hypothesis, ExperimentResult]],
    best: tuple[Hypothesis, ExperimentResult],
    on_new_best: Callable[[Hypothesis, ExperimentResult], None],
    max_iterations: int = 50,
) -> tuple[Hypothesis, ExperimentResult]:
    gib = spec.metric.greater_is_better
    for iteration in range(1, max_iterations + 1):
        # Новый эксперимент стоит примерно как прошлый: если не успеем — не начинаем
        if not budget.has_time(needed_seconds=best[1].fit_seconds * 1.2):
            log.info("Время на исходе — завершаю цикл после %d итераций", iteration - 1)
            break
        candidates = propose_hypotheses(spec, profile, history, k=1)
        if not candidates:
            log.info("Гипотезы закончились — завершаю цикл")
            break
        hyp = candidates[0]
        log.info("Итерация %d: %s — %s", iteration, hyp.id, hyp.title)
        result = run_experiment(spec, profile, hyp, train, budget)
        history.append((hyp, result))
        if result.status != ExperimentStatus.OK:
            log.info("  ✗ %s: %s", result.status.value, result.error)
            continue
        accepted, reason = accept(result, best[1], gib)
        if accepted:
            log.info("  ✓ лучше: %.5f → %.5f — %s", best[1].mean_score, result.mean_score, reason)
            best = (hyp, result)
            on_new_best(hyp, result)
        else:
            log.info("  · не лучше: %.5f (лучший %.5f) — %s", result.mean_score, best[1].mean_score, reason)
    return best
