"""Цикл AutoResearch: гипотеза → эксперимент → оценка → решение. Владелец — LLM/agent-инженер.

Правило принятия — agent/acceptance.py: попарное сравнение по фолдам со скорректированным t-тестом
(ADR 0011). Это защита от «улучшений», которые являются шумом.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from agent.acceptance import accept
from agent.critic import estimated_seconds, review
from agent.fallback import fallback_hypotheses, is_fallback
from agent.feature_code import MAX_FEATURE_TRIES, propose_feature_hypothesis
from agent.hypotheses import propose_hypotheses
from agent.llm_client import LLMClient
from agent.llm_critic import critique
from agent.llm_hypotheses import reflect
from agent.memory import ResearchMemory
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, Hypothesis, RunBudget, TaskSpec
from ml.experiment import run_experiment
from runtime.log import get_logger

log = get_logger("loop")

BATCH = 3  # гипотез за один запрос к LLM: дешевле, чем спрашивать на каждой итерации


def run_autoresearch(
    spec: TaskSpec,
    profile: DataProfile,
    train: pd.DataFrame,
    budget: RunBudget,
    history: list[tuple[Hypothesis, ExperimentResult]],
    best: tuple[Hypothesis, ExperimentResult],
    on_new_best: Callable[[Hypothesis, ExperimentResult], None],
    max_iterations: int = 200,  # с запасными направлениями (#200) предел — время, а не число итераций
    llm: LLMClient | None = None,
    memory: ResearchMemory | None = None,
) -> tuple[Hypothesis, ExperimentResult]:
    gib = spec.metric.greater_is_better
    if llm is None:
        llm = LLMClient(rub_limit=budget.llm_rub_limit)  # без ключа или с LLM_MOCK=1 — сразу недоступна
    memory = memory if memory is not None else ResearchMemory()
    if memory.best_id is None:
        memory.best_id = best[0].id  # на входе лучший — бейзлайн
    queue: list[Hypothesis] = []
    feature_tries = 0  # сколько раз просили у LLM код признаков
    batch_start = len(history)  # с какого места истории начинается текущая пачка гипотез
    baseline_seconds = best[1].fit_seconds  # на входе best — бейзлайн: от него считаем цену гипотез
    # Схему валидации выбрали один раз до цикла и посчитали в ней бейзлайн (#151, ADR 0008 D1, ADR 0013):
    # все гипотезы идут в ней же, иначе acceptance не сравнит их с бейзлайном
    validation = best[0].params.get("validation")
    for iteration in range(1, max_iterations + 1):
        # Не успеваем даже эксперимент ценой в бейзлайн — дальше пробовать нечего
        if not budget.has_time(needed_seconds=baseline_seconds * 1.2):
            log.info("Время на исходе — завершаю цикл после %d итераций", iteration - 1)
            break
        if not queue:
            batch = history[batch_start:]
            missed = batch and not any(memory.verdicts.get(h.id, "").startswith("принята") for h, _ in batch)
            # пачку запасных направлений придумали правила, а не LLM — рефлексия по ней тратит токены впустую
            if missed and not all(is_fallback(h) for h, _ in batch):
                reflect(spec, best, batch, llm, memory, budget)  # вся пачка мимо — спросим LLM, почему
            queue = propose_hypotheses(spec, profile, history, k=BATCH, budget=budget, llm=llm, memory=memory)
            if batch and feature_tries < MAX_FEATURE_TRIES:
                # после первой пачки, когда лучший известен: LLM пишет код признаков (A-L3-1), он идёт в песочнице
                feature_tries += 1
                feature_hyp = propose_feature_hypothesis(spec, profile, history, llm, best[0], budget)
                if feature_hyp is not None:
                    queue.append(feature_hyp)
            batch_start = len(history)
            rejected = critique(spec, profile, history, queue, llm, budget, memory)  # выключен без AGENT_LLM_CRITIC=1
            for hyp in [h for h in queue if h.id in rejected]:
                reason = f"критик LLM: {rejected[hyp.id]}"
                log.info("  ⊘ %s — %s", hyp.id, reason)
                history.append((hyp, _skipped(hyp, spec, reason)))
                memory.record(hyp.id, False, reason)
            queue = [h for h in queue if h.id not in rejected]
        if not queue:
            # playbook и LLM пусты, а время есть — не выходим, а берём запасные направления (#200)
            queue = fallback_hypotheses(spec, history, best, budget)
            if queue:
                log.info("Playbook и LLM исчерпаны — запасные направления: %d гипотез", len(queue))
        if not queue:
            log.info("Гипотезы закончились — завершаю цикл")
            break
        hyp = queue.pop(0)
        log.info("Итерация %d: %s — %s [%s]", iteration, hyp.id, hyp.title, hyp.source)
        reason = review(hyp, spec) or foreign_validation(hyp, validation)
        needed = estimated_seconds(hyp, baseline_seconds)
        if reason is None and not budget.has_time(needed_seconds=needed):
            reason = f"не успеет: оценка {needed:.0f} с, осталось {budget.seconds_left():.0f} с"
        if reason is not None:
            # в историю со статусом SKIPPED: гипотезу не предложат снова, а LLM увидит причину
            log.info("  ⊘ пропуск критиком: %s", reason)
            history.append((hyp, _skipped(hyp, spec, reason)))
            memory.record(hyp.id, False, reason)
            continue
        # run_experiment пишет в params запущенной копии _best_iteration, _proba_weights, _optuna_best —
        # финал должен обучаться на ней, а не на исходной гипотезе (#206: иначе 550 деревьев и без весов классов)
        run_hyp = with_validation(hyp, validation)
        result = run_experiment(spec, profile, run_hyp, train, budget)
        history.append((hyp, result))
        if result.status != ExperimentStatus.OK:
            log.info("  ✗ %s: %s", result.status.value, result.error)
            memory.record(hyp.id, False, result.status.value)
            continue
        accepted, reason = accept(result, best[1], gib)
        memory.record(hyp.id, accepted, reason)  # причина (t, k/5 фолдов) видна LLM в истории
        if accepted:
            log.info("  ✓ лучше: %.5f → %.5f — %s", best[1].mean_score, result.mean_score, reason)
            best = (run_hyp, result)
            on_new_best(run_hyp, result)
        else:
            log.info("  · не лучше: %.5f (лучший %.5f) — %s", result.mean_score, best[1].mean_score, reason)
    llm.log_summary()  # токены и время LLM — тай-брейк при равном балле; журнал вызовов — llm_calls.jsonl
    log.info(llm_decisions(history, memory))
    return best


def llm_decisions(history: list[tuple[Hypothesis, ExperimentResult]], memory: ResearchMemory) -> str:
    """Итог решений LLM за прогон: что предложила и что из этого вышло.

    Вебинар: детерминированный агент «не имеет смысла», топ проверят вручную — по этой строке в agent.log видно,
    какие гипотезы придумала LLM (включая код признаков) и какие из них агент принял.
    """
    ran = [(h, r) for h, r in history if h.source == "llm"]
    if not ran:
        return "Решения LLM: гипотез от LLM не было — работали playbook и правила"
    accepted = [h.id for h, _ in ran if memory.verdicts.get(h.id, "").startswith("принята")]
    skipped = sum(r.status == ExperimentStatus.SKIPPED for _, r in ran)
    failed = sum(r.status in (ExperimentStatus.FAILED, ExperimentStatus.TIMEOUT) for _, r in ran)
    worse = len(ran) - len(accepted) - skipped - failed
    taken = f"принято {len(accepted)} ({', '.join(accepted)})" if accepted else "принято 0"
    return (
        f"Решения LLM: предложено {len(ran)} → {taken}, не лучше {worse}, "
        f"пропущено критиком {skipped}, упало {failed}"
    )


def _skipped(hyp: Hypothesis, spec: TaskSpec, reason: str) -> ExperimentResult:
    return ExperimentResult(
        hypothesis_id=hyp.id, status=ExperimentStatus.SKIPPED, metric_name=spec.metric.name, error=reason
    )


def with_validation(hyp: Hypothesis, validation: str | None) -> Hypothesis:
    """Гипотеза в схеме валидации бейзлайна. В историю идёт исходная гипотеза: её params — то, что предложено."""
    if validation is None or "validation" in hyp.params:
        return hyp  # своя схема гипотезы — acceptance отклонит её как несравнимую, но не подменяем молча
    return hyp.model_copy(update={"params": {**hyp.params, "validation": validation}})


PLAIN_VALIDATION = ("kfold", "stratified")  # обычная CV — то же, что без validation


def foreign_validation(hyp: Hypothesis, validation: str | None) -> str | None:
    """Причина пропустить гипотезу со своей схемой валидации: результат несравним с бейзлайном (ADR 0008 D1).

    Без проверки она отработала бы целый эксперимент и только потом была бы отклонена правилом принятия.
    """
    own = hyp.params.get("validation")
    if own is None or own == validation or (validation is None and own in PLAIN_VALIDATION):
        return None
    return f"своя схема валидации {own} ≠ схемы прогона {validation or 'обычная CV'} — результаты несравнимы"
