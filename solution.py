"""Точка входа для платформы Альфа Хак. Владелец — капитан.

Интерфейс как в baseline.py организаторов:
    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... python solution.py path/to/task.md
Результат: submission.csv рядом с task.md.

Стратегия «сначала надёжно, потом умно»:
  0. сразу пишем тривиальный ответ (среднее / самый частый класс) — файл есть всегда;
  1. обучаем надёжный бейзлайн и перезаписываем ответ;
  2. цикл AutoResearch улучшает модель, пока есть время; каждый новый лучший сразу сохраняется.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from agent.hypotheses import baseline_hypothesis
from agent.loop import run_autoresearch
from agent.task_parser import parse_task
from evaluation.metrics import METRICS
from evaluation.submission import validate_submission, write_submission_atomic
from interfaces.schemas import DataProfile, ExperimentResult, Hypothesis, TaskSpec, TaskType
from ml.experiment import fit_predict_final, run_experiment
from ml.profiling import load_data, profile_data
from runtime.budget import make_budget
from runtime.log import setup_logging


def trivial_answer(spec: TaskSpec, train, test):
    """Ответ без модели: страховка на случай, если всё остальное упадёт."""
    y = train[spec.target]
    if spec.task_type == TaskType.REGRESSION:
        value = float(y.mean())
    elif METRICS[spec.metric.name].prediction == "proba" and spec.class_labels:
        value = float((y.astype(str) == spec.class_labels[-1]).mean())  # доля «положительного» класса
    else:
        value = str(y.astype(str).mode()[0])
    return test[[spec.id_column]].assign(**{spec.target: value})[spec.submission_columns]


def main(argv: list[str]) -> int:
    started = time.time()
    run_dir = Path(os.environ["AGENT_RUN_DIR"]) if os.getenv("AGENT_RUN_DIR") else None
    log = setup_logging(run_dir)
    budget = make_budget(started)
    task_md = Path(argv[1] if len(argv) > 1 else "task.md")
    log.info("Старт: %s, бюджет %.0f с, RAM %.0f МБ", task_md, budget.total_seconds, budget.ram_limit_mb)

    spec = parse_task(task_md)
    train, test = load_data(spec)
    write_submission_atomic(trivial_answer(spec, train, test), spec.submission_path)
    log.info("Шаг 0: тривиальный ответ записан")

    def save(hyp: Hypothesis, res: ExperimentResult, profile: DataProfile) -> None:
        try:
            answer = fit_predict_final(spec, profile, hyp, train, test)
            write_submission_atomic(answer[spec.submission_columns], spec.submission_path)
            report = validate_submission(spec.submission_path, spec, test)
            log.info("Сохранён ответ «%s» (CV %s=%.5f), проверка: %s", hyp.id, res.metric_name, res.mean_score,
                     "OK" if report.ok else report.problems)
        except Exception:  # noqa: BLE001 — старый валидный ответ остаётся на месте
            log.exception("Не удалось сохранить ответ гипотезы %s", hyp.id)

    profile = profile_data(spec, train, test)
    base = baseline_hypothesis()
    base_result = run_experiment(spec, profile, base, train, budget)
    log.info("Шаг 1: бейзлайн %s=%s ± %s", base_result.metric_name, base_result.mean_score, base_result.std_score)
    history = [(base, base_result)]
    if base_result.mean_score is None:
        log.error("Бейзлайн не обучился — остаёмся с тривиальным ответом")
        return 0
    save(base, base_result, profile)

    best = run_autoresearch(spec, profile, train, budget, history, (base, base_result),
                            on_new_best=lambda h, r: save(h, r, profile))
    log.info("Готово за %.0f с. Лучшая гипотеза: %s (%s=%.5f), всего экспериментов: %d",
             time.time() - started, best[0].id, best[1].metric_name, best[1].mean_score, len(history))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
