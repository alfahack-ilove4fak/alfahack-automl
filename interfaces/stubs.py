"""Заглушки: «фальшивые» реализации контрактов для параллельной разработки и тестов.

Пример: ML-инженер пишет run_experiment и подставляет stub_propose_hypotheses вместо
настоящего агента; агент-инженер отлаживает цикл с stub_run_experiment, не обучая модели.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from interfaces.schemas import (
    CodeResult,
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    MetricSpec,
    RunBudget,
    TaskSpec,
    TaskType,
    ValidationReport,
)


def stub_task_spec(task_dir: Path | None = None) -> TaskSpec:
    task_dir = task_dir or Path("tasks/public/regression")
    return TaskSpec(
        task_md=task_dir / "task.md",
        task_dir=task_dir,
        train_path=task_dir / "train.csv",
        test_path=task_dir / "test.csv",
        submission_path=task_dir / "submission.csv",
        task_type=TaskType.REGRESSION,
        target="site_eui",
        id_column="id",
        metric=MetricSpec(name="rmse", greater_is_better=False),
        submission_columns=["id", "site_eui"],
    )


def stub_profile_data(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> DataProfile:
    features = [c for c in train.columns if c not in (spec.target, spec.id_column)]
    return DataProfile(n_train=len(train), n_test=len(test), numeric_features=features, categorical_features=[])


def stub_propose_hypotheses(spec, profile, history, k: int = 3, budget: RunBudget | None = None) -> list[Hypothesis]:
    start = len(history)
    return [
        Hypothesis(id=f"stub-{start + i}", kind=HypothesisKind.HYPERPARAMS, title=f"Заглушка №{start + i}")
        for i in range(k)
    ]


def stub_run_experiment(spec, profile, hypothesis, train, budget: RunBudget) -> ExperimentResult:
    rnd = random.Random(hypothesis.id)
    scores = [rnd.uniform(0.5, 0.9) for _ in range(5)]
    mean = sum(scores) / len(scores)
    return ExperimentResult(
        hypothesis_id=hypothesis.id,
        status=ExperimentStatus.OK,
        metric_name=spec.metric.name,
        fold_scores=scores,
        mean_score=mean,
        std_score=(sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5,
    )


def stub_fit_predict_final(spec, profile, hypothesis, train, test) -> pd.DataFrame:
    value = 0.0 if spec.task_type == TaskType.REGRESSION else (spec.class_labels or ["0"])[0]
    return pd.DataFrame({spec.id_column: test[spec.id_column], spec.target: value})


def stub_run_code(code: str, workdir: Path, timeout: float, mem_mb: int) -> CodeResult:
    """Заглушка песочницы: ничего не исполняет, сообщает об успехе."""
    return CodeResult(exit_code=0, stdout="")


def stub_validate_submission(path: Path, spec: TaskSpec, test: pd.DataFrame) -> ValidationReport:
    return ValidationReport(ok=path.exists(), problems=[] if path.exists() else ["файла нет"])
