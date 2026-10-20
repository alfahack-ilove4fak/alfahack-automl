"""Сигнатуры функций, через которые модули общаются друг с другом.

Каждая функция реализована в своём модуле (колонка «где») и имеет заглушку в interfaces/stubs.py,
чтобы любой модуль можно было разрабатывать и тестировать, не дожидаясь остальных.

| функция               | где                         | владелец |
|-----------------------|-----------------------------|----------|
| parse_task            | agent/task_parser.py        | agent    |
| propose_hypotheses    | agent/hypotheses.py         | agent    |
| profile_data          | ml/profiling.py             | ml       |
| run_experiment        | ml/experiment.py            | ml       |
| fit_predict_final     | ml/experiment.py            | ml       |
| compute_metric        | evaluation/metrics.py       | qa       |
| validate_submission   | evaluation/submission.py    | qa       |
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    Hypothesis,
    RunBudget,
    TaskSpec,
    ValidationReport,
)


class ParseTask(Protocol):
    def __call__(self, task_md: Path) -> TaskSpec: ...


class ProposeHypotheses(Protocol):
    def __call__(
        self, spec: TaskSpec, profile: DataProfile, history: list[tuple[Hypothesis, ExperimentResult]], k: int
    ) -> list[Hypothesis]: ...


class ProfileData(Protocol):
    def __call__(self, spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> DataProfile: ...


class RunExperiment(Protocol):
    def __call__(
        self,
        spec: TaskSpec,
        profile: DataProfile,
        hypothesis: Hypothesis,
        train: pd.DataFrame,
        budget: RunBudget,
    ) -> ExperimentResult: ...


class FitPredictFinal(Protocol):
    """Обучает лучшую гипотезу на всём train и возвращает предсказания для test (в формате сабмита)."""

    def __call__(
        self,
        spec: TaskSpec,
        profile: DataProfile,
        hypothesis: Hypothesis,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> pd.DataFrame: ...


class ComputeMetric(Protocol):
    def __call__(self, name: str, y_true, y_pred) -> float: ...


class ValidateSubmission(Protocol):
    def __call__(self, path: Path, spec: TaskSpec, test: pd.DataFrame) -> ValidationReport: ...
