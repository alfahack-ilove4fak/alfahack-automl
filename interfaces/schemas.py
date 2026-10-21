"""Контракты данных между модулями. Меняются ТОЛЬКО через PR с двумя одобрениями (затронутых владельцев зови ревьюерами).

Перед изменением: /sync-check в Claude Code. После изменения: поднять CONTRACT_VERSION
и записать изменение в interfaces/CHANGELOG.md.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0.0"


class _Contract(BaseModel):
    # extra="forbid": опечатка в имени поля сразу даёт ошибку, а не тихо теряется
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- задача


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"
    OTHER = "other"


class MetricSpec(_Contract):
    """Метрика, которую оптимизируем. name — ключ из evaluation.metrics.METRICS."""

    name: str
    greater_is_better: bool


class TaskSpec(_Contract):
    """Что за задача. Создаёт agent (parse_task), читают все."""

    task_md: Path
    task_dir: Path
    train_path: Path
    test_path: Path
    sample_submission_path: Path | None = None
    data_dictionary_path: Path | None = None
    submission_path: Path  # куда писать ответ: рядом с task.md
    task_type: TaskType
    target: str
    id_column: str
    metric: MetricSpec
    submission_columns: list[str]
    class_labels: list[str] | None = None  # для классификации — допустимые метки
    description: str = ""  # текст task.md
    notes: list[str] = Field(default_factory=list)  # что заметили при разборе


# ---------------------------------------------------------------- данные


class DataProfile(_Contract):
    """Паспорт данных. Создаёт ml (profile_data), читает agent для гипотез."""

    n_train: int
    n_test: int
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_features: list[str] = Field(default_factory=list)  # id, константы и т.п.
    missing_share: dict[str, float] = Field(default_factory=dict)  # только где > 0
    target_summary: dict[str, Any] = Field(default_factory=dict)
    memory_mb: float = 0.0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- гипотезы


class HypothesisKind(str, Enum):
    BASELINE = "baseline"
    MODEL = "model"
    HYPERPARAMS = "hyperparams"
    FEATURES = "features"
    PREPROCESSING = "preprocessing"
    VALIDATION = "validation"
    ENSEMBLE = "ensemble"


class Hypothesis(_Contract):
    """Что пробуем. Создаёт agent, исполняет ml.

    params — словарь, который понимает ml.run_experiment:
      {"model": "lightgbm", "model_params": {...}, "feature_ops": [...], "n_folds": 5}
    """

    id: str
    kind: HypothesisKind
    title: str
    rationale: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    code: str | None = None  # код генерации признаков от LLM (исполняется в песочнице)
    parent_id: str | None = None
    source: Literal["baseline", "playbook", "llm"] = "playbook"


class ExperimentStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ExperimentResult(_Contract):
    """Чем закончился эксперимент. Создаёт ml, читают agent и evaluation."""

    hypothesis_id: str
    status: ExperimentStatus
    metric_name: str
    fold_scores: list[float] = Field(default_factory=list)
    mean_score: float | None = None
    std_score: float | None = None
    fit_seconds: float = 0.0
    peak_ram_mb: float = 0.0
    error: str | None = None
    oof_path: Path | None = None
    test_pred_path: Path | None = None

    def beats(self, other: ExperimentResult | None, greater_is_better: bool, min_delta: float = 0.0) -> bool:
        """Лучше ли этот результат, чем other, с запасом min_delta (защита от шума)."""
        if self.status != ExperimentStatus.OK or self.mean_score is None:
            return False
        if other is None or other.mean_score is None:
            return True
        diff = self.mean_score - other.mean_score
        return diff > min_delta if greater_is_better else -diff > min_delta


# ---------------------------------------------------------------- ресурсы и проверка


class RunBudget(_Contract):
    """Сколько можно потратить. Создаёт runtime, уважают все."""

    started_at: float = Field(default_factory=time.time)
    total_seconds: float = 3 * 3600
    safety_seconds: float = 15 * 60  # запас на финальное обучение и запись ответа
    ram_limit_mb: float = 30_000
    llm_rub_limit: float = 0.0  # 0 — без ограничения на стороне агента
    llm_rub_spent: float = 0.0

    def seconds_left(self) -> float:
        return self.started_at + self.total_seconds - self.safety_seconds - time.time()

    def has_time(self, needed_seconds: float = 0.0) -> bool:
        return self.seconds_left() > needed_seconds


class ValidationReport(_Contract):
    """Результат проверки submission.csv. Создаёт evaluation."""

    ok: bool
    problems: list[str] = Field(default_factory=list)
    n_rows: int = 0
