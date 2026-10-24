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

CONTRACT_VERSION = "1.1.2"

ROW_ID_COLUMN = "__row__"  # id_column задачи, где в test нет id, а в ответе нет колонки id: номер строки test (1.1.2)


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
    target: str  # основной таргет; по нему идут CV и цикл AutoResearch
    extra_targets: list[str] = Field(default_factory=list)  # остальные таргеты, если их несколько (#174, 1.1.1)
    id_column: str  # ROW_ID_COLUMN — id в test нет и в ответ он не входит: load_data добавляет колонку в test (1.1.2)
    metric: MetricSpec
    submission_columns: list[str]
    class_labels: list[str] | None = None  # для классификации — допустимые метки
    description: str = ""  # текст task.md
    notes: list[str] = Field(default_factory=list)  # что заметили при разборе


# ---------------------------------------------------------------- данные


class DataProfile(_Contract):
    """Паспорт данных. Создаёт ml (profile_data), читает agent для гипотез.

    target_summary (часть контракта, ADR 0008, D5): регрессия — `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`,
    `skew`; классификация — `class_counts` и `imbalance_ratio` (наибольший класс / наименьший; его читает playbook).
    """

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

    params — словарь, который понимает ml.run_experiment: список ключей — SUPPORTED_PARAMS в ml/experiment.py
      (пример: {"model": "lightgbm", "model_params": {...}, "n_folds": 5}).

    code — код признаков от LLM (ADR 0008, D4): обязан определять `add_features(df) -> df`; получает таблицу признаков
    train и test вместе БЕЗ target (иначе утечка) и возвращает её с новыми колонками, не меняя число строк и их порядок.
    Исполняется только в песочнице (runtime/sandbox.py), результат запуска — CodeResult.
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
    """Чем закончился эксперимент. Создаёт ml, читают agent и evaluation.

    validation — схема кросс-валидации в формате `<схема>-<число фолдов>` (`kfold-5`, `group-5`). Результаты разных
    схем несравнимы, поэтому `beats()` их не сравнивает: схему выбираем один раз до цикла (ADR 0008, D1).
    refit_params — что нужно знать финальному обучению (сейчас `n_estimators`); solution.py подставляет их в копию
    гипотезы перед `fit_predict_final` (ADR 0008, D2).
    oof_path и test_pred_path — файлы `.npy`, строки в порядке train и test; регрессия — вектор `(n,)`,
    классификация — вероятности `(n, n_классов)` в порядке `TaskSpec.class_labels` (ADR 0008, D5).
    """

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
    validation: str = "kfold-5"
    refit_params: dict[str, Any] = Field(default_factory=dict)

    def beats(self, other: ExperimentResult | None, greater_is_better: bool, min_delta: float = 0.0) -> bool:
        """Лучше ли этот результат, чем other, с запасом min_delta (защита от шума). Разные схемы CV — не лучше."""
        if self.status != ExperimentStatus.OK or self.mean_score is None:
            return False
        if other is None or other.mean_score is None:
            return True
        if self.validation != other.validation:
            return False
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


class CodeResult(_Contract):
    """Результат запуска кода в песочнице (runtime/sandbox.py). Создаёт captain, читает agent (ADR 0008, D4)."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    seconds: float = 0.0
