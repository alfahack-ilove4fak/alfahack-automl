"""Эксперимент = гипотеза → кросс-валидация → ExperimentResult. Владелец — ML-pipeline инженер.

Поддерживаемые params гипотезы (расширяйте — и документируйте в research/playbook.yaml):
  model          — "lightgbm" (пока единственная; добавить catboost/xgboost — задачи L1/L2 ML-инженера)
  model_params   — словарь гиперпараметров модели
  n_folds        — число фолдов (по умолчанию 5)
  log_target     — для регрессии: учиться на log1p(y), предсказывать expm1
  class_weight   — для классификации: "balanced" или None
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from evaluation.metrics import METRICS, compute_metric
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    RunBudget,
    TaskSpec,
    TaskType,
)
from ml.profiling import make_matrix
from runtime.budget import used_ram_mb
from runtime.log import get_logger

log = get_logger("experiment")

SEED = 42
BASE_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}
EARLY_STOPPING = 100


def _is_classification(spec: TaskSpec) -> bool:
    return spec.task_type in (TaskType.BINARY, TaskType.MULTICLASS)


def _encode_target(spec: TaskSpec, y: pd.Series) -> np.ndarray:
    if _is_classification(spec):
        index = {label: i for i, label in enumerate(spec.class_labels or [])}
        return y.astype(str).map(index).to_numpy()
    return y.to_numpy(dtype=float)


def _make_model(spec: TaskSpec, hypothesis: Hypothesis):
    params = {**BASE_PARAMS, **hypothesis.params.get("model_params", {})}
    if _is_classification(spec):
        return lgb.LGBMClassifier(class_weight=hypothesis.params.get("class_weight"), **params)
    return lgb.LGBMRegressor(**params)


def _predict(spec: TaskSpec, model, x: pd.DataFrame, log_target: bool):
    """Предсказание в том виде, который нужен метрике задачи."""
    if not _is_classification(spec):
        pred = model.predict(x)
        return np.expm1(pred) if log_target else pred
    proba = model.predict_proba(x)
    if METRICS[spec.metric.name].prediction == "proba":
        return proba
    return proba.argmax(axis=1)


def run_experiment(
    spec: TaskSpec, profile: DataProfile, hypothesis: Hypothesis, train: pd.DataFrame, budget: RunBudget
) -> ExperimentResult:
    started = time.time()
    peak = used_ram_mb()
    params = hypothesis.params
    log_target = bool(params.get("log_target")) and spec.task_type == TaskType.REGRESSION
    n_folds = int(params.get("n_folds", 5))
    try:
        x, _ = make_matrix(profile, train, train.head(0))
        y = _encode_target(spec, train[spec.target])
        y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
        splitter = (StratifiedKFold if _is_classification(spec) else KFold)(n_folds, shuffle=True, random_state=SEED)
        scores, best_iters = [], []
        for fold, (tr, va) in enumerate(splitter.split(x, y)):
            if not budget.has_time():
                return ExperimentResult(
                    hypothesis_id=hypothesis.id,
                    status=ExperimentStatus.TIMEOUT,
                    metric_name=spec.metric.name,
                    fold_scores=scores,
                    fit_seconds=time.time() - started,
                    error=f"кончилось время на фолде {fold}",
                )
            model = _make_model(spec, hypothesis)
            model.fit(
                x.iloc[tr],
                y_fit[tr],
                eval_set=[(x.iloc[va], y_fit[va])],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            best_iters.append(model.best_iteration_ or BASE_PARAMS["n_estimators"])
            scores.append(compute_metric(spec.metric.name, y[va], _predict(spec, model, x.iloc[va], log_target)))
            peak = max(peak, used_ram_mb())
        hypothesis.params.setdefault("_best_iteration", int(np.mean(best_iters)))
        return ExperimentResult(
            hypothesis_id=hypothesis.id,
            status=ExperimentStatus.OK,
            metric_name=spec.metric.name,
            fold_scores=[float(s) for s in scores],
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            fit_seconds=time.time() - started,
            peak_ram_mb=peak,
        )
    except Exception as exc:  # noqa: BLE001 — упавшая гипотеза не должна ронять агента
        log.exception("Гипотеза %s упала", hypothesis.id)
        return ExperimentResult(
            hypothesis_id=hypothesis.id,
            status=ExperimentStatus.FAILED,
            metric_name=spec.metric.name,
            fit_seconds=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def fit_predict_final(
    spec: TaskSpec, profile: DataProfile, hypothesis: Hypothesis, train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    """Обучение на всём train с числом деревьев из CV и предсказание test в формате сабмита."""
    params = hypothesis.params
    log_target = bool(params.get("log_target")) and spec.task_type == TaskType.REGRESSION
    x, x_test = make_matrix(profile, train, test)
    y = _encode_target(spec, train[spec.target])
    y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
    model = _make_model(spec, hypothesis)
    # На всех данных деревьев нужно чуть больше, чем на 4/5 из CV
    model.set_params(n_estimators=max(50, int(params.get("_best_iteration", 500) * 1.1)))
    model.fit(x, y_fit)
    pred = _predict(spec, model, x_test, log_target)

    if _is_classification(spec):
        labels = np.asarray(spec.class_labels)
        if METRICS[spec.metric.name].prediction == "proba":
            # Бинарная задача с ROC-AUC/logloss: в ответ — вероятность «положительного» класса
            answer = pred[:, -1] if spec.task_type == TaskType.BINARY else labels[pred.argmax(axis=1)]
        else:
            answer = labels[pred]
    else:
        answer = pred
    return pd.DataFrame({spec.id_column: test[spec.id_column].to_numpy(), spec.target: answer})
