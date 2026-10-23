"""Локальный балл по формуле организаторов (приложение 3 кейса) и наш простой бейзлайн вместо их бейзлайна.

Формула организаторов:
  регрессия (RMSE):        S = 100 · max(0, 1 − RMSE / B_RMSE)
  классификация (MacroF1): S = 100 · max(0, (F1 − B_F1) / (1 − B_F1))
Обобщаем на любую метрику реестра: «меньше — лучше» → как RMSE, «больше — лучше» → как F1
(у таких метрик потолок 1: F1, accuracy, ROC-AUC, R², Gini, MCC, QWK…).

B организаторов неизвестен, поэтому считаем два балла:
  score       — относительно тривиального ответа (среднее / частый класс / доли классов). Сильно завышен:
                годится только для сравнения версий агента между собой;
  model_score — относительно сильного бейзлайна: лучшего из LightGBM по умолчанию (без подбора и признаков)
                и тривиального ответа. Организаторы описали свой бейзлайн как «простейшую модель, которую DS
                обучил руками за 5 минут» — человек не сдал бы модель хуже среднего. Так на public/regression
                (отложены последние годы, LightGBM хуже среднего) средний балл двух задач 18.0 при public 17.46.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS, compute_metric
from interfaces.schemas import TaskSpec, TaskType

BEST_FOR_GREATER = 1.0  # потолок метрик «больше — лучше»
SEED = 42
MODEL_THREADS = 4
# Без force_*_wise LightGBM при старте замеряет row-wise и col-wise и берёт быстрый — выбор зависит от нагрузки
# машины, а ответы различаются: на public/multiclass F1 0.504 против 0.521. Бейзлайн обязан быть одинаковым
# в прогонах «было» и «стало», иначе таблица харнесса шумит (#219)
DETERMINISTIC = {"deterministic": True, "force_row_wise": True}


def organizer_score(metric_name: str, value: float, baseline: float) -> float:
    """Балл 0–100: насколько мы лучше бейзлайна, по формуле организаторов."""
    if not (np.isfinite(value) and np.isfinite(baseline)):
        return 0.0
    if METRICS[metric_name].greater_is_better:
        room = BEST_FOR_GREATER - baseline
        return float(100 * max(0.0, (value - baseline) / room)) if room > 0 else 0.0
    return float(100 * max(0.0, 1 - value / baseline)) if baseline > 0 else 0.0


def _encode(spec: TaskSpec, labels: pd.Series) -> np.ndarray:
    """Метки → коды 0..k-1 в порядке spec.class_labels (так же, как в ml/experiment.py)."""
    index = {label: i for i, label in enumerate(spec.class_labels or [])}
    codes = labels.astype(str).map(index)
    if codes.isna().any():
        bad = sorted(set(labels.astype(str)[codes.isna()]))[:5]
        raise ValueError(f"метки вне списка классов: {bad}")
    return codes.to_numpy(dtype=int)


def prediction_for_metric(spec: TaskSpec, pred: pd.Series) -> np.ndarray:
    """Ответ из submission.csv → вход метрики (число / код класса / вероятности)."""
    kind = METRICS[spec.metric.name].prediction
    if kind == "value":
        return pd.to_numeric(pred).to_numpy(dtype=float)
    if kind == "label":
        return _encode(spec, pred)
    if spec.task_type == TaskType.BINARY:
        return pd.to_numeric(pred).to_numpy(dtype=float)  # вероятность «положительного» (последнего) класса
    # Многоклассовая задача с метрикой по вероятностям: в сабмите метка → «уверенная» вероятность
    return np.eye(len(spec.class_labels or []))[_encode(spec, pred)]


def target_for_metric(spec: TaskSpec, y: pd.Series) -> np.ndarray:
    if METRICS[spec.metric.name].prediction == "value":
        return pd.to_numeric(y).to_numpy(dtype=float)
    return _encode(spec, y)


def trivial_prediction(spec: TaskSpec, y_train: pd.Series, n: int) -> np.ndarray:
    """Наш бейзлайн: ответ без модели в формате входа метрики."""
    kind = METRICS[spec.metric.name].prediction
    if kind == "value":
        return np.full(n, float(pd.to_numeric(y_train).mean()))
    codes = _encode(spec, y_train)
    k = len(spec.class_labels or [])
    if kind == "label":
        return np.full(n, np.bincount(codes, minlength=k).argmax())
    prior = np.bincount(codes, minlength=k) / len(codes)
    return np.full(n, prior[-1]) if spec.task_type == TaskType.BINARY else np.tile(prior, (n, 1))


def _model_frames(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Общие колонки; текстовые → category с категориями из train (новые значения в test → пропуск)."""
    cols = [c for c in x_train.columns if c in x_test.columns]
    train, test = x_train[cols].copy(), x_test[cols].copy()
    for col in cols:
        if not (pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_bool_dtype(train[col])):
            train[col] = train[col].astype(str).astype("category")
            test[col] = pd.Categorical(test[col].astype(str), categories=train[col].cat.categories)
    return train, test


def model_prediction(spec: TaskSpec, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    """Модельный бейзлайн: LightGBM по умолчанию, ответ в формате входа метрики (как trivial_prediction)."""
    import lightgbm as lgb  # тяжёлый импорт — только когда бейзлайн реально считаем

    train, test = _model_frames(x_train, x_test)
    params = {"random_state": SEED, "n_jobs": MODEL_THREADS, "verbose": -1, **DETERMINISTIC}
    kind = METRICS[spec.metric.name].prediction
    if kind == "value":
        model = lgb.LGBMRegressor(**params).fit(train, pd.to_numeric(y_train).to_numpy(dtype=float))
        return model.predict(test)
    codes = _encode(spec, y_train)
    model = lgb.LGBMClassifier(**params).fit(train, codes)
    if kind == "label":
        return model.predict(test).astype(int)
    proba = np.zeros((len(test), len(spec.class_labels or [])))
    proba[:, model.classes_] = model.predict_proba(test)  # классы, которых нет в train, — вероятность 0
    return proba[:, -1] if spec.task_type == TaskType.BINARY else proba


def score_submission(
    spec: TaskSpec,
    answers: pd.Series,
    submission: pd.Series,
    y_train: pd.Series,
    x_train: pd.DataFrame | None = None,
    x_test: pd.DataFrame | None = None,
) -> dict:
    """Метрика агента, метрики бейзлайнов и баллы. answers, submission и x_test — в одном порядке строк.

    Сильный бейзлайн считаем, только если переданы признаки x_train/x_test: model_baseline — лучший из LightGBM
    и тривиального ответа. Ошибка LightGBM не роняет оценку: model_baseline и model_score = NaN, причина —
    в model_error.
    """
    y = target_for_metric(spec, answers)
    value = compute_metric(spec.metric.name, y, prediction_for_metric(spec, submission))
    baseline = compute_metric(spec.metric.name, y, trivial_prediction(spec, y_train, len(y)))
    result = {
        "metric": spec.metric.name,
        "value": float(value),
        "baseline": float(baseline),
        "score": organizer_score(spec.metric.name, value, baseline),
    }
    if x_train is not None and x_test is not None:
        try:
            lgbm = float(compute_metric(spec.metric.name, y, model_prediction(spec, x_train, y_train, x_test)))
            better = max if METRICS[spec.metric.name].greater_is_better else min
            model_baseline = better(lgbm, float(baseline)) if np.isfinite(lgbm) else float(baseline)
            result["model_baseline"] = model_baseline
            result["model_score"] = organizer_score(spec.metric.name, value, model_baseline)
        except Exception as exc:  # noqa: BLE001 — бейзлайн вспомогательный, без него оценка агента остаётся
            result.update(
                model_baseline=float("nan"), model_score=float("nan"), model_error=f"{type(exc).__name__}: {exc}"
            )
    return result
