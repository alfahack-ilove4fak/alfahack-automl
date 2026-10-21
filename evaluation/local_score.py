"""Локальный балл по формуле организаторов (приложение 3 кейса) и наш простой бейзлайн вместо их бейзлайна.

Формула организаторов:
  регрессия (RMSE):        S = 100 · max(0, 1 − RMSE / B_RMSE)
  классификация (MacroF1): S = 100 · max(0, (F1 − B_F1) / (1 − B_F1))
Обобщаем на любую метрику реестра: «меньше — лучше» → как RMSE, «больше — лучше» → как F1
(у таких метрик потолок 1: F1, accuracy, ROC-AUC, R², Gini, MCC, QWK…).

B организаторов неизвестен — берём тривиальный ответ (среднее / частый класс / доли классов).
Поэтому абсолютный балл не равен баллу платформы: сравниваем версии агента между собой.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS, compute_metric
from interfaces.schemas import TaskSpec, TaskType

BEST_FOR_GREATER = 1.0  # потолок метрик «больше — лучше»


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


def score_submission(spec: TaskSpec, answers: pd.Series, submission: pd.Series, y_train: pd.Series) -> dict:
    """Метрика агента, метрика бейзлайна и балл. answers и submission — в одном порядке строк."""
    y = target_for_metric(spec, answers)
    value = compute_metric(spec.metric.name, y, prediction_for_metric(spec, submission))
    baseline = compute_metric(spec.metric.name, y, trivial_prediction(spec, y_train, len(y)))
    return {
        "metric": spec.metric.name,
        "value": float(value),
        "baseline": float(baseline),
        "score": organizer_score(spec.metric.name, value, baseline),
    }
