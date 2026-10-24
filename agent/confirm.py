"""Подтверждение нового лучшего на отложенной части train (#276, гипотеза штрафа −7). Владелец — agent.

За 3 часа цикл сравнивает с лучшим сотни гипотез на одних и тех же фолдах, и часть принятых «выигрышей» — шум
подбора под эти фолды. Их ловим двумя способами (оба выключены по умолчанию, включает AGENT_CONFIRM=1):
  - порог принятия растёт с числом сравнений (agent/acceptance.multi_factor);
  - здесь: до цикла от train отрезается кусок, который цикл не видит (последние ≈ 20% по времени для схемы
    time:, целые группы дублей для group:, стратифицированная случайная выборка для остальных). CV в цикле идёт
    на остальной части, а принятая по CV гипотеза становится лучшей, только если и на отложенном куске она лучше
    текущего лучшего. Финальное обучение (solution.py) — как раньше, на всём train.

AGENT_LLM_CONFIRM=1 — подтверждать только гипотезы LLM (#276, п. 3: LLM — источник идей, решения — после проверки).

Обучение на части и оценка на куске повторяют ml.experiment.fit_predict_final; если подтверждение станет
постоянным, функцию стоит перенести в ml/ публичной (владелец — ml).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS, compute_metric
from interfaces.schemas import DataProfile, Hypothesis, TaskSpec
from ml.experiment import (
    _encode_target,
    _final_estimators,
    _fit_model,
    _full_proba,
    _hide_unseen,
    _is_classification,
    _predict,
    _trees_scale,
    _use_log_target,
    apply_feature_ops,
    make_matrix,
    with_feature_code,
)
from ml.validation import DUPLICATES, duplicate_groups, parse_validation, time_values
from runtime.log import get_logger

log = get_logger("confirm")

CONFIRM_ENV = "AGENT_CONFIRM"  # "1" — поправка на число сравнений, подтверждение на отложенной части, терпение
LLM_CONFIRM_ENV = "AGENT_LLM_CONFIRM"  # "1" — подтверждать только гипотезы LLM
PATIENCE_ENV = "AGENT_CONFIRM_PATIENCE"  # итераций подряд без принятия, после которых цикл останавливается
PATIENCE_DEFAULT = 50
HOLDOUT_SHARE = 0.2
HOLDOUT_MIN_SHARE, HOLDOUT_MAX_SHARE = 0.15, 0.35  # для time: — целые последние периоды в этих пределах
MIN_TRAIN_ROWS = 300  # меньше — кусок слишком шумный, и CV без него теряет больше, чем даёт проверка
MIN_HOLDOUT_ROWS = 60
SEED = 42


def enabled() -> bool:
    return os.getenv(CONFIRM_ENV, "0") == "1"


def mode() -> str | None:
    """ "all" — подтверждать всех новых лучших, "llm" — только гипотезы LLM, None — выключено."""
    if enabled():
        return "all"
    if os.getenv(LLM_CONFIRM_ENV, "0") == "1":
        return "llm"
    return None


def patience() -> int | None:
    """Сколько итераций подряд без принятия терпим; None — без остановки (AGENT_CONFIRM выключен)."""
    if not enabled():
        return None
    try:
        value = int(os.getenv(PATIENCE_ENV, PATIENCE_DEFAULT))
    except ValueError:
        return PATIENCE_DEFAULT
    return value if value > 0 else None


def split_holdout(spec: TaskSpec, profile: DataProfile, train: pd.DataFrame, validation: str | None):
    """(часть для CV, отложенный кусок) по схеме валидации прогона. None — отложить нельзя (мало строк/периодов)."""
    n = len(train)
    if n < MIN_TRAIN_ROWS:
        return None
    rng = np.random.default_rng(SEED)
    scheme, column = parse_validation(validation, spec)
    if scheme == "time" and column in train.columns:
        t = time_values(train[column])
        if t is None:
            return None
        periods, counts = np.unique(t[~pd.isna(t)], return_counts=True)
        taken, share = 0, 0.0
        for count in counts[::-1]:  # с самого позднего периода
            if share >= HOLDOUT_MIN_SHARE:
                break
            taken, share = taken + 1, share + count / n
        if share > HOLDOUT_MAX_SHARE or taken >= len(periods) - 1:  # для CV нужно хотя бы 2 периода
            return None
        mask = np.isin(t, periods[-taken:])
    elif scheme == "group":
        if column == DUPLICATES:
            features = [c for c in profile.numeric_features + profile.categorical_features if c in train.columns]
            groups = duplicate_groups(train[features])
        elif column in train.columns:
            groups = train[column].astype(str).to_numpy()
        else:
            return None
        unique = np.unique(groups)  # группа целиком либо в CV, либо в куске — как в фолдах
        chosen = rng.choice(unique, size=max(1, round(len(unique) * HOLDOUT_SHARE)), replace=False)
        mask = np.isin(groups, chosen)
    else:
        mask = np.zeros(n, dtype=bool)
        if _is_classification(spec):
            y = train[spec.target].astype(str).to_numpy()
            for label in np.unique(y):  # стратификация: от каждого класса ≈ 20%, редкие (< 5) остаются в CV
                idx = np.flatnonzero(y == label)
                if len(idx) >= 5:
                    mask[rng.choice(idx, size=round(len(idx) * HOLDOUT_SHARE), replace=False)] = True
        else:
            mask[rng.choice(n, size=round(n * HOLDOUT_SHARE), replace=False)] = True
    if mask.sum() < MIN_HOLDOUT_ROWS or (~mask).sum() < MIN_TRAIN_ROWS:
        return None
    return train[~mask].reset_index(drop=True), train[mask].reset_index(drop=True)


def holdout_score(
    spec: TaskSpec, profile: DataProfile, hypothesis: Hypothesis, fit_part: pd.DataFrame, holdout: pd.DataFrame
) -> float:
    """Метрика гипотезы на отложенном куске: обучение на fit_part с числом деревьев из CV, как в финале."""
    params = hypothesis.params
    log_target = _use_log_target(spec, params)
    targets = [spec.target, *spec.extra_targets]
    hidden = holdout.drop(columns=[c for c in targets if c in holdout.columns])
    profile, fit_part, hidden = with_feature_code(spec, profile, hypothesis, fit_part, hidden)
    x, x_hold = make_matrix(profile, fit_part, hidden)
    x_hold = _hide_unseen(hypothesis, x, x_hold)
    y = _encode_target(spec, fit_part[spec.target])
    y_hold = _encode_target(spec, holdout[spec.target])
    y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
    ops = params.get("feature_ops")
    if ops:
        x, x_hold = apply_feature_ops(ops, x, y, _is_classification(spec), x_hold)
    n_estimators = _final_estimators(params.get("_best_iteration"), _trees_scale(params))
    model, _ = _fit_model(spec, hypothesis, x, y_fit, n_estimators=n_estimators)
    if not _is_classification(spec):
        return float(compute_metric(spec.metric.name, y_hold, _predict(spec, model, x_hold, log_target)))
    proba = _full_proba(spec, model, model.predict_proba(x_hold))
    if METRICS[spec.metric.name].prediction == "proba":
        return float(compute_metric(spec.metric.name, y_hold, proba))
    weights = params.get("_proba_weights")
    scored = proba * np.asarray(weights) if weights and len(weights) == proba.shape[1] else proba
    return float(compute_metric(spec.metric.name, y_hold, scored.argmax(axis=1)))


def better(new: float, best: float, greater_is_better: bool) -> bool:
    """Новая лучше на отложенном куске (строго). Равенство — не подтверждение: шум подбора даёт ровно это."""
    if not (np.isfinite(new) and np.isfinite(best)):
        return False
    return new > best if greater_is_better else new < best
