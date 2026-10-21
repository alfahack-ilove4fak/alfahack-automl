"""Паспорт данных (DataProfile) и подготовка признаков для моделей. Владелец — ML-pipeline инженер."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from interfaces.schemas import DataProfile, TaskSpec, TaskType
from runtime.log import get_logger

log = get_logger("profiling")


def load_data(spec: TaskSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(spec.train_path, low_memory=False)
    test = pd.read_csv(spec.test_path, low_memory=False)
    return train, test


def profile_data(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> DataProfile:
    features = [c for c in train.columns if c not in (spec.target, spec.id_column) and c in test.columns]
    numeric, categorical, dropped, warnings = [], [], [], []
    for col in features:
        series = train[col]
        if series.nunique(dropna=False) <= 1:
            dropped.append(col)
            continue
        if series.dtype == object or str(series.dtype) in ("category", "bool", "string"):
            categorical.append(col)
        else:
            numeric.append(col)

    missing = {c: round(float(train[c].isna().mean()), 4) for c in features if train[c].isna().any()}
    y = train[spec.target]
    if spec.task_type == TaskType.REGRESSION:
        target_summary = {k: float(v) for k, v in y.describe().items()}
        target_summary["skew"] = float(y.skew())
        if y.skew() > 2:
            warnings.append("Сильно скошенный target — стоит попробовать log1p-преобразование")
    else:
        counts = y.astype(str).value_counts()
        target_summary = {"class_counts": counts.to_dict(), "imbalance_ratio": float(counts.max() / counts.min())}
        if counts.max() / counts.min() > 10:
            warnings.append("Сильный дисбаланс классов — стоит попробовать class_weight='balanced'")

    profile = DataProfile(
        n_train=len(train),
        n_test=len(test),
        numeric_features=numeric,
        categorical_features=categorical,
        dropped_features=dropped,
        missing_share=missing,
        target_summary=target_summary,
        memory_mb=float((train.memory_usage(deep=True).sum() + test.memory_usage(deep=True).sum()) / 2**20),
        warnings=warnings,
    )
    log.info(
        "Данные: train=%d, test=%d, числовых=%d, категориальных=%d, выброшено=%d",
        len(train),
        len(test),
        len(numeric),
        len(categorical),
        len(dropped),
    )
    return profile


def make_matrix(profile: DataProfile, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Признаки для бустинга: числа как есть, категории → pandas category с общим словарём train+test."""
    cols = profile.numeric_features + profile.categorical_features
    x_train, x_test = train[cols].copy(), test[cols].copy()
    for col in profile.categorical_features:
        cats = pd.Index(pd.concat([x_train[col], x_test[col]]).astype(str).unique())
        x_train[col] = pd.Categorical(x_train[col].astype(str), categories=cats)
        x_test[col] = pd.Categorical(x_test[col].astype(str), categories=cats)
    for col in profile.numeric_features:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce").astype(np.float32)
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce").astype(np.float32)
    names = _safe_names(cols)
    x_train.columns, x_test.columns = names, names
    return x_train, x_test


def _safe_names(cols: list[str]) -> list[str]:
    """Имена признаков, которые примут все модели: LightGBM падает на спецсимволах JSON, XGBoost — на [ ] <.

    Всё, кроме букв, цифр и _, заменяется на _; совпавшие после замены имена получают суффикс _1, _2…
    """
    names, seen = [], set()
    for col in cols:
        base = re.sub(r"\W+", "_", str(col)).strip("_") or "feature"
        name, k = base, 1
        while name in seen:
            name, k = f"{base}_{k}", k + 1
        seen.add(name)
        names.append(name)
    return names
