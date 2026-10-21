"""Проверка submission.csv по правилам платформы и запись «без риска испортить старый файл»."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS
from interfaces.schemas import TaskSpec, TaskType, ValidationReport

PROBA_SUM_TOL = 1e-3  # насколько сумма вероятностей по строке может отличаться от 1


def per_class_columns(spec: TaskSpec) -> list[str]:
    """Колонки классов, если ответ — «колонка на класс» (`id, a, b, c` с вероятностями), иначе [].

    Признак формата (#110): в шаблоне больше двух колонок, и среди них нет target.
    """
    cols = spec.submission_columns
    if len(cols) > 2 and spec.target not in cols:
        return [c for c in cols if c != spec.id_column]
    return []


def validate_submission(path: Path, spec: TaskSpec, test: pd.DataFrame) -> ValidationReport:
    """Проверяет всё, за что платформа может не принять ответ."""
    problems: list[str] = []
    if not path.exists():
        return ValidationReport(ok=False, problems=[f"Файл {path} не найден"])
    try:
        # id — строкой, как в ml.profiling.load_data: иначе int из CSV ≠ str из test, и любой ответ «невалиден»
        sub = pd.read_csv(path, dtype={spec.id_column: str})
    except Exception as exc:  # noqa: BLE001 — любая ошибка чтения = невалидный файл
        return ValidationReport(ok=False, problems=[f"Не читается как CSV: {exc}"])

    if list(sub.columns) != spec.submission_columns:
        problems.append(f"Колонки {list(sub.columns)}, ожидались {spec.submission_columns}")
    if len(sub) != len(test):
        problems.append(f"Строк {len(sub)}, в test.csv {len(test)}")
    if spec.id_column in sub.columns:
        ids = sub[spec.id_column]
        if ids.duplicated().any():
            problems.append(f"Повторяющиеся {spec.id_column}: {int(ids.duplicated().sum())} шт.")
        if len(sub) == len(test) and not np.array_equal(
            ids.astype(str).to_numpy(), test[spec.id_column].astype(str).to_numpy()
        ):
            problems.append("Порядок или состав id не совпадает с test.csv")
    class_columns = per_class_columns(spec)
    if class_columns and set(class_columns) <= set(sub.columns):
        problems += _check_class_probabilities(sub[class_columns])
    if spec.target in sub.columns:
        pred = sub[spec.target]
        if pred.isna().any():
            problems.append(f"Пропуски в ответах: {int(pred.isna().sum())} шт.")
        expects_proba = METRICS[spec.metric.name].prediction == "proba"
        if spec.task_type == TaskType.REGRESSION or expects_proba:
            values = pd.to_numeric(pred, errors="coerce")
            if not np.isfinite(values).all():
                problems.append("Есть нечисловые или бесконечные значения")
            elif expects_proba and ((values < 0) | (values > 1)).any():
                problems.append("Вероятности вне отрезка [0, 1]")
        elif spec.class_labels:
            bad = set(pred.astype(str)) - set(map(str, spec.class_labels))
            if bad:
                problems.append(f"Метки вне списка классов: {sorted(bad)[:5]}")
    return ValidationReport(ok=not problems, problems=problems, n_rows=len(sub))


def _check_class_probabilities(probs: pd.DataFrame) -> list[str]:
    """Ответ «колонка на класс»: числа без пропусков, каждое в [0, 1], сумма по строке ≈ 1."""
    values = probs.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return [
            f"В колонках классов пропуски, нечисловые или бесконечные значения: {int((~np.isfinite(values)).sum())} шт."
        ]
    problems = []
    if ((values < 0) | (values > 1)).any():
        problems.append("Вероятности классов вне отрезка [0, 1]")
    sums = values.sum(axis=1)
    bad = np.abs(sums - 1) > PROBA_SUM_TOL
    if bad.any():
        problems.append(
            f"Сумма вероятностей классов по строке ≠ 1 в {int(bad.sum())} строках (например, {sums[bad][0]:.4f})"
        )
    return problems


def write_submission_atomic(df: pd.DataFrame, path: Path) -> None:
    """Пишем во временный файл и подменяем: если упадём посередине, старый ответ останется целым."""
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
