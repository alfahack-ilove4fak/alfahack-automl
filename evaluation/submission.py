"""Проверка submission.csv по правилам платформы и запись «без риска испортить старый файл»."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS
from interfaces.schemas import TaskSpec, TaskType, ValidationReport


def validate_submission(path: Path, spec: TaskSpec, test: pd.DataFrame) -> ValidationReport:
    """Проверяет всё, за что платформа может не принять ответ."""
    problems: list[str] = []
    if not path.exists():
        return ValidationReport(ok=False, problems=[f"Файл {path} не найден"])
    try:
        sub = pd.read_csv(path)
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
        if len(sub) == len(test) and not np.array_equal(ids.to_numpy(), test[spec.id_column].to_numpy()):
            problems.append("Порядок или состав id не совпадает с test.csv")
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


def write_submission_atomic(df: pd.DataFrame, path: Path) -> None:
    """Пишем во временный файл и подменяем: если упадём посередине, старый ответ останется целым."""
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
