"""Разбор настоящих публичных задач и поиск метрики в тексте."""

from pathlib import Path

import pytest

from agent.task_parser import parse_task
from evaluation.metrics import compute_metric, find_metric_in_text
from interfaces.schemas import TaskType

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "folder, task_type, target, metric, n_classes",
    [
        ("regression", TaskType.REGRESSION, "site_eui", "rmse", None),
        ("multiclass", TaskType.MULTICLASS, "attack_cat", "f1_macro", 10),
    ],
)
def test_public_tasks_parsed(folder, task_type, target, metric, n_classes):
    spec = parse_task(ROOT / "tasks" / "public" / folder / "task.md")
    assert spec.task_type == task_type
    assert spec.target == target
    assert spec.id_column == "id"
    assert spec.metric.name == metric
    assert spec.train_path.exists() and spec.test_path.exists()
    assert (len(spec.class_labels) if spec.class_labels else None) == n_classes


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Метрика оценки — **RMSE** на исходной шкале", "rmse"),
        ("Основная метрика — **Macro F1**. Дополнительно Accuracy", "f1_macro"),
        ("Quality is measured by ROC-AUC.", "roc_auc"),
        ("Метрика: MAE", "mae"),
        ("Оцениваем по log loss", "logloss"),
        ("Без метрики", None),
        # Дополнительная метрика стоит раньше основной — побеждает та, что после «основная метрика»
        ("Дополнительно смотрим на Accuracy, но основная метрика — Macro F1", "f1_macro"),
        # «mse» не должна находиться внутри «rmse» и наоборот
        ("Метрика оценки: MSE (не путать с RMSE)", "mse"),
        # «weighted f1» не должна проиграть простому «f1-score»
        ("Submissions are scored using Weighted F1-score.", "f1_weighted"),
        # Пока не находятся — чинить в задаче Q-L1-3 (реестр метрик), затем убрать xfail
        pytest.param(
            "Решения оцениваются по средней абсолютной ошибке",
            "mae",
            marks=pytest.mark.xfail(strict=True, reason="нет падежных форм в aliases"),
        ),
        pytest.param(
            "Evaluation metric: Root Mean Squared Logarithmic Error",
            "rmsle",
            marks=pytest.mark.xfail(strict=True, reason="нет полного названия в aliases"),
        ),
        pytest.param(
            "Submissions are evaluated on area under the ROC curve",
            "roc_auc",
            marks=pytest.mark.xfail(strict=True, reason="нет «area under the roc curve» в aliases"),
        ),
        pytest.param(
            "Метрика — macro-averaged F1",
            "f1_macro",
            marks=pytest.mark.xfail(strict=True, reason="нет «macro-averaged f1» в aliases"),
        ),
    ],
)
def test_find_metric_in_text(text, expected):
    assert find_metric_in_text(text) == expected


def test_metric_directions():
    assert compute_metric("rmse", [1, 2], [1, 2]) == 0.0
    assert compute_metric("f1_macro", [0, 1], [0, 1]) == 1.0
