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
    ],
)
def test_find_metric_in_text(text, expected):
    assert find_metric_in_text(text) == expected


def test_metric_directions():
    assert compute_metric("rmse", [1, 2], [1, 2]) == 0.0
    assert compute_metric("f1_macro", [0, 1], [0, 1]) == 1.0
