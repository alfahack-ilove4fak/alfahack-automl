"""Страховка выбора метрики в agent/task_parser.py: шаблон ответа требует вероятности, а метрика — метки (#85)."""

import pandas as pd
import pytest

from agent import task_parser
from agent.task_parser import parse_task
from evaluation.metrics import METRICS


def make_task(tmp_path, text, sample_values=(0, 0, 0, 0), target_values=(0, 1, 0, 1)):
    pd.DataFrame({"id": range(4), "x": [1.0, 2.0, 3.0, 4.0], "target": list(target_values)}).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame({"id": range(4), "x": [1.0, 2.0, 3.0, 4.0]}).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": range(4), "target": list(sample_values)}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    (tmp_path / "task.md").write_text(text, encoding="utf-8")
    return parse_task(tmp_path / "task.md")


@pytest.fixture
def search_says_accuracy(monkeypatch):
    """Поиск метрики ошибся и вернул accuracy — проверяем страховку отдельно от evaluation/metrics.py."""
    monkeypatch.setattr(task_parser, "find_metric_in_text", lambda text: "accuracy")


@pytest.mark.parametrize(
    "text",
    [
        "Метрика: accuracy здесь не подходит. Нужно предсказать вероятность класса 1, "
        "качество оценивается по binary cross-entropy.",
        "## Метрика\nВ отличие от accuracy, мы оцениваем вероятности: binary crossentropy.",
        "Evaluation metric: not accuracy — we score predicted probabilities with log loss.",
        "Метрика: Accuracy не используется. Вероятности оцениваются кросс-энтропией.",
        "Метрика: BCE",
    ],
)
def test_trap_ends_with_probability_metric(tmp_path, text):
    # сквозная проверка: поиск метрики + страховка вместе дают вероятностную метрику
    spec = make_task(tmp_path, text, sample_values=(0.5, 0.5, 0.5, 0.5))
    assert spec.metric.name == "logloss"
    assert METRICS[spec.metric.name].prediction == "proba"


def test_fractional_sample_switches_to_logloss(tmp_path, search_says_accuracy):
    spec = make_task(tmp_path, "Метрика: accuracy.", sample_values=(0.5, 0.5, 0.5, 0.5))
    assert spec.metric.name == "logloss"
    assert any("дробные значения" in note for note in spec.notes)


def test_fractional_sample_takes_proba_metric_from_text(tmp_path, search_says_accuracy):
    spec = make_task(tmp_path, "Метрика: accuracy (для справки также ROC-AUC).", sample_values=(0.1, 0.9, 0.3, 0.2))
    assert spec.metric.name == "roc_auc"


@pytest.mark.parametrize(
    "text",
    [
        "Метрика: accuracy.",
        "Метрика: accuracy. Для справки считаем log loss по вероятностям.",  # текст — не признак
        "Метрика: accuracy. Данные аукциона: auction price — вероятность продажи.",
    ],
)
def test_label_metric_kept_when_sample_has_labels(tmp_path, search_says_accuracy, text):
    spec = make_task(tmp_path, text, sample_values=(0, 1, 0, 1))
    assert spec.metric.name == "accuracy"
    assert not any("дробные значения" in note for note in spec.notes)


def test_auction_is_not_auc(tmp_path, search_says_accuracy):
    spec = make_task(tmp_path, "Метрика: accuracy. Цена на auction.", sample_values=(0.5, 0.5, 0.5, 0.5))
    assert spec.metric.name == "logloss"  # не roc_auc из-за «auction»


def test_multiclass_not_switched_yet(tmp_path, search_says_accuracy):
    # ответ на multiclass пока — метки (#110): переключение на logloss только навредит
    spec = make_task(tmp_path, "Метрика: accuracy.", sample_values=(0.5, 0.5, 0.5, 0.5), target_values=(0, 1, 2, 1))
    assert spec.task_type.value == "multiclass"
    assert spec.metric.name == "accuracy"


def test_regression_untouched(tmp_path):
    spec = make_task(
        tmp_path,
        "Метрика: RMSE. Вероятность выброса мала, log loss не используется.",
        sample_values=(0.5, 0.5, 0.5, 0.5),
        target_values=(1.5, 2.7, 3.1, 8.9),
    )
    assert spec.metric.name == "rmse"
