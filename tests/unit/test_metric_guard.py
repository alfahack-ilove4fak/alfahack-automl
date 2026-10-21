"""Страховка выбора метрики в agent/task_parser.py: задача ждёт вероятности, а в тексте первой стоит accuracy."""

import pandas as pd
import pytest

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


@pytest.mark.parametrize(
    "text",
    [
        # ловушка: accuracy упомянута первой, вероятности и cross-entropy — после
        "Метрика: accuracy здесь не подходит. Нужно предсказать вероятность класса 1, "
        "качество оценивается по binary cross-entropy.",
        "## Метрика\nВ отличие от accuracy, мы оцениваем вероятности: binary crossentropy.",
        "Evaluation metric: not accuracy — we score predicted probabilities with log loss.",
        "Метрика: Accuracy не используется. Вероятности оцениваются кросс-энтропией.",
    ],
)
def test_trap_switches_to_probability_metric(tmp_path, text):
    spec = make_task(tmp_path, text)
    assert spec.metric.name == "logloss"
    assert METRICS[spec.metric.name].prediction == "proba"
    assert any("ждёт вероятности" in note for note in spec.notes)


def test_fractional_sample_submission_means_probabilities(tmp_path):
    spec = make_task(tmp_path, "Метрика: accuracy.", sample_values=(0.5, 0.5, 0.5, 0.5))
    assert spec.metric.name == "logloss"


def test_fractional_sample_and_auc_in_text(tmp_path):
    spec = make_task(tmp_path, "Метрика: accuracy (для справки также ROC-AUC).", sample_values=(0.1, 0.9, 0.3, 0.2))
    assert spec.metric.name == "roc_auc"


def test_unknown_alias_bce_is_probability(tmp_path):
    spec = make_task(tmp_path, "Метрика: BCE")
    assert spec.metric.name == "logloss"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Метрика: accuracy.", "accuracy"),  # меток ждут — ничего не меняем
        ("Метрика: accuracy. Для справки считаем и ROC-AUC.", "accuracy"),  # нет ни дробей, ни «вероятности»
        ("Метрика: accuracy. Вероятность ошибки разметки мала.", "accuracy"),  # «вероятность» без proba-метрики
        ("Метрика: ROC-AUC", "roc_auc"),  # и так вероятности
    ],
)
def test_label_metric_kept_without_evidence(tmp_path, text, expected):
    spec = make_task(tmp_path, text, sample_values=(0, 1, 0, 1))
    assert spec.metric.name == expected
    assert not any("ждёт вероятности" in note for note in spec.notes)


def test_auction_is_not_auc(tmp_path):
    spec = make_task(tmp_path, "Метрика: accuracy. Данные аукциона: auction price — вероятность продажи.")
    assert spec.metric.name == "accuracy"


def test_regression_untouched(tmp_path):
    spec = make_task(
        tmp_path,
        "Метрика: RMSE. Вероятность выброса мала, log loss не используется.",
        sample_values=(0.5, 0.5, 0.5, 0.5),
        target_values=(1.5, 2.7, 3.1, 8.9),
    )
    assert spec.metric.name == "rmse"
