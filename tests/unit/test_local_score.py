"""Локальный балл: формула организаторов и пересчёт submission.csv во вход метрики."""

import numpy as np
import pandas as pd
import pytest

from evaluation.local_score import organizer_score, score_submission
from interfaces.schemas import MetricSpec, TaskSpec, TaskType


def _spec(task_type, metric, greater, labels=None):
    return TaskSpec(
        task_md="t/task.md",
        task_dir="t",
        train_path="t/train.csv",
        test_path="t/test.csv",
        submission_path="t/submission.csv",
        task_type=task_type,
        target="y",
        id_column="id",
        metric=MetricSpec(name=metric, greater_is_better=greater),
        submission_columns=["id", "y"],
        class_labels=labels,
    )


@pytest.mark.parametrize(
    "metric, value, baseline, expected",
    [
        ("rmse", 50, 100, 50.0),  # 100 · (1 − 50/100)
        ("rmse", 120, 100, 0.0),  # хуже бейзлайна — ноль, а не минус
        ("f1_macro", 0.6, 0.2, 50.0),  # 100 · (0.6 − 0.2) / (1 − 0.2)
        ("f1_macro", 0.1, 0.2, 0.0),
        ("roc_auc", 0.75, 0.5, 50.0),
        ("accuracy", 1.0, 1.0, 0.0),  # бейзлайн уже идеален — расти некуда
        ("rmse", 1.0, 0.0, 0.0),
        ("rmse", float("nan"), 1.0, 0.0),
    ],
)
def test_organizer_score(metric, value, baseline, expected):
    assert organizer_score(metric, value, baseline) == pytest.approx(expected)


def test_regression_perfect_answer_scores_100():
    spec = _spec(TaskType.REGRESSION, "rmse", False)
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = score_submission(spec, y, y, y_train=pd.Series([1.0, 5.0]))
    assert result["value"] == 0 and result["baseline"] > 0 and result["score"] == 100


def test_labels_are_encoded_like_ml():
    spec = _spec(TaskType.MULTICLASS, "f1_macro", True, labels=["a", "b", "c"])
    answers = pd.Series(["a", "b", "c", "c"])
    result = score_submission(spec, answers, answers.copy(), y_train=pd.Series(["c", "c", "a"]))
    assert result["value"] == 1.0 and result["score"] == 100


def test_binary_probability_and_trivial_baseline():
    spec = _spec(TaskType.BINARY, "roc_auc", True, labels=["no", "yes"])
    answers = pd.Series(["no", "yes", "no", "yes"])
    result = score_submission(spec, answers, pd.Series([0.1, 0.9, 0.2, 0.7]), y_train=answers)
    assert result["value"] == 1.0
    assert result["baseline"] == 0.5  # доля класса одинакова для всех строк — AUC 0.5


def test_trivial_answer_scores_zero():
    spec = _spec(TaskType.MULTICLASS, "accuracy", True, labels=["a", "b"])
    answers = pd.Series(["a", "a", "b"])
    result = score_submission(spec, answers, pd.Series(["a", "a", "a"]), y_train=pd.Series(["a", "a", "b"]))
    assert result["score"] == 0


def test_unknown_label_in_submission_is_an_error():
    spec = _spec(TaskType.MULTICLASS, "accuracy", True, labels=["a", "b"])
    with pytest.raises(ValueError, match="вне списка классов"):
        score_submission(spec, pd.Series(["a"]), pd.Series(["zzz"]), y_train=pd.Series(["a"]))


def test_multiclass_proba_metric_accepts_labels():
    """Для многих классов с logloss агент пишет метку — считаем её «уверенной» вероятностью."""
    spec = _spec(TaskType.MULTICLASS, "logloss", False, labels=["a", "b", "c"])
    answers = pd.Series(["a", "b", "c"])
    result = score_submission(spec, answers, answers.copy(), y_train=answers)
    assert np.isfinite(result["value"]) and result["value"] < result["baseline"]


def _xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["u", "v", "w"], n)})
    return x, x.a * 3 + (x.c == "u") * 2 + rng.normal(scale=0.1, size=n)


def test_model_baseline_is_stronger_than_trivial_on_learnable_data():
    spec = _spec(TaskType.REGRESSION, "rmse", False)
    x, y = _xy()
    result = score_submission(spec, y[200:], y[200:], y[:200], x_train=x[:200], x_test=x[200:])
    assert result["model_baseline"] < result["baseline"]  # LightGBM лучше среднего
    assert result["model_score"] == 100  # идеальный ответ — 100 и к модельному бейзлайну
    assert "model_error" not in result


@pytest.mark.parametrize(
    "task_type, metric, greater, labels",
    [
        (TaskType.BINARY, "roc_auc", True, ["no", "yes"]),
        (TaskType.MULTICLASS, "f1_macro", True, ["p", "q", "r"]),
        (TaskType.MULTICLASS, "logloss", False, ["p", "q", "r"]),
    ],
)
def test_model_baseline_for_classification(task_type, metric, greater, labels):
    spec = _spec(task_type, metric, greater, labels=labels)
    x, value = _xy()
    y = pd.Series(np.array(labels)[pd.qcut(value, len(labels), labels=False)])
    answer = (y[200:] == "yes").astype(float) if task_type == TaskType.BINARY else y[200:]  # бинарная — вероятность
    trivial = score_submission(spec, y[200:], answer, y[:200])
    result = score_submission(spec, y[200:], answer, y[:200], x_train=x[:200], x_test=x[200:])
    assert "model_baseline" not in trivial  # без признаков — только тривиальный бейзлайн, как раньше
    assert np.isfinite(result["model_baseline"]) and result["score"] == trivial["score"]


def test_model_baseline_class_missing_in_train_gets_zero_probability():
    spec = _spec(TaskType.MULTICLASS, "logloss", False, labels=["p", "q", "r"])
    x, value = _xy()
    y = pd.Series(np.where(value > 0, "q", "p"))
    y[250:] = "r"  # класс r есть только в отложенной части — модель его не видела
    result = score_submission(spec, y[200:], y[200:], y[:200], x_train=x[:200], x_test=x[200:])
    assert np.isfinite(result["model_baseline"])


def test_model_baseline_error_does_not_break_scoring():
    spec = _spec(TaskType.REGRESSION, "rmse", False)
    x, y = _xy()
    result = score_submission(spec, y[200:], y[200:], y[:200], x_train=x[:200].iloc[:0], x_test=x[200:])
    assert result["score"] == 100 and np.isnan(result["model_score"]) and result["model_error"]


def test_strong_baseline_is_never_weaker_than_trivial():
    """Сдвиг между train и test (как отложенные годы): LightGBM хуже среднего — бейзлайн берём тривиальный."""
    spec = _spec(TaskType.REGRESSION, "rmse", False)
    x, y = _xy()
    y_test = pd.Series(np.full(100, float(y[:200].mean())))  # на тесте связь с признаками пропала
    result = score_submission(spec, y_test, y_test + 1.0, y[:200], x_train=x[:200], x_test=x[200:])
    assert result["model_baseline"] == result["baseline"]  # тривиальный ответ здесь идеален и сильнее LightGBM
    assert result["model_score"] == result["score"] == 0


def test_strong_baseline_fixes_histogram_mode(monkeypatch):
    """#219: без force_row_wise LightGBM сам выбирает row/col-wise по замеру скорости — бейзлайн плавал между прогонами."""
    import lightgbm as lgb

    from evaluation import local_score

    seen = []

    class Spy(lgb.LGBMClassifier):
        def fit(self, X, y, **kwargs):
            seen.append(self.get_params())
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(lgb, "LGBMClassifier", Spy)
    spec = _spec(TaskType.MULTICLASS, "f1_macro", True, labels=["p", "q", "r"])
    x, value = _xy()
    y = pd.Series(np.array(["p", "q", "r"])[pd.qcut(value, 3, labels=False)])
    local_score.model_prediction(spec, x[:200], y[:200], x[200:])
    assert seen and seen[0]["deterministic"] is True and seen[0]["force_row_wise"] is True


def test_strong_baseline_same_for_any_thread_count(monkeypatch):
    from evaluation import local_score

    spec = _spec(TaskType.MULTICLASS, "logloss", False, labels=["p", "q", "r"])
    x, value = _xy(n=2000)
    y = pd.Series(np.array(["p", "q", "r"])[pd.qcut(value, 3, labels=False)])
    runs = []
    for threads in (1, 4):
        monkeypatch.setattr(local_score, "MODEL_THREADS", threads)
        runs.append(local_score.model_prediction(spec, x[:1500], y[:1500], x[1500:]))
    np.testing.assert_array_equal(runs[0], runs[1])
