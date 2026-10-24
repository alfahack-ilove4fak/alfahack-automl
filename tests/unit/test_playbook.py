"""Playbook: поля по шапке файла, params понимает ml/experiment.py, ни одна гипотеза не падает."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ml.experiment
from agent.hypotheses import load_playbook
from interfaces.schemas import (
    DataProfile,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    MetricSpec,
    RunBudget,
    TaskSpec,
    TaskType,
)
from ml.experiment import run_experiment

PLAYBOOK = load_playbook()
TASK_TYPES = {"regression", "binary", "multiclass"}  # значения task_types в шапке playbook
# Поддерживаемые ключи params — из docstring ml/experiment.py (строки вида «  ключ — описание»)
SUPPORTED_PARAMS = set(re.findall(r"^  (\w+)\s+—", ml.experiment.__doc__, flags=re.MULTILINE))


def test_supported_params_parsed():
    assert {"model", "model_params", "log_target", "class_weight"} <= SUPPORTED_PARAMS


REQUIREMENTS = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text(encoding="utf-8")
HEAVY_MODELS = {"xgboost", "catboost"}  # группа full: в CI их нет, на платформе — из requirements.txt


def _model(item: dict) -> str:
    return item["params"].get("model", "lightgbm")


def test_playbook_models_exist_on_platform():
    """Модель — из реестра ml/experiment.py, а тяжёлая — ещё и в requirements.txt платформы."""
    for item in PLAYBOOK:
        assert _model(item) in ml.experiment.MODEL_FITTERS, item["id"]
        if _model(item) in HEAVY_MODELS:
            assert re.search(rf"^{_model(item)}==", REQUIREMENTS, flags=re.MULTILINE), item["id"]


def test_playbook_has_unique_ids():
    ids = [item["id"] for item in PLAYBOOK]
    assert len(ids) >= 10
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("item", PLAYBOOK, ids=lambda item: item["id"])
def test_playbook_item_schema(item):
    assert re.fullmatch(r"[a-z0-9_]+", item["id"])
    HypothesisKind(item["kind"])
    assert item["title"] and item["rationale"]
    assert set(item.get("task_types", TASK_TYPES)) <= TASK_TYPES
    assert item.get("requires") in (None, "categorical", "imbalance", "metric_objective")
    assert (
        set(item["params"]) <= SUPPORTED_PARAMS
    ), f"новые ключи согласовать с ml: {set(item['params']) - SUPPORTED_PARAMS}"


def _toy_task(task_type: TaskType) -> tuple[TaskSpec, DataProfile, pd.DataFrame]:
    """Маленькая задача с числами, пропусками и категорией — чтобы проверить, что гипотеза не падает."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "x1": rng.normal(size=n),
            "x2": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
            "cat": rng.choice(["a", "b", "c", "d"], size=n),
        }
    )
    signal = df["x1"] + (df["cat"] == "a")
    metric = {
        TaskType.REGRESSION: MetricSpec(name="rmse", greater_is_better=False),
        TaskType.BINARY: MetricSpec(name="roc_auc", greater_is_better=True),
        TaskType.MULTICLASS: MetricSpec(name="f1_macro", greater_is_better=True),
    }[task_type]
    if task_type == TaskType.REGRESSION:
        df["y"] = np.exp(signal) + rng.random(n)
        labels = None
    else:
        bins = [0.0] if task_type == TaskType.BINARY else [-0.5, 0.5]
        df["y"] = np.digitize(signal, bins).astype(str)
        labels = sorted(df["y"].unique())
    spec = TaskSpec(
        task_md="task.md",
        task_dir=".",
        train_path="train.csv",
        test_path="test.csv",
        submission_path="submission.csv",
        task_type=task_type,
        target="y",
        id_column="id",
        metric=metric,
        submission_columns=["id", "y"],
        class_labels=labels,
    )
    profile = DataProfile(n_train=n, n_test=0, numeric_features=["x1", "x2"], categorical_features=["cat"])
    return spec, profile, df


@pytest.mark.parametrize(
    "task_type", [TaskType.REGRESSION, TaskType.BINARY, TaskType.MULTICLASS], ids=lambda t: t.value
)
def test_playbook_hypotheses_run(task_type, monkeypatch):
    # Проверяем «не падает», а не качество: мало деревьев, один поток, 2 фолда
    monkeypatch.setitem(ml.experiment.BASE_PARAMS, "n_estimators", 20)
    monkeypatch.setitem(ml.experiment.BASE_PARAMS, "n_jobs", 1)
    monkeypatch.setitem(ml.experiment.XGBOOST_PARAMS, "n_estimators", 20)
    monkeypatch.setitem(ml.experiment.CATBOOST_PARAMS, "iterations", 20)
    spec, profile, train = _toy_task(task_type)
    for item in PLAYBOOK:
        if item.get("task_types") and task_type.value not in item["task_types"]:
            continue
        if _model(item) in HEAVY_MODELS and importlib.util.find_spec(_model(item)) is None:
            continue  # CI без группы full; локально с `uv sync --group full` проверяется
        params = {**item["params"], "n_folds": 2}
        hyp = Hypothesis(id=item["id"], kind=HypothesisKind(item["kind"]), title=item["title"], params=params)
        result = run_experiment(spec, profile, hyp, train, RunBudget())
        assert result.status == ExperimentStatus.OK, f"{item['id']}: {result.error}"
        if item["params"].get("proba_weights") and spec.metric.name == "f1_macro":
            assert hyp.params.get("_proba_weights"), f"{item['id']}: множители классов не подобраны"


def test_xgboost_proba_weights_after_lightgbm_one():
    """Стенд v2: против бейзлайна xgboost_balanced_proba_weights на payments принимается зря (CV 0.21 против 0.10,
    честно F1 0.13 против 0.18), против balanced_proba_weights (CV 0.27) — нет. Порядок защищает от ложного принятия."""
    ids = [item["id"] for item in PLAYBOOK]
    assert ids.index("balanced_proba_weights") < ids.index("xgboost_balanced_proba_weights")
