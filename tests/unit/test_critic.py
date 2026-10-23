"""Критик гипотез по правилам (A-L3-2): пустые для задачи гипотезы, недопустимые значения, цена по времени."""

from pathlib import Path

import pytest

from agent import critic, loop
from agent.critic import MAX_SLOWDOWN, TIME_SAFETY, estimated_seconds, lightgbm_slowdown, review, slowdown
from interfaces.schemas import (
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    MetricSpec,
    RunBudget,
    TaskType,
)
from interfaces.stubs import stub_task_spec


def hyp(params, hid="h"):
    return Hypothesis(id=hid, kind=HypothesisKind.HYPERPARAMS, title="t", params=params)


def lgbm(hid="h", **model_params):
    return hyp({"model": "lightgbm", "model_params": model_params}, hid)


@pytest.fixture
def regression():
    return stub_task_spec(Path("tasks/demo"))


@pytest.fixture
def multiclass(regression):
    return regression.model_copy(
        update={"task_type": TaskType.MULTICLASS, "metric": MetricSpec(name="f1_macro", greater_is_better=True)}
    )


def test_noop_for_task_type(regression, multiclass):
    assert "только на регрессии" in review(hyp({"log_target": True}), multiclass)
    assert review(hyp({"log_target": True}), regression) is None
    assert "только на классификации" in review(hyp({"class_weight": "balanced"}), regression)
    assert review(hyp({"class_weight": "balanced"}), multiclass) is None


@pytest.mark.parametrize(
    "params",
    [
        {"class_weight": "heavy"},
        {"n_folds": 1},
        {"n_folds": 2.5},
        {"n_folds": True},
        {"model_params": {"learning_rate": 0}},
        {"model_params": {"learning_rate": -0.1}},
        {"model_params": {"learning_rate": 2}},
        {"model_params": {"learning_rate": "быстро"}},
        {"model_params": {"num_leaves": 1}},
        {"model_params": {"num_leaves": 10_000}},
        {"model_params": {"num_leaves": 31.5}},
    ],
)
def test_invalid_values_rejected(multiclass, params):
    assert review(hyp(params), multiclass) is not None


def test_playbook_like_hypotheses_pass(multiclass):
    for params in (
        {"model": "lightgbm"},
        {"model": "lightgbm", "model_params": {"learning_rate": 0.02}},
        {"model": "lightgbm", "model_params": {"num_leaves": 31, "min_child_samples": 50, "reg_lambda": 1.0}},
        {"model": "lightgbm", "class_weight": "balanced", "n_folds": 10},
    ):
        assert review(hyp(params), multiclass) is None, params


@pytest.mark.parametrize(
    "lr, leaves, measured",
    [
        (0.02, 63, 197 / 122),  # multiclass, замер 2026-10-21: бейзлайн тогда — learning_rate 0.05, num_leaves 63
        (0.05, 255, 217 / 122),
    ],
)
def test_slowdown_calibrated_on_measurements(lr, leaves, measured):
    # бейзлайн замера задан явно: калибровка не зависит от текущих BASE_PARAMS (#206)
    assert lightgbm_slowdown(lr, leaves, base_lr=0.05, base_leaves=63) == pytest.approx(measured, rel=0.15)


def test_slowdown_relative_to_current_base(monkeypatch):
    # смена бейзлайна (num_leaves 63 → 31): оценка — от нового бейзлайна, тест калибровки не падает
    monkeypatch.setitem(critic.BASE_PARAMS, "num_leaves", 31)
    assert slowdown(lgbm(num_leaves=31)) == 1.0
    assert slowdown(lgbm(num_leaves=63)) == pytest.approx((63 / 31) ** 0.4)


def test_slowdown_rules():
    assert slowdown(lgbm()) == 1.0
    assert slowdown(lgbm(num_leaves=31, learning_rate=0.1)) == 1.0  # дешевле бейзлайна не считаем
    assert slowdown(hyp({"n_folds": 10})) == pytest.approx(2.0)
    assert slowdown(lgbm(learning_rate=1e-6, num_leaves=4096)) == MAX_SLOWDOWN
    assert slowdown(lgbm(learning_rate="быстро")) == 1.0
    assert estimated_seconds(lgbm(learning_rate=0.0125), 100) == pytest.approx(100 * 2.0 * TIME_SAFETY)


def run_loop(spec, hyps, seconds_left, monkeypatch):
    ran = []

    def fake_experiment(spec, profile, h, train, budget):
        ran.append(h.id)
        return ExperimentResult(
            hypothesis_id=h.id, status=ExperimentStatus.OK, metric_name="rmse", mean_score=50.0, std_score=0.0
        )

    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    queue = list(hyps)
    monkeypatch.setattr(loop, "propose_hypotheses", lambda *a, **kw: [queue.pop(0)] if queue else [])
    base_h = hyp({"model": "lightgbm"}, "baseline")
    base_r = ExperimentResult(
        hypothesis_id="baseline",
        status=ExperimentStatus.OK,
        metric_name="rmse",
        mean_score=40.0,
        std_score=0.1,
        fit_seconds=100.0,
    )
    history = [(base_h, base_r)]
    budget = RunBudget(total_seconds=seconds_left, safety_seconds=0)
    loop.run_autoresearch(spec, None, None, budget, history, (base_h, base_r), lambda h, r: None, llm=None)
    return ran, {h.id: r for h, r in history}


def test_loop_skips_expensive_and_runs_cheap(regression, monkeypatch):
    slow = lgbm("slow", learning_rate=0.005)  # оценка: 100 с · √10 · 1.5 ≈ 474 с при бюджете 400 с
    cheap = lgbm("cheap", num_leaves=31)
    ran, results = run_loop(regression, [slow, cheap], seconds_left=400, monkeypatch=monkeypatch)
    assert ran == ["cheap"]
    assert results["slow"].status == ExperimentStatus.SKIPPED
    assert "не успеет" in results["slow"].error


def test_loop_skips_noop_without_running(regression, monkeypatch):
    noop = hyp({"class_weight": "balanced"}, "noop")
    ran, results = run_loop(regression, [noop], seconds_left=10_000, monkeypatch=monkeypatch)
    assert ran == []
    assert results["noop"].status == ExperimentStatus.SKIPPED


def test_model_slowdown():
    assert slowdown(hyp({"model": "catboost"})) == pytest.approx(8.0)
    assert slowdown(hyp({"model": "XGBoost"})) == pytest.approx(4.0)
    assert slowdown(hyp({"model": "something_new"})) == pytest.approx(4.0)  # неизвестная — осторожно
    # learning_rate/num_leaves откалиброваны только для LightGBM — у других моделей не умножаем
    assert slowdown(hyp({"model": "catboost", "model_params": {"learning_rate": 0.005}})) == pytest.approx(8.0)
    assert slowdown(hyp({"model": "catboost", "n_folds": 10})) == pytest.approx(16.0)


def test_num_leaves_only_for_lightgbm(multiclass):
    assert "не действует в catboost" in review(
        hyp({"model": "catboost", "model_params": {"num_leaves": 255}}), multiclass
    )
    lossguide = {"model": "catboost", "model_params": {"num_leaves": 255, "grow_policy": "Lossguide"}}
    assert review(hyp(lossguide), multiclass) is None
    assert review(hyp({"model": "xgboost", "model_params": {"max_depth": 8}}), multiclass) is None
