"""Потолок времени одной гипотезы (#276): доля остатка, поправка оценки по замеру, выключатель AGENT_HYP_TIME_SHARE."""

from pathlib import Path

import pytest

from agent import critic, loop
from agent.critic import capped_budget, hyp_time_share, hypothesis_limit, observed_slowdown
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, Hypothesis, HypothesisKind, RunBudget
from interfaces.stubs import stub_task_spec


def hyp(hid, model="lightgbm", **params):
    return Hypothesis(id=hid, kind=HypothesisKind.MODEL, title=hid, params={"model": model, **params})


def result(hid, score=40.0, seconds=1.0, status=ExperimentStatus.OK, folds=5):
    return ExperimentResult(
        hypothesis_id=hid,
        status=status,
        metric_name="rmse",
        fold_scores=[score] * folds,
        mean_score=score if status == ExperimentStatus.OK else None,
        std_score=0.0,
        fit_seconds=seconds,
    )


def run(monkeypatch, batch, baseline_seconds, history_extra=()):
    """Цикл на фейковом эксперименте; возвращает {id: seconds_left() бюджета, с которым гипотезу запустили}."""
    seen = {}

    def fake_experiment(spec, profile, h, train, budget):
        seen[h.id] = budget.seconds_left()
        return result(h.id, 41.0)

    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    batches = [batch]
    monkeypatch.setattr(loop, "propose_hypotheses", lambda *a, **kw: batches.pop(0) if batches else [])
    monkeypatch.setattr(loop, "fallback_hypotheses", lambda *a, **kw: [])
    base = (hyp("baseline"), result("baseline", 42.0, seconds=baseline_seconds))
    history = [base, *history_extra]
    spec = stub_task_spec(Path("tasks/demo"))
    profile = DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=[])
    loop.run_autoresearch(spec, profile, None, RunBudget(), history, base, lambda h, r: None, llm=None)
    return seen, {h.id: r for h, r in history}


def test_share_env(monkeypatch):
    monkeypatch.delenv(critic.HYP_TIME_SHARE_ENV, raising=False)
    assert hyp_time_share() == critic.DEFAULT_HYP_TIME_SHARE
    for value, expected in [
        ("0", 0.0),
        ("-1", 0.0),
        ("0.5", 0.5),
        ("3", 1.0),
        ("мусор", critic.DEFAULT_HYP_TIME_SHARE),
    ]:
        monkeypatch.setenv(critic.HYP_TIME_SHARE_ENV, value)
        assert hyp_time_share() == expected


def test_limit_floor_and_ceiling():
    assert hypothesis_limit(8000, 0.25, 100) == 2000
    assert hypothesis_limit(1000, 0.25, 100) == 300  # не меньше трёх бейзлайнов
    assert hypothesis_limit(200, 0.25, 100) == 200  # и не больше остатка


def test_capped_budget_limits_seconds_left():
    budget = RunBudget()
    capped = capped_budget(budget, 600)
    assert capped.seconds_left() == pytest.approx(600, abs=1)
    assert budget.seconds_left() > 9000  # исходный бюджет не тронут
    assert capped_budget(budget, 1e9) is budget


def test_observed_slowdown_from_finished_and_timeout_runs():
    xgb = hyp("x2", "xgboost")
    assert observed_slowdown(xgb, [], 100) == 1.0
    # прошлый XGBoost шёл 4000 с при оценке 100 × 4 → поправка ×10
    finished = [(hyp("x1", "xgboost"), result("x1", seconds=4000))]
    assert observed_slowdown(xgb, finished, 100) == pytest.approx(10)
    # оборван после 1 фолда из 5 на 500 с → полный запуск ≈ 2500 с → ×6.25
    cut = [(hyp("x1", "xgboost"), result("x1", seconds=500, status=ExperimentStatus.TIMEOUT, folds=1))]
    assert observed_slowdown(xgb, cut, 100) == pytest.approx(6.25)
    # чужая модель, быстрый запуск, ансамбль и optuna поправку не дают
    assert observed_slowdown(hyp("l", "lightgbm"), finished, 100) == 1.0
    assert observed_slowdown(xgb, [(hyp("x1", "xgboost"), result("x1", seconds=10))], 100) == 1.0
    assert observed_slowdown(hyp("o", "xgboost", optuna=True), finished, 100) == 1.0


def test_unmeasured_expensive_hypothesis_runs_under_cap(monkeypatch):
    monkeypatch.delenv(critic.HYP_TIME_SHARE_ENV, raising=False)
    # бейзлайн 1000 с: XGBoost по таблице ≈ 6000 с > 25% остатка, но замера ещё нет — не пропускаем, а запускаем
    # с бюджетом-потолком max(25% остатка, 3 бейзлайна) ≈ 3000 с: дорогую гипотезу оборвёт прогноз по фолду в ml/
    seen, results = run(monkeypatch, [hyp("xgb", "xgboost"), hyp("lgbm")], baseline_seconds=1000)
    assert seen["xgb"] == pytest.approx(3000, abs=5)
    assert seen["lgbm"] == pytest.approx(3000, abs=5)
    assert results["xgb"].status == ExperimentStatus.OK


def test_ensemble_not_skipped_by_table_estimate(monkeypatch):
    monkeypatch.delenv(critic.HYP_TIME_SHARE_ENV, raising=False)
    # регрессия расчёта по прогону A: ансамбль из трёх моделей оценён в ×13 бейзлайна, на деле дешевле — не пропускать
    ens = hyp("ens", ensemble=["lightgbm", "xgboost", "catboost"])
    seen, results = run(monkeypatch, [ens], baseline_seconds=300)
    assert "ens" in seen and results["ens"].status == ExperimentStatus.OK


def test_cap_off_keeps_old_behaviour(monkeypatch):
    monkeypatch.setenv(critic.HYP_TIME_SHARE_ENV, "0")
    seen, results = run(monkeypatch, [hyp("xgb", "xgboost")], baseline_seconds=1000)
    assert seen["xgb"] > 9000  # полный остаток, как до #276
    assert results["xgb"].status == ExperimentStatus.OK


def test_measured_slow_model_skipped_next_time(monkeypatch):
    monkeypatch.delenv(critic.HYP_TIME_SHARE_ENV, raising=False)
    # бейзлайн 100 с; первый XGBoost был оборван после 1 фолда на 1000 с (≈ 5000 с на 5 фолдов, ×12.5 к оценке)
    past = (hyp("xgb1", "xgboost"), result("xgb1", seconds=1000, status=ExperimentStatus.TIMEOUT, folds=1))
    seen, results = run(monkeypatch, [hyp("xgb2", "xgboost", model_params={"max_depth": 8})], 100, [past])
    assert "xgb2" not in seen  # оценка 100 × 4 × 1.5 × 12.5 = 7500 с > 25% остатка
    assert "потолка гипотезы" in results["xgb2"].error
