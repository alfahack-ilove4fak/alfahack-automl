"""Запасные направления цикла (#200): playbook и LLM пусты, а время есть — агент не выходит."""

from pathlib import Path

import pytest

from agent import loop
from agent.critic import estimated_seconds
from agent.fallback import base_params, fallback_hypotheses, near_misses
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, Hypothesis, HypothesisKind, RunBudget
from interfaces.stubs import stub_task_spec


def hyp(hid, **params):
    return Hypothesis(id=hid, kind=HypothesisKind.HYPERPARAMS, title=hid, params=params)


def result(hid, folds):
    return ExperimentResult(
        hypothesis_id=hid,
        status=ExperimentStatus.OK,
        metric_name="rmse",
        fold_scores=folds,
        mean_score=sum(folds) / len(folds),
        std_score=0.0,
    )


@pytest.fixture(autouse=True)
def fallback_on(monkeypatch):
    monkeypatch.setenv("AGENT_FALLBACK", "1")  # conftest выключает их для остальных тестов цикла


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))  # rmse: меньше — лучше


BASE = [40.0, 41.0, 42.0, 43.0, 44.0]


def test_models_and_optuna_keep_best_preprocessing(spec):
    best = (
        hyp("best", model="lightgbm", log_target=True, _best_iteration=120, validation="time:Year"),
        result("b", BASE),
    )
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    models = [h.params["model"] for h in hyps if "optuna" not in h.params]
    assert models == ["xgboost", "catboost", "hist_gb"]
    # предобработка лучшего сохраняется, служебные ключи и схема — нет (схему добавит цикл)
    assert all(h.params.get("log_target") is True for h in hyps)
    assert not any("_best_iteration" in h.params or "validation" in h.params for h in hyps)
    optuna = [h for h in hyps if "optuna" in h.params]
    assert [h.params["model"] for h in optuna] == ["lightgbm"]
    assert optuna[0].params["optuna"]["timeout"] == 600
    assert all(h.id.startswith("fb-") and h.parent_id == "best" for h in hyps)


def test_optuna_limited_by_budget_and_skipped_when_no_time(spec):
    best = (hyp("best"), result("b", BASE))
    short = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=200, safety_seconds=0))
    assert [h.params["optuna"]["timeout"] for h in short if "optuna" in h.params] == [60]  # 0.3 остатка
    none = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=60, safety_seconds=0))
    assert not any("optuna" in h.params for h in none)


def test_optuna_for_best_model_then_lightgbm(spec):
    best = (hyp("best", model="catboost", model_params={"depth": 8}), result("b", BASE))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    optuna = [h.params for h in hyps if "optuna" in h.params]
    assert [p["model"] for p in optuna] == ["catboost", "lightgbm"]
    assert optuna[0]["model_params"] == {"depth": 8} and "model_params" not in optuna[1]
    assert "catboost" not in [h.params["model"] for h in hyps if "optuna" not in h.params]


def test_history_repeats_are_skipped(spec):
    best = (hyp("best"), result("b", BASE))
    first = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    history = [best] + [(h, result(h.id, [50.0] * 5)) for h in first]
    assert fallback_hypotheses(spec, history, best, RunBudget(total_seconds=3600)) == []


def test_near_misses_combined_on_best(spec):
    best = (hyp("best", class_weight=None, feature_ops=["missing_flags"]), result("b", BASE))
    a = (hyp("a", model_params={"num_leaves": 31}), result("a", [39.0, 40.5, 41.5, 43.5, 44.5]))  # 3 из 5 лучше
    b = (hyp("b", feature_ops=["freq_encoding"]), result("b2", [39.5, 40.0, 42.5, 42.0, 43.0]))  # 4 из 5
    worse = (hyp("w", model_params={"min_child_samples": 50}), result("w", [45.0] * 5))
    other = (hyp("x", model="xgboost"), result("x", [30.0] * 5))  # чужая модель — не накладываем
    history = [best, a, b, worse, other]
    assert [h.id for h in near_misses(history, best, gib=False)] == ["x", "b", "a"]
    combos = [h for h in fallback_hypotheses(spec, history, best, RunBudget(total_seconds=3600)) if "+" in h.title]
    params = [h.params for h in combos]
    assert {"class_weight": None, "feature_ops": ["missing_flags", "freq_encoding"]} in params
    assert {"class_weight": None, "feature_ops": ["missing_flags"], "model_params": {"num_leaves": 31}} in params
    both = {"class_weight": None, "feature_ops": ["missing_flags", "freq_encoding"], "model_params": {"num_leaves": 31}}
    assert both in params
    assert not any(p.get("model") == "xgboost" for p in params)


def test_near_misses_ignore_incomparable_and_code(spec):
    best = (hyp("best"), result("b", BASE))
    three_folds = (hyp("t"), result("t", [1.0, 1.0, 1.0]))  # другое число фолдов — сравнивать нельзя
    code = (Hypothesis(id="c", kind=HypothesisKind.FEATURES, title="c", code="def add_features(df): return df"),)
    code = code + (result("c", [1.0] * 5),)
    tuned = (hyp("o", optuna={"timeout": 60}), result("o", [1.0] * 5))
    assert near_misses([best, three_folds, code, tuned], best, gib=False) == []


def test_switch_off(spec, monkeypatch):
    monkeypatch.setenv("AGENT_FALLBACK", "0")
    best = (hyp("best"), result("b", BASE))
    assert fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600)) == []


def test_base_params_strips_service_keys():
    assert base_params({"model": "lightgbm", "_proba_weights": [1], "validation": "kfold", "n_folds": 5}) == {
        "model": "lightgbm",
        "n_folds": 5,
    }


def test_optuna_cost_in_estimate():
    assert estimated_seconds(hyp("o", optuna={"timeout": 100}), 10) == pytest.approx(10 * 1.5 + 100)
    assert estimated_seconds(hyp("o", optuna={"timeout": "долго"}), 10) == pytest.approx(15)
    assert estimated_seconds(hyp("o", optuna=True), 10) == pytest.approx(15 + 600)  # #222: optuna: true от LLM


def test_loop_uses_fallback_instead_of_exit(spec, monkeypatch):
    """Playbook пуст: раньше цикл выходил сразу, теперь идёт по запасным направлениям, пока они есть."""
    ran = []

    def fake_experiment(spec, profile, h, train, budget):
        ran.append(h)
        return result(h.id, [50.0] * 5)  # всё хуже бейзлайна

    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    monkeypatch.setattr(loop, "propose_hypotheses", lambda *a, **kw: [])
    reflected = []
    monkeypatch.setattr(loop, "reflect", lambda *a, **kw: reflected.append(a))
    base = (hyp("baseline", validation="time:Year"), result("baseline", BASE).model_copy(update={"fit_seconds": 1.0}))
    profile = DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=[])
    loop.run_autoresearch(spec, profile, None, RunBudget(total_seconds=3600), [base], base, lambda h, r: None, llm=None)
    assert {h.params["model"] for h in ran} >= {"xgboost", "catboost", "hist_gb"}
    assert any("optuna" in h.params for h in ran)
    assert all(h.params["validation"] == "time:Year" for h in ran)  # в схеме прогона, как все гипотезы
    assert len({h.id for h in ran}) == len(ran)  # без повторов
    assert reflected == []  # пачки запасных направлений придумала не LLM — рефлексию не просим
