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


def _short(h):
    """Коротко, что за направление: модель, ансамбль или Optuna."""
    if "ensemble" in h.params:
        return "ensemble:" + "+".join(m if isinstance(m, str) else m["model"] for m in h.params["ensemble"])
    return ("optuna:" if "optuna" in h.params else "") + h.params["model"]


def test_models_and_optuna_keep_best_preprocessing(spec):
    best = (
        hyp("best", model="lightgbm", log_target=True, _best_iteration=120, validation="kfold"),
        result("b", BASE),
    )
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    # порядок danil по стенду (#200): xgboost → ансамбль из двух → catboost → Optuna → ансамбль из трёх → hist_gb
    assert [_short(h) for h in hyps[:5]] == [
        "xgboost",
        "ensemble:lightgbm+xgboost",
        "catboost",
        "optuna:lightgbm",
        "ensemble:lightgbm+xgboost+catboost",
    ]
    assert _short(hyps[5]) == "hist_gb"
    # предобработка лучшего сохраняется, служебные ключи и схема — нет (схему добавит цикл)
    assert all(h.params.get("log_target") is True for h in hyps)
    assert not any("_best_iteration" in h.params or "validation" in h.params for h in hyps)
    assert hyps[3].params["optuna"] == {"timeout": 600}
    assert all(h.id.startswith("fb-") and h.parent_id == "best" for h in hyps)


@pytest.mark.parametrize("scheme", ["time:Year", "group:duplicates"])
def test_no_optuna_in_time_and_group_schemes(spec, scheme):
    """На time:/group: Optuna переобучается под фолды (public/regression: 53.1 против 44.5) — не запускаем (#236)."""
    best = (hyp("best", model="lightgbm", validation=scheme), result("b", BASE))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    assert hyps and not any("optuna" in h.params for h in hyps)  # остальные направления на месте
    kfold = (hyp("best", model="lightgbm", validation="kfold"), result("b", BASE))
    assert any("optuna" in h.params for h in fallback_hypotheses(spec, [kfold], kfold, RunBudget(total_seconds=3600)))


def test_optuna_limited_by_budget_and_skipped_when_no_time(spec):
    best = (hyp("best"), result("b", BASE))
    short = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=200, safety_seconds=0))
    assert {h.params["optuna"]["timeout"] for h in short if "optuna" in h.params} == {60}  # 0.3 остатка
    none = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=60, safety_seconds=0))
    assert not any("optuna" in h.params for h in none)


def test_optuna_for_best_model_then_lightgbm(spec):
    best = (hyp("best", model="catboost", model_params={"depth": 8}), result("b", BASE))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    optuna = [h.params for h in hyps if "optuna" in h.params and "trials" not in h.params["optuna"]]
    assert [p["model"] for p in optuna] == ["catboost", "lightgbm", "xgboost"]
    assert optuna[0]["model_params"] == {"depth": 8} and "model_params" not in optuna[1]
    assert "catboost" not in [h.params.get("model") for h in hyps if "optuna" not in h.params]
    # ансамбль начинается с лучшей модели и её параметров
    assert next(h for h in hyps if "ensemble" in h.params).params["ensemble"] == [
        {"model": "catboost", "model_params": {"depth": 8}},
        "lightgbm",
    ]


def test_history_repeats_are_skipped(spec):
    best = (hyp("best"), result("b", BASE))
    first = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    history = [best] + [(h, result(h.id, [50.0] * 5)) for h in first]
    assert fallback_hypotheses(spec, history, best, RunBudget(total_seconds=3600)) == []


def test_optuna_not_repeated_when_budget_shrinks(spec):
    """#230: timeout Optuna считается от остатка бюджета — с меньшим остатком это та же гипотеза, а не новая."""
    best = (hyp("best"), result("b", BASE))
    first = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    history = [best] + [(h, result(h.id, [50.0] * 5)) for h in first]
    later = fallback_hypotheses(spec, history, best, RunBudget(total_seconds=1000, safety_seconds=0))
    assert later == []
    assert [h.id for h in first if "optuna" in h.params] == [
        h.id
        for h in fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=1000, safety_seconds=0))
        if "optuna" in h.params
    ]  # id не зависит от остатка бюджета


def test_accepted_optuna_not_tuned_again(spec):
    """Принятая Optuna становится лучшей с подобранными model_params — подбирать ту же модель снова незачем."""
    tuned = hyp("fb-x", model="lightgbm", optuna={"timeout": 600}, model_params={"num_leaves": 17}, _optuna_best={})
    best = (tuned, result("fb-x", [30.0] * 5))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    same = [h.params["optuna"] for h in hyps if h.params.get("optuna") and h.params["model"] == "lightgbm"]
    assert same == [{"timeout": 600, "trials": t} for t in (60, 120, 240)]  # только подбор глубже
    seeds = [h for h in hyps if h.title.startswith("запасное: усреднение")]
    assert seeds[0].params["ensemble"][0] == {
        "model": "lightgbm",
        "model_params": {"num_leaves": 17, "random_state": 42},
    }


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
    base = (hyp("baseline", validation="kfold"), result("baseline", BASE).model_copy(update={"fit_seconds": 1.0}))
    profile = DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=[])
    loop.run_autoresearch(spec, profile, None, RunBudget(total_seconds=3600), [base], base, lambda h, r: None, llm=None)
    assert {h.params.get("model") for h in ran} >= {"xgboost", "catboost", "hist_gb"}
    assert any("ensemble" in h.params for h in ran)
    assert any("optuna" in h.params for h in ran)
    assert all(h.params["validation"] == "kfold" for h in ran)  # в схеме прогона, как все гипотезы
    assert len({h.id for h in ran}) == len(ran)  # без повторов
    assert reflected == []  # пачки запасных направлений придумала не LLM — рефлексию не просим


def test_playbook_on_best_keeps_accepted_preprocessing(spec):
    """Идеи playbook поверх лучшего: своя модель — слияние с его params, чужая — только его предобработка."""
    best = (hyp("best", model="lightgbm", log_target=True, model_params={"learning_rate": 0.02}), result("b", BASE))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    on_best = {h.title.removeprefix("запасное: ").split()[0]: h.params for h in hyps if "поверх лучшего" in h.title}
    assert on_best["freq_encoding"] == {
        "model": "lightgbm",
        "log_target": True,
        "model_params": {"learning_rate": 0.02},
        "feature_ops": ["freq_encoding"],
    }
    # чужие model_params на другую модель не тянем (catboost поверх лучшего совпал с направлением «модели» — повтор)
    assert "catboost" not in on_best
    assert all("model_params" not in p for p in on_best.values() if p.get("model") != "lightgbm")
    assert "balanced_classes" not in on_best  # классификационные гипотезы на регрессии не берём


def test_tuned_best_overlays_without_new_tuning(spec):
    """Лучший после Optuna: наложения берут подобранные model_params, но не запускают подбор снова."""
    tuned = hyp("fb-x", model="lightgbm", optuna={"timeout": 600}, model_params={"num_leaves": 17}, _optuna_best={})
    best = (tuned, result("fb-x", [30.0] * 5))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    overlays = [h.params for h in hyps if "поверх лучшего" in h.title]
    assert overlays and not any("optuna" in p for p in overlays)
    assert {"model": "lightgbm", "model_params": {"num_leaves": 17}, "feature_ops": ["pair_ops"]} in overlays


@pytest.mark.parametrize("model", ["linear", "hist_gb"])
def test_no_seed_bagging_for_deterministic_models(spec, model):
    best = (hyp("best", model=model), result("b", BASE))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    assert not any("усреднение" in h.title for h in hyps)


def test_ensemble_best_gets_no_model_directions(spec):
    """Лучший — ансамбль: модели «с предобработкой лучшего» ещё имеют смысл, ансамбли из ансамбля и seed — нет."""
    best = (hyp("best", ensemble=["lightgbm", "xgboost"], log_target=True), result("b", BASE))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    assert not any("ensemble" in h.params or "усреднение" in h.title for h in hyps)
    assert all("ensemble" not in h.params for h in hyps)
    assert {"model": "lightgbm", "log_target": True} in [h.params for h in hyps]


def test_best_code_carried_to_fallback(spec):
    best_hyp = Hypothesis(
        id="feat", kind=HypothesisKind.FEATURES, title="f", code="def add_features(df): return df", source="llm"
    )
    best = (best_hyp, result("feat", BASE))
    hyps = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    assert hyps and all(h.code == best_hyp.code for h in hyps)


def test_ensemble_cost_is_sum_of_members():
    assert estimated_seconds(hyp("e", ensemble=["lightgbm", "xgboost"]), 10) == pytest.approx(10 * (1 + 4) * 1.5)
    slow = {"model": "lightgbm", "model_params": {"learning_rate": 0.0125}}  # ×2 по калибровке lr
    assert estimated_seconds(hyp("e", ensemble=[slow, "lightgbm"]), 10) == pytest.approx(10 * 3 * 1.5)
    # 13 CatBoost — ×104, а не ×20: предел оценки — у каждого участника
    assert estimated_seconds(hyp("e", ensemble=["catboost"] * 13), 10) == pytest.approx(10 * 8 * 13 * 1.5)


def test_loop_rebuilds_fallback_from_new_best(spec, monkeypatch):
    """Запасные берутся порциями; после принятой запасной следующая порция строится от нового лучшего."""
    ran = []

    def fake_experiment(spec, profile, h, train, budget):
        ran.append(h)
        if h.params.get("model") == "xgboost" and len(ran) == 1:
            return result(h.id, [30.0, 31.0, 32.0, 33.0, 34.0])  # xgboost явно лучше — новый лучший
        return result(h.id, [50.0] * 5)

    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    monkeypatch.setattr(loop, "propose_hypotheses", lambda *a, **kw: [])
    base = (hyp("baseline"), result("baseline", BASE).model_copy(update={"fit_seconds": 1.0}))
    profile = DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=[])
    best = loop.run_autoresearch(
        spec, profile, None, RunBudget(total_seconds=3600), [base], base, lambda h, r: None, llm=None
    )
    assert best[0].params["model"] == "xgboost"
    # второй шёл уже от xgboost: ансамбль начинается с него, а не с LightGBM прошлого лучшего
    assert ran[1].params["ensemble"][0] == "xgboost"
    assert len(ran) > loop.FALLBACK_CHUNK  # порций было несколько


def _metric_spec(spec, metric):
    return spec.model_copy(update={"metric": spec.metric.model_copy(update={"name": metric})})


def test_metric_objective_first_for_best_xgboost_catboost(spec):
    """#237: цель под метрику — первой у лучшей модели и xgboost, у catboost — перед сменой модели на catboost."""
    best = (hyp("best", model="lightgbm"), result("b", BASE))
    hyps = fallback_hypotheses(_metric_spec(spec, "pinball@0.1"), [best], best, RunBudget(total_seconds=3600))
    objective = [(i, h.params["model"]) for i, h in enumerate(hyps) if h.params.get("objective") == "metric"]
    assert objective[:2] == [(0, "lightgbm"), (1, "xgboost")]
    catboost = next(i for i, m in objective if m == "catboost")
    assert catboost < next(i for i, h in enumerate(hyps) if _short(h) == "catboost" and "objective" not in h.params)


def test_metric_objective_skipped_without_own_objective(spec):
    """RMSE учится на L2 и так; у XGBoost нет цели MAPE — повтор на L2 не добавляем."""
    best = (hyp("best", model="lightgbm"), result("b", BASE))
    rmse = fallback_hypotheses(spec, [best], best, RunBudget(total_seconds=3600))
    assert not any(h.params.get("objective") == "metric" for h in rmse)
    mape = fallback_hypotheses(_metric_spec(spec, "mape"), [best], best, RunBudget(total_seconds=3600))
    models = {h.params["model"] for h in mape if h.params.get("objective") == "metric" and "model" in h.params}
    assert models == {"lightgbm", "catboost"}


def test_metric_objective_not_repeated_for_best_with_it(spec):
    best = (hyp("best", model="xgboost", objective="metric"), result("b", BASE))
    hyps = fallback_hypotheses(_metric_spec(spec, "mae"), [best], best, RunBudget(total_seconds=3600))
    assert {"model": "xgboost", "objective": "metric"} not in [h.params for h in hyps]
    # цель лучшего (objective) — его предобработка: её наследуют и другие модели
    assert all(h.params.get("objective") == "metric" for h in hyps)


def test_metric_objective_replaces_explicit_objective_of_best(spec):
    """Лучший с явной целью (objective_huber) — направление для его модели без неё, иначе повтор лучшего."""
    best = (hyp("best", model="lightgbm", model_params={"objective": "huber", "num_leaves": 31}), result("b", BASE))
    hyps = fallback_hypotheses(_metric_spec(spec, "pinball@0.9"), [best], best, RunBudget(total_seconds=3600))
    assert hyps[0].params == {"model": "lightgbm", "model_params": {"num_leaves": 31}, "objective": "metric"}
