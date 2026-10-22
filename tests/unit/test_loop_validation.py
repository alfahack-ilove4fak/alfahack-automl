"""Цикл ведёт все гипотезы в схеме валидации бейзлайна (#151, ADR 0008 D1)."""

from pathlib import Path

import pytest

from agent import loop
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, Hypothesis, HypothesisKind, RunBudget
from interfaces.stubs import stub_task_spec


def hyp(hid, **params):
    return Hypothesis(id=hid, kind=HypothesisKind.HYPERPARAMS, title=hid, params={"model": "lightgbm", **params})


def result(hid, score, validation="kfold-5"):
    return ExperimentResult(
        hypothesis_id=hid,
        status=ExperimentStatus.OK,
        metric_name="rmse",
        fold_scores=[score] * 5,
        mean_score=score,
        std_score=0.0,
        validation=validation,
    )


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))


@pytest.fixture
def profile():
    return DataProfile(n_train=500, n_test=200, numeric_features=["a"], categorical_features=[])


def run(spec, profile, monkeypatch, base, batch):
    seen = []

    def fake_experiment(spec, profile, h, train, budget):
        seen.append(h)
        validation = h.params.get("validation", "kfold")
        return result(h.id, 40.0, f"{validation}-5")

    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    batches = [batch]
    monkeypatch.setattr(loop, "propose_hypotheses", lambda *a, **kw: batches.pop(0) if batches else [])
    history = [base]
    loop.run_autoresearch(spec, profile, None, RunBudget(), history, base, lambda h, r: None, llm=None)
    return seen, history


def test_hypotheses_run_in_baseline_scheme(spec, profile, monkeypatch):
    base = (hyp("baseline", validation="time:Year"), result("baseline", 42.0, "time:Year-5"))
    seen, history = run(spec, profile, monkeypatch, base, [hyp("a"), hyp("b", model_params={"num_leaves": 15})])
    assert [h.params["validation"] for h in seen] == ["time:Year", "time:Year"]
    # в историю — исходная гипотеза: повторы ищутся по предложенным params
    assert all("validation" not in h.params for h, _ in history[1:])
    # результаты в той же схеме → правило принятия их сравнивает: «a» лучше бейзлайна и принята
    assert history[1][1].validation == "time:Year-5"


def test_no_scheme_keeps_hypothesis_unchanged(spec, profile, monkeypatch):
    base = (hyp("baseline"), result("baseline", 42.0))
    seen, _ = run(spec, profile, monkeypatch, base, [hyp("a")])
    assert "validation" not in seen[0].params


def test_scheme_kept_after_new_best(spec, profile, monkeypatch):
    # принятая гипотеза в истории без validation — следующие всё равно идут в схеме бейзлайна
    base = (hyp("baseline", validation="time:Year"), result("baseline", 42.0, "time:Year-5"))
    seen, _ = run(spec, profile, monkeypatch, base, [hyp("a"), hyp("b", model_params={"min_child_samples": 50})])
    assert all(h.params["validation"] == "time:Year" for h in seen)


def test_with_validation_helper():
    h = hyp("a")
    assert loop.with_validation(h, None) is h
    assert loop.with_validation(h, "time:Year").params["validation"] == "time:Year"
    assert "validation" not in h.params  # исходная гипотеза не меняется


def test_validation_is_not_part_of_repeat_key():
    # solution.py кладёт схему в params бейзлайна: «LightGBM по умолчанию» из playbook/LLM без схемы — всё равно повтор
    from agent.hypotheses import params_key

    assert params_key({"model": "lightgbm", "validation": "time:Year"}) == params_key({"model": "lightgbm"})
    assert params_key({"model": "lightgbm", "class_weight": "balanced"}) != params_key({"model": "lightgbm"})
@pytest.mark.parametrize(
    "run_scheme, own, skipped",
    [
        ("time:Year", "group:duplicates", True),  # чужая схема — несравнима, не запускаем
        ("time:Year", "time:Year", False),  # та же — запускаем
        (None, "time:Year", True),  # прогон в обычной CV, гипотеза во времени — несравнима
        (None, "kfold", False),  # kfold = обычная CV
    ],
)
def test_foreign_scheme_skipped_before_run(spec, profile, monkeypatch, run_scheme, own, skipped):
    # ревью #175: гипотеза со своей схемой отрабатывала целый эксперимент и только потом отклонялась
    base_params = {"validation": run_scheme} if run_scheme else {}
    base = (hyp("baseline", **base_params), result("baseline", 42.0, f"{run_scheme or 'kfold'}-5"))
    seen, history = run(spec, profile, monkeypatch, base, [hyp("g", validation=own)])
    assert (seen == []) is skipped
    if skipped:
        assert history[1][1].status == ExperimentStatus.SKIPPED and "несравнимы" in history[1][1].error
