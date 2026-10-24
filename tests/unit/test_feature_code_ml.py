"""Код признаков от LLM в ml/ и цикле (#23, A-L3-1): песочница, CV, финальное обучение, генерация в цикле."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent import loop
from agent.feature_code import FEATURE_ENV, MAX_FEATURE_TRIES, propose_feature_hypothesis
from interfaces.schemas import DataProfile, ExperimentResult, ExperimentStatus, Hypothesis, HypothesisKind, RunBudget
from interfaces.stubs import stub_task_spec
from ml.experiment import fit_predict_final, run_experiment, with_feature_code

RATIO = "def add_features(df):\n    df = df.copy()\n    df['ratio'] = df['a'] / (df['b'].abs() + 1)\n    return df\n"


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))  # регрессия, target site_eui, id id


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["site_eui"] = df["a"] / (df["b"].abs() + 1) * 10 + rng.normal(scale=0.1, size=n)
    return df.iloc[:300], df.iloc[300:].drop(columns="site_eui")


@pytest.fixture
def profile():
    return DataProfile(n_train=300, n_test=100, numeric_features=["a", "b"], categorical_features=[])


def hyp(code=None, hid="feat"):
    params = {"model": "lightgbm", "n_folds": 3, "model_params": {"n_estimators": 60}}
    return Hypothesis(id=hid, kind=HypothesisKind.FEATURES, title=hid, params=params, code=code, source="llm")


def test_no_code_changes_nothing(spec, profile, data):
    train, test = data
    assert with_feature_code(spec, profile, hyp(), train, test) == (profile, train, test)


def test_code_adds_numeric_columns_to_train_test_and_profile(spec, profile, data):
    train, test = data
    new_profile, new_train, new_test = with_feature_code(spec, profile, hyp(RATIO), train, test)
    assert new_profile.numeric_features == ["a", "b", "ratio"]
    expected = train["a"] / (train["b"].abs() + 1)
    np.testing.assert_allclose(new_train["ratio"], expected)
    assert "ratio" in new_test.columns and len(new_test) == len(test)
    assert "ratio" not in train.columns  # исходные таблицы не меняются


def test_code_sees_no_target_and_no_id(spec, profile, data):
    # колонок у кода ровно столько, сколько признаков: без site_eui и id (контракт D4)
    code = "def add_features(df):\n    df = df.copy()\n    df['z'] = df['a'] + df.shape[1]\n    return df\n"
    _, new_train, _ = with_feature_code(spec, profile, hyp(code), *data)
    np.testing.assert_allclose(new_train["z"] - new_train["a"], 2)


def test_run_experiment_with_code_ok_and_better(spec, profile, data):
    train, _ = data
    plain = run_experiment(spec, profile, hyp(hid="plain"), train, RunBudget())
    feat = run_experiment(spec, profile, hyp(RATIO), train, RunBudget())
    assert plain.status == feat.status == ExperimentStatus.OK
    assert feat.mean_score < plain.mean_score  # признак-формула target делает RMSE меньше


def test_bad_code_is_failed_not_crash(spec, profile, data):
    code = "def add_features(df):\n    raise ValueError('boom')\n"
    result = run_experiment(spec, profile, hyp(code), data[0], RunBudget())
    assert result.status == ExperimentStatus.FAILED and "код признаков" in result.error


def test_fit_predict_final_with_code(spec, profile, data):
    train, test = data
    feature_hyp = hyp(RATIO)
    run_experiment(spec, profile, feature_hyp, train, RunBudget())
    answer = fit_predict_final(spec, profile, feature_hyp, train, test)
    assert list(answer.columns) == ["id", "site_eui"] and len(answer) == len(test)
    truth = test["a"] / (test["b"].abs() + 1) * 10
    assert np.mean(np.abs(answer["site_eui"].to_numpy() - truth.to_numpy())) < 1.0


class FakeLLM:
    def __init__(self, answer, available=True):
        self.answer, self._available, self.requests = answer, available, 0
        self.rub_spent, self.calls = 0.0, 0

    def available(self):
        return self._available

    def chat_json(self, messages, **kwargs):
        self.requests += 1
        return self.answer

    def chat(self, messages, **kwargs):
        return None

    def log_summary(self):
        pass


FEATURE_ANSWER = {"id": "ratio_ab", "title": "Отношение a к b", "rationale": "форма зависимости", "code": RATIO}


def test_propose_feature_hypothesis_takes_best(spec, profile, monkeypatch):
    monkeypatch.setenv(FEATURE_ENV, "1")
    best = Hypothesis(id="best", kind=HypothesisKind.HYPERPARAMS, title="b", params={"model": "xgboost"})
    h = propose_feature_hypothesis(spec, profile, [], FakeLLM(FEATURE_ANSWER), best, RunBudget())
    assert h.kind == HypothesisKind.FEATURES and h.code == RATIO
    assert h.parent_id == "best" and h.params == {"model": "xgboost"}


@pytest.mark.parametrize("env, available", [("0", True), ("1", False)])
def test_propose_feature_hypothesis_disabled(spec, profile, monkeypatch, env, available):
    monkeypatch.setenv(FEATURE_ENV, env)
    llm = FakeLLM(FEATURE_ANSWER, available=available)
    assert propose_feature_hypothesis(spec, profile, [], llm, None) is None
    assert llm.requests == 0


def run_loop(spec, profile, monkeypatch, llm, n_batches=4):
    seen = []

    def fake_experiment(spec, profile, h, train, budget):
        seen.append(h)
        return ExperimentResult(
            hypothesis_id=h.id, status=ExperimentStatus.OK, metric_name="rmse", mean_score=50.0, std_score=0.0
        )

    batches = [
        [Hypothesis(id=f"p{i}{j}", kind=HypothesisKind.HYPERPARAMS, title="t", params={"n_folds": 3 + j})]
        for i in range(n_batches)
        for j in range(1)
    ]
    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    monkeypatch.setattr(loop, "propose_hypotheses", lambda *a, **kw: batches.pop(0) if batches else [])
    base = (
        Hypothesis(id="baseline", kind=HypothesisKind.BASELINE, title="b", params={"model": "lightgbm"}),
        ExperimentResult(
            hypothesis_id="baseline", status=ExperimentStatus.OK, metric_name="rmse", mean_score=40.0, std_score=0.1
        ),
    )
    history = [base]
    loop.run_autoresearch(spec, profile, None, RunBudget(), history, base, lambda h, r: None, llm=llm)
    return seen


def test_loop_asks_for_feature_code_after_first_batch_at_most_max(spec, profile, monkeypatch):
    monkeypatch.setenv(FEATURE_ENV, "1")
    llm = FakeLLM(FEATURE_ANSWER)
    seen = run_loop(spec, profile, monkeypatch, llm)
    features = [h for h in seen if h.kind == HypothesisKind.FEATURES]
    assert seen[0].kind != HypothesisKind.FEATURES  # первая пачка — без кода: лучший ещё не известен
    assert 1 <= len(features) <= MAX_FEATURE_TRIES
    assert features[0].parent_id == "baseline"


def test_loop_without_feature_code_when_disabled(spec, profile, monkeypatch):
    monkeypatch.setenv(FEATURE_ENV, "0")
    seen = run_loop(spec, profile, monkeypatch, FakeLLM(FEATURE_ANSWER))
    assert all(h.kind != HypothesisKind.FEATURES for h in seen)


def test_feature_code_only_after_playbook_exhausted(spec, profile, monkeypatch):
    """ADR 0014: пока в пачке есть гипотезы playbook, код признаков у LLM не просим — он идёт после playbook."""
    monkeypatch.setenv(FEATURE_ENV, "1")
    seen = run_loop(spec, profile, monkeypatch, FakeLLM(FEATURE_ANSWER), n_batches=3)
    kinds = [h.kind for h in seen]
    first_feature = kinds.index(HypothesisKind.FEATURES)
    assert all(h.source == "playbook" for h in seen[:first_feature])
    assert first_feature == 3  # все три пачки playbook прошли раньше кода признаков


def test_feature_code_disabled_by_default(spec, profile, monkeypatch):
    """#276: без переменной код признаков у LLM не просим — безопасное значение по умолчанию."""
    monkeypatch.delenv(FEATURE_ENV, raising=False)
    llm = FakeLLM(FEATURE_ANSWER)
    assert propose_feature_hypothesis(spec, profile, [], llm, None) is None
    assert llm.requests == 0
    seen = run_loop(spec, profile, monkeypatch, FakeLLM(FEATURE_ANSWER))
    assert all(h.kind != HypothesisKind.FEATURES for h in seen)


def test_loop_accepts_llm_hypotheses_strictly(spec, profile, monkeypatch):
    """#276: цикл передаёт в правило принятия strict=True только для гипотез от LLM."""
    monkeypatch.setenv(FEATURE_ENV, "1")
    calls = []

    def fake_accept(new, best, gib, strict=False):
        calls.append((new.hypothesis_id, strict))
        return False, "тест"

    monkeypatch.setattr(loop, "accept", fake_accept)
    seen = run_loop(spec, profile, monkeypatch, FakeLLM(FEATURE_ANSWER))
    by_id = {h.id: h.source for h in seen}
    assert any(strict for _, strict in calls) and any(not strict for _, strict in calls)
    assert all(strict == (by_id[hid] == "llm") for hid, strict in calls)
