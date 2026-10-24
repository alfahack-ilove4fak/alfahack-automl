"""Подтверждение нового лучшего на отложенной части train и терпение цикла (#276, AGENT_CONFIRM)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent import confirm, loop
from agent.confirm import CONFIRM_ENV, LLM_CONFIRM_ENV, PATIENCE_ENV, holdout_score, split_holdout
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    HypothesisKind,
    RunBudget,
    TaskType,
)
from interfaces.stubs import stub_task_spec

ENVS = (CONFIRM_ENV, LLM_CONFIRM_ENV, PATIENCE_ENV, "AGENT_FEATURE_CODE", "AGENT_FALLBACK")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_FALLBACK", "0")


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))  # регрессия, target site_eui, id id, rmse


def frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["year"] = np.repeat(np.arange(1, 7), -(-n // 6))[:n]  # 6 периодов
    df["site_eui"] = df["a"] * 3 + rng.normal(scale=0.5, size=n)
    return df


PROFILE = DataProfile(n_train=1000, n_test=0, numeric_features=["a", "b", "year"], categorical_features=[])


# ---------- отложенный кусок ----------


def test_split_time_takes_latest_whole_periods(spec):
    cv, hold = split_holdout(spec, PROFILE, frame(), "time:year")
    assert set(hold["year"]) == {6} and 6 not in set(cv["year"])  # последний период целиком, CV его не видит
    assert 0.15 <= len(hold) / 1000 <= 0.35


def test_split_time_refuses_when_too_few_periods(spec):
    df = frame()
    df["year"] = np.where(df.index < 500, 1, 2)  # 2 периода по 50%: кусок > 35% и для CV остался бы один
    assert split_holdout(spec, PROFILE, df, "time:year") is None


def test_split_group_keeps_groups_whole(spec):
    df = frame()
    df["shop"] = df.index % 50
    cv, hold = split_holdout(spec, PROFILE, df, "group:shop")
    assert not set(cv["shop"]) & set(hold["shop"])


def test_split_stratified_for_classes():
    spec = stub_task_spec(Path("tasks/demo")).model_copy(
        update={"task_type": TaskType.BINARY, "target": "y", "class_labels": ["0", "1"]}
    )
    df = frame()
    df["y"] = (np.arange(1000) % 10 == 0).astype(int)  # 10% положительных
    cv, hold = split_holdout(spec, PROFILE, df, None)
    assert len(hold) == pytest.approx(200, abs=2)
    assert hold["y"].mean() == pytest.approx(0.1, abs=0.01)


def test_split_small_train_refused(spec):
    assert split_holdout(spec, PROFILE, frame(n=200), None) is None


def test_holdout_score_real_model(spec):
    df = frame()
    cv, hold = df.iloc[:800], df.iloc[800:]
    hyp = Hypothesis(id="lgbm", kind=HypothesisKind.BASELINE, title="t", params={"model": "lightgbm", "n_folds": 3})
    hyp.params["_best_iteration"] = 100
    rmse = holdout_score(spec, PROFILE, hyp, cv, hold)
    assert 0 < rmse < 1.5  # шум 0.5, std(target) ≈ 3: модель учится


# ---------- цикл: ложное принятие на шуме отклоняется ----------

# CV «верит» гипотезе: она лучше на всех фолдах (t ≫ 3), а на отложенном куске — хуже бейзлайна.
# Так выглядит подгонка под фолды: выигрыш на тех же фолдах, которого нет на данных, которых цикл не видел.
BASE_FOLDS = [50.0, 49.0, 51.0, 50.5, 49.5]
LUCKY_FOLDS = [f - 1.0 + d for f, d in zip(BASE_FOLDS, [0.01, -0.02, 0.0, 0.02, -0.01], strict=True)]


def result(hid, folds, status=ExperimentStatus.OK):
    return ExperimentResult(
        hypothesis_id=hid,
        status=status,
        metric_name="rmse",
        fold_scores=folds,
        mean_score=float(np.mean(folds)) if folds else None,
        std_score=float(np.std(folds)) if folds else None,
        fit_seconds=1.0,
    )


def run(spec, monkeypatch, hyps, holdout_scores=None, folds_for=None, holdout_fn=None):
    """Цикл на фейковых экспериментах. holdout_scores — метрика на куске по id гипотезы (baseline — бейзлайн)."""
    seen_rows, new_best = [], []

    def fake_experiment(spec, profile, h, train, budget):
        seen_rows.append(len(train))
        return result(h.id, (folds_for or {}).get(h.id, LUCKY_FOLDS if h.id != "baseline" else BASE_FOLDS))

    batches = [list(hyps)]
    monkeypatch.setattr(loop, "run_experiment", fake_experiment)
    monkeypatch.setattr(loop, "propose_hypotheses", lambda *a, **kw: batches.pop(0) if batches else [])
    fn = holdout_fn or (lambda spec, profile, h, cv, hold: (holdout_scores or {})[h.id])
    monkeypatch.setattr(confirm, "holdout_score", fn)
    base = (Hypothesis(id="baseline", kind=HypothesisKind.BASELINE, title="b", params={}), result("b", BASE_FOLDS))
    best = loop.run_autoresearch(
        spec, PROFILE, frame(), RunBudget(), [base], base, lambda h, r: new_best.append(h.id), llm=None
    )
    return best, new_best, seen_rows


def hyp(hid, source="playbook"):
    return Hypothesis(id=hid, kind=HypothesisKind.HYPERPARAMS, title=hid, params={"model": "xgboost"}, source=source)


def test_default_off_accepts_by_cv_and_uses_full_train(spec, monkeypatch):
    best, new_best, rows = run(spec, monkeypatch, [hyp("lucky")])
    assert new_best == ["lucky"] and rows == [1000]  # как раньше: принято по CV, CV на всём train


def test_noise_win_rejected_on_holdout(spec, monkeypatch):
    """AGENT_CONFIRM=1: по CV лучше, на отложенном куске хуже бейзлайна — не принимаем."""
    monkeypatch.setenv(CONFIRM_ENV, "1")
    best, new_best, rows = run(spec, monkeypatch, [hyp("lucky")], holdout_scores={"baseline": 50.0, "lucky": 50.4})
    assert new_best == [] and best[0].id == "baseline"
    assert rows == [800, 800]  # бейзлайн пересчитан на части без куска, гипотеза — там же


def test_honest_win_confirmed_on_holdout(spec, monkeypatch):
    monkeypatch.setenv(CONFIRM_ENV, "1")
    best, new_best, _ = run(spec, monkeypatch, [hyp("good")], holdout_scores={"baseline": 50.0, "good": 49.1})
    assert new_best == ["good"] and best[0].id == "good"


def test_llm_confirm_only_checks_llm(spec, monkeypatch):
    """AGENT_LLM_CONFIRM=1: гипотеза playbook — по CV, гипотеза LLM — только с подтверждением (#276, п. 3)."""
    monkeypatch.setenv(LLM_CONFIRM_ENV, "1")
    folds = {"pb": LUCKY_FOLDS, "llm_idea": [f - 0.5 for f in LUCKY_FOLDS]}
    scores = {"baseline": 50.0, "pb": 50.4, "llm_idea": 50.6}  # на куске обе хуже
    _, new_best, _ = run(spec, monkeypatch, [hyp("pb"), hyp("llm_idea", "llm")], scores, folds)
    assert new_best == ["pb"]


def test_patience_stops_loop(spec, monkeypatch):
    monkeypatch.setenv(CONFIRM_ENV, "1")
    monkeypatch.setenv(PATIENCE_ENV, "3")
    hyps = [hyp(f"h{i}") for i in range(6)]
    folds = {h.id: [f + 1.0 for f in BASE_FOLDS] for h in hyps}  # все хуже
    _, _, rows = run(spec, monkeypatch, hyps, {"baseline": 50.0}, folds)
    assert len(rows) == 1 + 3  # бейзлайн + 3 гипотезы, дальше стоп


def test_holdout_failure_does_not_break_loop(spec, monkeypatch):
    """Проверка на куске упала — гипотеза принимается по CV, цикл не падает."""
    monkeypatch.setenv(CONFIRM_ENV, "1")

    def boom(spec, profile, h, cv, hold):
        if h.id == "baseline":
            return 50.0
        raise RuntimeError("сломалось")

    _, new_best, _ = run(spec, monkeypatch, [hyp("lucky")], holdout_fn=boom)
    assert new_best == ["lucky"]
