"""Разбор правила принятия (research/accept_rule_study.py): правила и проигрывание цикла агента по стенду."""

import json

import pandas as pd

from research.accept_rule_study import rule_current, rule_last, simulate


def test_last_period_rule_only_for_time_scheme():
    gains = [-0.5, 0.2, 0.9]  # шумные ранние фолды, явный выигрыш на последнем периоде
    assert not rule_current(gains, time_scheme=True)
    assert rule_last(gains, time_scheme=True)
    assert rule_last(gains, time_scheme=False) == rule_current(gains, time_scheme=False)
    assert not rule_last([0.9, 0.2, -0.1], time_scheme=True)  # последний период хуже — не принимаем


def _rows(validation: str) -> pd.DataFrame:
    """RMSE (меньше — лучше): гипотеза h честно лучше бейзлайна и лучше на последнем фолде."""
    base = {"task": "t", "id": "baseline", "metric": "rmse", "validation": validation, "honest": 10.0}
    hyp = {"task": "t", "id": "h", "metric": "rmse", "validation": validation, "honest": 9.0, "honest_gain": 1.0}
    base["fold_scores"] = json.dumps([10.0, 10.0, 10.0])
    hyp["fold_scores"] = json.dumps([10.4, 9.9, 9.0])
    return pd.DataFrame([base, hyp])


def test_simulate_follows_agent_order_and_reports_honest_gain():
    time_rows = _rows("time:year")
    assert simulate(time_rows, ["h"], rule_current)["final"] == "baseline"
    res = simulate(time_rows, ["h"], rule_last)
    assert res["final"] == "h" and abs(res["honest_gain_pct"] - 10.0) < 1e-9 and res["wrong_steps"] == 0
    assert simulate(_rows("kfold"), ["h"], rule_last)["final"] == "baseline"  # вне time-схемы правило прежнее
