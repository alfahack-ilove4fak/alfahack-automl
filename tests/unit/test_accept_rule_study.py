"""Разбор правила принятия (research/accept_rule_study.py): правила и проигрывание цикла агента по стенду."""

import json

import pandas as pd

from research.accept_rule_study import RULES, rule_current, rule_deployed, rule_last, simulate


def test_last_period_rule_only_for_time_scheme():
    gains = [-0.5, 0.2, 0.9]  # шумные ранние фолды, явный выигрыш на последнем периоде
    assert not rule_current(gains, time_scheme=True)
    assert rule_last(gains, time_scheme=True)
    assert rule_last(gains, time_scheme=False) == rule_current(gains, time_scheme=False)
    assert not rule_last([0.9, 0.2, -0.1], time_scheme=True)  # последний период хуже — не принимаем


def _rows(validation: str, hyp_folds: tuple = (10.4, 9.9, 9.0)) -> pd.DataFrame:
    """RMSE (меньше — лучше): гипотеза h честно лучше бейзлайна и лучше на последнем фолде."""
    base = {"task": "t", "id": "baseline", "metric": "rmse", "validation": validation, "honest": 10.0}
    hyp = {"task": "t", "id": "h", "metric": "rmse", "validation": validation, "honest": 9.0, "honest_gain": 1.0}
    base["fold_scores"] = json.dumps([10.0, 10.0, 10.0])
    hyp["fold_scores"] = json.dumps(list(hyp_folds))
    return pd.DataFrame([base, hyp])


def test_simulate_follows_agent_order_and_reports_honest_gain():
    time_rows = _rows("time:year")
    assert simulate(time_rows, ["h"], rule_current)["final"] == "baseline"
    res = simulate(time_rows, ["h"], rule_last)
    assert res["final"] == "h" and abs(res["honest_gain_pct"] - 10.0) < 1e-9 and res["wrong_steps"] == 0
    assert simulate(_rows("kfold"), ["h"], rule_last)["final"] == "baseline"  # вне time-схемы правило прежнее


def test_deployed_rule_is_agent_rule_with_one_percent_gate():
    base = [10.0, 10.0, 10.0]
    # шумные фолды, выигрыш на последнем периоде, в среднем 2.3% RMSE — проходит и last, и действующее правило
    assert rule_deployed([10.4, 9.9, 9.0], base, gib=False, validation="time:year")
    # в среднем 0.1% (< TIME_MIN_REL_GAIN = 1%): last принимает, агент — нет (#273)
    assert rule_last([-0.02, 0.01, 0.04], time_scheme=True)
    assert not rule_deployed([10.02, 9.99, 9.96], base, gib=False, validation="time:year")
    # вне time-схемы нужен t ≥ 2, как у current
    assert not rule_deployed([10.4, 9.9, 9.0], base, gib=False, validation="kfold")
    assert rule_deployed([9.0, 9.1, 8.9], base, gib=False, validation="kfold")
    assert "deployed" in RULES


def test_simulate_with_deployed_rule():
    assert simulate(_rows("time:year"), ["h"], rule_deployed)["final"] == "h"
    small = _rows("time:year", hyp_folds=(10.02, 9.99, 9.96))
    assert simulate(small, ["h"], rule_deployed)["final"] == "baseline"
    assert simulate(small, ["h"], rule_last)["final"] == "h"
