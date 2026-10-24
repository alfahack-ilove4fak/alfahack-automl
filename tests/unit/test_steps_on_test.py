"""Разбор agent.log в research/external/steps_on_test.py: какие гипотезы приняты и по какому правилу."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("steps_on_test", ROOT / "research" / "external" / "steps_on_test.py")
steps_on_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(steps_on_test)

LOG = """\
[    1.5s] INFO  agent: Шаг 1: бейзлайн rmse=0.14548 ± 0.0135
[    1.7s] INFO  agent.loop: Итерация 1: log_target — Обучение на log1p(target) [playbook]
[    2.2s] INFO  agent.loop: Итерация 2: log_target_slow — log1p(target) + медленный шаг (комбинация) [playbook]
[    3.2s] INFO  agent.loop:   ✓ лучше: 0.14549 → 0.14458 — улучшение 0.0009095, лучше на 5/5 фолдах, t=3.08 (порог 2)
[    3.6s] INFO  agent: Сохранён ответ «log_target_slow» (CV rmse=0.14458), проверка: OK
[    3.6s] INFO  agent.loop: Итерация 3: xgboost — XGBoost [playbook]
    return MODEL_FITTERS[name](spec, hypothesis, x, y, eval_set, n_estimators)
[    3.7s] INFO  agent.loop:   ✗ failed: ModuleNotFoundError: No module named 'xgboost'
[    9.2s] INFO  agent.loop: Итерация 12: lgbm_extra_trees — LightGBM со случайными порогами (extra_trees) [playbook]
[    9.8s] INFO  agent.loop:   ✓ лучше: 0.14458 → 0.14116 — улучшение 0.003413, лучше на 4/5 фолдах, t=0.85 (порог 2); \
лучше на последнем периоде (0.003413) и в среднем — time-схема
[   10.0s] INFO  agent.loop: Итерация 13: fb-994c0e1e — запасное: лучший + lgbm_extra_trees [playbook]
[   11.0s] INFO  agent.loop:   ✓ лучше: 0.14116 → 0.14000 — улучшение 0.00116, лучше на 5/5 фолдах, t=inf (порог 2)
"""


def test_accepted_steps_take_id_from_iteration_and_rule_from_reason():
    steps = steps_on_test.accepted_steps(LOG)
    assert [s["id"] for s in steps] == ["log_target_slow", "lgbm_extra_trees", "fb-994c0e1e"]
    assert [s["rule"] for s in steps] == ["t ≥ 2", steps_on_test.LAST_PERIOD, "t ≥ 2"]
    assert steps[0]["t"] == 3.08 and steps[1]["t"] == 0.85 and steps[2]["t"] == float("inf")


def test_accepted_steps_empty_log():
    assert steps_on_test.accepted_steps("") == []
    assert steps_on_test.accepted_steps("[ 1.0s] INFO  agent.loop:   ✓ лучше: без итерации до этого") == []
