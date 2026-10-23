"""#236: стенд research/bench_playbook.py с ансамблями вместо playbook (playbook.yaml не трогаем).

    uv run python research/night/r236_bench.py [--tasks tasks/public/regression ...]

Кандидаты — ансамбли из params.ensemble (#222, ml/experiment.py) и их участники по отдельности.
cv — как видит агент (его схема CV), honest — финал на train-части → честная отложенная часть, accepted — правило агента.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

import bench_playbook  # noqa: E402

ET = {"model": "lightgbm", "model_params": {"extra_trees": True}}
CANDIDATES = [
    {"id": "xgboost", "kind": "model", "title": "XGBoost", "params": {"model": "xgboost"}},
    {"id": "extra_trees", "kind": "hyperparams", "title": "LightGBM extra_trees", "params": ET},
    {"id": "ens_lgb_xgb", "kind": "ensemble", "title": "LightGBM + XGBoost", "params": {"ensemble": ["lightgbm", "xgboost"]}},
    {"id": "ens_lgb_xgb_et", "kind": "ensemble", "title": "LightGBM + XGBoost + extra_trees",
     "params": {"ensemble": ["lightgbm", "xgboost", ET]}},
]

bench_playbook.load_playbook = lambda: CANDIDATES

if __name__ == "__main__":
    sys.argv[0] = "bench_playbook.py"
    bench_playbook.main()
