"""Ночной аудит: даёт ли выбор цели обучения (objective) под метрику выигрыш на метриках, где агент учит L2.

    PYTHONPATH=. uv run python research/night/objective_check.py
5-fold CV LightGBM со значениями по умолчанию, метрика считается репозиторным compute_metric (для pinball — sklearn).
Задачи: практические tasks/practice (MAE, MAPE) и синтетические /tmp/night_tasks (pinball@0.9, quantile).
"""

import os

os.environ.setdefault("LLM_MOCK", "1")  # parse_task не должен звать настоящую LLM

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold

from agent.task_parser import parse_task
from evaluation.metrics import compute_metric
from ml.profiling import load_data, make_matrix, profile_data

CASES = [
    ("tasks/practice/diabetes_regression_mae/task.md", [("l2", {}), ("l1", {"objective": "l1"}), ("huber", {"objective": "huber"})]),
    ("tasks/practice/claims_regression_mape_rowno/task.md", [("l2", {}), ("l1", {"objective": "l1"}), ("mape-obj", {"objective": "mape"})]),
    ("tasks/practice/sales_regression_dates_rmsle/task.md", [("l2", {}), ("l2 on log1p", "log")]),
]
for path, variants in CASES:
    spec = parse_task(__import__("pathlib").Path(path))
    train, test = load_data(spec)
    profile = profile_data(spec, train, test)
    X, _ = make_matrix(profile, train, test)
    y = pd.to_numeric(train[spec.target]).to_numpy(float)
    print(f"\n{path.split('/')[2]}  метрика={spec.metric.name}")
    for name, kw in variants:
        scores = []
        for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
            if kw == "log":
                m = lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=4).fit(X.iloc[tr], np.log1p(y[tr]))
                p = np.expm1(m.predict(X.iloc[va]))
            else:
                m = lgb.LGBMRegressor(random_state=0, verbose=-1, n_jobs=4, **kw).fit(X.iloc[tr], y[tr])
                p = m.predict(X.iloc[va])
            scores.append(compute_metric(spec.metric.name, y[va], p))
        print(f"  {name:12} {np.mean(scores):.4f} ± {np.std(scores):.4f}")
