"""Состязательная проверка «train против test»: AUC классификатора как детектор сдвига (идея Д отчёта ночного аудита)."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "public/regression (сдвиг по годам)": "tasks/public/regression",
    "public/multiclass": "tasks/public/multiclass",
    "practice/sales (test позже)": "tasks/practice/sales_regression_dates_rmsle",
    "practice/payments (test = след. месяц)": "tasks/practice/payments_binary_f1_rare",
    "practice/claims (сдвига нет)": "tasks/practice/claims_regression_mape_rowno",
    "practice/cancer (случайный split)": "tasks/practice/cancer_binary_auc",
    "t07 смешанные даты (test позже)": "runs/night/tasks/t07_mixed_dates_logloss",
    "t14 новые категории в test": "runs/night/tasks/t14_ids_overlap_newcat_balacc",
}
for name, rel in CASES.items():
    d = ROOT / rel
    tr = next(d.rglob("train*.csv"))
    te = next(d.rglob("test*.csv"))
    a = pd.read_csv(tr, nrows=20000)
    b = pd.read_csv(te, nrows=20000)
    cols = [c for c in b.columns if c in a.columns and c.lower() != "id"]
    single = {}
    for c in cols:  # одна колонка отделяет train от test почти идеально — это номер записи/время, а не «сдвиг модели»
        v = pd.concat([a[c], b[c]], ignore_index=True)
        if v.dtype == object:
            v = pd.Series(pd.factorize(v)[0])
        yy = np.r_[np.zeros(len(a)), np.ones(len(b))]
        single[c] = abs(roc_auc_score(yy, v.fillna(-1)) - 0.5) + 0.5
    drop = [c for c, auc in single.items() if auc > 0.97]
    cols = [c for c in cols if c not in drop]
    x = pd.concat([a[cols], b[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    for c in x.columns:
        if x[c].dtype == object:
            x[c] = pd.Categorical(x[c])
    p = cross_val_predict(
        lgb.LGBMClassifier(verbose=-1, n_jobs=2, n_estimators=100), x, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba"
    )[:, 1]
    print(f"{name}: AUC = {roc_auc_score(y, p):.3f}; отброшены колонки-«часы» (одна колонка AUC>0.97): {drop}", flush=True)
