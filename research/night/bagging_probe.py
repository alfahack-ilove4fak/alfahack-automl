# ruff: noqa: B023
"""Проба идеи «простаивающие часы → бэггинг/усреднение»: одна LightGBM против среднего из 7 моделей с разными seed и подвыборкой (повторная 5×2 CV)."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold


def prep(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.Categorical(df[c])
    return df


def go(name, df, target, kind, metric, extra=None):
    df = prep(df.dropna(subset=[target]).reset_index(drop=True))
    X, y = df.drop(columns=[target]), df[target]
    if kind == "clf":
        y = pd.Series(pd.factorize(y)[0])
    cv = (RepeatedStratifiedKFold if kind == "clf" else RepeatedKFold)(n_splits=5, n_repeats=2, random_state=969)
    one, bag = [], []
    for tr, va in cv.split(X, y):

        def fit(seed, sub):
            kw = dict(verbose=-1, n_jobs=2, n_estimators=300, learning_rate=0.05, random_state=seed, **(extra or {}))
            if sub:
                kw.update(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
            M = lgb.LGBMClassifier if kind == "clf" else lgb.LGBMRegressor
            m = M(**kw).fit(X.iloc[tr], y.iloc[tr])
            return m.predict_proba(X.iloc[va])[:, 1] if kind == "clf" else m.predict(X.iloc[va])

        p1 = fit(0, False)
        pb = np.mean([fit(s, True) for s in range(7)], axis=0)
        one.append(metric(y.iloc[va], p1))
        bag.append(metric(y.iloc[va], pb))
    print(
        name, f"одна {np.mean(one):.4f}±{np.std(one):.4f}  бэггинг×7 {np.mean(bag):.4f}±{np.std(bag):.4f}", flush=True
    )


# dia = pd.read_csv("tasks/practice/diabetes_regression_mae/train.csv").drop(columns=["id"], errors="ignore")
# go("diabetes MAE", dia, dia.columns[-1], "reg", skm.mean_absolute_error)
net = pd.read_csv("tasks/practice/network_binary_logloss_crooked/data/train_flows.csv")
net = net.drop(columns=[c for c in net.columns if c.endswith("_id")], errors="ignore")
tgt = [
    c
    for c in net.columns
    if c not in pd.read_csv("tasks/practice/network_binary_logloss_crooked/data/test_flows.csv", nrows=1).columns
][0]
go("network logloss", net.sample(8000, random_state=0), tgt, "clf", skm.log_loss)
go(
    "t05 wide PR-AUC",
    pd.read_csv("runs/night/tasks/t05_wide_1500_prauc/train.csv").drop(columns=["id"]),
    "target",
    "clf",
    skm.average_precision_score,
)
