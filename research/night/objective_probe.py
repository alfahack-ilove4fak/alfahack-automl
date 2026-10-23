"""Проба идеи «функция потерь по метрике»: LightGBM L2 против objective, соответствующего метрике (повторная 5×3 CV)."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics as skm

def prep(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.Categorical(df[c])
    return df

def run(name, df, target, metric, variants):
    df = prep(df.dropna(subset=[target]).reset_index(drop=True))
    X, y = df.drop(columns=[target]), df[target].to_numpy()
    res = {k: [] for k in variants}
    for tr, va in RepeatedKFold(n_splits=5, n_repeats=3, random_state=969).split(X):
        for k, kw in variants.items():
            m = lgb.LGBMRegressor(verbose=-1, n_jobs=2, n_estimators=300, learning_rate=0.05, **kw).fit(X.iloc[tr], y[tr])
            res[k].append(metric(y[va], m.predict(X.iloc[va])))
    print(name, {k: f"{np.mean(v):.4f}±{np.std(v):.4f}" for k, v in res.items()}, flush=True)

mae = skm.mean_absolute_error
p = "runs/night/tasks/"
dia = pd.read_csv("tasks/practice/diabetes_regression_mae/train.csv").drop(columns=["id"], errors="ignore")
run("diabetes MAE", dia, dia.columns[-1], mae, {"l2": {}, "l1": {"objective": "l1"}, "huber": {"objective": "huber", "alpha": 20}})
t8 = pd.read_csv(p + "t08_target_nan_mae/train.csv").drop(columns=["id"])
run("t08 MAE", t8, "y", mae, {"l2": {}, "l1": {"objective": "l1"}, "huber": {"objective": "huber", "alpha": 1.0}})
cl = pd.read_csv("tasks/practice/claims_regression_mape_rowno/train.csv")
cl = cl.drop(columns=[c for c in cl.columns if c in ("claim_id", "record_no", "id")], errors="ignore")
run("claims MAPE", cl, "claim_amount", skm.mean_absolute_percentage_error, {"l2": {}, "mape": {"objective": "mape"}})
t15 = pd.read_csv(p + "t15_wrong_paths_quantile/participants/v2/train.csv").drop(columns=["id"])
run("t15 pinball0.9", t15, "y", lambda a, b: skm.mean_pinball_loss(a, b, alpha=0.9), {"l2": {}, "quantile0.9": {"objective": "quantile", "alpha": 0.9}})
