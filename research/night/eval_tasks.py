"""Ночной аудит: независимая оценка submission.csv на синтетических задачах gen_tasks.py.

    uv run python research/night/eval_tasks.py <папка_прогона_с_задачей> <runs/night/tasks/_answers/имя>

Не использует TaskSpec агента: формат и метрику берёт из meta.json («паспорта» задачи), поэтому ловит и неверный
разбор. Считает метрику агента, тривиального ответа и LightGBM по умолчанию (на верно разобранных данных) и балл
по формуле организаторов к сильному бейзлайну (лучший из LightGBM и тривиального).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics as skm


def metric(name: str, y, p, pos=None) -> float:
    y = np.asarray(y)
    if name == "mae":
        return skm.mean_absolute_error(y, p)
    if name == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y, p)))
    if name == "mean_rmse":
        return float(np.mean([np.sqrt(skm.mean_squared_error(y[:, i], p[:, i])) for i in range(y.shape[1])]))
    if name == "r2":
        return skm.r2_score(y, p)
    if name == "mape":
        return skm.mean_absolute_percentage_error(y, p)
    if name == "smape":
        p = np.asarray(p, float)
        d = np.abs(y) + np.abs(p)
        return float(100 * np.mean(np.divide(2 * np.abs(p - y), d, out=np.zeros_like(d), where=d != 0)))
    if name == "rmsle":
        return float(np.sqrt(skm.mean_squared_log_error(y, np.clip(p, 0, None))))
    if name.startswith("pinball@"):
        return skm.mean_pinball_loss(y, p, alpha=float(name.split("@")[1]))
    if name == "roc_auc":
        return skm.roc_auc_score(y, p)
    if name == "pr_auc":
        return skm.average_precision_score(y, p)
    if name == "logloss":
        return skm.log_loss(y, p, labels=None if np.ndim(p) == 1 else list(range(np.shape(p)[1])))
    if name == "brier":
        return skm.brier_score_loss(y, p)
    if name == "accuracy":
        return skm.accuracy_score(y.astype(str), np.asarray(p).astype(str))
    if name == "balanced_accuracy":
        return skm.balanced_accuracy_score(y.astype(str), np.asarray(p).astype(str))
    if name == "cohen_kappa":
        return skm.cohen_kappa_score(y.astype(str), np.asarray(p).astype(str))
    if name == "f2":
        return skm.fbeta_score(y.astype(str), np.asarray(p).astype(str), beta=2, pos_label=pos)
    raise KeyError(name)


def org_score(value: float, base: float, greater: bool) -> float:
    if not (np.isfinite(value) and np.isfinite(base)):
        return 0.0
    if greater:
        return 100 * max(0.0, (value - base) / (1 - base)) if base < 1 else 0.0
    return 100 * max(0.0, 1 - value / base) if base > 0 else 0.0


def prepare(meta: dict, train: pd.DataFrame, test: pd.DataFrame):
    drop = set(meta["targets"]) | {meta["id"], "amount_check"}
    cols = [c for c in train.columns if c not in drop and c in test.columns]
    xtr, xte = train[cols].copy(), test[cols].copy()
    for c in cols:
        if xtr[c].dtype == object:
            cats = pd.Index(pd.concat([xtr[c], xte[c]]).astype(str).unique())
            xtr[c] = pd.Categorical(xtr[c].astype(str), categories=cats)
            xte[c] = pd.Categorical(xte[c].astype(str), categories=cats)
    xtr.columns = xte.columns = [f"c{i}" for i in range(len(cols))]
    return xtr, xte


def lgbm_baseline(meta: dict, train: pd.DataFrame, test: pd.DataFrame):
    """LightGBM по умолчанию на верно разобранной задаче — «простейшая модель DS за 5 минут»."""
    xtr, xte = prepare(meta, train, test)
    kind, targets = meta["kind"], meta["targets"]
    params = {"verbose": -1, "n_jobs": 4, "random_state": 0}
    if kind == "value":
        if meta["metric"].startswith("pinball@"):
            params.update(objective="quantile", alpha=float(meta["metric"].split("@")[1]))
        outs = []
        for t in targets:
            ok = train[t].notna()
            outs.append(lgb.LGBMRegressor(**params).fit(xtr[ok], train.loc[ok, t]).predict(xte))
        return np.column_stack(outs) if len(outs) > 1 else outs[0]
    y = train[targets[0]].astype(str)
    model = lgb.LGBMClassifier(**params).fit(xtr, y)
    if kind == "proba":
        return model.predict_proba(xte)[:, list(model.classes_).index("1")]
    if kind == "proba_cols":
        proba = pd.DataFrame(model.predict_proba(xte), columns=model.classes_)
        return proba[meta["submission"][1:]].to_numpy()
    return model.predict(xte)


def trivial(meta: dict, train: pd.DataFrame, n: int):
    kind, targets = meta["kind"], meta["targets"]
    if kind == "value":
        cols = [np.full(n, train[t].mean()) for t in targets]
        return np.column_stack(cols) if len(cols) > 1 else cols[0]
    y = train[targets[0]].astype(str)
    if kind == "proba":
        return np.full(n, (y == "1").mean())
    if kind == "proba_cols":
        shares = y.value_counts(normalize=True)
        return np.tile([shares.get(c, 0) for c in meta["submission"][1:]], (n, 1))
    return np.full(n, y.mode()[0])


def truth(meta: dict, test: pd.DataFrame):
    kind, targets = meta["kind"], meta["targets"]
    if kind == "value":
        return test[targets].to_numpy() if len(targets) > 1 else test[targets[0]].to_numpy()
    if kind == "proba":
        return test[targets[0]].astype(int).to_numpy()
    if kind == "proba_cols":
        return pd.Categorical(test[targets[0]], categories=meta["submission"][1:]).codes
    return test[targets[0]].astype(str).to_numpy()


def evaluate(run_task_dir: Path, ans_dir: Path) -> dict:
    meta = json.loads((ans_dir / "meta.json").read_text())
    train, test = pd.read_pickle(ans_dir / "train.pkl"), pd.read_pickle(ans_dir / "test_full.pkl")
    train = train[train[meta["targets"][0]].notna()]
    name, greater = meta["metric"], meta["greater"]
    y = truth(meta, test)
    res = {"task": ans_dir.name, "metric": name}
    res["trivial"] = metric(name, y, trivial(meta, train, len(test)), meta.get("pos"))
    try:
        res["lgbm"] = metric(name, y, lgbm_baseline(meta, train, test), meta.get("pos"))
    except Exception as exc:  # noqa: BLE001
        res["lgbm"] = float("nan")
        res["lgbm_error"] = str(exc)[:120]
    candidates = [v for v in (res["trivial"], res["lgbm"]) if np.isfinite(v)]
    res["strong"] = (max if greater else min)(candidates)
    sub_path = run_task_dir / "submission.csv"
    problems = []
    if not sub_path.exists():
        res.update(valid=False, problems=["нет submission.csv"], agent=float("nan"), score=0.0)
        return res
    sub = pd.read_csv(sub_path, dtype=str)
    if list(sub.columns) != meta["submission"]:
        problems.append(f"колонки {list(sub.columns)[:4]} ≠ {meta['submission'][:4]}")
    if len(sub) != len(test):
        problems.append(f"строк {len(sub)} ≠ {len(test)}")
    expected_ids = (
        np.arange(len(test)).astype(str) if meta.get("id_is_rowno") else test[meta["id"]].astype(str).to_numpy()
    )
    if not problems and not np.array_equal(sub[meta["id"]].to_numpy(), expected_ids):
        problems.append("id не совпадают с test по порядку/составу")
    agent = float("nan")
    if not problems:
        try:
            cols = meta["submission"][1:]
            if meta["kind"] in ("value", "proba", "proba_cols"):
                pred = sub[cols].apply(pd.to_numeric, errors="raise").to_numpy()
                pred = pred[:, 0] if pred.shape[1] == 1 else pred
            else:
                pred = sub[cols[0]].to_numpy()
                bad = set(pred) - set(train[meta["targets"][0]].astype(str))
                if bad:
                    problems.append(f"метки вне классов train: {sorted(bad)[:3]}")
            if not problems:
                agent = metric(name, y, pred, meta.get("pos"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"метрика не считается: {type(exc).__name__}: {str(exc)[:100]}")
    res.update(valid=not problems, problems=problems, agent=agent)
    res["score"] = org_score(agent, res["strong"], greater) if not problems else 0.0
    res["score_vs_trivial"] = org_score(agent, res["trivial"], greater) if not problems else 0.0
    return res


if __name__ == "__main__":
    out = evaluate(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(out, ensure_ascii=False, default=float))
