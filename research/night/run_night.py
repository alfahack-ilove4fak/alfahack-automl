"""Прогон агента на синтетических задачах ночного аудита и независимая проверка результата.

    uv run python research/night/run_night.py runs/night/tasks t01_semicolon_bom_ru --minutes 5 [--mock-llm]

Печатает одну JSON-строку: упал ли агент, валиден ли ответ, как разобрана метрика (TaskSpec) против ожидаемой,
свой независимый скор (sklearn) против тривиального ответа и против LightGBM по умолчанию, секунды, пик RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path: Path, sep: str, decimal: str):
    return pd.read_csv(path, sep=sep, decimal=decimal, encoding="utf-8-sig")


def map_at_k(y, pred, k=3):
    s = []
    for t, p in zip(y, pred, strict=True):
        items = str(p).split()[:k]
        s.append(1 / (items.index(t) + 1) if t in items else 0.0)
    return float(np.mean(s))


def score(metric: str, y, pred):
    """pred: для logloss/auc/pr_auc/pinball — числа/вероятности, для остальных — метки."""
    if metric == "mae":
        return float(skm.mean_absolute_error(y, pred))
    if metric == "r2":
        return float(skm.r2_score(y, pred))
    if metric == "mape":
        return float(skm.mean_absolute_percentage_error(y, pred))
    if metric == "smape":
        y, pred = np.asarray(y, float), np.asarray(pred, float)
        return float(np.mean(2 * np.abs(y - pred) / (np.abs(y) + np.abs(pred) + 1e-12)) * 100)
    if metric.startswith("pinball"):
        return float(skm.mean_pinball_loss(y, pred, alpha=0.9))
    if metric == "roc_auc":
        return float(skm.roc_auc_score(y, pred))
    if metric == "pr_auc":
        return float(skm.average_precision_score(y, pred))
    if metric == "accuracy":
        return float(skm.accuracy_score(y, pred))
    if metric == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(y, pred))
    if metric == "qwk":
        return float(skm.cohen_kappa_score(y, pred, weights="quadratic"))
    if metric == "fbeta":
        return float(skm.fbeta_score(y, pred, beta=2))
    if metric == "map@3":
        return map_at_k(y, pred)
    raise KeyError(metric)


def my_score(name, exp, ans, sub, train_target_col, sep_dec):
    """Скор присланного submission.csv независимым кодом. Возвращает (value, comment)."""
    metric = exp["metric"]
    cols = [c for c in ans.columns[1:]]
    try:
        if name == "t13_multitarget_mae":
            return float(np.mean([skm.mean_absolute_error(ans[c], sub[c]) for c in cols])), ""
        if name == "t09_50class_logloss":
            classes = sorted(sub.columns[1:])
            y = ans[ans.columns[1]].to_numpy()
            p = sub[classes].to_numpy(float)
            p = p / p.sum(1, keepdims=True)
            return float(skm.log_loss(y, p, labels=classes)), ""
        y = ans[ans.columns[-1]].to_numpy()
        col = sub.columns[-1]
        pred = sub[col].to_numpy()
        if metric in ("logloss",):
            return float(skm.log_loss(y, np.clip(pred.astype(float), 1e-15, 1 - 1e-15))), ""
        return score(metric, y, pred), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"скор не посчитан: {type(exc).__name__}: {exc}"[:200]


def trivial_value(name, exp, tr, ans, target):
    metric = exp["metric"]
    y = ans[ans.columns[-1]].to_numpy()
    n = len(ans)
    try:
        if name == "t13_multitarget_mae":
            return float(
                np.mean([skm.mean_absolute_error(ans[c], np.full(n, tr[c].median())) for c in ans.columns[1:]])
            )
        t = tr[target].dropna()
        if metric in ("mae", "r2", "mape", "smape") or metric.startswith("pinball"):
            const = t.quantile(0.9) if metric.startswith("pinball") else (t.mean() if metric == "r2" else t.median())
            return score(metric, y, np.full(n, const))
        if metric == "logloss" and name != "t09_50class_logloss":
            return float(skm.log_loss(y, np.full(n, t.mean())))
        if metric in ("roc_auc",):
            return 0.5
        if metric == "pr_auc":
            return float(np.mean(y))
        if name == "t09_50class_logloss":
            return float(np.log(50))
        if metric == "map@3":
            top = " ".join(t.value_counts().index[:3])
            return map_at_k(y, [top] * n)
        return score(metric, y, np.full(n, t.mode().iloc[0]))
    except Exception:  # noqa: BLE001
        return None


def lgbm_default(name, exp, tr, te, ans, target):
    """LightGBM с параметрами по умолчанию — как «сильный бейзлайн»; None, если задача не для него."""
    import lightgbm as lgb

    if name in ("t13_multitarget_mae", "t16_map3_ranking"):
        return None
    metric = exp["metric"]
    X = tr.drop(columns=[target]).copy()
    T = te.copy()
    for c in X.columns:
        if X[c].dtype == object:
            cats = pd.Categorical(X[c]).categories
            X[c] = pd.Categorical(X[c], categories=cats)
            T[c] = pd.Categorical(T[c], categories=cats)
    X = X.loc[:, [c for c in X.columns if c in T.columns]]
    T = T[X.columns]
    y = tr[target]
    mask = y.notna()
    X, y = X[mask], y[mask]
    yt = ans[ans.columns[-1]].to_numpy()
    if metric in ("mae", "r2", "mape", "smape") or metric.startswith("pinball"):
        kw = {"objective": "quantile", "alpha": 0.9} if metric.startswith("pinball") else {}
        pred = lgb.LGBMRegressor(verbose=-1, n_jobs=4, **kw).fit(X, y).predict(T)
        return score(metric, yt, pred)
    clf = lgb.LGBMClassifier(verbose=-1, n_jobs=4).fit(X, y)
    if name == "t09_50class_logloss":
        return float(skm.log_loss(yt, clf.predict_proba(T), labels=list(clf.classes_)))
    if metric in ("logloss", "roc_auc", "pr_auc"):
        return score(metric, yt, np.clip(clf.predict_proba(T)[:, 1], 1e-15, 1 - 1e-15))
    return score(metric, yt, clf.predict(T))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("name")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--no-lgbm", action="store_true")
    a = ap.parse_args()
    exp = json.loads((a.root / "_expected.json").read_text())[a.name]
    src = a.root / a.name
    run = ROOT / "runs" / "night" / "work" / a.name
    shutil.rmtree(run, ignore_errors=True)
    shutil.copytree(src, run / "task")
    task_md = next((run / "task").glob("task.md"))
    sep, dec = (";", ",") if a.name == "t01_semicolon_bom_ru" else (",", ".")
    env = {**os.environ, "AGENT_TIME_BUDGET_SEC": str(int(a.minutes * 60)), "AGENT_RUN_DIR": str(run)}
    if a.mock_llm:
        env["LLM_MOCK"] = "1"
    out: dict = {"task": a.name, "expected_metric": exp["metric"]}
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "solution.py"), str(task_md)],
        env=env,
        cwd=run,
        stdout=(run / "stdout.txt").open("w"),
        stderr=subprocess.STDOUT,
    )
    peak = [0]

    def watch():
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                peak[0] = max(peak[0], rss)
            except psutil.Error:
                pass
            time.sleep(0.5)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        proc.wait(timeout=a.minutes * 60 + 120)
        out["hung"] = False
    except subprocess.TimeoutExpired:
        proc.kill()
        out["hung"] = True
    out["seconds"] = round(time.time() - t0)
    out["exit"] = proc.returncode
    out["peak_ram_mb"] = round(peak[0] / 1e6)
    sub_path = task_md.parent / "submission.csv"
    out["submission_exists"] = sub_path.exists()

    # как разобран TaskSpec
    try:
        from agent.task_parser import parse_task

        spec = parse_task(task_md)
        out["spec_metric"] = spec.metric.name
        out["spec_greater"] = spec.metric.greater_is_better
        out["spec_target"] = spec.target
        out["spec_id"] = spec.id_column
        out["spec_type"] = spec.task_type.value
        out["metric_ok"] = spec.metric.name == exp["metric"] and spec.metric.greater_is_better == exp["greater"]
    except Exception as exc:  # noqa: BLE001
        out["spec_error"] = f"{type(exc).__name__}: {exc}"[:300]
        spec = None

    # независимая проверка формата ответа и скор
    ans = pd.read_csv(a.root / "_answers" / f"{a.name}.csv")
    te_path = next((run / "task").rglob("test.csv"))
    te = read(te_path, sep, dec)
    out["test_rows"] = len(te)
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        problems = []
        if len(sub) != len(te):
            problems.append(f"строк {len(sub)} вместо {len(te)}")
        samp = next((run / "task").rglob("sample_submission.csv"), None)
        if samp is not None:
            want = list(read(samp, sep, dec).columns)
            if list(sub.columns) != want:
                problems.append(f"колонки {list(sub.columns)[:5]} вместо {want[:5]}")
        if sub.isna().any().any():
            problems.append("NaN в ответе")
        idc = ans.columns[0]
        if idc in te.columns and idc in sub.columns and not (sub[idc].to_numpy() == te[idc].to_numpy()).all():
            problems.append("id не совпадают с test по порядку")
        out["my_format_problems"] = problems
        if len(sub) == len(ans):
            val, comment = my_score(a.name, exp, ans, sub, None, (sep, dec))
            out["my_value"] = val
            if comment:
                out["my_comment"] = comment
            tr = read(next((run / "task").rglob("train.csv")), sep, dec)
            target = ans.columns[-1] if a.name != "t13_multitarget_mae" else "y1"
            if a.name == "t01_semicolon_bom_ru":
                target = "Цена, тыс. руб"
            if a.name == "t09_50class_logloss":
                target = "cls"
            out["trivial"] = trivial_value(a.name, exp, tr, ans, target)
            if not a.no_lgbm:
                try:
                    out["lgbm_default"] = lgbm_default(a.name, exp, tr, te, ans, target)
                except Exception as exc:  # noqa: BLE001
                    out["lgbm_error"] = f"{type(exc).__name__}: {exc}"[:200]
    log = run / "agent.log"
    if log.exists():
        text = log.read_text(errors="replace")
        out["log_failed"] = text.count("FAILED")
        out["log_errors"] = text.count("Traceback")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
