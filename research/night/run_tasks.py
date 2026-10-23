"""Ночной аудит: прогнать агента на сгенерированных задачах и независимо проверить результат.

    PYTHONPATH=. uv run python research/night/run_tasks.py --tasks /tmp/night_tasks --minutes 4 [--only a,b] [--llm]

Для каждой задачи: запуск `solution.py task.md` как на платформе (копия в runs/night/<имя>/), замер времени и пика
ОЗУ (сумма RSS процесса и детей), затем
  1) разбор TaskSpec (parse_task) против правильного разбора из _truth/<имя>.json (метрика, направление, target, id);
  2) валидность submission.csv (evaluation.submission.validate_submission);
  3) НЕЗАВИСИМЫЙ скор через sklearn (не через реестр метрик репозитория) + тривиальный ответ и LightGBM по умолчанию;
  4) признаки в agent.log: число итераций, FAILED, Traceback.
Итог — строка JSON на задачу в runs/night/results.jsonl и таблица в консоль.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_csv(path: Path, truth: dict, **kw) -> pd.DataFrame:
    return pd.read_csv(path, sep=truth["sep"], decimal=truth["decimal"], encoding=truth["encoding"], **kw)


def watch_ram(proc: subprocess.Popen, stop: threading.Event, box: dict) -> None:
    try:
        parent = psutil.Process(proc.pid)
    except psutil.Error:
        return
    while not stop.is_set():
        try:
            rss = parent.memory_info().rss + sum(c.memory_info().rss for c in parent.children(recursive=True))
            box["peak_mb"] = max(box.get("peak_mb", 0), rss / 2**20)
        except psutil.Error:
            pass
        stop.wait(0.5)


def run_agent(task_dir: Path, run_dir: Path, minutes: float, llm: bool) -> dict:
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.copytree(task_dir, run_dir / "task")
    env = {**os.environ, "AGENT_TIME_BUDGET_SEC": str(int(minutes * 60)), "AGENT_RUN_DIR": str(run_dir)}
    env.pop("OPENAI_API_KEY", None) if not llm else None
    if not llm:
        env["LLM_MOCK"] = "1"
    started = time.time()
    # solution.py в корне репозитория; рабочая папка = папка задачи, как на платформе
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "solution.py"), str(run_dir / "task" / "task.md")],
        env=env,
        cwd=run_dir / "task",
        stdout=(run_dir / "stdout.txt").open("w"),
        stderr=subprocess.STDOUT,
    )
    stop, box = threading.Event(), {}
    thread = threading.Thread(target=watch_ram, args=(proc, stop, box), daemon=True)
    thread.start()
    limit = minutes * 60 * 1.5 + 120
    try:
        code = proc.wait(timeout=limit)
        status = "OK" if code == 0 else f"EXIT_{code}"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        code, status = None, "OVERRUN_KILLED"
    stop.set()
    thread.join(timeout=2)
    return {
        "seconds": round(time.time() - started, 1),
        "exit": code,
        "status": status,
        "peak_ram_mb": round(box.get("peak_mb", 0)),
    }


def log_facts(run_dir: Path) -> dict:
    log = run_dir / "agent.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    lines = text.splitlines()
    saved = [ln for ln in lines if "Сохранён ответ" in ln]
    return {
        "iterations": sum("Итерация" in ln for ln in lines),
        "failed": sum("FAILED" in ln or "упала" in ln for ln in lines),
        "tracebacks": text.count("Traceback"),
        "last_saved": saved[-1][-160:] if saved else "",
        "validation_line": next((ln[-140:] for ln in lines if "алидаци" in ln and "схем" in ln), ""),
    }


def lgbm_baseline(truth, train, test, target, id_col, kind, class_names):
    import lightgbm as lgb

    feats = [c for c in train.columns if c not in (target, id_col) and c in test.columns]
    xtr, xte = train[feats].copy(), test[feats].copy()
    for c in feats:
        if xtr[c].dtype == object or str(xtr[c].dtype) in ("bool", "string"):
            cats = pd.Categorical(pd.concat([xtr[c], xte[c]]).astype(str)).categories
            xtr[c] = pd.Categorical(xtr[c].astype(str), categories=cats)
            xte[c] = pd.Categorical(xte[c].astype(str), categories=cats)
    y = train[target]
    ok = y.notna()
    params = dict(random_state=0, n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
    if kind == "reg":
        return lgb.LGBMRegressor(**params).fit(xtr[ok], y[ok]).predict(xte)
    codes = pd.Categorical(y[ok].astype(str), categories=class_names).codes
    model = lgb.LGBMClassifier(**params).fit(xtr[ok], codes)
    proba = np.zeros((len(xte), len(class_names)))
    proba[:, model.classes_] = model.predict_proba(xte)
    return proba


def independent_score(truth, answers, sub, train, test, target):
    """Скор агента, тривиального ответа и LightGBM по умолчанию в метрике из truth (sklearn, не реестр репозитория)."""
    from sklearn import metrics as sk

    m = truth["metric"] or ("fbeta2" if truth.get("fbeta") else None)
    id_col = truth["id"]
    n = len(test)
    reg = m in ("mae", "rmse", "r2", "mape", "pinball@0.9")
    y = answers.iloc[:, 0]
    if reg:
        pred = pd.to_numeric(sub[target], errors="coerce").to_numpy(float)
        if m == "pinball@0.9":
            fn = lambda yt, p: sk.mean_pinball_loss(yt, p, alpha=0.9)  # noqa: E731
            triv = np.full(n, pd.to_numeric(train[target]).quantile(0.9))
        else:
            fn = {
                "mae": sk.mean_absolute_error,
                "rmse": lambda a, b: float(np.sqrt(sk.mean_squared_error(a, b))),
                "r2": sk.r2_score,
                "mape": lambda a, b: float(np.mean(np.abs(a - b) / np.maximum(np.abs(a), 1e-9))),
            }[m]
            triv = np.full(n, pd.to_numeric(train[target]).mean())
        yv = y.to_numpy(float)
        value, trivial = fn(yv, pred), fn(yv, triv)
        lg = fn(yv, lgbm_baseline(truth, train, test, target, id_col, "reg", None))
        greater = m == "r2"
    else:
        classes = sorted(train[target].dropna().astype(str).unique())
        yc = pd.Categorical(y.astype(str), categories=classes).codes
        prior = (
            pd.Categorical(train[target].dropna().astype(str), categories=classes)
            .value_counts(normalize=True)
            .to_numpy()
        )
        proba_lg = lgbm_baseline(truth, train, test, target, id_col, "clf", classes)
        if m == "logloss":  # колонки по классам в порядке образца
            cols = [c for c in sub.columns if c != id_col]
            P = sub[cols].to_numpy(float)
            order = [cols.index(c) for c in classes]
            P = P[:, order]
            fn = lambda yt, p: sk.log_loss(
                yt, np.clip(p / p.sum(1, keepdims=True), 1e-15, 1), labels=range(len(classes))
            )  # noqa: E731
            value, trivial, lg = fn(yc, P), fn(yc, np.tile(prior, (n, 1))), fn(yc, np.clip(proba_lg, 1e-6, 1))
            greater = False
        elif m in ("roc_auc", "pr_auc"):
            pos = len(classes) - 1
            p = pd.to_numeric(sub[target], errors="coerce").to_numpy(float)
            f = sk.roc_auc_score if m == "roc_auc" else sk.average_precision_score
            value, trivial, lg = f(yc == pos, p), f(yc == pos, np.full(n, prior[pos])), f(yc == pos, proba_lg[:, pos])
            greater = True
        else:
            labels = pd.Categorical(sub[target].astype(str), categories=classes).codes
            if (labels < 0).any():
                raise ValueError(
                    f"в ответе метки вне классов train: {sorted(set(sub[target].astype(str)) - set(classes))[:5]}"
                )
            lab_triv = np.full(n, int(np.argmax(prior)))
            lab_lg = proba_lg.argmax(1)
            if m == "accuracy":
                fn = sk.accuracy_score
            elif m == "balanced_accuracy":
                fn = sk.balanced_accuracy_score
            else:  # fbeta2 — класс 1
                fn = lambda a, b: sk.fbeta_score(a, b, beta=2)  # noqa: E731
            value, trivial, lg = fn(yc, labels), fn(yc, lab_triv), fn(yc, lab_lg)
            greater = True
    best = max if greater else min
    base = best(lg, trivial)
    if greater:
        score = 100 * max(0.0, (value - base) / (1 - base)) if base < 1 else 0.0
    else:
        score = 100 * max(0.0, 1 - value / base) if base > 0 else 0.0
    return dict(value=float(value), trivial=float(trivial), lgbm=float(lg), score=round(score, 2), greater=greater)


def check_one(name: str, tasks: Path, run_dir: Path, run: dict) -> dict:
    from agent.task_parser import parse_task
    from evaluation.submission import validate_submission

    truth = json.loads((tasks / "_truth" / f"{name}.json").read_text(encoding="utf-8"))
    out: dict = {"task": name, **run, **log_facts(run_dir)}
    task_md = run_dir / "task" / "task.md"
    spec = None
    try:
        spec = parse_task(task_md)
        want = truth["metric"]
        out["spec"] = (
            f"{spec.target}|{spec.id_column}|{spec.metric.name}|{'↑' if spec.metric.greater_is_better else '↓'}|{spec.task_type.value}"
        )
        out["parse_ok"] = (
            spec.target == truth["target"]
            and spec.id_column == truth["id"]
            and (want is None or spec.metric.name == want)
            and spec.metric.greater_is_better == truth["greater_is_better"]
        )
        out["spec_notes"] = "; ".join(spec.notes)[:200]
    except Exception as exc:  # noqa: BLE001 — для аудита важно увидеть сам сбой разбора
        out["parse_ok"] = False
        out["spec"] = f"PARSE_FAIL {type(exc).__name__}: {str(exc)[:120]}"
    sub_path = run_dir / "task" / "submission.csv"
    out["sub_exists"] = sub_path.exists()
    if not sub_path.exists():
        return out
    try:
        test = read_csv(run_dir / "task" / "test.csv", truth)
        train = read_csv(run_dir / "task" / "train.csv", truth)
        if spec is not None:
            try:
                rep = validate_submission(sub_path, spec, test)
                out["valid"] = bool(rep.ok)
                out["problems"] = "; ".join(rep.problems)[:200]
            except Exception as exc:  # noqa: BLE001
                out["valid"] = False
                out["problems"] = f"validate: {type(exc).__name__}: {exc}"[:200]
        sub = pd.read_csv(sub_path)
        answers = pd.read_csv(tasks / "_truth" / f"{name}.csv")
        if len(sub) != len(answers):
            out["score_error"] = f"строк в ответе {len(sub)}, в test {len(answers)}"
            return out
        out["score"] = independent_score(truth, answers, sub, train, test, truth["target"])
    except Exception as exc:  # noqa: BLE001
        out["score_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=Path("/tmp/night_tasks"))
    ap.add_argument("--minutes", type=float, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "night")
    args = ap.parse_args()
    names = [p.name for p in sorted(args.tasks.iterdir()) if p.is_dir() and not p.name.startswith("_")]
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        run_dir = args.out / name
        print(f"▶ {name} ({args.minutes} мин)", flush=True)
        run = run_agent(args.tasks / name, run_dir, args.minutes, args.llm)
        row = check_one(name, args.tasks, run_dir, run)
        with (args.out / "results.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        sc = row.get("score")
        print(
            f"  {row['status']} {row['seconds']} с, RAM {row['peak_ram_mb']} МБ, parse_ok={row.get('parse_ok')}, "
            f"valid={row.get('valid')}, score={sc['score'] if sc else row.get('score_error')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
