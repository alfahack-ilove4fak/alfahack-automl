"""Прогон агента на «чужих» задачах (research/night/make_alien_tasks.py) с независимой проверкой ответа.

    uv run python research/night/run_alien.py --minutes 6                    # все задачи
    uv run python research/night/run_alien.py --minutes 6 --only cp1251_r2 no_id_rmse

Для каждой задачи:
  1. разбор задачи агентом (parse_task, без LLM) — что он понял: target, id, метрика и направление, тип, колонки;
  2. solution.py в копии папки задачи, как на платформе (LLM_MOCK=1, бюджет --minutes): время, код выхода, пик RAM;
  3. своя проверка submission.csv по ответам генератора (не evaluation/, чтобы не повторять ошибки разбора агента):
     колонки, строки, id, пропуски; настоящая метрика задачи — своей реализацией;
  4. ориентиры: тривиальный ответ, LightGBM по умолчанию (≈ бейзлайн «DS за 5 минут») и LightGBM с целевой
     функцией под метрику (квантиль, L1, MAPE) — сколько агент недобирает из-за цели обучения;
  5. из agent.log: схема валидации, CV бейзлайна и лучшей гипотезы, число экспериментов, FAILED, когда агент закончил.
Строка результата → runs/night/alien_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ALIEN = ROOT / "runs" / "night" / "alien"
RUNS = ROOT / "runs" / "night" / "alien_runs"
RESULTS = ROOT / "runs" / "night" / "alien_results.jsonl"


# ------------------------------------------------------------------ метрики (своя реализация)


def _smape(y, p):
    denom = np.abs(y) + np.abs(p)
    ratio = np.divide(2 * np.abs(p - y), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(100 * ratio.mean())


def _pinball(y, p, a=0.9):
    d = y - p
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def true_metric(meta: dict, answers: pd.DataFrame, sub: pd.DataFrame, classes: list[str] | None) -> float:
    m = meta["metric"]
    targets = meta["targets"]
    if m == "pinball@0.9":
        return float(np.mean([_pinball(answers[t].to_numpy(float), sub[t].to_numpy(float)) for t in targets]))
    t = targets[0]
    if "per_class" in meta:
        cols = [meta["per_class"][c] for c in classes]
        proba = sub[cols].to_numpy(float)
        proba = np.clip(proba, 1e-15, 1)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return float(skm.log_loss(answers[t].astype(str), proba, labels=classes))
    y = answers[t]
    p = sub[t]
    if meta["task_type"] == "regression":
        y, p = y.to_numpy(float), pd.to_numeric(p).to_numpy(float)
        return {
            "mae": lambda: skm.mean_absolute_error(y, p),
            "rmse": lambda: float(np.sqrt(skm.mean_squared_error(y, p))),
            "r2": lambda: skm.r2_score(y, p),
            "rmsle": lambda: float(np.sqrt(skm.mean_squared_log_error(y, np.clip(p, 0, None)))),
            "smape": lambda: _smape(y, p),
            "mape": lambda: skm.mean_absolute_percentage_error(y, p),
        }[m]()
    pos = meta.get("positive")
    if m in ("roc_auc", "pr_auc", "logloss"):
        yb = (y.astype(str) == pos).astype(int).to_numpy()
        pp = pd.to_numeric(p).to_numpy(float)
        if m == "roc_auc":
            return float(skm.roc_auc_score(yb, pp))
        if m == "pr_auc":
            return float(skm.average_precision_score(yb, pp))
        return float(skm.log_loss(yb, np.clip(pp, 1e-15, 1 - 1e-15)))
    ys, ps = y.astype(str).to_numpy(), p.astype(str).to_numpy()
    if m == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(ys, ps))
    if m == "qwk":
        return float(skm.cohen_kappa_score(y.astype(int), pd.to_numeric(p).round().astype(int), weights="quadratic"))
    if m == "f2":
        return float(skm.fbeta_score(ys, ps, beta=2, pos_label=pos, zero_division=0))
    raise KeyError(m)


def organizer_score(gib: bool, value: float, base: float) -> float:
    if not (np.isfinite(value) and np.isfinite(base)):
        return 0.0
    if gib:
        room = 1.0 - base
        return float(100 * max(0.0, (value - base) / room)) if room > 0 else 0.0
    return float(100 * max(0.0, 1 - value / base)) if base > 0 else 0.0


# ------------------------------------------------------------------ ориентиры


def read_kw(meta: dict) -> dict:
    kw = {}
    if meta.get("sep"):
        kw["sep"] = meta["sep"]
    if meta.get("decimal"):
        kw["decimal"] = meta["decimal"]
    if meta.get("encoding"):
        kw["encoding"] = meta["encoding"]
    return kw


def load_task(name: str, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    task = ALIEN / name
    train_path = next(task.rglob("train*.csv"))
    test_path = next(task.rglob("test*.csv"))
    kw = read_kw(meta)
    return pd.read_csv(train_path, low_memory=False, **kw), pd.read_csv(test_path, low_memory=False, **kw)


def _frames(train: pd.DataFrame, test: pd.DataFrame, meta: dict):
    drop = set(meta["targets"]) | ({meta["id"]} if meta["id"] else set())
    cols = [c for c in train.columns if c in test.columns and c not in drop and not test[c].isna().all()]
    xtr, xte = train[cols].copy(), test[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(xtr[c]):
            xtr[c] = xtr[c].astype(str).astype("category")
            xte[c] = pd.Categorical(xte[c].astype(str), categories=xtr[c].cat.categories)
    xtr.columns = xte.columns = [re.sub(r"\W+", "_", str(c)) + f"_{i}" for i, c in enumerate(cols)]
    return xtr, xte


def references(meta: dict, train: pd.DataFrame, test: pd.DataFrame, answers: pd.DataFrame, classes) -> dict:
    """Тривиальный ответ, LightGBM по умолчанию и LightGBM с целью под метрику — в формате сабмита → метрика."""
    import lightgbm as lgb

    out = {}
    m, targets = meta["metric"], meta["targets"]
    train = train.dropna(subset=targets).reset_index(drop=True)
    xtr, xte = _frames(train, test, meta)
    params = {"random_state": 0, "n_jobs": 4, "verbose": -1}

    def as_sub(preds: dict) -> pd.DataFrame:
        return pd.DataFrame(preds)

    # тривиальный
    triv = {}
    for t in targets:
        y = train[t]
        if meta["task_type"] == "regression":
            triv[t] = np.full(len(test), y.mean())
        elif "per_class" in meta:
            shares = y.astype(str).value_counts(normalize=True)
            for c in classes:
                triv[meta["per_class"][c]] = np.full(len(test), shares.get(c, 1e-6))
        elif m in ("roc_auc", "pr_auc", "logloss"):
            triv[t] = np.full(len(test), (y.astype(str) == meta["positive"]).mean())
        else:
            triv[t] = np.full(len(test), y.astype(str).mode()[0], dtype=object)
    out["trivial"] = true_metric(meta, answers, as_sub(triv), classes)

    def lgbm(objective: dict | None = None) -> pd.DataFrame:
        preds = {}
        for t in targets:
            y = train[t]
            if meta["task_type"] == "regression":
                model = lgb.LGBMRegressor(**params, **(objective or {})).fit(xtr, y.to_numpy(float))
                preds[t] = model.predict(xte)
            else:
                labels = sorted(y.astype(str).unique())
                codes = y.astype(str).map({c: i for i, c in enumerate(labels)}).to_numpy()
                model = lgb.LGBMClassifier(**params).fit(xtr, codes)
                proba = model.predict_proba(xte)
                if "per_class" in meta:
                    for i, c in enumerate(labels):
                        preds[meta["per_class"][c]] = proba[:, i]
                elif m in ("roc_auc", "pr_auc", "logloss"):
                    preds[t] = proba[:, labels.index(meta["positive"])]
                else:
                    preds[t] = np.array(labels, dtype=object)[proba.argmax(axis=1)]
        return as_sub(preds)

    out["lgbm_default"] = true_metric(meta, answers, lgbm(), classes)
    aligned = {
        "pinball@0.9": {"objective": "quantile", "alpha": 0.9},
        "mae": {"objective": "l1"},
        "mape": {"objective": "mape"},
    }.get(m)
    if aligned:
        out["lgbm_aligned"] = true_metric(meta, answers, lgbm(aligned), classes)
    return out


# ------------------------------------------------------------------ проверка сабмита


def check_submission(path: Path, meta: dict, answers: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    problems = []
    if not path.exists():
        return None, ["нет submission.csv"]
    try:
        sub = pd.read_csv(path, dtype={meta["id"]: str} if meta["id"] else None)
    except Exception as exc:  # noqa: BLE001
        return None, [f"не читается: {exc}"]
    expected = meta["submission_columns"]
    if list(sub.columns) != expected:
        problems.append(f"колонки {list(sub.columns)} ≠ {expected}")
    if len(sub) != len(answers):
        problems.append(f"строк {len(sub)} ≠ {len(answers)}")
    if meta["id"] and meta["id"] in sub.columns and len(sub) == len(answers):
        if list(sub[meta["id"]].astype(str)) != list(answers[meta["id"]].astype(str)):
            if set(sub[meta["id"]].astype(str)) == set(answers[meta["id"]].astype(str)):
                problems.append("id те же, но порядок другой")
                sub = (
                    sub.set_index(sub[meta["id"]].astype(str))
                    .loc[answers[meta["id"]].astype(str)]
                    .reset_index(drop=True)
                )
            else:
                problems.append("id не совпадают с test")
    cols = [c for c in expected if c != meta["id"] and c in sub.columns]
    if cols and sub[cols].isna().any().any():
        problems.append(f"пропуски в ответе: {int(sub[cols].isna().sum().sum())}")
    return sub, problems


# ------------------------------------------------------------------ разбор лога агента


def parse_log(log: Path) -> dict:
    if not log.exists():
        return {"log": "нет agent.log"}
    text = log.read_text(encoding="utf-8", errors="replace")
    info = {}
    m = re.search(r"Задача: (\w+), target=(.+?), метрика=(\S+), классов=(\S+)", text)
    if m:
        info["log_task"] = m.groups()
    m = re.search(r"Схема валидации: (.+)", text)
    info["validation"] = m.group(1)[:120] if m else None
    m = re.search(r"Шаг 1: бейзлайн (\S+)=(\S+) ± (\S+)", text)
    info["baseline_cv"] = m.group(2) if m else None
    saves = re.findall(r"Сохранён ответ «(.+?)» \((.+?)\), проверка: (.+)", text)
    if saves:
        info["final_answer"] = saves[-1][0]
        info["final_cv"] = saves[-1][1]
        info["save_check"] = saves[-1][2][:120]
    info["iterations"] = len(re.findall(r"Итерация \d+:", text))
    info["accepted"] = len(re.findall(r"✓ лучше", text))
    info["failed"] = len(re.findall(r"✗ (failed|timeout)", text))
    info["failed_reasons"] = sorted(set(re.findall(r"✗ (?:failed|timeout): (.{0,100})", text)))[:4]
    info["tracebacks"] = text.count("Traceback")
    info["errors"] = [ln[:200] for ln in text.splitlines() if " ERROR " in ln or "| ERROR" in ln][:6]
    m = re.search(r"Готово за (\d+) с", text)
    info["done_sec"] = int(m.group(1)) if m else None
    info["loop_end"] = (
        re.findall(r"(Время на исходе.*|Гипотезы закончились.*|цикл AutoResearch пропускаем.*)", text) or [None]
    )[-1]
    info["warnings"] = sorted(set(re.findall(r"WARNING.*?\| (.{0,140})", text)))[:6]
    return info


# ------------------------------------------------------------------ прогон


def spec_summary(task_md: Path) -> dict:
    """Что агент понял из задачи (отдельным процессом: падение разбора не роняет раннер)."""
    code = (
        "import json,sys; from agent.task_parser import parse_task; s=parse_task(__import__('pathlib').Path(sys.argv[1]));"
        "print(json.dumps({'task_type':s.task_type.value,'target':s.target,'extra':s.extra_targets,'id':s.id_column,"
        "'metric':s.metric.name,'gib':s.metric.greater_is_better,'cols':s.submission_columns,"
        "'classes':(s.class_labels or [])[:8],'n_classes':len(s.class_labels or []),'notes':s.notes},ensure_ascii=False))"
    )
    env = {**os.environ, "LLM_MOCK": "1", "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c", code, str(task_md)], capture_output=True, text=True, env=env, cwd=ROOT, timeout=600
    )
    if proc.returncode != 0:
        return {"parse_error": (proc.stderr.strip().splitlines() or ["?"])[-1][:300]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_agent(task_dir: Path, minutes: float, extra_env: dict | None = None) -> dict:
    env = {
        **os.environ,
        "AGENT_TIME_BUDGET_SEC": str(int(minutes * 60)),
        "AGENT_RUN_DIR": str(task_dir.parent),
        "LLM_MOCK": "1",
        **(extra_env or {}),
    }
    started = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "solution.py"), str(task_dir / "task.md")],
        env=env,
        cwd=task_dir,
        stdout=subprocess.DEVNULL,
        stderr=open(task_dir.parent / "stderr.txt", "w"),
    )
    peak = 0.0
    ps = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
            peak = max(peak, rss / 2**20)
        except psutil.Error:
            pass
        time.sleep(0.5)
    return {"seconds": round(time.time() - started, 1), "exit_code": proc.returncode, "peak_ram_mb": round(peak)}


def run_one(name: str, minutes: float, tag: str = "", extra_env: dict | None = None) -> dict:
    meta = json.loads((ALIEN / "_answers" / f"{name}.json").read_text(encoding="utf-8"))
    answers = pd.read_csv(ALIEN / "_answers" / f"{name}.csv", dtype={meta["id"]: str} if meta["id"] else None)
    run_dir = RUNS / f"{name}{tag}"
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.copytree(ALIEN / name, run_dir / "task")
    row = {"task": name, "tag": tag, "minutes": minutes, "metric": meta["metric"], "checks": meta["checks"]}
    row["spec"] = spec_summary(run_dir / "task" / "task.md")
    row.update(run_agent(run_dir / "task", minutes, extra_env))
    sub, problems = check_submission(run_dir / "task" / "submission.csv", meta, answers)
    row["problems"] = problems
    train, test = load_task(name, meta)
    t = meta["targets"][0]
    classes = sorted(train[t].dropna().astype(str).unique()) if meta["task_type"] != "regression" else None
    row["value"] = None
    if sub is not None and not any(p.startswith(("колонки", "строк", "id не")) for p in problems):
        try:
            row["value"] = true_metric(meta, answers, sub, classes)
        except Exception as exc:  # noqa: BLE001
            row["value_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        row["ref"] = references(meta, train, test, answers, classes)
    except Exception as exc:  # noqa: BLE001
        row["ref_error"] = f"{type(exc).__name__}: {exc}"[:200]
    gib = meta["greater_is_better"]
    ref = row.get("ref", {})
    if row["value"] is not None and ref:
        strong = (max if gib else min)(ref["trivial"], ref["lgbm_default"])
        row["score_vs_trivial"] = round(organizer_score(gib, row["value"], ref["trivial"]), 2)
        row["score_vs_strong"] = round(organizer_score(gib, row["value"], strong), 2)
    row["log"] = parse_log(run_dir / "agent.log")
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(
        f"{'✅' if not problems else '❌'} {name}{tag}: {row['seconds']} с, выход {row['exit_code']}, RAM {row['peak_ram_mb']} МБ, "
        f"{meta['metric']}={row['value']} | ref {ref} | баллы {row.get('score_vs_trivial')} / {row.get('score_vs_strong')} | "
        f"проблемы {problems} | spec {row['spec']}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=6)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    names = sorted(p.stem for p in (ALIEN / "_answers").glob("*.json"))
    for name in names:
        if not args.only or name in args.only:
            try:
                run_one(name, args.minutes, args.tag)
            except Exception as exc:  # noqa: BLE001 — одна задача не роняет ночной прогон
                print(f"💥 {name}: раннер упал: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
