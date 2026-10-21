"""Стенд для playbook: прогоняет каждую гипотезу на задачах так же, как агент, и сравнивает с бейзлайном.

Две оценки на гипотезу:
  cv      — та же кросс-валидация, что в агенте (ml.experiment.run_experiment);
  honest  — одна «честная» отложенная выборка без утечек из EDA (docs/research/eda.md):
            если есть колонка времени, по которой test лежит после train, — последний период train;
            иначе — фолд StratifiedGroupKFold, где полные дубликаты держатся вместе.
Гипотеза, которая выигрывает на cv, но проигрывает на honest, скорее всего запоминает данные.

Запуск (задачи — tasks/public/* и tasks/practice/*, если есть):
    uv run python research/bench_playbook.py
    uv run python research/bench_playbook.py --tasks tasks/public/regression --only log_target lgbm_slow_lr
Результат — таблица в консоли и CSV в runs/bench/ (на платформу не едет).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.hypotheses import _applies, baseline_hypothesis, load_playbook  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from evaluation.metrics import compute_metric  # noqa: E402
from interfaces.schemas import ExperimentStatus, Hypothesis, HypothesisKind, RunBudget, TaskType  # noqa: E402
from ml.experiment import (  # noqa: E402
    EARLY_STOPPING,
    SEED,
    _encode_target,
    _is_classification,
    _make_model,
    _predict,
    run_experiment,
)
from ml.profiling import load_data, make_matrix, profile_data  # noqa: E402


def find_tasks() -> list[Path]:
    return sorted(ROOT.glob("tasks/public/*/task.md")) + sorted(ROOT.glob("tasks/practice/*/task.md"))


def time_column(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> str | None:
    """Целочисленная колонка с небольшим числом значений, где весь test лежит после train, — «период»."""
    for col in features:
        if not (pd.api.types.is_integer_dtype(train[col]) and pd.api.types.is_integer_dtype(test[col])):
            continue
        if 1 < train[col].nunique() <= 50 and test[col].min() > train[col].max():
            return col
    return None


def honest_split(spec, profile, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    features = profile.numeric_features + profile.categorical_features
    col = time_column(train, test, features)
    if col:
        last = train[col].max()
        return np.where(train[col] < last)[0], np.where(train[col] == last)[0], f"время: {col}={last}"
    groups = pd.util.hash_pandas_object(train[features], index=False).to_numpy()
    strata = train[spec.target].astype(str) if _is_classification(spec) else np.zeros(len(train))
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, strata, groups))
    return tr, va, "группы дубликатов"


def honest_score(spec, profile, hyp: Hypothesis, train: pd.DataFrame, split) -> float:
    """Одно обучение с ранней остановкой на отложенной части — тот же код модели, что в агенте."""
    tr, va, _ = split
    log_target = bool(hyp.params.get("log_target")) and spec.task_type == TaskType.REGRESSION
    x, _ = make_matrix(profile, train, train.head(0))
    y = _encode_target(spec, train[spec.target])
    y_fit = np.log1p(np.clip(y, 0, None)) if log_target else y
    model = _make_model(spec, hyp)
    model.fit(
        x.iloc[tr],
        y_fit[tr],
        eval_set=[(x.iloc[va], y_fit[va])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    return float(compute_metric(spec.metric.name, y[va], _predict(spec, model, x.iloc[va], log_target)))


def bench_task(task_md: Path, only: set[str] | None) -> list[dict]:
    spec = parse_task(task_md)
    train, test = load_data(spec)
    profile = profile_data(spec, train, test)
    split = honest_split(spec, profile, train, test)
    print(f"\n== {task_md.parent.name}: {spec.task_type.value}, {spec.metric.name}, честная часть — {split[2]}")
    hyps = [baseline_hypothesis()]
    for item in load_playbook():
        if (only and item["id"] not in only) or not _applies(item, spec, profile):
            continue
        hyps.append(
            Hypothesis(
                id=item["id"],
                kind=HypothesisKind(item["kind"]),
                title=item["title"],
                params=dict(item.get("params", {})),
                source="playbook",
            )
        )
    rows = []
    for hyp in hyps:
        started = time.time()
        res = run_experiment(spec, profile, hyp, train, RunBudget())
        row = {
            "task": task_md.parent.name,
            "id": hyp.id,
            "metric": spec.metric.name,
            "status": res.status.value,
            "cv_mean": res.mean_score,
            "cv_std": res.std_score,
            "honest": None,
            "error": res.error,
        }
        if res.status == ExperimentStatus.OK:
            try:
                row["honest"] = honest_score(spec, profile, hyp, train, split)
            except Exception as exc:  # noqa: BLE001 — стенд не должен падать из-за одной гипотезы
                row["error"] = f"honest: {type(exc).__name__}: {exc}"
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        print(
            f"  {hyp.id:28s} cv {row['cv_mean'] or float('nan'):.5f} ± {row['cv_std'] or 0:.5f}"
            f"  honest {row['honest'] or float('nan'):.5f}  {row['seconds']:.0f} с  {row['error'] or ''}"
        )
    gib = spec.metric.greater_is_better
    base = rows[0]
    for row in rows:
        for key in ("cv_mean", "honest"):
            if row[key] is not None and base[key] is not None:
                delta = row[key] - base[key]
                row[f"{key}_gain"] = delta if gib else -delta  # > 0 — лучше бейзлайна
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка гипотез playbook на задачах")
    parser.add_argument("--tasks", nargs="*", type=Path, help="папки задач (по умолчанию все public и practice)")
    parser.add_argument("--only", nargs="*", help="id гипотез (бейзлайн считается всегда)")
    args = parser.parse_args()
    tasks = [t / "task.md" for t in args.tasks] if args.tasks else find_tasks()
    rows = [r for t in tasks for r in bench_task(t, set(args.only) if args.only else None)]
    out_dir = ROOT / "runs" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"playbook-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nСохранено: {out}")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
