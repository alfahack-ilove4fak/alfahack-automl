"""Стенд для playbook: прогоняет каждую гипотезу на задачах так же, как агент, и сравнивает с бейзлайном.

Train задачи делится на train-часть и честную отложенную часть без утечек из EDA (docs/research/eda.md):
если есть колонка времени (целый «период» или дата строкой), по которой test лежит после train, — последний
период train; иначе — фолд StratifiedGroupKFold, где полные дубликаты держатся вместе.

Две оценки на гипотезу — ровно публичным путём агента, поэтому работают любые params (модели реестра,
feature_ops, proba_weights, optuna…):
  cv      — ml.experiment.run_experiment на train-части: то, по чему агент принимает гипотезу;
  honest  — ml.experiment.fit_predict_final на train-части → ответ для отложенной части → метрика задачи
            (evaluation.local_score.score_submission, как в харнессе). Отложенная часть не участвует ни в обучении,
            ни в ранней остановке.
Гипотеза, которая выигрывает на cv, но проигрывает на honest, скорее всего запоминает данные.

Как агент (#151, ADR 0013): схема CV выбирается один раз на задачу (ml.validation.choose_validation по train-части и
отложенной части в роли test) и одна у бейзлайна и всех гипотез; accepted — примет ли агент гипотезу против бейзлайна
по своему правилу (agent.acceptance.accept). Итог по гипотезам — сводка в конце: на скольких задачах выиграла честно,
сколько раз агент принял бы её зря (cv выиграла, honest — нет).

Запуск (задачи — tasks/public/* и tasks/practice/*):
    uv run python research/bench_playbook.py
    uv run python research/bench_playbook.py --tasks tasks/public/regression --only log_target lgbm_slow_lr
Результат — таблица в консоли и CSV в runs/bench/ (на платформу не едет).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.acceptance import accept  # noqa: E402
from agent.hypotheses import _applies, baseline_hypothesis, load_playbook  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from evaluation.local_score import score_submission  # noqa: E402
from interfaces.schemas import ExperimentStatus, Hypothesis, HypothesisKind, RunBudget, TaskType  # noqa: E402
from ml.experiment import SEED, fit_predict_final, run_experiment  # noqa: E402
from ml.profiling import load_data, profile_data  # noqa: E402
from ml.validation import choose_validation  # noqa: E402

DATE_SHARE = 0.95  # колонку считаем датой, если так разбирается не меньше этой доли значений
LAST_PERIOD = 0.2  # для дат: в честную часть уходят последние 20% train по времени


def find_tasks() -> list[Path]:
    return sorted(ROOT.glob("tasks/public/*/task.md")) + sorted(ROOT.glob("tasks/practice/*/task.md"))


def _as_dates(series: pd.Series) -> pd.Series | None:
    """Строки-даты → datetime, если разбирается почти всё; иначе None."""
    if pd.api.types.is_numeric_dtype(series):
        return None
    values = series.dropna()
    if values.empty:
        return None
    parsed = pd.to_datetime(values.astype(str), errors="coerce", format="mixed")
    if parsed.notna().mean() < DATE_SHARE:
        return None
    return pd.to_datetime(series.astype(str), errors="coerce", format="mixed")


def time_column(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[str, str] | None:
    """Колонка, по которой весь test лежит после train: («period», имя) — целый код периода, («date», имя) — дата."""
    for col in features:
        if pd.api.types.is_integer_dtype(train[col]) and pd.api.types.is_integer_dtype(test[col]):
            if 1 < train[col].nunique() <= 50 and test[col].min() > train[col].max():
                return "period", col
            continue
        tr_dates, te_dates = _as_dates(train[col]), _as_dates(test[col])
        if tr_dates is not None and te_dates is not None and te_dates.min() > tr_dates.max():
            return "date", col
    return None


def honest_split(spec, profile, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    features = profile.numeric_features + profile.categorical_features
    found = time_column(train, test, features)
    if found and found[0] == "period":
        col = found[1]
        last = train[col].max()
        return np.where(train[col] < last)[0], np.where(train[col] == last)[0], f"время: {col}={last}"
    if found:
        col = found[1]
        dates = _as_dates(train[col])
        cut = dates.quantile(1 - LAST_PERIOD)
        return np.where(dates < cut)[0], np.where(dates >= cut)[0], f"время: {col} ≥ {cut.date()}"
    groups = pd.util.hash_pandas_object(train[features], index=False).to_numpy()
    is_class = spec.task_type in (TaskType.BINARY, TaskType.MULTICLASS)
    strata = train[spec.target].astype(str) if is_class else np.zeros(len(train))
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, strata, groups))
    return tr, va, "группы дубликатов"


PLAIN_SCHEMES = (None, "kfold", "stratified")  # обычная CV — как без validation (solution.py)


def with_validation(hyp: Hypothesis, validation: str | None) -> Hypothesis:
    """Схема прогона в params гипотезы — как agent.loop.with_validation (своя схема гипотезы не трогается)."""
    params = copy.deepcopy(hyp.params)  # run_experiment дописывает в params
    if validation not in PLAIN_SCHEMES:
        params.setdefault("validation", validation)
    return hyp.model_copy(update={"params": params})


def evaluate(spec, hyp: Hypothesis, part: pd.DataFrame, hold: pd.DataFrame) -> tuple[object, float | None]:
    """cv агента на train-части и честная метрика на отложенной части — только публичными функциями ml."""
    profile = profile_data(spec, part, hold.drop(columns=[spec.target]))
    res = run_experiment(spec, profile, hyp, part, RunBudget())
    if res.status != ExperimentStatus.OK:
        return res, None
    answer = fit_predict_final(spec, profile, hyp, part, hold.drop(columns=[spec.target]))
    scored = score_submission(spec, hold[spec.target], answer[spec.target], part[spec.target])
    return res, float(scored["value"])


def bench_task(task_md: Path, only: set[str] | None) -> list[dict]:
    spec = parse_task(task_md)
    train, test = load_data(spec)
    profile = profile_data(spec, train, test)
    tr, va, how = honest_split(spec, profile, train, test)
    part, hold = train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)
    hold_x = hold.drop(columns=[spec.target])
    validation = choose_validation(spec, profile_data(spec, part, hold_x), part, hold_x)
    print(
        f"\n== {task_md.parent.name}: {spec.task_type.value}, {spec.metric.name}, честная часть — {how} "
        f"({len(part)} / {len(hold)} строк), схема CV — {validation}"
    )
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
    base_res = None
    for hyp in hyps:
        started = time.time()
        hyp = with_validation(hyp, validation)
        res = None
        try:
            res, honest = evaluate(spec, hyp, part, hold)
            row = {"status": res.status.value, "cv_mean": res.mean_score, "cv_std": res.std_score, "error": res.error}
            row["fold_scores"] = json.dumps([round(v, 6) for v in res.fold_scores])  # для разбора правила принятия
        except Exception as exc:  # noqa: BLE001 — стенд не должен падать из-за одной гипотезы
            honest, row = None, {"status": "failed", "cv_mean": None, "cv_std": None, "error": f"{exc!r}"[:200]}
        if hyp.id == "baseline":
            base_res = res
        elif res is not None and base_res is not None:
            row["accepted"], row["accept_reason"] = accept(res, base_res, spec.metric.greater_is_better)
        row = {"task": task_md.parent.name, "id": hyp.id, "metric": spec.metric.name, "split": how, **row}
        row["validation"] = validation
        row["honest"] = honest
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        mark = {True: "✓", False: "·"}.get(row.get("accepted"), " ")
        print(
            f"  {mark} {hyp.id:28s} cv {row['cv_mean'] or float('nan'):.5f} ± {row['cv_std'] or 0:.5f}"
            f"  honest {row['honest'] or float('nan'):.5f}  {row['seconds']:.0f} с  {row['error'] or ''}",
            flush=True,
        )
    gib = spec.metric.greater_is_better
    base = rows[0]
    for row in rows:
        for key in ("cv_mean", "honest"):
            if row[key] is not None and base[key] is not None:
                delta = row[key] - base[key]
                row[f"{key}_gain"] = delta if gib else -delta  # > 0 — лучше бейзлайна
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Сводка по гипотезам — порядок playbook: доля задач, где гипотеза честно лучше бейзлайна, потом медиана выигрыша.

    Медиана, а не среднее: на метриках с маленьким бейзлайном (F1 редкого класса ≈ 0.18) любой сдвиг даёт ±50–100%
    и перекрывает остальные задачи. accepted — сколько раз агент принял бы гипотезу, accepted_and_win — из них по делу.
    """
    rows = rows.drop_duplicates(["task", "id"], keep="last")  # несколько CSV: бейзлайн задачи считается в каждом
    df = rows[rows["id"] != "baseline"].copy()
    if df.empty or "honest_gain" not in df:
        return pd.DataFrame()
    df["honest_win"] = df["honest_gain"] > 0
    df["accepted"] = df["accepted"].eq(True) if "accepted" in df else False
    # выигрыш в % от бейзлайна — чтобы сравнивать задачи с разными метриками
    base = rows[rows["id"] == "baseline"].set_index("task")["honest"]
    df["honest_rel"] = 100 * df["honest_gain"] / df["task"].map(base).abs()
    summary = df.groupby("id", sort=False).agg(
        tasks=("task", "size"),
        wins=("honest_win", "sum"),
        median_gain_pct=("honest_rel", "median"),
        accepted=("accepted", "sum"),
        accepted_and_win=("honest_win", lambda w: int((w & df.loc[w.index, "accepted"]).sum())),
        seconds=("seconds", "mean"),
    )
    summary.insert(2, "win_share", summary["wins"] / summary["tasks"])
    return summary.sort_values(["win_share", "median_gain_pct"], ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка гипотез playbook на задачах")
    parser.add_argument("--tasks", nargs="*", type=Path, help="папки задач (по умолчанию все public и practice)")
    parser.add_argument("--tag", default="", help="метка в имени CSV (параллельные прогоны не перетирают друг друга)")
    parser.add_argument("--only", nargs="*", help="id гипотез (бейзлайн считается всегда)")
    parser.add_argument("--from-csv", nargs="*", type=Path, help="не считать, а свести сохранённые прогоны стенда")
    args = parser.parse_args()
    if args.from_csv:
        rows = pd.concat([pd.read_csv(path) for path in args.from_csv], ignore_index=True)
        out = None
    else:
        tasks = [t / "task.md" for t in args.tasks] if args.tasks else find_tasks()
        out_dir = ROOT / "runs" / "bench"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"playbook-{args.tag + '-' if args.tag else ''}{time.strftime('%Y%m%d-%H%M%S')}.csv"
        found = []
        for t in tasks:
            try:
                found += bench_task(t, set(args.only) if args.only else None)
            except Exception as exc:  # noqa: BLE001 — задача, которую агент не разобрал, не роняет весь стенд
                print(f"\n== {t.parent.name}: пропуск — {exc!r}"[:300], flush=True)
            if found:  # после каждой задачи: долгий стенд, прерванный на середине, не теряет готовое
                pd.DataFrame(found).to_csv(out, index=False)
        rows = pd.DataFrame(found)
    summary = summarize(rows)
    if not summary.empty:
        print(
            "\nСводка (wins — честно лучше бейзлайна; accepted — агент принял бы; accepted_and_win — принял по делу):"
        )
        print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    if out is not None:
        print(f"\nСохранено: {out}")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
