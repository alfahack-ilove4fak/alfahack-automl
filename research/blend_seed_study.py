"""Стенд для E1 (бленд по OOF) и E3 (сиды финала), #261/#262. Владелец — research; ml/ не трогаем.

    uv run python research/blend_seed_study.py                      # 12 задач харнесса
    uv run python research/blend_seed_study.py --tasks tasks/public/regression --top 6

Для каждой задачи — та же честная отложенная часть и та же схема CV, что у стенда playbook (bench_playbook):
  1. гипотезы playbook → run_experiment на train-части (CV агента), берём top-K по CV (OK-эксперименты);
  2. «OOF» для весов бленда — последний фолд схемы агента внутри train-части (во time-схеме — самый поздний
     период): каждый кандидат учится на остальном и предсказывает этот фолд. Настоящий OOF по всем фолдам —
     в ml/blend.py у Данила; если бленд не выигрывает даже на одном фолде, делать его незачем;
  3. жадный бленд Caruana с возвратом (≤ 10 шагов) на этом фолде: веса = доли выбора;
  4. честно на отложенной части: (а) лучший по CV одиночный, (б) он же со средним 3 сидов, (в) бленд.
Классификация смешивается по вероятностям (шаблон «колонка на класс»), ответ — argmax; множители классов
(proba_weights) здесь не применяются ни к одиночному, ни к бленду — сравнение честное между собой.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.hypotheses import _applies, baseline_hypothesis, load_playbook  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from evaluation.local_score import score_submission  # noqa: E402
from evaluation.metrics import METRICS  # noqa: E402
from interfaces.schemas import ExperimentStatus, Hypothesis, HypothesisKind, RunBudget, TaskType  # noqa: E402
from ml.experiment import _split, fit_predict_final, run_experiment  # noqa: E402
from ml.profiling import load_data, profile_data  # noqa: E402
from ml.validation import choose_validation, make_folds, parse_validation  # noqa: E402
from research.bench_playbook import honest_split, with_validation  # noqa: E402

SEEDS = (42, 7, 2026)
BLEND_STEPS = 10
SEED_KEY = {"lightgbm": "random_state", "xgboost": "random_state", "catboost": "random_seed", "hist_gb": "random_state"}
HARNESS_TASKS = sorted(ROOT.glob("tasks/public/*/task.md")) + sorted(ROOT.glob("tasks/practice/*/task.md"))


def _is_class(spec) -> bool:
    return spec.task_type in (TaskType.BINARY, TaskType.MULTICLASS)


def proba_spec(spec):
    """Классификация — шаблон «колонка на класс»: fit_predict_final отдаёт вероятности, их и смешиваем."""
    if not _is_class(spec):
        return spec
    return spec.model_copy(update={"submission_columns": [spec.id_column, *spec.class_labels], "extra_targets": []})


def predict(spec, hyp: Hypothesis, fit: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Прогноз модели гипотезы: регрессия — вектор, классификация — матрица вероятностей (n × k)."""
    sp = proba_spec(spec)
    x_rows = rows.drop(columns=[spec.target, *spec.extra_targets], errors="ignore")
    answer = fit_predict_final(sp, profile_data(sp, fit, x_rows), hyp, fit, x_rows)
    if _is_class(spec):
        return answer[[str(c) for c in spec.class_labels]].to_numpy(dtype=float)
    return answer[spec.target].to_numpy(dtype=float)


def to_answer(spec, pred: np.ndarray) -> pd.Series:
    """Прогноз → ответ в форме, которую ждёт метрика: число, вероятность класса 1 или метка."""
    if not _is_class(spec):
        return pd.Series(pred)
    kind = METRICS[spec.metric.name].prediction
    if kind == "proba" and spec.task_type == TaskType.BINARY:
        return pd.Series(pred[:, -1])
    return pd.Series(np.asarray(spec.class_labels)[pred.argmax(axis=1)])


def score(spec, y_true: pd.Series, pred: np.ndarray, y_train: pd.Series) -> float:
    return float(score_submission(spec, y_true.reset_index(drop=True), to_answer(spec, pred), y_train)["value"])


def better(spec, a: float, b: float) -> bool:
    return a > b if spec.metric.greater_is_better else a < b


def caruana(spec, preds: dict[str, np.ndarray], y_val: pd.Series, y_train: pd.Series) -> dict[str, float]:
    """Жадный отбор с возвратом: начинаем с лучшего на фолде, добавляем модель, пока метрика смеси растёт."""
    scores = {k: score(spec, y_val, p, y_train) for k, p in preds.items()}
    first = max(scores, key=lambda k: scores[k] if spec.metric.greater_is_better else -scores[k])
    chosen, best = [first], scores[first]
    for _ in range(BLEND_STEPS - 1):
        step = None
        for key, p in preds.items():
            mix = (sum(preds[c] for c in chosen) + p) / (len(chosen) + 1)
            s = score(spec, y_val, mix, y_train)
            if better(spec, s, best if step is None else step[1]):
                step = (key, s)
        if step is None:
            break
        chosen.append(step[0])
        best = step[1]
    return {k: chosen.count(k) / len(chosen) for k in dict.fromkeys(chosen)}


def with_seed(hyp: Hypothesis, seed: int) -> Hypothesis:
    params = copy.deepcopy(hyp.params)
    key = SEED_KEY.get(params.get("model", "lightgbm"))
    if key:
        params["model_params"] = {**(params.get("model_params") or {}), key: seed}
    return hyp.model_copy(update={"params": params})


def study_task(task_md: Path, top: int) -> dict:
    started = time.time()
    spec = parse_task(task_md)
    train, test = load_data(spec)
    tr, va, how = honest_split(spec, profile_data(spec, train, test), train, test)
    part, hold = train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)
    hold_x = hold.drop(columns=[spec.target])
    profile = profile_data(spec, part, hold_x)
    validation = choose_validation(spec, profile, part, hold_x)

    hyps = [baseline_hypothesis()] + [
        Hypothesis(id=i["id"], kind=HypothesisKind(i["kind"]), title=i["title"], params=dict(i.get("params", {})))
        for i in load_playbook()
        if _applies(i, spec, profile)
    ]
    done = []
    for hyp in hyps:
        hyp = with_validation(hyp, validation)
        try:
            res = run_experiment(spec, profile, hyp, part, RunBudget())
        except Exception:  # noqa: BLE001
            continue
        if res.status == ExperimentStatus.OK and res.mean_score is not None:
            done.append((hyp, res.mean_score))
    done.sort(key=lambda t: t[1], reverse=spec.metric.greater_is_better)
    cands = done[:top]

    # «OOF» для весов — последний фолд схемы агента внутри train-части
    scheme, column = parse_validation(validation if validation not in (None, "kfold", "stratified") else None, spec)
    features = profile.numeric_features + profile.categorical_features
    y_enc = part[spec.target].astype(str).to_numpy() if _is_class(spec) else part[spec.target].to_numpy(float)
    if _is_class(spec):
        y_enc = pd.Series(y_enc).map({c: i for i, c in enumerate(spec.class_labels)}).to_numpy()
    fit_idx, val_idx = make_folds(spec, scheme, column, part, features, y_enc, 5, _split)[-1]
    fit, val = part.iloc[fit_idx].reset_index(drop=True), part.iloc[val_idx].reset_index(drop=True)

    val_preds, hold_preds = {}, {}
    for hyp, _ in cands:
        try:
            val_preds[hyp.id] = predict(spec, hyp, fit, val)
            hold_preds[hyp.id] = predict(spec, hyp, part, hold)
        except Exception:  # noqa: BLE001 — кандидат не обучился — без него
            continue
    weights = caruana(spec, val_preds, val[spec.target], fit[spec.target])
    blend = sum(w * hold_preds[k] for k, w in weights.items())

    best_hyp = cands[0][0]
    single = hold_preds[best_hyp.id]
    seeded = np.mean([predict(spec, with_seed(best_hyp, s), part, hold) for s in SEEDS], axis=0)

    y_hold, y_part = hold[spec.target], part[spec.target]
    base = score(spec, y_hold, predict(spec, with_validation(baseline_hypothesis(), validation), part, hold), y_part)
    out = {
        "task": task_md.parent.name,
        "metric": spec.metric.name,
        "scheme": validation or "kfold",
        "n_part": len(part),
        "baseline": base,
        "best": best_hyp.id,
        "single": score(spec, y_hold, single, y_part),
        "seeds3": score(spec, y_hold, seeded, y_part),
        "blend": score(spec, y_hold, blend, y_part),
        "blend_members": len(weights),
        "blend_weights": "; ".join(f"{k} {w:.2f}" for k, w in sorted(weights.items(), key=lambda t: -t[1])),
        "seconds": round(time.time() - started),
    }
    sign = 1 if spec.metric.greater_is_better else -1
    for key in ("seeds3", "blend"):
        out[f"{key}_gain_pct"] = 100 * sign * (out[key] - out["single"]) / abs(out["single"])
    print(
        f"{out['task']:34s} {out['metric']:10s} single {out['single']:.5f}  seeds3 {out['seeds3_gain_pct']:+.2f}%  "
        f"blend {out['blend_gain_pct']:+.2f}% ({out['blend_members']} мод.)  {out['seconds']} с",
        flush=True,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="E1/E3: бленд и сиды на честной отложенной части")
    parser.add_argument("--tasks", nargs="*", type=Path, help="папки задач (по умолчанию 12 задач харнесса)")
    parser.add_argument("--top", type=int, default=8, help="сколько лучших по CV экспериментов берём в бленд")
    args = parser.parse_args()
    tasks = [t / "task.md" for t in args.tasks] if args.tasks else HARNESS_TASKS
    rows = []
    for task_md in tasks:
        try:
            rows.append(study_task(task_md, args.top))
        except Exception as exc:  # noqa: BLE001 — стенд не падает из-за одной задачи
            print(f"{task_md.parent.name}: ошибка {exc!r}"[:200], flush=True)
    out = ROOT / "runs" / "bench" / f"blend-seeds-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Сохранено: {out}")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
