"""Шаги агента на настоящем test (наборы beyond, external): что сделал с ответом каждый принятый шаг. Владелец — research.

    uv run python research/night/run_alien.py --set beyond --minutes 5          # сначала прогоны агента
    LLM_MOCK=1 uv run --with xgboost==2.1.4 --with catboost==1.2.8 python research/external/steps_on_test.py --set beyond

Зачем (#295): стенд (research/bench_playbook.py → accept_rule_study.py) судит гипотезы по последнему периоду train,
а здесь — по настоящему test задачи, которого агент не видел. Для каждой задачи с прогоном агента:
  1. из agent.log — принятые гипотезы по порядку и правило, по которому их приняли: t ≥ 2 или «последний период»
     (time-схема, agent/acceptance.py);
  2. бейзлайн и каждая принятая гипотеза из playbook ОТДЕЛЬНО — в основном цикле гипотезы не складываются, принятая
     заменяет лучшую. Как в агенте: схема — choose_validation на train/test, затем run_experiment и fit_predict_final
     на всём train (без лимита времени, поэтому CV может отличаться от CV в прогоне на 5 минут). Запасные комбинации
     (fb-…, «поверх лучшего») так не пересобрать — для них только итог агента;
  3. итог агента — его submission.csv;
  4. всё — независимой метрикой research/night/run_alien.true_metric по ответам генератора.
Результат — таблица в консоли и CSV runs/bench/<набор>/steps_on_test.csv. xgboost и catboost — через --with: без
них гипотезы на этих моделях не пересобрать. Запускать по очереди: параллельно с другими прогонами на 4 CPU агент
и обучение здесь заметно медленнее.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.hypotheses import baseline_hypothesis, load_playbook  # noqa: E402
from agent.loop import PLAIN_VALIDATION, with_validation  # noqa: E402
from agent.task_parser import parse_task  # noqa: E402
from interfaces.schemas import ExperimentStatus, Hypothesis, HypothesisKind, RunBudget  # noqa: E402
from ml.experiment import fit_predict_final, run_experiment  # noqa: E402
from ml.profiling import load_data, profile_data  # noqa: E402
from ml.validation import choose_validation  # noqa: E402

ITERATION = re.compile(r"Итерация \d+: (\S+) —")
T_VALUE = re.compile(r"t=(-?(?:inf|\d+(?:\.\d+)?))")
LAST_PERIOD = "последний период"


def _alien():
    """research/night/run_alien.py — своя проверка ответа и метрика (не evaluation/, как и в прогонах)."""
    spec = importlib.util.spec_from_file_location("run_alien", ROOT / "research" / "night" / "run_alien.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted_steps(log_text: str) -> list[dict]:
    """Принятые гипотезы по порядку: id — из ближайшей «Итерация N: <id> — …» перед «✓ лучше», t и правило."""
    steps, current = [], None
    for line in log_text.splitlines():
        found = ITERATION.search(line)
        if found:
            current = found.group(1)
        elif "✓ лучше" in line and current:
            t = T_VALUE.search(line)
            rule = LAST_PERIOD if "последнем периоде" in line else "t ≥ 2"
            steps.append({"id": current, "t": float(t.group(1)) if t else None, "rule": rule})
    return steps


def step_on_test(spec, profile, hyp: Hypothesis, validation: str | None, train, test) -> tuple[float | None, object]:
    """CV и ответ для test гипотезы, как в агенте: копия в схеме бейзлайна, CV, финал на той же копии (ADR 0008)."""
    run_hyp = with_validation(hyp, validation)
    res = run_experiment(spec, profile, run_hyp, train, RunBudget())
    if res.status != ExperimentStatus.OK:
        return None, None
    return res.mean_score, fit_predict_final(spec, profile, run_hyp, train, test)


def check_task(name: str, set_dir: Path, runs_dir: Path, alien) -> list[dict]:
    meta = json.loads((set_dir / "_answers" / f"{name}.json").read_text(encoding="utf-8"))
    target = meta["targets"][0]
    answers = pd.read_csv(set_dir / "_answers" / f"{name}.csv", dtype={meta["id"]: str} if meta["id"] else None)
    spec = parse_task(set_dir / name / "task.md", use_analyst=False)
    train, test = load_data(spec)
    profile = profile_data(spec, train, test)
    validation = choose_validation(spec, profile, train, test)
    scheme = None if validation in PLAIN_VALIDATION else validation
    classes = sorted(train[target].dropna().astype(str).unique()) if meta["task_type"] != "regression" else None
    book = {item["id"]: item for item in load_playbook()}
    steps = accepted_steps((runs_dir / name / "agent.log").read_text(encoding="utf-8", errors="replace"))
    rows = [{"id": "baseline", "t": None, "rule": "—"}, *steps]
    out = []
    for step in rows:
        cv, value = None, None
        if step["id"] == "baseline" or step["id"] in book:
            item = book.get(step["id"])
            hyp = baseline_hypothesis(spec) if item is None else Hypothesis(
                id=item["id"], kind=HypothesisKind(item["kind"]), title=item["title"],
                params=dict(item.get("params", {})), source="playbook",
            )  # fmt: skip
            cv, answer = step_on_test(spec, profile, hyp, scheme, train, test)
            if answer is not None:
                value = alien.true_metric(meta, answers, answer.rename(columns={spec.target: target}), classes)
        out.append({"task": name, "metric": meta["metric"], "validation": validation, **step, "cv": cv, "test": value})
    sub, problems = alien.check_submission(runs_dir / name / "task" / "submission.csv", meta, answers)
    final = alien.true_metric(meta, answers, sub, classes) if sub is not None else None
    out.append({"task": name, "metric": meta["metric"], "validation": validation, "id": "итог агента", "t": None,
                "rule": "; ".join(problems) or "—", "cv": None, "test": final})  # fmt: skip
    base = out[0]["test"]
    sign = 1 if meta["greater_is_better"] else -1
    for row in out:
        ok = row["test"] is not None and base not in (None, 0)
        row["vs_baseline_pct"] = round(100 * sign * (row["test"] - base) / abs(base), 2) if ok else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="beyond", help="набор run_alien.py --set: beyond | external | …")
    parser.add_argument("--only", nargs="*", help="имена задач")
    args = parser.parse_args()
    set_dir = ROOT / "runs" / "night" / args.set
    runs_dir = ROOT / "runs" / "night" / f"{args.set}_runs"
    alien = _alien()
    names = sorted(p.stem for p in (set_dir / "_answers").glob("*.json"))
    rows = []
    for name in names:
        if (args.only and name not in args.only) or not (runs_dir / name / "agent.log").exists():
            continue  # агента на задаче не запускали
        try:
            task_rows = check_task(name, set_dir, runs_dir, alien)
        except Exception as exc:  # noqa: BLE001 — одна задача не роняет остальные
            print(f"💥 {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows += task_rows
        print(f"\n== {name}: {task_rows[0]['metric']}, схема {task_rows[0]['validation']}", flush=True)
        for r in task_rows:
            test = "—" if r["test"] is None else f"{r['test']:.4f}"
            delta = "" if r["vs_baseline_pct"] is None else f" ({r['vs_baseline_pct']:+.2f}% к бейзлайну)"
            how = r["rule"] if r["t"] is None else f"t={r['t']:.2f}, {r['rule']}"
            print(f"   {r['id']:<28} {how:<28} test {test}{delta}", flush=True)
    if not rows:
        print(f"Нет прогонов агента в {runs_dir}: сначала run_alien.py --set {args.set}")
        return 2
    out = ROOT / "runs" / "bench" / args.set / "steps_on_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nСохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
