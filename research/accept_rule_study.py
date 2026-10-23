"""Какое правило принятия гипотез лучше переносится на скрытый test (#206, #236). Владелец — research.

    uv run python research/accept_rule_study.py runs/bench/playbook-<время>.csv [ещё CSV …]

Вход — прогоны стенда research/bench_playbook.py с колонкой fold_scores (оценка CV по фолдам). Для каждой задачи
скрипт проигрывает цикл агента: гипотезы по порядку playbook, каждую сравнивает с ТЕКУЩЕЙ лучшей по правилу,
в конце смотрит честную метрику итоговой модели на отложенной части. Так видно, сколько честного выигрыша
правило доводит до ответа и сколько раз принимает то, что честно хуже бейзлайна.

Правила (все без снижения порога t для обычных схем):
  current   — как agent.acceptance.accept: скорректированная t-статистика по фолдам ≥ 2;
  last      — для схемы по времени: лучше на последнем (самом позднем) периоде и в среднем по фолдам;
              для остальных схем — как current;
  last_or_t — current ИЛИ last (то есть last добавляет принятия только во time-схеме).
Код агента не меняется: это исследование для владельцев agent/ и ml/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.acceptance import T_MIN, corrected_t  # noqa: E402
from agent.hypotheses import load_playbook  # noqa: E402


def _gib(task_rows: pd.DataFrame) -> bool:
    """Направление метрики — из самого стенда: honest_gain = ±(honest − honest бейзлайна), знак «+» — лучше."""
    base = task_rows[task_rows.id == "baseline"].honest.iloc[0]
    for _, r in task_rows[task_rows.honest_gain.notna()].iterrows():
        if r.honest != base and r.honest_gain != 0:
            return (r.honest - base) * r.honest_gain > 0
    return True


def gains(new: list[float], best: list[float], gib: bool) -> list[float] | None:
    if len(new) != len(best) or len(new) < 2:
        return None
    sign = 1.0 if gib else -1.0
    return [sign * (a - b) for a, b in zip(new, best, strict=True)]


def rule_current(g: list[float], time_scheme: bool) -> bool:
    return sum(g) / len(g) > 0 and corrected_t(g) >= T_MIN


def rule_last(g: list[float], time_scheme: bool) -> bool:
    if not time_scheme:
        return rule_current(g, time_scheme)
    return g[-1] > 0 and sum(g) / len(g) > 0


def rule_last_or_t(g: list[float], time_scheme: bool) -> bool:
    return rule_current(g, time_scheme) or rule_last(g, time_scheme)


RULES = {"current": rule_current, "last": rule_last, "last_or_t": rule_last_or_t}


def simulate(task_rows: pd.DataFrame, order: list[str], rule) -> dict:
    """Цикл агента на одной задаче: итоговая гипотеза и её честный выигрыш против бейзлайна (в %)."""
    base = task_rows[task_rows.id == "baseline"].iloc[0]
    gib = _gib(task_rows)
    time_scheme = str(base.get("validation", "")).startswith("time:")
    best_id, best_folds, best_honest = "baseline", json.loads(base.fold_scores), base.honest
    accepted, wrong = 0, 0
    by_id = task_rows.set_index("id")
    for hyp in order:
        if hyp not in by_id.index:
            continue
        r = by_id.loc[hyp]
        if pd.isna(r.honest) or not isinstance(r.fold_scores, str):
            continue
        g = gains(json.loads(r.fold_scores), best_folds, gib)
        if g is None or not rule(g, time_scheme):
            continue
        accepted += 1
        sign = 1 if gib else -1
        wrong += int(sign * (r.honest - best_honest) < 0)  # принята, а честно хуже текущей лучшей
        best_id, best_folds, best_honest = hyp, json.loads(r.fold_scores), r.honest
    sign = 1 if gib else -1
    gain_pct = 100 * sign * (best_honest - base.honest) / abs(base.honest)
    return {"final": best_id, "honest_gain_pct": gain_pct, "accepted": accepted, "wrong_steps": wrong}


def main(paths: list[str]) -> int:
    rows = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).drop_duplicates(["task", "id"], keep="last")
    order = [item["id"] for item in load_playbook()]
    out = []
    for task, task_rows in rows.groupby("task", sort=True):
        for name, rule in RULES.items():
            out.append({"task": task, "rule": name, **simulate(task_rows, order, rule)})
    df = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(df.pivot(index="task", columns="rule", values="honest_gain_pct").round(2).to_string())
    print()
    print(df.pivot(index="task", columns="rule", values="final").to_string())
    print()
    summary = df.groupby("rule").agg(
        mean_gain_pct=("honest_gain_pct", "mean"),
        tasks_worse=("honest_gain_pct", lambda s: int((s < 0).sum())),
        accepted=("accepted", "sum"),
        wrong_steps=("wrong_steps", "sum"),
    )
    print(summary.round(2).to_string())
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
