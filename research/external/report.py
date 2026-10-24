"""Отчёт по прогону внешних задач и сравнение с базой. Владелец — research.

    uv run python research/night/run_alien.py --set external --minutes 5     # прогон (≈ 6 мин на задачу)
    uv run python research/external/report.py                                # таблица + сравнение с базой
    uv run python research/external/report.py --update-baseline              # текущий результат → новая база

Что смотрим (код выхода 1, если нарушено):
  1. нули (агент упал, ответ невалиден или «балл к сильному бейзлайну» ≤ 0) — число не выросло и ни одна задача
     с баллом не стала нулём;
  2. разбор: метрика и таргет, которые понял агент, совпадают с теми, что записаны в task.md (meta);
  3. балл любой задачи не упал больше чем на NOISE относительно базы.
База — docs/external_baseline.json; без неё печатаем только таблицу. Подгонять playbook под этот набор нельзя:
он для гейта «стало ли хуже», а не для настройки параметров.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "runs" / "night" / "external_results.jsonl"
ANSWERS = ROOT / "runs" / "night" / "external" / "_answers"
BASELINE = ROOT / "docs" / "external_baseline.json"
NOISE = 2.0


def is_zero(row: dict) -> bool:
    if row.get("exit_code") not in (0, None) or row.get("problems") or row.get("value") is None:
        return True
    score = row.get("score_vs_strong")
    return score is None or score <= 0


def summarize(rows: list[dict]) -> dict[str, dict]:
    last = {row["task"]: row for row in rows}
    out = {}
    for name, row in sorted(last.items()):
        meta_path = ANSWERS / f"{name}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        spec = row.get("spec", {})
        out[name] = {
            "zero": is_zero(row),
            "score": row.get("score_vs_strong"),
            "seconds": row.get("seconds"),
            "type": meta.get("task_type"),
            "metric": meta.get("metric"),
            "parsed_metric": spec.get("metric"),
            "parse_ok": bool(meta) and spec.get("metric") == meta.get("metric") and spec.get("target") == (meta.get("targets") or [None])[0],
            "problems": row.get("problems") or [],
        }  # fmt: skip
    return out


def compare(current: dict[str, dict], base: dict) -> list[str]:
    problems = []
    zeros_now = sum(v["zero"] for v in current.values())
    zeros_before = sum(v["zero"] for v in base.values())
    if zeros_now > zeros_before:
        problems.append(f"нулей стало больше: {zeros_before} → {zeros_now}")
    for name, now in current.items():
        before = base.get(name)
        if before is None:
            continue
        if now["zero"] and not before["zero"]:
            problems.append(f"{name}: был балл {before['score']}, стал ноль")
        elif not now["zero"] and not before["zero"] and now["score"] < before["score"] - NOISE:
            problems.append(f"{name}: балл к сильному {before['score']} → {now['score']} (шум ±{NOISE})")
    for name, now in current.items():
        if not now["parse_ok"]:
            problems.append(f"{name}: агент понял метрику/таргет иначе — {now['parsed_metric']} вместо {now['metric']}")
    return problems


def table(current: dict[str, dict], base: dict) -> str:
    lines = [
        "| Задача | Тип | Метрика | Балл к сильному | База | Ноль | Разбор | Время, с |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, v in current.items():
        before = base.get(name, {}).get("score")
        lines.append(
            f"| {name} | {v['type']} | {v['metric']} | {v['score']} | {before if before is not None else '—'} "
            f"| {'❌' if v['zero'] else ''} | {'✅' if v['parse_ok'] else '❌ ' + str(v['parsed_metric'])} | {v['seconds']} |"
        )
    zeros = sum(v["zero"] for v in current.values())
    scores = [v["score"] for v in current.values() if v["score"] is not None]
    lines.append(
        f"\nЗадач: {len(current)}, нулей: **{zeros}**, средний балл к сильному: **{sum(scores) / max(len(scores), 1):.2f}**"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    if not RESULTS.exists():
        print(f"Нет {RESULTS}: сначала прогон run_alien.py --set external")
        return 2
    rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    current = summarize(rows)
    base = json.loads(BASELINE.read_text(encoding="utf-8")).get("tasks", {}) if BASELINE.exists() else {}
    print(table(current, base))
    if args.update_baseline:
        BASELINE.write_text(json.dumps({"tasks": current}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"\nБаза записана: {BASELINE}")
        return 0
    problems = compare(current, base) if base else []
    for p in problems:
        print(f"❌ {p}")
    if not base:
        print("\nБазы нет — сравнивать не с чем (--update-baseline).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
