"""Живой монитор прогона внешних задач: таблица готовых, текущая задача и оценка времени до конца.

    uv run python research/external/watch.py                 # обновляется каждые 5 с, выход — Ctrl+C
    uv run python research/external/watch.py --once          # один снимок и выход
    uv run python research/external/watch.py --root /путь/к/репозиторию --interval 10

Читает runs/night/external_results.jsonl (готовые задачи), runs/night/external/_answers (весь список) и
agent.log текущей задачи. Ничего не запускает и не меняет — можно держать в соседнем терминале.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def load(root: Path) -> tuple[list[str], dict[str, dict], dict[str, dict]]:
    base = root / "runs" / "night"
    names = sorted(p.stem for p in (base / "external" / "_answers").glob("*.json"))
    meta = {n: json.loads((base / "external" / "_answers" / f"{n}.json").read_text(encoding="utf-8")) for n in names}
    rows: dict[str, dict] = {}
    results = base / "external_results.jsonl"
    if results.exists():
        for line in results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["task"]] = row
    return names, meta, rows


def is_zero(row: dict) -> bool:
    if row.get("exit_code") not in (0, None) or row.get("problems") or row.get("value") is None:
        return True
    score = row.get("score_vs_strong")
    return score is None or score <= 0


def current_task(root: Path, names: list[str], rows: dict[str, dict]) -> tuple[str | None, float | None, str]:
    runs = root / "runs" / "night" / "external_runs"
    pending = [n for n in names if n not in rows and (runs / n).exists()]
    if not pending:
        return None, None, ""
    name = max(pending, key=lambda n: (runs / n).stat().st_mtime)
    log = runs / name / "agent.log"
    started = (runs / name).stat().st_ctime
    last = ""
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace").splitlines()
        last = re.sub(r"^\[\s*[\d.]+s\]\s*INFO\s+", "", text[-1])[:110] if text else ""
    return name, time.time() - started, last


def render(root: Path) -> str:
    names, meta, rows = load(root)
    if not names:
        return "Нет задач: сначала make_external_tasks.py и run_alien.py --set external"
    done = [n for n in names if n in rows]
    lines = [
        "| # | Задача | Тип | Метрика | Балл к сильному | Ноль | Время, с | Разбор |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, name in enumerate(names, 1):
        m = meta[name]
        row = rows.get(name)
        if row is None:
            lines.append(f"| {i} | {name} | {m['task_type']} | {m['metric']} | … | | | |")
            continue
        spec = row.get("spec", {})
        parsed = "✅" if spec.get("metric") == m["metric"] else f"❌ {spec.get('metric')}"
        lines.append(
            f"| {i} | {name} | {m['task_type']} | {m['metric']} | {row.get('score_vs_strong')} "
            f"| {'❌' if is_zero(row) else ''} | {row.get('seconds')} | {parsed} |"
        )
    zeros = sum(is_zero(rows[n]) for n in done)
    scores = [rows[n]["score_vs_strong"] for n in done if rows[n].get("score_vs_strong") is not None]
    mean = sum(scores) / len(scores) if scores else 0.0
    seconds = [rows[n]["seconds"] for n in done if rows[n].get("seconds")]
    avg = sum(seconds) / len(seconds) if seconds else None
    cur, elapsed, last = current_task(root, names, rows)
    head = [f"Готово {len(done)} из {len(names)} · нулей: {zeros} · средний балл к сильному: {mean:.2f}"]
    if cur:
        head.append(f"Сейчас: {cur} — {elapsed / 60:.1f} мин · {last}")
    if avg and len(done) < len(names):
        left = (len(names) - len(done)) * (avg + 40)  # +≈ 40 с на ориентиры LightGBM после агента
        head.append(f"Осталось ≈ {left / 60:.0f} мин (по среднему {avg:.0f} с на задачу)")
    if len(done) == len(names):
        head.append("Прогон закончен: uv run python research/external/report.py")
    return "\n".join(head) + "\n\n" + "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            text = render(args.root)
            print(
                "\033[2J\033[H" + time.strftime("%H:%M:%S") + "  (Ctrl+C — выход)\n\n" + text if not args.once else text
            )
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
