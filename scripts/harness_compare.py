"""Таблица «было/стало» по двум прогонам харнесса — для комментария в PR (workflow harness-pr.yml).

    uv run python scripts/harness_compare.py base.csv head.csv --minutes 1 --rows 5000 --out comment.md

base.csv — агент из dev, head.csv — агент из PR (слитого с dev). Оба — CSV лидерборда evaluation/harness.py
на одних и тех же задачах и отложенных частях. Из каждого файла берётся последняя строка по задаче.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MARKER = "<!-- harness-report -->"
NOISE = 2.0  # разница в баллах, которую при одном коротком прогоне считаем шумом


def _last_by_task(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.groupby("task", sort=True).tail(1).set_index("task")


def _fmt(value) -> str:
    return "—" if pd.isna(value) else f"{value:.5g}"


def _int(value) -> str:
    return "—" if pd.isna(value) else f"{value:.0f}"


def _mark(delta: float, head_status: str) -> str:
    if head_status != "OK":
        return "🔴 упал"
    if delta > NOISE:
        return "🟢 лучше"
    if delta < -NOISE:
        return "🔴 хуже"
    return "⚪ в пределах шума"


def render(base: pd.DataFrame, head: pd.DataFrame, minutes: float | None = None, rows: int | None = None) -> str:
    base_commit = str(base["commit"].iloc[0]) if len(base) else "?"
    head_commit = str(head["commit"].iloc[0]) if len(head) else "?"
    lines = [
        MARKER,
        f"### 📊 Харнесс: было `{base_commit}` (dev) → стало `{head_commit}` (PR + dev)",
        "",
        "| Задача | Метрика | Было | Стало | Балл: было → стало | Δ балла | Итог | Время, с | Пик RAM, МБ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for task in sorted(set(base.index) | set(head.index)):
        b = base.loc[task] if task in base.index else None
        h = head.loc[task] if task in head.index else None
        b_score = float(b["score"]) if b is not None else 0.0
        h_score = float(h["score"]) if h is not None else 0.0
        delta = h_score - b_score
        status = str(h["status"]) if h is not None else "нет данных"
        metric = (h if h is not None else b)["metric"]
        lines.append(
            f"| {task} | {metric if not pd.isna(metric) else '—'} | {_fmt(b['value']) if b is not None else '—'} | "
            f"{_fmt(h['value']) if h is not None else '—'} | {b_score:.1f} → {h_score:.1f} | {delta:+.1f} | "
            f"{_mark(delta, status)} | {_int(h['seconds']) if h is not None else '—'} | "
            f"{_int(h['peak_ram_mb']) if h is not None else '—'} |"
        )
    mean_b = base["score"].mean() if len(base) else 0.0
    mean_h = head["score"].mean() if len(head) else 0.0
    lines += ["", f"**Средний балл: {mean_b:.1f} → {mean_h:.1f} ({mean_h - mean_b:+.1f})**", ""]
    errors = [f"- `{t}`: {e}" for t, e in head["error"].items() if isinstance(e, str) and e]
    if errors:
        lines += ["Ошибки в PR:", *errors, ""]
    setup = []
    if minutes is not None:
        setup.append(f"бюджет {minutes:g} мин на задачу")
    if rows:
        setup.append(f"train урезан до {rows} строк")
    setup += ["без LLM", "один прогон"]
    lines.append(
        f"<sub>{', '.join(setup)}. Балл — по формуле организаторов относительно тривиального ответа "
        f"(`evaluation/local_score.py`). Разница меньше {NOISE:g} балла может быть шумом. "
        "Перезапуск — снять и снова поставить метку `harness`.</sub>"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Таблица «было/стало» по двум прогонам харнесса")
    parser.add_argument("base", type=Path, help="CSV харнесса для dev")
    parser.add_argument("head", type=Path, help="CSV харнесса для PR")
    parser.add_argument("--minutes", type=float, help="бюджет прогона (для подписи)")
    parser.add_argument("--rows", type=int, help="урезание train (для подписи)")
    parser.add_argument("--out", type=Path, help="куда записать markdown (иначе — в консоль)")
    args = parser.parse_args(argv)

    text = render(_last_by_task(args.base), _last_by_task(args.head), args.minutes, args.rows)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → кириллица и эмодзи роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
