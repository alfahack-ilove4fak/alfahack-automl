"""Гейт перед сабмитом: не стало ли хуже, чем в прошлом сабмите. Владелец — qa.

    uv run python scripts/gate.py --minutes 5                      # прогнать и сравнить с docs/gate_baseline.json
    uv run python scripts/gate.py --minutes 5 --update-baseline    # после сабмита: текущий результат → новая база
    uv run python scripts/gate.py --no-run                         # только сравнить уже готовые результаты последнего прогона

Что проверяем (код выхода 1, если что-то из этого нарушено):
  1. «Чужие» задачи (research/night/run_alien.py, 15 штук): число нулей не выросло и ни одна задача,
     которая раньше давала балл, не стала нулём. Ноль = агент упал, ответ невалиден, метрика не посчиталась
     или «балл к сильному бейзлайну» ≤ 0.
  2. Public-пара (харнесс, отложенная часть train): «балл к сильному» каждой задачи не ниже базы больше чем на NOISE.
  3. Время: агент не должен уходить за бюджет (секунды задачи ≤ бюджет + запас на финальное обучение).
Результаты и отчёт — runs/gate/<время>/. База — docs/gate_baseline.json (коммитится вместе с записью в submissions.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs" / "gate_baseline.json"
ALIEN_RESULTS = ROOT / "runs" / "night" / "alien_results.jsonl"
PUBLIC_TASKS = ["tasks/public/regression/task.md", "tasks/public/multiclass/task.md"]
NOISE = 2.0  # как в harness_compare.py: меньшую разницу при коротком прогоне считаем шумом
OVERRUN = 1.25  # допустимое превышение бюджета (финальное переобучение и запись ответа)


def alien_zero(row: dict) -> bool:
    """Задача «чужая» дала ноль: упал агент, невалидный ответ, нет метрики или не лучше сильного бейзлайна."""
    if row.get("exit_code") not in (0, None) or row.get("problems") or row.get("value") is None:
        return True
    score = row.get("score_vs_strong")
    return score is None or score <= 0


def summarize_alien(rows: list[dict]) -> dict[str, dict]:
    """Последняя строка по задаче → {задача: {zero, score, seconds}}."""
    last = {row["task"]: row for row in rows}
    return {
        name: {
            "zero": alien_zero(row),
            "score": row.get("score_vs_strong"),
            "seconds": row.get("seconds"),
            "minutes": row.get("minutes"),
        }
        for name, row in sorted(last.items())
    }


def summarize_public(board: pd.DataFrame) -> dict[str, dict]:
    last = board.groupby("task", sort=True).tail(1)
    out = {}
    for _, row in last.iterrows():
        score = row.get("model_score")
        out[Path(str(row["task"])).name] = {
            "score": None if pd.isna(score) else float(score),
            "status": str(row["status"]),
            "seconds": None if pd.isna(row["seconds"]) else float(row["seconds"]),
            "minutes": float(row["minutes"]),
        }
    return out


def compare(base: dict | None, alien: dict[str, dict], public: dict[str, dict]) -> tuple[list[str], list[str]]:
    """(нарушения, замечания). Пустые нарушения — гейт пройден."""
    fails, notes = [], []
    for name, cur in alien.items():
        if cur["seconds"] and cur["minutes"] and cur["seconds"] > cur["minutes"] * 60 * OVERRUN:
            fails.append(f"время: «{name}» работала {cur['seconds']:.0f} с при бюджете {cur['minutes'] * 60:.0f} с")
    for name, cur in public.items():
        if cur["status"] != "OK":
            fails.append(f"public «{name}»: статус {cur['status']}")
        if cur["seconds"] and cur["seconds"] > cur["minutes"] * 60 * OVERRUN:
            fails.append(
                f"время: public «{name}» работала {cur['seconds']:.0f} с при бюджете {cur['minutes'] * 60:.0f} с"
            )
    if base is None:
        notes.append(
            "базы нет (docs/gate_baseline.json): сравнивать не с чем — после сабмита запусти с --update-baseline"
        )
        return fails, notes

    zeros, base_zeros = sum(v["zero"] for v in alien.values()), sum(v["zero"] for v in base["alien"].values())
    if zeros > base_zeros:
        fails.append(f"нулей на чужих задачах стало больше: {base_zeros} → {zeros}")
    for name, cur in alien.items():
        was = base["alien"].get(name)
        if was is None:
            notes.append(f"чужая «{name}» новая — в базе её нет")
        elif cur["zero"] and not was["zero"]:
            fails.append(f"чужая «{name}»: раньше был балл ({was['score']}), теперь ноль")
    for name, cur in public.items():
        was = base["public"].get(name)
        if was is None or was["score"] is None or cur["score"] is None:
            notes.append(f"public «{name}»: нечего сравнивать (нет балла в базе или в прогоне)")
        elif cur["score"] < was["score"] - NOISE:
            fails.append(f"public «{name}»: балл к сильному {was['score']:.1f} → {cur['score']:.1f} (шум ±{NOISE})")
    return fails, notes


def render(base: dict | None, alien: dict, public: dict, fails: list[str], notes: list[str]) -> str:
    lines = ["# Гейт перед сабмитом", "", f"**{'❌ НЕ ПРОХОДИТ' if fails else '✅ проходит'}**", ""]
    lines += [f"- ❌ {f}" for f in fails] + [f"- ℹ️ {n}" for n in notes] + [""]
    lines += ["| Public | Балл к сильному | База | Время, с |", "|---|---|---|---|"]
    for name, cur in public.items():
        was = (base or {}).get("public", {}).get(name, {}).get("score")
        lines.append(f"| {name} | {cur['score']} | {was} | {cur['seconds']} |")
    zeros = sum(v["zero"] for v in alien.values())
    lines += ["", f"Чужие задачи: нулей **{zeros}** из {len(alien)}", ""]
    lines += ["| Чужая | Балл к сильному | Ноль | База: ноль | Время, с |", "|---|---|---|---|---|"]
    for name, cur in alien.items():
        was = (base or {}).get("alien", {}).get(name, {}).get("zero")
        lines.append(f"| {name} | {cur['score']} | {'🔴' if cur['zero'] else ''} | {was} | {cur['seconds']} |")
    return "\n".join(lines) + "\n"


def _git_head() -> str:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip() or "?"


def run_checks(minutes: float, out_dir: Path) -> None:
    """Прогнать чужие задачи и public-пару харнессом (долго: ≈ (15 + 2) × minutes)."""
    alien_dir = ROOT / "runs" / "night" / "alien"
    if not (alien_dir / "_answers").exists():
        subprocess.run([sys.executable, "research/night/make_alien_tasks.py"], cwd=ROOT, check=True)
    ALIEN_RESULTS.unlink(missing_ok=True)  # свежие строки только этого прогона
    subprocess.run([sys.executable, "research/night/run_alien.py", "--minutes", str(minutes)], cwd=ROOT)
    cmd = [sys.executable, "-m", "evaluation.harness", "--minutes", str(minutes), "--out", str(out_dir / "harness.csv")]
    for task in PUBLIC_TASKS:
        cmd += ["--task", task]
    subprocess.run(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=5, help="бюджет агента на задачу, минут")
    parser.add_argument("--no-run", action="store_true", help="не запускать, взять последний прогон")
    parser.add_argument(
        "--update-baseline", action="store_true", help="записать результат как новую базу (после сабмита)"
    )
    parser.add_argument("--out", type=Path, help="папка отчёта (по умолчанию runs/gate/<время>)")
    args = parser.parse_args()

    runs = ROOT / "runs" / "gate"
    if args.no_run:
        previous = sorted(runs.glob("*/harness.csv"))
        if not previous or not ALIEN_RESULTS.exists():
            print("Нет готового прогона: запусти без --no-run", file=sys.stderr)
            return 2
        out_dir = args.out or previous[-1].parent
    else:
        out_dir = args.out or runs / time.strftime("%Y%m%d-%H%M")
        out_dir.mkdir(parents=True, exist_ok=True)
        run_checks(args.minutes, out_dir)

    alien_rows = [json.loads(ln) for ln in ALIEN_RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not alien_rows or not (out_dir / "harness.csv").exists():
        print("Прогон не оставил результатов (alien_results.jsonl / harness.csv)", file=sys.stderr)
        return 2
    alien = summarize_alien(alien_rows)
    public = summarize_public(pd.read_csv(out_dir / "harness.csv"))
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None

    fails, notes = compare(base, alien, public)
    report = render(base, alien, public, fails, notes)
    (out_dir / "gate.md").write_text(report, encoding="utf-8")
    print(report)

    if args.update_baseline:
        BASELINE.write_text(
            json.dumps(
                {"commit": _git_head(), "date": time.strftime("%Y-%m-%d %H:%M"), "alien": alien, "public": public},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"База обновлена: {BASELINE.relative_to(ROOT)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
