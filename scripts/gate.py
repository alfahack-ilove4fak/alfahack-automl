"""Гейт перед сабмитом: не стало ли хуже, чем в прошлом сабмите. Владелец — qa.

    uv run python scripts/gate.py --minutes 5                      # прогнать и сравнить с docs/gate_baseline.json
    uv run python scripts/gate.py --minutes 5 --update-baseline    # после сабмита: текущий результат → новая база
    uv run python scripts/gate.py --no-run                         # только сравнить уже готовые результаты последнего прогона
    uv run python scripts/gate.py --minutes 5 --llm                # то же, но с настоящей Qwen (тратит общий бюджет ₽!)

Что проверяем (код выхода 1, если что-то из этого нарушено):
  1. «Чужие» задачи (research/night/run_alien.py, 15 штук): число нулей не выросло и ни одна задача,
     которая раньше давала балл, не стала нулём. Ноль = агент упал, ответ невалиден, метрика не посчиталась
     или «балл к сильному бейзлайну» ≤ 0.
  2. Public-пара (харнесс, отложенная часть train): «балл к сильному» каждой задачи не ниже базы больше чем на NOISE.
  3. Время: агент не должен уходить за бюджет (секунды задачи ≤ бюджет + запас на финальное обучение).
С --llm агент работает с Qwen, как на платформе (по умолчанию LLM выключена: LLM_MOCK=1): нужны OPENAI_API_KEY и
OPENAI_MODEL; в отчёте — запросы, ошибки, токены и ₽; гейт не пройден, если Qwen не ответила ни разу. База снята без
LLM, поэтому с --llm её не обновляем.
Результаты и отчёт — runs/gate/<время>/. База — docs/gate_baseline.json (коммитится вместе с записью в submissions.md).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


def llm_problem(env: dict | os._Environ = os.environ) -> str | None:
    """Чего не хватает для прогона с настоящей Qwen; None — всё есть."""
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_MODEL") if not env.get(name)]
    return f"нет переменных окружения: {', '.join(missing)}" if missing else None


def llm_totals(alien_usage: dict[str, dict], board: pd.DataFrame) -> dict:
    """Расход Qwen за прогон: чужие задачи (журналы агентов) + public-пара (колонки llm_* лидерборда харнесса)."""
    keys = ("llm_calls", "llm_errors", "llm_tokens", "llm_rub")
    total = {key: 0.0 for key in keys}
    for usage in alien_usage.values():
        for key in keys:
            total[key] += float(usage.get(key, 0) or 0)
    for key in keys:
        if key in board.columns:
            total[key] += float(board.groupby("task", sort=True).tail(1)[key].fillna(0).sum())
    return total


def llm_checks(llm: dict | None) -> list[str]:
    """Нарушения для прогона с --llm: Qwen не ответила ни разу — прогон ничего не проверил."""
    if llm is None:
        return []
    if llm["llm_calls"] == 0:
        return [
            f"Qwen не ответила ни на один запрос (ошибок: {llm['llm_errors']:.0f}) — прогон не проверяет работу с LLM"
        ]
    return []


def render(
    base: dict | None, alien: dict, public: dict, fails: list[str], notes: list[str], llm: dict | None = None
) -> str:
    lines = ["# Гейт перед сабмитом", "", f"**{'❌ НЕ ПРОХОДИТ' if fails else '✅ проходит'}**", ""]
    lines += [f"- ❌ {f}" for f in fails] + [f"- ℹ️ {n}" for n in notes] + [""]
    lines += ["| Public | Балл к сильному | База | Время, с |", "|---|---|---|---|"]
    for name, cur in public.items():
        was = (base or {}).get("public", {}).get(name, {}).get("score")
        lines.append(f"| {name} | {cur['score']} | {was} | {cur['seconds']} |")
    if llm is not None:
        lines += [
            "",
            f"LLM (Qwen): запросов **{llm['llm_calls']:.0f}** (ошибок {llm['llm_errors']:.0f}), "
            f"токенов {llm['llm_tokens']:.0f}, **{llm['llm_rub']:.2f} ₽**",
        ]
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


def run_alien_llm(minutes: float) -> dict[str, dict]:
    """Чужие задачи с настоящей Qwen: run_alien.py всегда ставит LLM_MOCK=1, поэтому зовём его run_one напрямую."""
    spec = importlib.util.spec_from_file_location("run_alien", ROOT / "research" / "night" / "run_alien.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(ROOT))
    from evaluation.harness import llm_usage

    usage = {}
    for name in sorted(p.stem for p in (module.ALIEN / "_answers").glob("*.json")):
        try:
            module.run_one(name, minutes, "_llm", {"LLM_MOCK": "0"})
        except Exception as exc:  # noqa: BLE001 — одна задача не роняет гейт
            print(f"💥 {name}: раннер упал: {type(exc).__name__}: {exc}", flush=True)
        usage[name] = llm_usage(module.RUNS / f"{name}_llm")
    return usage


def run_checks(minutes: float, out_dir: Path, use_llm: bool = False) -> None:
    """Прогнать чужие задачи и public-пару харнессом (долго: ≈ (15 + 2) × minutes)."""
    alien_dir = ROOT / "runs" / "night" / "alien"
    if not (alien_dir / "_answers").exists():
        subprocess.run([sys.executable, "research/night/make_alien_tasks.py"], cwd=ROOT, check=True)
    ALIEN_RESULTS.unlink(missing_ok=True)  # свежие строки только этого прогона
    if use_llm:
        (out_dir / "llm_alien.json").write_text(json.dumps(run_alien_llm(minutes)), encoding="utf-8")
    else:
        subprocess.run([sys.executable, "research/night/run_alien.py", "--minutes", str(minutes)], cwd=ROOT)
    cmd = [sys.executable, "-m", "evaluation.harness", "--minutes", str(minutes), "--out", str(out_dir / "harness.csv")]
    if use_llm:
        cmd.append("--llm")
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
    parser.add_argument("--llm", action="store_true", help="с настоящей Qwen (OPENAI_*), тратит общий бюджет ₽")
    args = parser.parse_args()

    if args.llm and args.update_baseline:
        print("--llm и --update-baseline вместе нельзя: база снята без LLM", file=sys.stderr)
        return 2
    if args.llm and not args.no_run:
        problem = llm_problem()
        if problem:
            print(f"--llm: {problem}", file=sys.stderr)
            return 2
        print("⚠️ --llm: агент ходит в Qwen на всех задачах — это тратит общий бюджет ₽ команды", flush=True)

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
        run_checks(args.minutes, out_dir, args.llm)

    alien_rows = [json.loads(ln) for ln in ALIEN_RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not alien_rows or not (out_dir / "harness.csv").exists():
        print("Прогон не оставил результатов (alien_results.jsonl / harness.csv)", file=sys.stderr)
        return 2
    alien = summarize_alien(alien_rows)
    public = summarize_public(pd.read_csv(out_dir / "harness.csv"))
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None

    llm = None
    if (out_dir / "llm_alien.json").exists():
        board = pd.read_csv(out_dir / "harness.csv")
        llm = llm_totals(json.loads((out_dir / "llm_alien.json").read_text(encoding="utf-8")), board)

    fails, notes = compare(base, alien, public)
    fails += llm_checks(llm)
    report = render(base, alien, public, fails, notes, llm)
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
