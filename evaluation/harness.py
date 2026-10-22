"""Харнесс — локальный лидерборд: насколько хорош агент, без сабмитов на платформу.

    uv run python -m evaluation.harness --all --minutes 5          # tasks/public/* и tasks/practice/*
    uv run python -m evaluation.harness --task tasks/public/regression/task.md --minutes 10
    uv run python -m evaluation.harness --all --minutes 1 --rows 5000   # быстро: урезанный train

Для каждой задачи:
  1. из train.csv откладываем ~25% строк — «скрытый тест» с известными ответами. Как у стенда research
     (research/bench_playbook.py): если есть колонка времени, по которой test идёт после train, — последние
     периоды; иначе случайно, но полные дубликаты строк держим по одну сторону (иначе оценка завышена);
  2. собираем копию задачи (task.md, остальной train, новый test без target, sample_submission)
     и запускаем solution.py отдельным процессом с бюджетом времени — как платформа;
  3. считаем метрику задачи и два балла по формуле организаторов (evaluation/local_score.py): относительно
     тривиального ответа (score) и относительно сильного бейзлайна — лучшего из LightGBM по умолчанию и
     тривиального ответа (model_score); второй ближе к бейзлайну организаторов, по нему сверяемся с public.
Ответы лежат вне папки задачи — агент их не видит. Результаты дописываются в runs/harness/leaderboard.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from agent.task_parser import parse_task
from evaluation.local_score import score_submission
from evaluation.submission import validate_submission
from interfaces.schemas import TaskSpec, TaskType

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "runs" / "harness" / "leaderboard.csv"
COLUMNS = [
    "date",
    "commit",
    "task",
    "split",
    "metric",
    "value",
    "baseline",
    "score",
    "model_baseline",
    "model_score",
    "status",
    "seconds",
    "peak_ram_mb",
    "minutes",
    "error",
]
SEED = 42
GRACE_SEC = 60  # сколько ждём сверх бюджета, прежде чем считать, что агент «завис»


# --- 1. Отложенная часть


def time_column(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> str | None:
    """Целочисленная колонка с небольшим числом значений, где весь test лежит после train, — «период»."""
    for col in features:
        if not (pd.api.types.is_integer_dtype(train[col]) and pd.api.types.is_integer_dtype(test[col])):
            continue
        if 1 < train[col].nunique() <= 50 and test[col].min() > train[col].max():
            return col
    return None


def holdout_mask(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame, frac: float) -> tuple[np.ndarray, str]:
    """Какие строки train уйдут в «скрытый тест» и как выбраны (для отчёта)."""
    features = [c for c in train.columns if c not in (spec.target, spec.id_column) and c in test.columns]
    col = time_column(train, test, features)
    if col:
        counts = train[col].value_counts().sort_index(ascending=False)
        last = counts.index[(counts.cumsum() >= frac * len(train)).argmax()]
        mask = (train[col] >= last).to_numpy()
        if 0 < mask.mean() < 0.6:
            return mask, f"time:{col}>={last}"

    groups = pd.util.hash_pandas_object(train[features], index=False).to_numpy()
    rows = np.arange(len(train))
    mask = np.zeros(len(train), dtype=bool)
    if spec.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        try:
            folds = StratifiedGroupKFold(max(2, round(1 / frac)), shuffle=True, random_state=SEED)
            with warnings.catch_warnings():  # «в редком классе меньше объектов, чем фолдов» — ожидаемо
                warnings.simplefilter("ignore", UserWarning)
                _, hold = next(folds.split(rows, train[spec.target].astype(str), groups))
            mask[hold] = True
            return mask, "stratified_groups"
        except ValueError:
            pass  # редкие классы — ниже обычное разбиение по группам
    _, hold = next(GroupShuffleSplit(1, test_size=frac, random_state=SEED).split(rows, groups=groups))
    mask[hold] = True
    return mask, "groups"


def make_holdout_task(task_md: Path, out: Path, frac: float = 0.25, rows: int | None = None) -> dict:
    """Собрать в out/task копию задачи с отложенной частью train вместо test; ответы — в out/answers.csv."""
    spec = parse_task(task_md)
    train = pd.read_csv(spec.train_path, low_memory=False)
    test = pd.read_csv(spec.test_path, low_memory=False, nrows=1000)
    if rows and len(train) > rows:
        train = train.sample(rows, random_state=SEED).sort_index()
    if spec.id_column not in train.columns:
        train.insert(0, spec.id_column, np.arange(len(train)))
    mask, how = holdout_mask(spec, train, test, frac)
    part, hold = train[~mask], train[mask]

    task_dir = out / "task"
    data_files = {spec.train_path, spec.test_path, spec.sample_submission_path, spec.submission_path}
    shutil.copytree(
        spec.task_dir,
        task_dir,
        ignore=lambda folder, names: [n for n in names if Path(folder, n) in data_files or n == "__pycache__"],
    )

    def place(path: Path) -> Path:
        target = task_dir / path.relative_to(spec.task_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    part.to_csv(place(spec.train_path), index=False)
    test_columns = [c for c in test.columns if c in hold.columns and c != spec.target]
    if spec.id_column not in test_columns:
        test_columns.insert(0, spec.id_column)
    hold[test_columns].to_csv(place(spec.test_path), index=False)
    if spec.sample_submission_path:
        sample = pd.read_csv(spec.sample_submission_path, nrows=1)
        filler = sample[spec.target].iloc[0] if spec.target in sample.columns and len(sample) else 0
        hold[[spec.id_column]].assign(**{spec.target: filler})[spec.submission_columns].to_csv(
            place(spec.sample_submission_path), index=False
        )
    hold[[spec.id_column, spec.target]].to_csv(out / "answers.csv", index=False)
    return {"spec": spec, "task_md": task_dir / task_md.name, "split": how, "y_train": part[spec.target]}


# --- 2. Запуск агента


def run_with_limits(cmd: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[int | None, float, float]:
    """Запустить процесс и следить за ним: код выхода (None — убит по таймауту), секунды, пик RAM всего дерева."""
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak, killed = 0.0, False
    try:
        parent = psutil.Process(proc.pid)
        while proc.poll() is None:
            if time.time() - started > timeout:
                for child in parent.children(recursive=True):
                    child.kill()
                proc.kill()
                proc.wait()
                killed = True
                break
            try:
                tree = [parent, *parent.children(recursive=True)]
                peak = max(peak, sum(p.memory_info().rss for p in tree) / 2**20)
            except psutil.Error:
                pass  # процесс завершился между проверками
            time.sleep(0.5)
    except psutil.NoSuchProcess:
        proc.wait()
    return (None if killed else proc.returncode), time.time() - started, peak


def run_task(
    task_md: Path,
    workdir: Path,
    minutes: float,
    frac: float,
    rows: int | None,
    use_llm: bool,
    agent_root: Path = ROOT,
) -> dict:
    """Одна строка лидерборда. Любая ошибка — строка со status=FAILED и score=0, а не исключение."""
    row = {"task": task_md.parent.relative_to(ROOT).as_posix() if task_md.is_relative_to(ROOT) else str(task_md)}
    row.update(minutes=minutes, score=0.0, status="FAILED", error="")
    out = workdir / task_md.parent.name
    try:
        prepared = make_holdout_task(task_md, out, frac, rows)
        row["split"] = prepared["split"]
        env = {
            **os.environ,
            "AGENT_TIME_BUDGET_SEC": str(int(minutes * 60)),
            "AGENT_RUN_DIR": str(out / "run"),
        }
        env.pop("PYTHONPATH", None)
        if not use_llm:
            env["LLM_MOCK"] = "1"
        code, row["seconds"], row["peak_ram_mb"] = run_with_limits(
            [sys.executable, str(agent_root / "solution.py"), str(prepared["task_md"])],
            prepared["task_md"].parent,
            env,
            minutes * 60 + GRACE_SEC,
        )
        spec: TaskSpec = prepared["spec"]
        new_spec = parse_task(prepared["task_md"])
        test = pd.read_csv(new_spec.test_path)
        report = validate_submission(new_spec.submission_path, new_spec, test)
        if not report.ok:
            row["error"] = "; ".join(report.problems)[:300]
            return row
        answers = pd.read_csv(out / "answers.csv")
        submission = pd.read_csv(new_spec.submission_path)
        merged = answers.merge(submission, on=spec.id_column, suffixes=("_true", "_pred"))
        features = [c for c in test.columns if c != spec.id_column]
        x_test = merged[[spec.id_column]].merge(test, on=spec.id_column, how="left")[features]
        x_train = pd.read_csv(new_spec.train_path, low_memory=False).drop(columns=[spec.target], errors="ignore")
        scores = score_submission(
            spec,
            merged[f"{spec.target}_true"],
            merged[f"{spec.target}_pred"],
            prepared["y_train"],
            x_train=x_train.drop(columns=[spec.id_column], errors="ignore"),
            x_test=x_test,
        )
        if "model_error" in scores:
            print(f"  ⚠️ модельный бейзлайн не посчитан: {scores.pop('model_error')}", flush=True)
        row.update(scores)
        row["status"] = "OK" if code == 0 else ("TIMEOUT" if code is None else f"EXIT_{code}")
    except Exception as exc:  # noqa: BLE001 — упавшая задача не должна ронять весь прогон
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return row


# --- 3. Лидерборд


def commit_id(root: Path = ROOT) -> str:
    proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip()
    return (proc.stdout.strip() or "unknown") + ("-dirty" if dirty else "")


def append_rows(rows: list[dict], path: Path = LEADERBOARD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != COLUMNS:  # старый лидерборд без новых колонок — переписываем с новой шапкой, строки сохраняем
            pd.read_csv(path).reindex(columns=COLUMNS).to_csv(path, index=False)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict]) -> str:
    lines = [
        "| Задача | Разбиение | Метрика | Агент | Бейзлайн | Балл | Сильный бейзлайн | Балл к сильному | Статус | Время, с "
        "| Пик RAM, МБ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        status = r["status"] + (f" — {r['error']}" if r.get("error") else "")
        model_score = r.get("model_score", float("nan"))
        lines.append(
            f"| {r['task']} | {r.get('split', '—')} | {r.get('metric', '—')} | {r.get('value', float('nan')):.5g} | "
            f"{r.get('baseline', float('nan')):.5g} | {r['score']:.1f} | {r.get('model_baseline', float('nan')):.5g} | "
            f"{'—' if np.isnan(model_score) else f'{model_score:.1f}'} | {status} | "
            f"{r.get('seconds', 0):.0f} | {r.get('peak_ram_mb', 0):.0f} |"
        )
    lines.append(f"\nСредний балл: **{np.mean([r['score'] for r in rows]):.1f}** (платформа берёт среднее по задачам)")
    model_scores = [r.get("model_score", float("nan")) for r in rows]
    if not np.all(np.isnan(model_scores)):
        lines.append(
            f"Средний балл к сильному бейзлайну (лучший из LightGBM и тривиального): "
            f"**{np.nanmean(model_scores):.1f}** — ближе к платформе, сверяем с public"
        )
    return "\n".join(lines)


def find_tasks() -> list[Path]:
    return sorted(ROOT.glob("tasks/public/*/task.md")) + sorted(ROOT.glob("tasks/practice/*/task.md"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Локальный лидерборд: отложенная часть train → агент → балл")
    parser.add_argument("--all", action="store_true", help="все задачи tasks/public и tasks/practice")
    parser.add_argument("--task", type=Path, action="append", help="task.md (можно несколько)")
    parser.add_argument("--minutes", type=float, default=5, help="бюджет агента на задачу, минут (по умолчанию 5)")
    parser.add_argument("--frac", type=float, default=0.25, help="доля train в «скрытый тест» (по умолчанию 0.25)")
    parser.add_argument("--rows", type=int, help="урезать train до N строк до разбиения (быстрый прогон)")
    parser.add_argument("--llm", action="store_true", help="не выключать LLM (тратит общий бюджет ₽)")
    parser.add_argument("--out", type=Path, default=LEADERBOARD, help="CSV лидерборда (дописываем строки)")
    parser.add_argument(
        "--agent-root",
        type=Path,
        default=ROOT,
        help="папка с solution.py другой версии агента (сравнение «было/стало»: задачи и разбиение — те же)",
    )
    args = parser.parse_args(argv)

    tasks = find_tasks() if args.all else [t.resolve() for t in args.task or []]
    if not tasks:
        parser.error("укажи --all или --task")
    commit, date = commit_id(args.agent_root.resolve()), time.strftime("%Y-%m-%d %H:%M")
    rows = []
    # Дочерние процессы агента могут дописывать файлы в момент удаления папки («Directory not empty») —
    # ошибка уборки не должна терять результаты: строки пишем внутри with, уборку не роняем
    with tempfile.TemporaryDirectory(prefix="harness_", ignore_cleanup_errors=True) as tmp:
        for task_md in tasks:
            print(f"… {task_md.parent.name}: бюджет {args.minutes} мин", flush=True)
            row = run_task(task_md, Path(tmp), args.minutes, args.frac, args.rows, args.llm, args.agent_root.resolve())
            rows.append({"date": date, "commit": commit, **row})
            print(f"  {row['status']}: балл {row['score']:.1f}", flush=True)
        append_rows(rows, args.out)
    print("\n" + render(rows))
    print(f"\nСтроки дописаны в {args.out}")
    return 0 if all(r["status"] == "OK" for r in rows) else 1


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → кириллица роняет вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
