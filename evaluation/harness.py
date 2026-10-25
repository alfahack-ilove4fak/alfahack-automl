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
  3. считаем метрику задачи и баллы по формуле организаторов (evaluation/local_score.py): относительно
     тривиального ответа (score), сильного бейзлайна — лучшего из LightGBM по умолчанию и тривиального ответа
     (model_score) и сильнейшего из набора «тривиальный, LightGBM, RandomForest, линейная» (best_score, #276).
Сдвиг test (#276): в каждой строке shift_auc — AUC классификатора «строка из train или из test». С --holdout test_like
отложенная часть — группы строк train, самые похожие на test (если shift_auc ≥ 0.7; иначе обычная), с --holdout both —
оба прогона подряд и «чувствительность к сдвигу» в отчёте. Метки test при этом не нужны: только его признаки.
Ответы лежат вне папки задачи — агент их не видит. Результаты дописываются в runs/harness/leaderboard.csv.
Из CLI .env в корне подхватывается сам (ключ для --llm); сравнить «без LLM / с Qwen» — scripts/qwen.py compare.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from evaluation.local_score import DETERMINISTIC, MODEL_THREADS, organizer_score, score_submission
from evaluation.submission import validate_submission
from interfaces.schemas import TaskSpec, TaskType

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = ROOT / "runs" / "harness" / "leaderboard.csv"
COLUMNS = [
    "date",
    "commit",
    "task",
    "split",
    "shift_auc",
    "metric",
    "value",
    "baseline",
    "score",
    "model_baseline",
    "model_score",
    "rf_baseline",
    "linear_baseline",
    "best_baseline",
    "best_name",
    "best_score",
    "status",
    "seconds",
    "peak_ram_mb",
    "llm_calls",
    "llm_errors",
    "llm_tokens",
    "llm_rub",
    "minutes",
    "error",
]
SEED = 42
DATE_SHARE = 0.95  # колонку считаем датой, если так разбирается не меньше этой доли значений (как в стенде research)
GRACE_SEC = 60  # сколько ждём сверх бюджета, прежде чем считать, что агент «завис»
# Сдвиг test (#276): на public/multiclass AUC 0.88, на regression 1.0 (год 6). Порог 0.7 — первая оценка капитана,
# на других задачах не проверен
SHIFT_AUC_MIN = 0.7
SHIFT_MAX_ROWS = 50_000  # строк с каждой стороны для классификатора train/test (≈ 10–30 с на 4 потоках)
SHIFT_FOLDS = 4
HOLDOUTS = ("standard", "test_like")


# --- 1. Отложенная часть


def as_dates(series: pd.Series) -> pd.Series | None:
    """Строки-даты → datetime, если разбирается почти всё (≥ DATE_SHARE непустых); иначе None."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    values = series.dropna()
    if values.empty:
        return None
    with warnings.catch_warnings():  # pandas предупреждает, что формат угадывается по строкам
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(values.astype(str), errors="coerce", format="mixed")
        if parsed.notna().mean() < DATE_SHARE:
            return None
        return pd.to_datetime(series.astype(str), errors="coerce", format="mixed")


def time_column(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[str, pd.Series] | None:
    """Колонка, по которой весь test лежит после train, и её значения в train для упорядочивания.

    Как в стенде research (research/bench_playbook.py): целочисленный код периода (2–50 значений, «год 1–6»)
    или дата строкой. Номер записи и прочие «растущие» числа периодом не считаем.
    """
    for col in features:
        if pd.api.types.is_integer_dtype(train[col]) and pd.api.types.is_integer_dtype(test[col]):
            if 1 < train[col].nunique() <= 50 and test[col].min() > train[col].max():
                return col, train[col]
            continue
        tr_dates, te_dates = as_dates(train[col]), as_dates(test[col])
        if tr_dates is not None and te_dates is not None and te_dates.min() > tr_dates.max():
            return col, tr_dates
    return None


def targets_of(spec: TaskSpec) -> list[str]:
    """Все колонки ответа-таргеты: основной и остальные (несколько таргетов, #174, #220)."""
    return [spec.target, *spec.extra_targets]


def holdout_mask(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame, frac: float) -> tuple[np.ndarray, str]:
    """Какие строки train уйдут в «скрытый тест» и как выбраны (для отчёта)."""
    features = [c for c in train.columns if c not in (*targets_of(spec), spec.id_column) and c in test.columns]
    found = time_column(train, test, features)
    if found:
        col, values = found
        counts = values.value_counts().sort_index(ascending=False)  # последние периоды / даты — первыми
        last = counts.index[(counts.cumsum() >= frac * len(train)).argmax()]
        mask = (values >= last).to_numpy()
        if 0 < mask.mean() < 0.6:
            label = last.date() if isinstance(last, pd.Timestamp) else last
            return mask, f"time:{col}>={label}"

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


def shift_features(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Признаки для классификатора «train против test»: общие колонки без id, таргетов, констант и идентификаторов.

    Идентификатор — целые числа или строки, все разные и в train, и в test: по нему test отделяется всегда (AUC 1.0),
    хотя модели он не нужен. Дробные числа с уникальными значениями (площадь, сумма) — обычные признаки.
    """
    features = []
    for col in train.columns:
        if col in (*targets_of(spec), spec.id_column) or col not in test.columns:
            continue
        tr, te = train[col], test[col]
        if tr.nunique(dropna=False) <= 1 and te.nunique(dropna=False) <= 1:
            continue
        id_like = not pd.api.types.is_float_dtype(tr) or not pd.api.types.is_float_dtype(te)
        if id_like and tr.nunique(dropna=False) == len(tr) and te.nunique(dropna=False) == len(te):
            continue
        features.append(col)
    return features


def shift_probability(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, np.ndarray] | None:
    """AUC «train против test» по out-of-fold и вероятность «это test» для каждой строки train.

    Классификатор учится на подвыборке до SHIFT_MAX_ROWS строк с каждой стороны; строкам train вне подвыборки —
    среднее моделей фолдов. None — не из чего считать (нет признаков или строк).
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    features = shift_features(spec, train, test)
    if not features or len(train) < 4 * SHIFT_FOLDS or len(test) < 4 * SHIFT_FOLDS:
        return None
    both = pd.concat([train[features], test[features]], ignore_index=True)
    for col in features:
        if not (pd.api.types.is_numeric_dtype(both[col]) or pd.api.types.is_bool_dtype(both[col])):
            both[col] = both[col].astype(str).astype("category")  # общий словарь категорий train и test
    x_train, x_test = both.iloc[: len(train)], both.iloc[len(train) :]
    rng = np.random.default_rng(SEED)
    tr_rows = np.sort(rng.choice(len(train), min(len(train), SHIFT_MAX_ROWS), replace=False))
    te_rows = np.sort(rng.choice(len(test), min(len(test), SHIFT_MAX_ROWS), replace=False))
    x = pd.concat([x_train.iloc[tr_rows], x_test.iloc[te_rows]], ignore_index=True)
    is_test = np.r_[np.zeros(len(tr_rows)), np.ones(len(te_rows))]
    rest = np.setdiff1d(np.arange(len(train)), tr_rows)
    oof, p_train = np.zeros(len(x)), np.zeros(len(train))
    for fit_rows, pred_rows in StratifiedKFold(SHIFT_FOLDS, shuffle=True, random_state=SEED).split(x, is_test):
        model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.1, random_state=SEED, n_jobs=MODEL_THREADS, verbose=-1, **DETERMINISTIC
        )
        model.fit(x.iloc[fit_rows], is_test[fit_rows])
        oof[pred_rows] = model.predict_proba(x.iloc[pred_rows])[:, 1]
        if len(rest):
            p_train[rest] += model.predict_proba(x_train.iloc[rest])[:, 1] / SHIFT_FOLDS
    p_train[tr_rows] = oof[: len(tr_rows)]
    return float(roc_auc_score(is_test, oof)), p_train


def shifted_holdout_mask(spec: TaskSpec, train: pd.DataFrame, test: pd.DataFrame, p_test: np.ndarray, frac: float):
    """Отложенная часть «как test»: целые группы одинаковых строк train с наибольшей вероятностью «это test».

    Группы — как в holdout_mask, поэтому копия строки не бывает по обе стороны. Берём группы по убыванию средней
    вероятности, пока не наберётся frac строк.
    """
    features = [c for c in train.columns if c not in (*targets_of(spec), spec.id_column) and c in test.columns]
    groups = pd.util.hash_pandas_object(train[features], index=False).to_numpy()
    per_group = pd.Series(p_test).groupby(groups).agg(["mean", "size"])
    per_group = per_group.sort_values("mean", ascending=False, kind="stable")
    take = (per_group["size"].cumsum() <= frac * len(train)).to_numpy()
    take[0] = True  # хотя бы одна группа, даже если она больше frac
    return np.isin(groups, per_group.index[take])


def make_holdout_task(
    task_md: Path, out: Path, frac: float = 0.25, rows: int | None = None, holdout: str = "standard"
) -> dict:
    """Собрать в out/task копию задачи с отложенной частью train вместо test; ответы — в out/answers.csv.

    holdout="test_like" — отложенная часть из строк, самых похожих на test (shifted_holdout_mask), если сдвиг есть
    (shift_auc ≥ SHIFT_AUC_MIN); иначе обычная, а в split — пометка, почему.
    """
    spec = parse_task(task_md)
    train = pd.read_csv(spec.train_path, low_memory=False)
    test = pd.read_csv(spec.test_path, low_memory=False, nrows=1000)
    if rows and len(train) > rows:
        train = train.sample(rows, random_state=SEED).sort_index()
    if spec.id_column not in train.columns:
        train.insert(0, spec.id_column, np.arange(len(train)))
    try:
        shift = shift_probability(spec, train, pd.read_csv(spec.test_path, low_memory=False))
    except Exception as exc:  # noqa: BLE001 — замер сдвига вспомогательный: без него обычная отложенная часть
        print(f"  ⚠️ сдвиг train/test не посчитан: {type(exc).__name__}: {exc}", flush=True)
        shift = None
    shift_auc = shift[0] if shift else float("nan")
    if holdout == "test_like" and shift and shift_auc >= SHIFT_AUC_MIN:
        mask, how = shifted_holdout_mask(spec, train, test, shift[1], frac), f"test_like(auc={shift_auc:.3f})"
    else:
        mask, how = holdout_mask(spec, train, test, frac)
        if holdout == "test_like":
            how += f" (сдвига нет: AUC {shift_auc:.3f} < {SHIFT_AUC_MIN})"
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
    targets = targets_of(spec)
    test_columns = [c for c in test.columns if c in hold.columns and c not in targets]
    if spec.id_column not in test_columns:
        test_columns.insert(0, spec.id_column)
    hold[test_columns].to_csv(place(spec.test_path), index=False)
    if spec.sample_submission_path:
        sample = pd.read_csv(spec.sample_submission_path, nrows=1)
        fillers = {t: sample[t].iloc[0] if t in sample.columns and len(sample) else 0 for t in targets}
        hold[[spec.id_column]].assign(**fillers)[spec.submission_columns].to_csv(
            place(spec.sample_submission_path), index=False
        )
    hold[[spec.id_column, *targets]].to_csv(out / "answers.csv", index=False)
    return {
        "spec": spec,
        "task_md": task_dir / task_md.name,
        "split": how,
        "y_train": part[targets],
        "shift_auc": shift_auc,
    }


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


def llm_usage(run_dir: Path) -> dict:
    """Расход LLM за прогон из журнала агента run_dir/llm_calls.jsonl — для тай-брейка (вебинар: токены, затем время).

    Ответы из кэша не считаем: на платформе кэша нет, а здесь он прячет настоящий расход. Нет журнала — нули.
    """
    usage = {"llm_calls": 0, "llm_errors": 0, "llm_tokens": 0, "llm_rub": 0.0}
    ledger = run_dir / "llm_calls.jsonl"
    if not ledger.exists():
        return usage
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            call = json.loads(line)
        except json.JSONDecodeError:
            continue  # строку дописывал убитый по таймауту агент
        if not isinstance(call, dict) or call.get("cached"):
            continue
        if not call.get("ok"):
            usage["llm_errors"] += 1
            continue
        try:  # журнал пишет клиент агента; кривое поле — пропускаем строку, а не роняем прогон в FAILED
            tokens = int(call.get("prompt") or 0) + int(call.get("completion") or 0)
            rub = float(call.get("cost") or 0.0)
        except (TypeError, ValueError):
            continue
        usage["llm_calls"] += 1
        usage["llm_tokens"] += tokens
        usage["llm_rub"] += rub if np.isfinite(rub) else 0.0
    usage["llm_rub"] = round(usage["llm_rub"], 4)
    return usage


def score_targets(
    spec: TaskSpec,
    merged: pd.DataFrame,
    y_train: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    extra_baselines: bool = True,
) -> dict:
    """Балл по ответу; merged — answers + submission с суффиксами _true/_pred.

    Несколько таргетов (#220): метрика каждого таргета считается отдельно, итог — среднее (так их сводят в task.md),
    бейзлайны — тоже средние, балл — по формуле организаторов от средних. Сильнейший бейзлайн — среднее сильнейших
    по таргетам (у каждого таргета свой).
    """
    parts = []
    for target in targets_of(spec):
        spec_t = spec
        if target != spec.target:
            labels = None  # классы остальных таргетов в TaskSpec не хранятся — берём из данных
            if spec.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
                labels = sorted(pd.concat([y_train[target], merged[f"{target}_true"]]).dropna().astype(str).unique())
            spec_t = spec.model_copy(update={"target": target, "class_labels": labels})
        parts.append(
            score_submission(
                spec_t,
                merged[f"{target}_true"],
                merged[f"{target}_pred"],
                y_train[target],
                x_train=x_train,
                x_test=x_test,
                extra_baselines=extra_baselines,
            )
        )
    if len(parts) == 1:
        return parts[0]
    name = spec.metric.name
    result = {"metric": name, **{k: float(np.mean([p[k] for p in parts])) for k in ("value", "baseline")}}
    result["score"] = organizer_score(name, result["value"], result["baseline"])
    model_baselines = [p.get("model_baseline", float("nan")) for p in parts]
    if all(np.isfinite(model_baselines)):
        result["model_baseline"] = float(np.mean(model_baselines))
        result["model_score"] = organizer_score(name, result["value"], result["model_baseline"])
    for key in ("rf_baseline", "linear_baseline"):
        if any(key in p for p in parts):
            result[key] = float(np.mean([p.get(key, float("nan")) for p in parts]))
    best = [p.get("best_baseline", float("nan")) for p in parts]
    if all(np.isfinite(best)):
        result["best_baseline"] = float(np.mean(best))
        result["best_name"] = "/".join(dict.fromkeys(p["best_name"] for p in parts))
        result["best_score"] = organizer_score(name, result["value"], result["best_baseline"])
    errors = [p["model_error"] for p in parts if "model_error" in p]
    if errors:
        result.update(model_baseline=float("nan"), model_score=float("nan"), model_error=errors[0])
    return result


def run_task(
    task_md: Path,
    workdir: Path,
    minutes: float,
    frac: float,
    rows: int | None,
    use_llm: bool,
    agent_root: Path = ROOT,
    holdout: str = "standard",
    extra_baselines: bool = True,
) -> dict:
    """Одна строка лидерборда. Любая ошибка — строка со status=FAILED и score=0, а не исключение."""
    row = {"task": task_md.parent.relative_to(ROOT).as_posix() if task_md.is_relative_to(ROOT) else str(task_md)}
    row.update(minutes=minutes, score=0.0, status="FAILED", error="")
    out = workdir / (task_md.parent.name + ("" if holdout == "standard" else f"_{holdout}"))
    try:
        prepared = make_holdout_task(task_md, out, frac, rows, holdout)
        row["split"], row["shift_auc"] = prepared["split"], prepared["shift_auc"]
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
        row.update(llm_usage(out / "run"))  # до проверки ответа: расход важен и у невалидного сабмита
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
        x_train = pd.read_csv(new_spec.train_path, low_memory=False).drop(columns=targets_of(spec), errors="ignore")
        scores = score_targets(
            spec,
            merged,
            prepared["y_train"],
            x_train=x_train.drop(columns=[spec.id_column], errors="ignore"),
            x_test=x_test,
            extra_baselines=extra_baselines,
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


def _num(value, fmt: str = ".1f") -> str:
    try:
        return "—" if value is None or not np.isfinite(float(value)) else format(float(value), fmt)
    except (TypeError, ValueError):
        return "—"


def holdout_of(row: dict) -> str:
    """Какой отложенной частью получена строка: test_like или standard (по полю split)."""
    return "test_like" if str(row.get("split", "")).startswith("test_like") else "standard"


def render(rows: list[dict]) -> str:
    lines = [
        "| Задача | Разбиение | AUC сдвига | Метрика | Агент | Бейзлайн | Балл | Сильный бейзлайн | Балл к сильному "
        "| Сильнейший бейзлайн | Балл к сильнейшему | Статус | Время, с | Пик RAM, МБ | LLM: запросов (ошибок) "
        "| Токены LLM | ₽ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        status = r["status"] + (f" — {r['error']}" if r.get("error") else "")
        best = f"{_num(r.get('best_baseline'), '.5g')} ({r['best_name']})" if r.get("best_name") else "—"
        lines.append(
            f"| {r['task']} | {r.get('split', '—')} | {_num(r.get('shift_auc'), '.3f')} | {r.get('metric', '—')} | "
            f"{_num(r.get('value'), '.5g')} | {_num(r.get('baseline'), '.5g')} | {r['score']:.1f} | "
            f"{_num(r.get('model_baseline'), '.5g')} | {_num(r.get('model_score'))} | {best} | "
            f"{_num(r.get('best_score'))} | {status} | {r.get('seconds', 0):.0f} | {r.get('peak_ram_mb', 0):.0f} | "
            f"{r.get('llm_calls', 0)} ({r.get('llm_errors', 0)}) | {r.get('llm_tokens', 0)} | {r.get('llm_rub', 0.0):.2f} |"
        )
    standard = [r for r in rows if holdout_of(r) == "standard"] or rows
    lines.append(
        f"\nСредний балл: **{np.mean([r['score'] for r in standard]):.1f}** (платформа берёт среднее по задачам)"
    )
    for key, title in (
        ("model_score", "к сильному бейзлайну (лучший из LightGBM и тривиального)"),
        ("best_score", "к сильнейшему бейзлайну (тривиальный, LightGBM, RandomForest, линейная) — оценка снизу"),
    ):
        values = [r.get(key, float("nan")) for r in standard]
        if not np.all(np.isnan(np.asarray(values, dtype=float))):
            lines.append(f"Средний балл {title}: **{np.nanmean(values):.1f}**")
    shifted = {r["task"]: r for r in rows if holdout_of(r) == "test_like"}
    for r in standard:
        other = shifted.get(r["task"])
        if other is not None and r is not other:
            lines.append(
                f"Чувствительность к сдвигу, {r['task']}: {r.get('metric', '')} {_num(r.get('value'), '.4g')} → "
                f"{_num(other.get('value'), '.4g')} на test-подобной части; балл к сильному {_num(r.get('model_score'))}"
                f" → {_num(other.get('model_score'))}, к сильнейшему {_num(r.get('best_score'))} → "
                f"{_num(other.get('best_score'))}"
            )
    calls = sum(r.get("llm_calls", 0) + r.get("llm_errors", 0) for r in rows)
    if calls:
        tokens, rub = sum(r.get("llm_tokens", 0) for r in rows), sum(r.get("llm_rub", 0.0) for r in rows)
        lines.append(
            f"LLM за прогон: {calls} запросов, **{tokens} токенов** ({tokens / len(rows):.0f} на задачу), {rub:.2f} ₽ "
            "— тай-брейк при равном балле"
        )
    return "\n".join(lines)


def load_env_file(path: Path = ROOT / ".env") -> bool:
    """Подхватить .env из корня репозитория (ключ Qwen и настройки), не перезаписывая заданные переменные.

    Только для локального запуска: на платформе .env нет, ключ задаёт сама платформа.
    """
    if not path.exists():
        return False
    from dotenv import load_dotenv

    return load_dotenv(path, override=False)


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
    parser.add_argument(
        "--holdout",
        choices=[*HOLDOUTS, "both"],
        default="standard",
        help="отложенная часть: standard — как обычно; test_like — строки train, самые похожие на test "
        f"(при AUC сдвига ≥ {SHIFT_AUC_MIN}); both — оба прогона подряд и чувствительность к сдвигу",
    )
    parser.add_argument(
        "--no-extra-baselines",
        action="store_true",
        help="не считать RandomForest и линейную модель (быстрее; best_score тогда — из тривиального и LightGBM)",
    )
    args = parser.parse_args(argv)

    tasks = find_tasks() if args.all else [t.resolve() for t in args.task or []]
    if not tasks:
        parser.error("укажи --all или --task")
    if args.llm and os.getenv("LLM_CACHE_DIR"):  # на платформе кэша нет: ответы из него не считаются в расход
        print(f"⚠️ LLM_CACHE_DIR={os.getenv('LLM_CACHE_DIR')}: ответы из кэша не попадут в токены и ₽ — расход занижен")
    commit, date = commit_id(args.agent_root.resolve()), time.strftime("%Y-%m-%d %H:%M")
    rows = []
    # Дочерние процессы агента могут дописывать файлы в момент удаления папки («Directory not empty») —
    # ошибка уборки не должна терять результаты: строки пишем внутри with, уборку не роняем
    with tempfile.TemporaryDirectory(prefix="harness_", ignore_cleanup_errors=True) as tmp:
        holdouts = HOLDOUTS if args.holdout == "both" else (args.holdout,)
        for task_md in tasks:
            for holdout in holdouts:
                if holdout == "test_like" and args.holdout == "both":
                    auc = rows[-1].get("shift_auc", float("nan")) if rows else float("nan")
                    if not (np.isfinite(auc) and auc >= SHIFT_AUC_MIN):
                        print(f"  test_like пропущен: AUC сдвига {_num(auc, '.3f')} < {SHIFT_AUC_MIN}", flush=True)
                        continue
                print(f"… {task_md.parent.name} [{holdout}]: бюджет {args.minutes} мин", flush=True)
                row = run_task(
                    task_md,
                    Path(tmp),
                    args.minutes,
                    args.frac,
                    args.rows,
                    args.llm,
                    args.agent_root.resolve(),
                    holdout,
                    not args.no_extra_baselines,
                )
                rows.append({"date": date, "commit": commit, **row})
                print(
                    f"  {row['status']}: балл {row['score']:.1f}, к сильнейшему {_num(row.get('best_score'))}",
                    flush=True,
                )
        append_rows(rows, args.out)
    print("\n" + render(rows))
    print(f"\nСтроки дописаны в {args.out}")
    return 0 if all(r["status"] == "OK" for r in rows) else 1


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → кириллица роняет вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    load_env_file()  # только CLI: тесты зовут main() без .env
    sys.exit(main())
