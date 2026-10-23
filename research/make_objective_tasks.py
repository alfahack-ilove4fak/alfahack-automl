"""Задачи стенда для #237 — цель обучения под метрику: pinball@0.9, Poisson, Huber, Tweedie. Владелец — research.

    uv run python research/make_objective_tasks.py          # → runs/objective/<задача>/ (в git не едет)
    uv run python research/bench_playbook.py --tasks runs/objective/* tasks/practice/claims_regression_mape_rowno ...

Зачем отдельно от tasks/practice/: это данные для стенда гипотез, а не тренировочные задачи харнесса —
не меняют набор задач `harness --all` у команды. Метрики — из реестра evaluation/metrics.py; в каждой задаче
цель обучения L2 (по умолчанию у LightGBM) расходится с метрикой: асимметрия (квантиль), счётные данные
(Пуассон), тяжёлые хвосты (Huber), много нулей (Tweedie). Всё детерминировано (SEED), без интернета.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.make_practice_tasks import _dictionary, _standard_md, _write  # noqa: E402

OUT = ROOT / "runs" / "objective"
SEED = 2037
N = 8000
TASKS: dict[str, Callable[[Path], None]] = {}


def task(name: str):
    def register(fn):
        TASKS[name] = fn
        return fn

    return register


def _features(rng: np.random.Generator) -> pd.DataFrame:
    """Общие признаки: числа, категория с эффектом, пропуски."""
    regions = np.array([f"R{i}" for i in range(8)])
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=N),
            "x2": rng.uniform(0, 10, N),
            "x3": rng.integers(0, 50, N),
            "segment": rng.choice(regions, N),
        }
    )
    df.loc[rng.random(N) < 0.05, "x2"] = np.nan
    return df


def _effect(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    seg = dict(zip([f"R{i}" for i in range(8)], rng.normal(0, 0.3, 8), strict=True))
    return 0.4 * df["x1"] + 0.08 * df["x2"].fillna(5) + 0.01 * df["x3"] + df["segment"].map(seg)


def _save(task_dir: Path, df: pd.DataFrame, target: str, title: str, about: str, target_text: str, metric_text: str):
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df.insert(0, "id", rng.permutation(len(df)) + 10000)
    train, test_full = train_test_split(df, test_size=0.25, random_state=SEED)
    train, test_full = train.reset_index(drop=True), test_full.sort_values("id").reset_index(drop=True)
    dictionary = _dictionary(
        [
            ("id", "id", "int", "Идентификатор"),
            ("x1", "feature", "numeric", "Признак 1"),
            ("x2", "feature", "numeric", "Признак 2; бывают пропуски"),
            ("x3", "feature", "int", "Признак 3"),
            ("segment", "feature", "category", "Сегмент"),
            (target, "target", "numeric", target_text),
        ]
    )
    md = _standard_md(
        title,
        about,
        len(train),
        len(test_full),
        "`x1`, `x2`, `x3` — числа; `segment` — категория.",
        f"Для каждой строки `test.csv` — число `{target}`.",
        metric_text,
        f"id,{target}",
    )
    _write(
        task_dir,
        {
            "train.csv": train,
            "test.csv": test_full.drop(columns=[target]),
            "data_dictionary.csv": dictionary,
            "sample_submission.csv": test_full[["id"]].assign(**{target: 0.0}),
        },
        md,
    )


@task("demand_pinball90")
def demand(task_dir: Path) -> None:
    """Спрос со скошенным шумом, разброс растёт с уровнем: нужен 90-й квантиль, а не среднее."""
    rng = np.random.default_rng(SEED + 1)
    df = _features(rng)
    level = np.exp(1.5 + _effect(df, rng))
    df["demand"] = (level * rng.lognormal(0, 0.6, N)).round(2)
    _save(
        task_dir,
        df,
        "demand",
        "Запас под спрос",
        "Оцените запас товара на неделю: недозаказ обходится в 9 раз дороже перезаказа.",
        "Спрос за неделю, шт.",
        "pinball loss (квантильная потеря) для квантиля 0.9 (меньше — лучше)",
    )


@task("visits_poisson")
def visits(task_dir: Path) -> None:
    """Счётные данные: число визитов ~ Пуассон с лог-линейной интенсивностью."""
    rng = np.random.default_rng(SEED + 2)
    df = _features(rng)
    df["visits"] = rng.poisson(np.exp(0.3 + _effect(df, rng)))
    _save(
        task_dir,
        df,
        "visits",
        "Визиты клиента",
        "Сколько раз клиент придёт в отделение за месяц.",
        "Число визитов за месяц",
        "mean Poisson deviance, пуассоновская девианса (меньше — лучше)",
    )


@task("latency_huber")
def latency(task_dir: Path) -> None:
    """Тяжёлые хвосты шума (Стьюдент, 2 степени свободы): квадрат ошибки гоняется за выбросами."""
    rng = np.random.default_rng(SEED + 3)
    df = _features(rng)
    df["latency"] = (2.0 + 1.5 * _effect(df, rng) + 0.8 * rng.standard_t(2, N)).round(3)
    _save(
        task_dir,
        df,
        "latency",
        "Задержка ответа сервиса",
        "Оцените задержку ответа; в логах бывают редкие огромные значения из-за сбоев сети.",
        "Задержка, с",
        "Huber loss (меньше — лучше)",
    )


@task("losses_tweedie")
def losses(task_dir: Path) -> None:
    """Страховые потери: ~65% нулей, остальное — гамма с лог-линейным средним (Tweedie)."""
    rng = np.random.default_rng(SEED + 4)
    df = _features(rng)
    eff = _effect(df, rng)
    claim = rng.random(N) < 1 / (1 + np.exp(-(-0.8 + eff)))
    df["loss"] = np.where(claim, rng.gamma(2.0, np.exp(1.0 + 0.5 * eff) / 2.0), 0.0).round(2)
    _save(
        task_dir,
        df,
        "loss",
        "Убыток по полису",
        "Оцените убыток по полису за год; у большинства полисов убытка нет.",
        "Убыток за год, тыс. ₽",
        "mean Tweedie deviance (меньше — лучше)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Задачи стенда для целей обучения под метрику (#237)")
    parser.add_argument("--only", nargs="*", choices=sorted(TASKS))
    args = parser.parse_args()
    for name in args.only or sorted(TASKS):
        task_dir = OUT / name
        shutil.rmtree(task_dir, ignore_errors=True)
        TASKS[name](task_dir)
        print(f"✅ {name} → {task_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
