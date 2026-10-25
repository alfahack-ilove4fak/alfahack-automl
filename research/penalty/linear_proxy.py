"""Проверка гипотезы «штраф public − харнесс = регрессия на платформе даёт 0» (#276). Владелец — research.

    uv run python research/penalty/linear_proxy.py                       # линейный бейзлайн на отложенной части + прогноз public
    uv run python research/penalty/linear_proxy.py --reg 44.14 --f1 0.647   # свои значения агента

Идея: бейзлайн организаторов, судя по сабмиту №1, — простая линейная модель (docs/research/validation.md:
RMSE 49.34 / F1 0.421 на отложенной части, при агенте 51.25 / 0.6227 это даёт 0.0 / 34.8, среднее 17.4 = public 17.4595).
Здесь тот же линейный бейзлайн считается заново на отложенной части харнесса (evaluation/harness.py), затем для набора
версий агента (RMSE регрессии и F1 multiclass на отложенной части) выводится «прогноз public» = среднее двух баллов к линейной
модели. Сходится с реальным public — значит харнессу нужен линейный бейзлайн (иначе он завышает регрессию), а резерв
качества — именно в регрессии. Не сходится — штраф лежит в другом месте (бисекция по коммитам).

Известные версии: значения агента на отложенной части и public платформы — из docs/experiments.md и docs/submissions.md.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import harness  # noqa: E402

REG, MULTI = ROOT / "tasks" / "public" / "regression" / "task.md", ROOT / "tasks" / "public" / "multiclass" / "task.md"

# (версия, RMSE регрессии на отложенной части, Macro F1 multiclass на отложенной части, public платформы)
KNOWN = [
    ("№1 b6c6e0c", 51.254, 0.6227, 17.4595),
    ("№5 3fc774f (60 мин, без LLM)", 44.138, 0.64736, 17.55),
    ("№6 3fc774f без LLM (60 мин)", 44.138, 0.64736, 15.64),
]


def linear_model(kind: str, x: pd.DataFrame):
    numeric = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical = [c for c in x.columns if c not in numeric]
    parts = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric)]
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
        parts.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), encoder), categorical))
    prep = ColumnTransformer(parts, sparse_threshold=0.3)
    model = Ridge(alpha=1.0) if kind == "reg" else LogisticRegression(max_iter=300)
    return make_pipeline(prep, model)


def linear_baseline(task_md: Path, kind: str) -> tuple[float, str]:
    """Метрика линейной модели на отложенной части харнесса и описание разбиения."""
    with tempfile.TemporaryDirectory() as tmp:
        info = harness.make_holdout_task(task_md, Path(tmp))
        spec = info["spec"]
        task_dir = Path(tmp) / "task"
        train = pd.read_csv(task_dir / spec.train_path.relative_to(spec.task_dir), low_memory=False)
        test = pd.read_csv(task_dir / spec.test_path.relative_to(spec.task_dir), low_memory=False)
        answers = pd.read_csv(Path(tmp) / "answers.csv", low_memory=False)
        features = [c for c in train.columns if c in test.columns and c not in (spec.target, spec.id_column)]
        model = linear_model(kind, train[features]).fit(train[features], train[spec.target])
        pred = model.predict(test[features])
        truth = answers[spec.target].to_numpy()
        if kind == "reg":
            return float(np.sqrt(np.mean((truth.astype(float) - pred) ** 2))), info["split"]
        return float(f1_score(truth.astype(str), pred.astype(str), average="macro")), info["split"]


def score_lower(value: float, base: float) -> float:
    return 100 * max(0.0, 1 - value / base)


def score_higher(value: float, base: float) -> float:
    return 100 * max(0.0, (value - base) / (1 - base))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reg", type=float, help="RMSE агента на отложенной части public/regression")
    parser.add_argument("--f1", type=float, help="Macro F1 агента на отложенной части public/multiclass")
    args = parser.parse_args()
    rmse_lin, split_r = linear_baseline(REG, "reg")
    f1_lin, split_m = linear_baseline(MULTI, "multi")
    print(f"Линейный бейзлайн на отложенной части харнесса: regression RMSE {rmse_lin:.3f} ({split_r}), "
          f"multiclass Macro F1 {f1_lin:.4f} ({split_m})")  # fmt: skip
    print("(у Ромы в docs/research/validation.md: 49.34 / 0.421)\n")
    rows = list(KNOWN)
    if args.reg is not None and args.f1 is not None:
        rows.append(("ваша версия", args.reg, args.f1, float("nan")))
    lines = ["| Версия | RMSE | F1 | Балл рег. к линейной | Балл multi к линейной | Прогноз public | Реальный public | Разница |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, reg, f1, public in rows:
        s_reg, s_multi = score_lower(reg, rmse_lin), score_higher(f1, f1_lin)
        pred = (s_reg + s_multi) / 2
        diff = f"{public - pred:+.1f}" if np.isfinite(public) else ""
        lines.append(f"| {name} | {reg} | {f1} | {s_reg:.1f} | {s_multi:.1f} | **{pred:.1f}** | "
                     f"{public if np.isfinite(public) else '—'} | {diff} |")  # fmt: skip
    print("\n".join(lines))
    print(
        "\nЕсли «Разница» у №1 около 0, а у №5/№6 заметно отрицательна — линейный бейзлайн воспроизводит №1, "
        "но не объясняет всё падение; смотрим, какая из двух задач его даёт (баллы по задачам платформы)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
