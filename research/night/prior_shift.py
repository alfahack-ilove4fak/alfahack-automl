"""Подстройка долей классов под test без меток (EM, Saerens et al., 2002) — прототип для #276, multiclass.

Идея: модель обучена при долях классов train p_tr. Если в test доли другие (q), правильные вероятности на test —
P(y|x)·q/p_tr с нормировкой. q неизвестна, её оценивает EM прямо по вероятностям модели на test:
q ← среднее скорректированных вероятностей, пока не сойдётся. Меток test не нужно, работает на любой классификации.

Сравниваем четыре способа ответа (одна и та же модель LightGBM):
  argmax        — как бейзлайн без множителей (№1);
  multipliers   — множители под метрику, подобранные на OOF train (как proba_weights в агенте сейчас);
  em            — EM-доли test, затем argmax;
  em+multipliers — EM-доли test, затем те же множители.
Проверки: отложенная часть train без сдвига; та же часть, пересэмплированная под случайные доли классов
(Dirichlet α=1); под доли, оценённые EM на настоящем test; строки настоящего test с точной копией в train.

    uv run python research/night/prior_shift.py tasks/public/multiclass/task.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.task_parser import parse_task  # noqa: E402
from evaluation import harness  # noqa: E402
from evaluation.metrics import compute_metric  # noqa: E402
from ml.class_weights import fit_weights  # noqa: E402

SEED = 7
N_SHIFTS = 30
MODEL = {"n_estimators": 400, "learning_rate": 0.05, "verbose": -1, "random_state": SEED}


def em_priors(proba: np.ndarray, p_train: np.ndarray, iters: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Оценка долей классов test по вероятностям модели (Saerens, Latinne, Decaestecker, 2002)."""
    q = p_train.copy()
    for _ in range(iters):
        adj = proba * (q / p_train)
        adj /= adj.sum(axis=1, keepdims=True)
        q_new = adj.mean(axis=0)
        if np.abs(q_new - q).max() < tol:
            return q_new
        q = q_new
    return q


def adjust(proba: np.ndarray, q: np.ndarray, p_train: np.ndarray) -> np.ndarray:
    adj = proba * (q / p_train)
    return adj / adj.sum(axis=1, keepdims=True)


def matrix(train: pd.DataFrame, other: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, xo = train[features].copy(), other[features].copy()
    for col in x.columns:
        if x[col].dtype == object:
            cats = pd.Categorical(pd.concat([x[col], xo[col]]).astype(str)).categories
            x[col] = pd.Categorical(x[col].astype(str), categories=cats)
            xo[col] = pd.Categorical(xo[col].astype(str), categories=cats)
    return x, xo


def fit_proba(train: pd.DataFrame, y: np.ndarray, other: pd.DataFrame, features: list[str], k: int) -> np.ndarray:
    x, xo = matrix(train, other, features)
    model = lgb.LGBMClassifier(**MODEL).fit(x, y)
    proba = np.zeros((len(other), k))
    proba[:, model.classes_] = model.predict_proba(xo)
    return proba


def answers(proba: np.ndarray, p_train: np.ndarray, w: np.ndarray) -> dict[str, np.ndarray]:
    q = em_priors(proba, p_train)
    adj = adjust(proba, q, p_train)
    return {
        "argmax": proba.argmax(axis=1),
        "multipliers": (proba * w).argmax(axis=1),
        "em": adj.argmax(axis=1),
        "em+multipliers": (adj * w).argmax(axis=1),
    }


def main(task_md: Path) -> None:
    spec = parse_task(task_md)
    train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
    features = [c for c in test.columns if c != spec.id_column and c in train.columns]
    classes = list(spec.class_labels)
    k = len(classes)
    code = {c: i for i, c in enumerate(classes)}
    y_all = train[spec.target].astype(str).map(code).to_numpy()
    metric = spec.metric.name

    harness.SEED = SEED
    mask, how = harness.holdout_mask(spec, train, test, 0.25)
    part, hold = train[~mask].reset_index(drop=True), train[mask].reset_index(drop=True)
    y_part, y_hold = y_all[~mask], y_all[mask]
    p_train = np.bincount(y_part, minlength=k) / len(y_part)
    print(f"Разбиение {how}, train-часть {len(part)}, отложено {len(hold)}, метрика {metric}", flush=True)

    # OOF train-части — для множителей, как в агенте
    groups = pd.util.hash_pandas_object(part[features], index=False).to_numpy()
    oof = np.zeros((len(part), k))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(part, y_part, groups):
        oof[va] = fit_proba(part.iloc[tr], y_part[tr], part.iloc[va], features, k)
    w = fit_weights(metric, True, y_part, oof)
    print("множители", dict(zip(classes, w.round(2), strict=True)), flush=True)

    proba_hold = fit_proba(part, y_part, hold, features, k)
    rows = []

    def score(name: str, idx: np.ndarray) -> None:
        for method, pred in answers(proba_hold[idx], p_train, w).items():
            rows.append({"проверка": name, "способ": method, "score": compute_metric(metric, y_hold[idx], pred)})

    score("без сдвига", np.arange(len(hold)))
    rng = np.random.default_rng(SEED)
    by_class = [np.flatnonzero(y_hold == c) for c in range(k)]
    present = [c for c in range(k) if len(by_class[c])]

    def resample(q: np.ndarray) -> np.ndarray:
        counts = rng.multinomial(len(hold), q / q.sum())
        return np.concatenate([rng.choice(by_class[c], counts[c]) for c in present if counts[c]])

    for _ in range(N_SHIFTS):
        q = np.zeros(k)
        q[present] = rng.dirichlet(np.ones(len(present)))
        score("случайные доли (Dirichlet)", resample(q))

    # настоящий test: модель на всём train, EM-доли, строки с точной копией в train
    proba_test = fit_proba(train, y_all, test, features, k)
    p_all = np.bincount(y_all, minlength=k) / len(y_all)
    q_test = em_priors(proba_test, p_all)
    print("доли классов: train / EM-оценка test", flush=True)
    print(pd.DataFrame({"train": p_all, "test_EM": q_test}, index=classes).round(3).to_string(), flush=True)
    q_like = np.where(np.isin(np.arange(k), present), q_test, 0)
    for _ in range(N_SHIFTS):
        score("доли как EM-оценка test", resample(q_like))

    g_tr = pd.util.hash_pandas_object(train[features], index=False)
    g_te = pd.util.hash_pandas_object(test[features], index=False)
    label = pd.Series(y_all).groupby(g_tr.to_numpy()).agg(lambda s: s.value_counts().index[0])
    known = g_te.isin(label.index).to_numpy()
    y_known = g_te[known].map(label).to_numpy()
    for method, pred in answers(proba_test, p_all, w).items():
        rows.append(
            {
                "проверка": f"test, строки с копией ({known.sum()})",
                "способ": method,
                "score": compute_metric(metric, y_known, pred[known]),
            }
        )
        if method in ("argmax", "em"):
            share = np.bincount(pred, minlength=k) / len(pred)
            print(f"доли ответа на test, {method}:", dict(zip(classes, share.round(3), strict=True)), flush=True)

    table = pd.DataFrame(rows).groupby(["проверка", "способ"], sort=False).score.agg(["mean", "std", "count"])
    print(table.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
