"""Харнесс и сдвиг test (#276): AUC «train против test», test-подобная отложенная часть, набор бейзлайнов."""

import numpy as np
import pandas as pd

from agent.task_parser import parse_task
from evaluation.harness import make_holdout_task, render, shift_features, shift_probability
from evaluation.local_score import score_submission


def _task(dst, shift=0.0, n=600, kind="binary", ids=True):
    """Задача с признаком x: в test он сдвинут на shift; колонки-идентификатор, константа и дробная «площадь»."""
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(0, 1, n), 1)  # округление — есть полные дубликаты строк
    x_test = np.round(rng.normal(shift, 1, n // 2), 1)
    target = {
        "binary": lambda v: np.where(v > 0.5, "yes", "no"),
        "multiclass": lambda v: np.select([v < -0.5, v < 0.5], ["a", "b"], "c"),
        "regression": lambda v: 3 * v + rng.normal(0, 0.1, len(v)),
    }[kind]
    train = pd.DataFrame({"row_id": range(n), "rec": range(n), "const": 1, "x": x, "label": target(x)})
    test = pd.DataFrame({"row_id": range(n, n + len(x_test)), "rec": range(n, n + len(x_test)), "const": 1})
    test["x"] = x_test
    if not ids:  # без колонки-номера записи: одинаковые x — полные дубликаты строк
        train, test = train.drop(columns="rec"), test.drop(columns="rec")
    dst.mkdir(parents=True)
    train.to_csv(dst / "train.csv", index=False)
    test.to_csv(dst / "test.csv", index=False)
    filler = 0.0 if kind == "regression" else "no"
    test[["row_id"]].assign(label=filler).to_csv(dst / "sample_submission.csv", index=False)
    metric = {"binary": "accuracy", "multiclass": "macro F1", "regression": "RMSE"}[kind]
    (dst / "task.md").write_text(f"Предсказать `label`. Метрика — {metric}.\n", encoding="utf-8")
    return dst / "task.md"


def test_shift_features_skip_ids_and_constants(tmp_path):
    spec = parse_task(_task(tmp_path / "src"))
    train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
    train["area"], test["area"] = np.arange(len(train)) + 0.5, np.arange(len(test)) + 0.25  # дробные, все разные
    features = shift_features(spec, train, test)
    assert "rec" not in features and "const" not in features and "row_id" not in features
    assert "x" in features and "area" in features


def test_shift_auc_high_with_shift_and_low_without(tmp_path):
    for shift, check in ((2.0, lambda auc: auc > 0.85), (0.0, lambda auc: auc < 0.7)):
        spec = parse_task(_task(tmp_path / f"s{shift}", shift=shift))
        train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
        auc, p_test = shift_probability(spec, train, test)
        assert check(auc), (shift, auc)
        assert len(p_test) == len(train) and np.all((0 <= p_test) & (p_test <= 1))


def test_test_like_holdout_takes_rows_closest_to_test(tmp_path):
    prepared = make_holdout_task(_task(tmp_path / "src", shift=2.0, ids=False), tmp_path / "out", holdout="test_like")
    assert prepared["split"].startswith("test_like(auc=") and prepared["shift_auc"] > 0.85
    part = pd.read_csv(tmp_path / "out" / "task" / "train.csv")
    hold = pd.read_csv(tmp_path / "out" / "task" / "test.csv")
    assert hold.x.mean() > part.x.mean() + 1  # отложены строки, похожие на test (x сдвинут вправо)
    assert not set(hold.x) & set(part.x)  # группы копий целиком по одну сторону
    assert "label" not in hold.columns and 0.1 < len(hold) / (len(hold) + len(part)) < 0.4


def test_test_like_falls_back_to_standard_without_shift(tmp_path):
    prepared = make_holdout_task(_task(tmp_path / "src"), tmp_path / "out", holdout="test_like")
    assert not prepared["split"].startswith("test_like") and "сдвига нет" in prepared["split"]
    standard = make_holdout_task(_task(tmp_path / "src2"), tmp_path / "out2")
    assert standard["split"] in ("stratified_groups", "groups") and np.isfinite(standard["shift_auc"])


def test_extra_baselines_give_strongest_baseline(tmp_path):
    for kind in ("multiclass", "regression"):
        spec = parse_task(_task(tmp_path / kind, kind=kind))
        train = pd.read_csv(spec.train_path)
        part, hold = train.iloc[:450], train.iloc[450:]
        x_part, x_hold = part[["x", "const"]], hold[["x", "const"]]
        result = score_submission(spec, hold.label, hold.label, part.label, x_train=x_part, x_test=x_hold)
        assert np.isfinite(result["rf_baseline"]) and np.isfinite(result["linear_baseline"])
        assert result["best_name"] in ("trivial", "lightgbm", "random_forest", "linear")
        assert result["best_score"] <= result["model_score"] + 1e-9  # сильнейший не слабее LightGBM и тривиального
        light = score_submission(
            spec, hold.label, hold.label, part.label, x_train=x_part, x_test=x_hold, extra_baselines=False
        )
        assert "rf_baseline" not in light and light["model_score"] == result["model_score"]


def test_render_shows_shift_sensitivity_and_old_rows():
    base = {"task": "tasks/public/multiclass", "metric": "f1_macro", "score": 50.0, "status": "OK"}
    rows = [
        {
            **base,
            "split": "stratified_groups",
            "shift_auc": 0.88,
            "value": 0.64,
            "model_score": 27.0,
            "best_score": 17.0,
            "best_baseline": 0.57,
            "best_name": "random_forest",
        },
        {
            **base,
            "split": "test_like(auc=0.880)",
            "shift_auc": 0.88,
            "value": 0.58,
            "model_score": 25.0,
            "best_score": 11.5,
            "best_baseline": 0.53,
            "best_name": "random_forest",
        },
        {"task": "old", "score": 10.0, "status": "OK"},  # строка старого лидерборда без новых полей
    ]
    text = render(rows)
    assert "Чувствительность к сдвигу, tasks/public/multiclass" in text and "17.0 → 11.5" in text
    assert "0.57 (random_forest)" in text and "| old |" in text
