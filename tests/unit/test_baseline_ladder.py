"""Лестница отступления бейзлайна (#171, #151): упавшая CV не оставляет в ответе тривиальное среднее."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

import solution
from interfaces.schemas import ExperimentResult, ExperimentStatus
from runtime.budget import make_budget, plan_stages


@pytest.fixture
def task_md(tmp_path, monkeypatch) -> Path:
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    data = load_breast_cancer(as_frame=True).frame.rename(columns=lambda c: c.replace(" ", "_"))
    data.insert(0, "row_id", range(len(data)))
    train, test = data.iloc[:450], data.iloc[450:].drop(columns="target")
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    test[["row_id"]].assign(target=0.5).to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task.md").write_text("# Диагноз\n\nПредскажите `target`. Метрика — ROC-AUC.\n", encoding="utf-8")
    return tmp_path / "task.md"


def run(task_md, monkeypatch, ok_ids: set[str], validation: str | None = None, fail=ExperimentStatus.FAILED):
    """run_pipeline с подменённой CV: успешны только гипотезы из ok_ids. → (запущенные, гипотеза цикла или None)."""
    tried, loop_base = [], []

    def fake_experiment(spec, profile, hyp, train, budget):
        key = f"{hyp.id}/{hyp.params.get('validation', 'plain')}"
        tried.append(key)
        if key not in ok_ids:
            return ExperimentResult(hypothesis_id=hyp.id, status=fail, metric_name="roc_auc", error="не успела")
        return ExperimentResult(
            hypothesis_id=hyp.id, status=ExperimentStatus.OK, metric_name="roc_auc", mean_score=0.98, std_score=0.01
        )

    def fake_loop(spec, profile, train, budget, history, best, on_new_best, **kwargs):
        loop_base.append(best[0])
        return best

    monkeypatch.setattr(solution, "run_experiment", fake_experiment)
    monkeypatch.setattr(solution, "run_autoresearch", fake_loop)
    monkeypatch.setattr(solution, "choose_validation", lambda *a: validation)
    budget = make_budget()
    keeper = solution.AnswerKeeper(task_md)
    solution.run_pipeline(task_md, budget, plan_stages(budget), keeper, budget.started_at)
    return tried, (loop_base[0] if loop_base else None), pd.read_csv(task_md.parent / "submission.csv")


def is_model_answer(sub: pd.DataFrame) -> bool:
    return sub["target"].nunique() > 2  # тривиальный ответ — одна константа на все строки


def test_chosen_scheme_ok_goes_to_loop(task_md, monkeypatch):
    tried, base, sub = run(task_md, monkeypatch, {"baseline/time:mean_radius"}, validation="time:mean_radius")
    assert tried == ["baseline/time:mean_radius"]
    assert base.params["validation"] == "time:mean_radius"  # цикл получает схему через бейзлайн (#175)
    assert is_model_answer(sub)


def test_scheme_failed_falls_back_to_plain_cv(task_md, monkeypatch):
    tried, base, sub = run(task_md, monkeypatch, {"baseline/plain"}, validation="time:mean_radius")
    assert tried == ["baseline/time:mean_radius", "baseline/plain"]
    assert base is not None and "validation" not in base.params
    assert is_model_answer(sub)


def test_plain_scheme_not_tried_twice(task_md, monkeypatch):
    tried, base, _ = run(task_md, monkeypatch, {"baseline/plain"}, validation="stratified")
    assert tried == ["baseline/plain"] and base is not None


def test_only_light_baseline_saved_without_loop(task_md, monkeypatch):
    tried, base, sub = run(task_md, monkeypatch, {f"{solution.LIGHT_ID}/plain"})
    assert tried == ["baseline/plain", f"{solution.LIGHT_ID}/plain"]
    assert base is None  # облегчённая CV несравнима с гипотезами — цикл пропущен
    assert is_model_answer(sub)


def test_all_cv_failed_still_model_answer(task_md, monkeypatch):
    # ступени 1–3 упали: раньше здесь оставался тривиальный ответ = 0 баллов (#171)
    tried, base, sub = run(task_md, monkeypatch, set(), validation="group:duplicates")
    assert tried == ["baseline/group:duplicates", "baseline/plain", f"{solution.LIGHT_ID}/plain"]
    assert base is None
    assert is_model_answer(sub)
    assert sub["target"].between(0, 1).all() and len(sub) == 119


def test_timeout_goes_straight_to_light(task_md, monkeypatch):
    # ревью #176: после нехватки времени обычная CV той же модели не успеет и снова — сразу облегчённая
    tried, _, sub = run(
        task_md,
        monkeypatch,
        {f"{solution.LIGHT_ID}/plain"},
        validation="time:mean_radius",
        fail=ExperimentStatus.TIMEOUT,
    )
    assert tried == ["baseline/time:mean_radius", f"{solution.LIGHT_ID}/plain"]
    assert is_model_answer(sub)


@pytest.mark.parametrize(
    "status, error, expected",
    [
        (ExperimentStatus.TIMEOUT, "кончилось время на фолде 1", True),
        (ExperimentStatus.FAILED, "MemoryError: Unable to allocate 3.2 GiB", True),
        (ExperimentStatus.FAILED, "ValueError: в фолде один класс", False),  # схема — пробуем обычную CV
    ],
)
def test_out_of_resources(status, error, expected):
    res = ExperimentResult(hypothesis_id="b", status=status, metric_name="m", error=error)
    assert solution._out_of_resources(res) is expected
