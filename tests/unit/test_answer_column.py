"""Колонка ответа в шаблоне называется не как таргет (#276): train `target`, шаблон `id, prediction`.

Раньше `answer[spec.submission_columns]` падал с KeyError, агент падал целиком и сдавал шаблон из нулей.
"""

import numpy as np
import pandas as pd

from agent.hypotheses import baseline_hypothesis
from agent.task_parser import parse_task
from evaluation.submission import validate_submission
from ml.experiment import fit_predict_final
from ml.profiling import load_data, profile_data
from solution import to_template, trivial_answer


def make_task(folder, sample_columns=("id", "prediction"), n=300, n_test=60):
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"id": np.arange(n + n_test), "a": rng.normal(size=n + n_test), "b": rng.normal(size=n + n_test)})
    train, test = x.iloc[:n].copy(), x.iloc[n:].reset_index(drop=True)
    train["target"] = train["a"] * 3 + train["b"]
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    pd.DataFrame({c: (test["id"] if c == "id" else np.zeros(n_test)) for c in sample_columns}).to_csv(
        folder / "sample_submission.csv", index=False
    )
    (folder / "task.md").write_text("Предскажите target. В ответе колонка prediction. Метрика — RMSE.", encoding="utf-8")
    return folder / "task.md"


def test_answer_column_other_than_target_is_renamed_in_every_answer(tmp_path):
    spec = parse_task(make_task(tmp_path), use_analyst=False)
    assert spec.target == "target" and spec.submission_columns == ["id", "prediction"]
    train, test = load_data(spec)
    path = tmp_path / "submission.csv"
    for answer in (
        trivial_answer(spec, train, test),
        to_template(
            fit_predict_final(spec, profile_data(spec, train, test), baseline_hypothesis(), train, test), spec
        ),
    ):
        assert list(answer.columns) == ["id", "prediction"] and len(answer) == len(test)
        answer.to_csv(path, index=False)
        assert validate_submission(path, spec, pd.read_csv(spec.test_path)).ok


def test_template_with_target_name_is_unchanged(tmp_path):
    spec = parse_task(make_task(tmp_path, sample_columns=("id", "target")), use_analyst=False)
    train, test = load_data(spec)
    answer = trivial_answer(spec, train, test)
    assert list(answer.columns) == ["id", "target"]
    assert to_template(answer, spec) is not None and list(to_template(answer, spec).columns) == ["id", "target"]
