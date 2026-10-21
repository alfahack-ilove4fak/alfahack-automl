"""Сквозной тест (#117): задача, где ответ — «колонка на класс» (`id, a, b, c` с вероятностями).

Отдельный файл, а не случай в test_robustness.py: там же сейчас правит #108, так меньше конфликтов.
Тест валидного ответа больше не xfail (формат поддержан, #111); модельный ответ — пока xfail(strict=True). Чинят несколько модулей (#110, #117):
agent/task_parser.py (#111), solution.py::trivial_answer, ml/experiment.py::fit_predict_final.
Когда всё починят, тесты станут XPASS и покраснеют — тогда снимите отметки ниже.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_robustness import base_task, run_agent, write_task  # noqa: E402

CLASSES = ["a", "b", "c"]
XFAIL_NO_MODEL = pytest.mark.xfail(
    strict=True, reason="#117: fit_predict_final пишет одну колонку target, а не вероятности по классам"
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("per_class")
    task = base_task("multiclass")
    task.sample = False
    task.text = "# Задача\n\nПредсказать `target`. Метрика — log loss: в ответе вероятность каждого класса.\n"
    task_md = write_task(task, root / "task")
    sample = task.test[["id"]].assign(**{c: 1 / len(CLASSES) for c in CLASSES})
    sample.to_csv(task_md.parent / "sample_submission.csv", index=False)
    return task, task_md, run_agent(task_md, root)


def test_valid_per_class_answer(run):
    task, task_md, proc = run
    log = proc.stderr[-3000:]
    path = task_md.parent / "submission.csv"
    assert path.exists(), f"нет submission.csv\n{log}"
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["id", *CLASSES], f"колонки {list(sub.columns)}\n{log}"
    assert sub["id"].tolist() == task.test["id"].tolist(), "id не как в test"
    probs = sub[CLASSES].to_numpy(dtype=float)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all(), "вероятности вне [0, 1]"
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-3), "сумма по строке ≠ 1"
    assert proc.returncode == 0, log


@XFAIL_NO_MODEL
def test_per_class_answer_from_model(run):
    _, _, proc = run
    assert "Сохранён ответ «" in proc.stderr, "сдан только тривиальный ответ\n" + proc.stderr[-3000:]
