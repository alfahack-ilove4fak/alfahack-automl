"""Харнесс: «скрытый тест» честный (агент не видит ответы, дубликаты не разрезаны) и прогон даёт балл."""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.harness import main, make_holdout_task, time_column

ROOT = Path(__file__).resolve().parents[2]


def _task(dst, n=400, period=False):
    """Бинарная задача с файлами во вложенной папке, полными дубликатами строк и (опционально) годом."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, n)
    train = pd.DataFrame({"row_id": range(n), "x": x, "z": x * 2, "label": np.where(x > 2, "yes", "no")})
    test = pd.DataFrame({"row_id": range(n, n + 50), "x": rng.integers(0, 5, 50)}).assign(z=lambda d: d.x * 2)
    if period:
        train["year"], test["year"] = np.repeat(np.arange(2010, 2018), n // 8), 2020
    files = dst / "data"
    files.mkdir(parents=True)
    train.to_csv(files / "train.csv", index=False)
    test.to_csv(files / "test.csv", index=False)
    test[["row_id"]].assign(label="no").to_csv(files / "sample_submission.csv", index=False)
    (dst / "task.md").write_text("Предсказать `label`. Метрика — accuracy.\n", encoding="utf-8")
    return dst / "task.md"


def test_holdout_is_hidden_and_duplicates_stay_together(tmp_path):
    prepared = make_holdout_task(_task(tmp_path / "src"), tmp_path / "out")
    task_dir = tmp_path / "out" / "task"
    train = pd.read_csv(task_dir / "data" / "train.csv")
    test = pd.read_csv(task_dir / "data" / "test.csv")
    answers = pd.read_csv(tmp_path / "out" / "answers.csv")

    assert "label" not in test.columns and list(test.columns) == ["row_id", "x", "z"]
    assert not (task_dir / "answers.csv").exists()  # ответы вне папки задачи
    assert set(test.row_id) == set(answers.row_id) and not set(test.row_id) & set(train.row_id)
    assert 0.1 < len(test) / (len(test) + len(train)) < 0.4
    # x определяет строку признаков целиком — одна и та же строка не бывает по обе стороны
    assert not set(test.x) & set(train.x)
    sample = pd.read_csv(task_dir / "data" / "sample_submission.csv")
    assert list(sample.columns) == ["row_id", "label"] and len(sample) == len(test)
    assert prepared["split"] == "stratified_groups" or prepared["split"] == "groups"


def test_time_column_gives_last_periods(tmp_path):
    prepared = make_holdout_task(_task(tmp_path / "src", period=True), tmp_path / "out")
    train = pd.read_csv(tmp_path / "out" / "task" / "data" / "train.csv")
    test = pd.read_csv(tmp_path / "out" / "task" / "data" / "test.csv")
    assert prepared["split"].startswith("time:year")
    assert test.year.min() > train.year.max()


def test_date_strings_give_last_dates(tmp_path):
    """Дата строкой, test — следующий месяц (как payments и sales): отложенная часть — последние даты, а не случайная."""
    src = tmp_path / "src"
    _task(src)
    data = src / "data"
    train, test = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    train["date"] = pd.date_range("2024-01-01", periods=len(train), freq="h").strftime("%Y-%m-%d")
    test["date"] = "2024-03-01"
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    prepared = make_holdout_task(src / "task.md", tmp_path / "out")
    part = pd.read_csv(tmp_path / "out" / "task" / "data" / "train.csv")
    hold = pd.read_csv(tmp_path / "out" / "task" / "data" / "test.csv")
    assert prepared["split"].startswith("time:date>=2024-")
    assert pd.to_datetime(hold.date).min() > pd.to_datetime(part.date).max()
    assert 0.1 < len(hold) / (len(hold) + len(part)) < 0.4


def test_time_column_rejects_overlap_text_and_row_numbers():
    dates = pd.Series(pd.date_range("2024-01-01", periods=60, freq="D").strftime("%Y-%m-%d"))
    train = pd.DataFrame({"date": dates, "city": ["moscow", "kazan"] * 30, "record_no": range(60)})
    test = pd.DataFrame({"date": dates[:5], "city": ["moscow"] * 5, "record_no": range(60, 65)})
    assert time_column(train, test, ["date", "city", "record_no"]) is None  # даты пересекаются, номер записи — не время


def test_harness_run_on_public_task(tmp_path):
    out = tmp_path / "leaderboard.csv"
    task = ROOT / "tasks" / "public" / "multiclass" / "task.md"
    code = main(["--task", str(task), "--minutes", "0.5", "--rows", "2000", "--out", str(out)])
    board = pd.read_csv(out)
    assert code == 0, board.to_dict("records")
    row = board.iloc[0]
    assert row.status == "OK" and row.metric == "f1_macro"
    assert 0 <= row.score <= 100 and row.value > row.baseline


def test_agent_root_runs_other_agent(tmp_path):
    """--agent-root: харнесс запускает solution.py из другой папки (так считается «было» в PR-отчёте)."""
    agent = tmp_path / "other_agent"
    agent.mkdir()
    (agent / "solution.py").write_text(
        "import shutil, sys\nfrom pathlib import Path\n"
        "task = Path(sys.argv[1]).parent\n"
        "shutil.copy(task / 'sample_submission.csv', task / 'submission.csv')\n",
        encoding="utf-8",
    )
    out = tmp_path / "leaderboard.csv"
    task = ROOT / "tasks" / "public" / "multiclass" / "task.md"
    main(["--task", str(task), "--minutes", "0.5", "--rows", "1500", "--agent-root", str(agent), "--out", str(out)])
    row = pd.read_csv(out).iloc[0]
    # Ответ-копия шаблона валиден, но это константа — балл 0, а не балл настоящего агента
    assert row.status == "OK" and row.score == 0
    assert row.seconds < 30
