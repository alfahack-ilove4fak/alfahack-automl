"""Монитор прогона внешних задач (research/external/watch.py): состояние и текст по файлам прогона."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("watch", ROOT / "research" / "external" / "watch.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def make_run(root: Path) -> None:
    base = root / "runs" / "night"
    answers = base / "external" / "_answers"
    answers.mkdir(parents=True)
    for name, metric in (("alpha", "rmse"), ("beta", "roc_auc"), ("gamma", "accuracy")):
        meta = {"task_type": "regression", "metric": metric, "source": f"https://example.org/{name}"}
        (answers / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    rows = [
        {"task": "alpha", "exit_code": 0, "problems": [], "value": 1.0, "score_vs_strong": 12.5, "seconds": 300,
         "ref": {"lgbm_default": 1.2}, "spec": {"metric": "rmse"}},
        {"task": "beta", "exit_code": 0, "problems": [], "value": 0.7, "score_vs_strong": 0.0, "seconds": 310,
         "ref": {"lgbm_default": 0.7}, "spec": {"metric": "logloss"}},
    ]  # fmt: skip
    (base / "external_results.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    run = base / "external_runs" / "gamma"
    run.mkdir(parents=True)
    (run / "agent.log").write_text(
        "[ 1.0s] INFO  agent.loop: Итерация 1: x\n[ 2.0s] INFO  agent.loop: ✓ лучше: 1 → 0\n", encoding="utf-8"
    )


def test_state_counts_done_zero_running_and_parse(tmp_path):
    make_run(tmp_path)
    s = watch.state(tmp_path)
    by = {t["name"]: t for t in s["tasks"]}
    assert (s["total"], s["done"], s["zeros"], s["unparsed"]) == (3, 2, 1, 1)
    assert by["alpha"]["status"] == "ok" and by["beta"]["status"] == "zero"
    assert by["beta"]["note"] == "не лучше LightGBM по умолчанию" and by["beta"]["parsed"] is False
    assert by["gamma"]["status"] == "running" and by["gamma"]["accepted"] == 1 and by["gamma"]["iterations"] == 1
    assert s["eta_seconds"] and not s["finished"]


def test_text_render_shows_every_task_and_warning(tmp_path):
    make_run(tmp_path)
    text = watch.render_text(watch.state(tmp_path), color=False)
    assert "2/3" in text and "alpha" in text and "gamma" in text and "метрика понята как logloss" in text


def test_empty_root_does_not_crash(tmp_path):
    (tmp_path / "runs" / "night" / "external" / "_answers").mkdir(parents=True)
    s = watch.state(tmp_path)
    assert s["total"] == 0 and watch.render_text(s, color=False)
