"""Ночной аудит: только разбор задач (TaskSpec) против «паспорта» meta.json, без обучения.

    LLM_MOCK=1 uv run python research/night/parse_probe.py runs/night/tasks
"""
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.task_parser import parse_task  # noqa: E402

root = Path(sys.argv[1])
for task in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
    meta = json.loads((root / "_answers" / task.name / "meta.json").read_text())
    try:
        s = parse_task(task / "task.md")
        got = dict(target=[s.target, *s.extra_targets], id=s.id_column, metric=s.metric.name,
                   greater=s.metric.greater_is_better, sub=s.submission_columns, type=s.task_type.value)
        flags = []
        if set(got["target"]) != set(meta["targets"]):
            flags.append("TARGET")
        if got["id"] != meta["id"]:
            flags.append("ID")
        if got["metric"] != meta["metric"]:
            flags.append("METRIC")
        if got["greater"] != meta["greater"]:
            flags.append("DIRECTION")
        if got["sub"] != meta["submission"]:
            flags.append("SUBCOLS")
        print(f"{task.name}: {'OK' if not flags else ' '.join(flags)} | {got}")
    except Exception as exc:
        print(f"{task.name}: CRASH {type(exc).__name__}: {str(exc)[:200]}")
