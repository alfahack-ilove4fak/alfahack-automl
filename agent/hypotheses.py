"""Откуда берутся гипотезы. Владелец — LLM/agent-инженер (playbook — Research/DS).

Сейчас (скелет): только research/playbook.yaml, по порядку, без повторов.
Дальше (задачи L1–L3 agent-инженера): LLM смотрит на DataProfile и историю экспериментов
и предлагает новые гипотезы в формате Hypothesis; playbook остаётся запасным источником.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from interfaces.schemas import DataProfile, ExperimentResult, Hypothesis, HypothesisKind, TaskSpec, TaskType

PLAYBOOK_PATH = Path(__file__).resolve().parent.parent / "research" / "playbook.yaml"


def baseline_hypothesis() -> Hypothesis:
    return Hypothesis(id="baseline", kind=HypothesisKind.BASELINE, title="LightGBM с параметрами по умолчанию",
                      params={"model": "lightgbm"}, source="baseline")


def load_playbook(path: Path = PLAYBOOK_PATH) -> list[dict]:
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text(encoding="utf-8")).get("hypotheses", [])


def _applies(item: dict, spec: TaskSpec, profile: DataProfile) -> bool:
    task_types = item.get("task_types")
    if task_types and spec.task_type.value not in task_types:
        return False
    if item.get("requires") == "categorical" and not profile.categorical_features:
        return False
    if item.get("requires") == "imbalance" and spec.task_type != TaskType.REGRESSION:
        return profile.target_summary.get("imbalance_ratio", 1) > 3
    return item.get("requires") != "imbalance"


def propose_hypotheses(
    spec: TaskSpec, profile: DataProfile, history: list[tuple[Hypothesis, ExperimentResult]], k: int = 3
) -> list[Hypothesis]:
    tried = {h.id for h, _ in history}
    out = []
    for item in load_playbook():
        if item["id"] in tried or not _applies(item, spec, profile):
            continue
        out.append(Hypothesis(id=item["id"], kind=HypothesisKind(item["kind"]), title=item["title"],
                              rationale=item.get("rationale", ""), params=item.get("params", {}), source="playbook"))
        if len(out) == k:
            break
    return out
