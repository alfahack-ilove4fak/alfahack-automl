"""Контракты: схемы валидны, заглушки совместимы, код импортирует только разрешённые библиотеки."""

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from interfaces import stubs
from interfaces.schemas import CONTRACT_VERSION, ExperimentResult, ExperimentStatus, RunBudget, TaskSpec

ROOT = Path(__file__).resolve().parents[2]
# Код, который поедет на платформу в zip-архиве
SHIPPED = ["solution.py", "interfaces", "runtime", "agent", "ml", "evaluation"]
LOCAL_PACKAGES = {"interfaces", "runtime", "agent", "ml", "evaluation", "research"}
# имя при импорте → имя пакета в requirements.txt
IMPORT_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "category_encoders": "category-encoders",
    "feature_engine": "feature-engine",
    "imblearn": "imbalanced-learn",
    "autogluon": "autogluon.tabular",
    "langchain_openai": "langchain-openai",
}


def allowed_packages() -> set[str]:
    names = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(line.split("==")[0].split("[")[0].strip().lower())
    return names


def test_contract_version_is_semver():
    assert len(CONTRACT_VERSION.split(".")) == 3


def test_schemas_roundtrip_json():
    spec = stubs.stub_task_spec()
    assert TaskSpec.model_validate_json(spec.model_dump_json()) == spec


def test_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ExperimentResult(hypothesis_id="x", status="ok", metric_name="rmse", typo_field=1)


def test_beats_respects_direction():
    a = ExperimentResult(hypothesis_id="a", status=ExperimentStatus.OK, metric_name="rmse", mean_score=1.0)
    b = ExperimentResult(hypothesis_id="b", status=ExperimentStatus.OK, metric_name="rmse", mean_score=2.0)
    assert a.beats(b, greater_is_better=False)
    assert b.beats(a, greater_is_better=True)
    assert not a.beats(b, greater_is_better=False, min_delta=5.0)


def test_stubs_chain_works():
    spec = stubs.stub_task_spec()
    train = pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "site_eui": [1.0, 2.0]})
    test = pd.DataFrame({"id": [3], "x": [0.3]})
    profile = stubs.stub_profile_data(spec, train, test)
    hyps = stubs.stub_propose_hypotheses(spec, profile, [], k=2)
    result = stubs.stub_run_experiment(spec, profile, hyps[0], train, RunBudget())
    answer = stubs.stub_fit_predict_final(spec, profile, hyps[0], train, test)
    assert result.status == ExperimentStatus.OK and list(answer.columns) == ["id", "site_eui"]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_only_allowed_imports():
    """На платформе нет интернета: всё, чего нет в requirements.txt, упадёт с ImportError."""
    allowed = allowed_packages()
    bad = []
    for item in SHIPPED:
        for py in [ROOT / item] if item.endswith(".py") else (ROOT / item).rglob("*.py"):
            for name in _imports(py):
                if name in sys.stdlib_module_names or name in LOCAL_PACKAGES or name == "__future__":
                    continue
                if IMPORT_TO_PACKAGE.get(name, name).lower() not in allowed:
                    bad.append(f"{py.relative_to(ROOT)}: import {name}")
    assert not bad, "Импорты вне requirements.txt:\n" + "\n".join(bad)


def test_beats_refuses_to_compare_different_validation_schemes():
    base = ExperimentResult(hypothesis_id="a", status=ExperimentStatus.OK, metric_name="rmse", mean_score=2.0)
    honest = ExperimentResult(
        hypothesis_id="b", status=ExperimentStatus.OK, metric_name="rmse", mean_score=1.0, validation="group-5"
    )
    assert not honest.beats(base, greater_is_better=False)  # другая схема CV: несравнимо
    same = honest.model_copy(update={"validation": "kfold-5"})
    assert same.beats(base, greater_is_better=False)


def test_refit_params_and_code_result_roundtrip():
    from interfaces.schemas import CodeResult

    result = ExperimentResult(
        hypothesis_id="a", status="ok", metric_name="rmse", refit_params={"n_estimators": 321}, validation="kfold-5"
    )
    assert ExperimentResult.model_validate_json(result.model_dump_json()) == result
    code = CodeResult(exit_code=1, stderr="Traceback", timed_out=False, seconds=0.5)
    assert CodeResult.model_validate_json(code.model_dump_json()) == code


def test_stubs_accept_budget_and_run_code():
    spec = stubs.stub_task_spec()
    hyps = stubs.stub_propose_hypotheses(spec, None, [], k=1, budget=RunBudget())
    assert len(hyps) == 1
    assert stubs.stub_run_code("print(1)", Path("."), timeout=1.0, mem_mb=64).exit_code == 0


def test_extra_targets_optional_and_roundtrip():
    # 1.1.1 (#174): поле необязательное — старые TaskSpec без него остаются валидными
    spec = stubs.stub_task_spec(Path("tasks/demo"))
    assert spec.extra_targets == []
    multi = spec.model_copy(update={"extra_targets": ["y2", "y3"]})
    assert TaskSpec.model_validate_json(multi.model_dump_json()).extra_targets == ["y2", "y3"]
