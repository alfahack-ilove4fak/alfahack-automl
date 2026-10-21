"""Код признаков от LLM (A-L3-1): проверка до песочницы, промпт и разбор ответа."""

from pathlib import Path

import pytest

from agent.feature_code import build_feature_messages, check_feature_code, parse_feature_hypothesis
from interfaces.schemas import DataProfile, HypothesisKind
from interfaces.stubs import stub_task_spec

GOOD = """
import numpy as np
import pandas as pd

def add_features(df):
    df = df.copy()
    df["area_per_floor"] = df["area"] / df["floors"].replace(0, np.nan)
    df["log_area"] = np.log1p(df["area"].clip(lower=0))
    return df
"""


@pytest.fixture
def spec():
    return stub_task_spec(Path("tasks/demo"))


@pytest.fixture
def profile():
    return DataProfile(n_train=100, n_test=50, numeric_features=["area", "floors"], categorical_features=["city"])


def test_good_code_passes():
    assert check_feature_code(GOOD) is None


@pytest.mark.parametrize(
    "code, fragment",
    [
        ("", "пустой"),
        ("def add_features(df)\n    return df", "синтаксическая"),
        ("def make(df):\n    return df", "нет функции"),
        ("def add_features(df, y):\n    return df", "ровно один"),
        ("import os\ndef add_features(df):\n    return df", "импорт: os"),
        ("from subprocess import run\ndef add_features(df):\n    return df", "импорт: subprocess"),
        ("def add_features(df):\n    open('/etc/passwd')\n    return df", "open"),
        ("def add_features(df):\n    eval('1')\n    return df", "eval"),
        ("def add_features(df):\n    df.to_csv('x.csv')\n    return df", "файлами"),
        ("import pandas as pd\ndef add_features(df):\n    pd.read_csv('t.csv')\n    return df", "файлами"),
        ("def add_features(df):\n    return ().__class__.__bases__", "служебному"),
        ("def add_features(df):\n    getattr(df, 'x')\n    return df", "getattr"),
        ("def add_features(df):\n    df['t2'] = df['site_eui']\n    return df", "target"),
        ("def add_features(df):\n    return df\n" + "#" * 6000, "длиннее"),
    ],
)
def test_bad_code_rejected(code, fragment):
    reason = check_feature_code(code, target="site_eui")
    assert reason is not None and fragment in reason, reason


def test_prompt_renders(spec, profile):
    messages = build_feature_messages(spec, profile, tried=["отношения площадей"])
    user = messages[1]["content"]
    assert "$" not in user
    assert "area (число)" in user and "city (категория)" in user
    assert "- отношения площадей" in user
    assert "add_features(df)" in user and "pandas" in user


def test_parse_good_answer(spec):
    hyp = parse_feature_hypothesis({"id": "area ratio!", "title": "Отношения", "code": GOOD, "extra": 1}, spec, [])
    assert hyp.kind == HypothesisKind.FEATURES and hyp.source == "llm"
    assert hyp.id == "area_ratio" and hyp.code == GOOD


@pytest.mark.parametrize(
    "raw",
    [None, "текст", [], {"id": "x", "title": "t"}, {"id": "x", "title": "t", "code": "import os"}, {"code": GOOD}],
)
def test_parse_bad_answer(spec, raw):
    assert parse_feature_hypothesis(raw, spec, []) is None
