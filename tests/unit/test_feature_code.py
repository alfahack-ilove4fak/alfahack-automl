"""Код признаков от LLM (A-L3-1): проверка до песочницы, промпт и разбор ответа."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.feature_code import (
    MAX_NEW_COLUMNS,
    apply_feature_code,
    build_feature_messages,
    check_feature_code,
    parse_feature_hypothesis,
)
from interfaces.schemas import DataProfile, Hypothesis, HypothesisKind
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
        # обходы предфильтра из ревью Данила (#101): доступ к ОС и файлам через атрибуты pandas/numpy
        ("import pandas as pd\ndef add_features(df):\n    pd.io.common.os.system('echo x')\n    return df", "атрибут"),
        ("import numpy as np\ndef add_features(df):\n    np.genfromtxt('/etc/hostname')\n    return df", "атрибут"),
        ("import numpy as np\ndef add_features(df):\n    np.fromregex('/etc/x', 'a', 'S1')\n    return df", "атрибут"),
        ("import numpy as np\ndef add_features(df):\n    np.memmap('/tmp/x')\n    return df", "атрибут"),
        ("import numpy as np\ndef add_features(df):\n    np.lib.npyio.DataSource()\n    return df", "атрибут"),
        ("import pandas as pd\ndef add_features(df):\n    pd.core.sys.exit()\n    return df", "атрибут"),
        ("def add_features(df):\n    df['a2'] = df.eval('area * 2')\n    return df", "атрибут"),
        ("def add_features(df):\n    df = df.query('area > 0')\n    return df", "атрибут"),
        # операции по порядку строк: на склеенных train+test — утечка соседних строк
        ("def add_features(df):\n    df['prev'] = df['area'].shift(1)\n    return df", "порядка строк"),
        ("def add_features(df):\n    df['d'] = df['area'].diff()\n    return df", "порядка строк"),
        ("def add_features(df):\n    df['r'] = df['area'].rolling(3).mean()\n    return df", "порядка строк"),
        ("def add_features(df):\n    df['e'] = df['area'].expanding().mean()\n    return df", "порядка строк"),
        ("def add_features(df):\n    df['w'] = df['area'].ewm(span=3).mean()\n    return df", "порядка строк"),
        ("def add_features(df):\n    df['c'] = df['area'].cumsum()\n    return df", "порядка строк"),
        ("def add_features(df):\n    df['c'] = df.groupby('city').cumcount()\n    return df", "порядка строк"),
        ("def add_features(df):\n    df['p'] = df['area'].pct_change()\n    return df", "порядка строк"),
        ("def add_features(df):\n    df['pos'] = df.index / len(df)\n    return df", "порядка строк"),
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


# ---------------------------------------------------------------- исполнение в песочнице (настоящий run_code)


@pytest.fixture
def table():
    # train + test вместе, без target; индекс не с нуля — проверяем, что порядок и индекс сохраняются
    return pd.DataFrame(
        {"area": [100.0, 250.0, np.nan, 80.0], "floors": [1, 0, 3, 2], "city": ["a", "b", "a", "c"]},
        index=[10, 11, 12, 13],
    )


def apply(code, table, tmp_path, **kw):
    return apply_feature_code(code, table, target="site_eui", workdir=tmp_path, timeout=60, mem_mb=4096, **kw)


def test_apply_adds_new_numeric_columns(table, tmp_path):
    out, note = apply(GOOD, table, tmp_path)
    assert note == ""
    assert list(out.columns) == ["area", "floors", "city", "area_per_floor", "log_area"]
    assert out.index.equals(table.index)
    assert out.loc[10, "area_per_floor"] == 100.0
    assert np.isnan(out.loc[11, "area_per_floor"])  # деление на 0 этажей → NaN, а не inf
    pd.testing.assert_frame_equal(out[table.columns], table)


def test_apply_ignores_changes_to_existing_columns(table, tmp_path):
    code = "def add_features(df):\n    df['area'] = 0\n    df['twice'] = df['floors'] * 2\n    return df"
    out, _ = apply(code, table, tmp_path)
    pd.testing.assert_series_equal(out["area"], table["area"])  # исходные признаки — из нашего df
    assert out["twice"].tolist() == [2.0, 0.0, 6.0, 4.0]


def test_apply_inf_becomes_nan(table, tmp_path):
    code = "import numpy as np\ndef add_features(df):\n    df['r'] = df['area'] / df['floors']\n    return df"
    out, _ = apply(code, table, tmp_path)
    assert np.isnan(out.loc[11, "r"])  # 250 / 0 = inf → NaN


def test_apply_skips_text_and_constant_columns_with_note(table, tmp_path):
    code = (
        "def add_features(df):\n    df['txt'] = df['city'] + '!'\n    df['one'] = 1\n"
        "    df['f'] = df['floors'] + 1\n    return df"
    )
    out, note = apply(code, table, tmp_path)
    assert "f" in out.columns and "txt" not in out.columns and "one" not in out.columns
    assert "txt" in note and "one" in note


@pytest.mark.parametrize(
    "code, fragment",
    [
        ("import os\ndef add_features(df):\n    return df", "импорт"),  # до песочницы
        ("def add_features(df):\n    raise ValueError('boom')", "ValueError: boom"),
        ("def add_features(df):\n    return df.iloc[:2]", "изменил строки"),
        ("def add_features(df):\n    return df.iloc[::-1]", "изменил строки"),
        ("def add_features(df):\n    return df.drop(columns=['city'])", "удалил исходные"),
        ("def add_features(df):\n    return df", "ни одной числовой"),
        ("def add_features(df):\n    return 42", "не DataFrame"),
        ("def add_features(df):\n    df['c'] = 7\n    return df", "константные"),
        (
            f"def add_features(df):\n    for i in range({MAX_NEW_COLUMNS + 2}):\n"
            "        df[f'x{i}'] = df['floors'] * (i + 1)\n    return df",
            "больше",
        ),
    ],
)
def test_apply_rejects_bad_results(table, tmp_path, code, fragment):
    out, reason = apply(code, table, tmp_path)
    assert out is None and fragment in reason, reason


def test_apply_timeout(table, tmp_path):
    code = "def add_features(df):\n    while True:\n        pass"
    out, reason = apply_feature_code(code, table, workdir=tmp_path, timeout=3, mem_mb=4096)
    assert out is None and "не уложился" in reason


def test_apply_refuses_table_with_target(table, tmp_path):
    out, reason = apply(GOOD, table.assign(site_eui=1.0), tmp_path)
    assert out is None and "target" in reason


def test_apply_leaves_no_files(table, tmp_path):
    apply(GOOD, table, tmp_path)
    assert list(tmp_path.iterdir()) == []  # обмен через временную папку, она удаляется


def test_whole_table_statistics_still_allowed():
    # разрешено то, что не зависит от порядка: статистики по всей таблице и groupby-transform
    code = (
        "def add_features(df):\n"
        "    df = df.copy()\n"
        "    df['area_rank'] = df['area'].rank(pct=True)\n"
        "    df['city_mean_area'] = df.groupby('city')['area'].transform('mean')\n"
        "    return df"
    )
    assert check_feature_code(code, target="site_eui") is None


def test_feature_hypothesis_takes_params_of_best(spec):
    best = Hypothesis(
        id="xgb_tuned",
        kind=HypothesisKind.HYPERPARAMS,
        title="XGBoost",
        params={"model": "xgboost", "model_params": {"max_depth": 6}, "_best_iteration": 420, "_proba_weights": [1.0]},
    )
    hyp = parse_feature_hypothesis({"id": "ratios", "title": "Отношения", "code": GOOD}, spec, [], best=best)
    assert hyp.params == {"model": "xgboost", "model_params": {"max_depth": 6}}  # без служебных `_…`
    assert hyp.parent_id == "xgb_tuned"
    hyp.params["model_params"]["max_depth"] = 1
    assert best.params["model_params"]["max_depth"] == 6  # копия, лучший не меняется


def test_feature_hypothesis_without_best_is_default_lightgbm(spec):
    hyp = parse_feature_hypothesis({"id": "ratios", "title": "Отношения", "code": GOOD}, spec, [])
    assert hyp.params == {"model": "lightgbm"} and hyp.parent_id is None


def test_prompt_forbids_row_order_features(spec, profile):
    user = build_feature_messages(spec, profile, tried=[])[1]["content"]
    assert "порядку строк" in user and "groupby-transform" in user
