"""LLM-аналитик задачи (agent/task_analyst.py): сверка ответа LLM с данными и вшивание в parse_task.

Настоящая LLM не вызывается: ответы подделываются через FakeLLM. Главное правило, которое проверяем:
данные главнее LLM — она меняет только то, что эвристика лишь угадала, и только если колонка реально существует.
"""

import numpy as np
import pandas as pd
import pytest

from agent import task_analyst
from agent.task_analyst import ANALYST_ENV, PREFIX, ask_llm, build_messages, metric_key, reconcile
from agent.task_parser import parse_task

TRAIN_COLS = ["id", "a", "b", "y", "y2"]
TEST_COLS = ["id", "a", "b"]


class FakeLLM:
    """Подмена LLMClient: chat_json отдаёт заранее заданный ответ (или бросает исключение)."""

    def __init__(self, answer, available=True):
        self.answer, self._available = answer, available
        self.requests = 0
        self.messages = None
        self.kwargs = None

    def available(self):
        return self._available

    def chat_json(self, messages, **kwargs):
        self.requests += 1
        self.messages, self.kwargs = messages, kwargs
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def heur(target="y", id_column="id", metric="rmse"):
    """Результат детерминированного разбора, который сверяет reconcile."""
    return {"target": target, "id_column": id_column, "metric": metric}


def run(view, heuristic=None, guessed=(), train=TRAIN_COLS, test=TEST_COLS):
    return reconcile(view, heuristic or heur(), set(guessed), train, test)


# ---------- metric_key ----------


@pytest.mark.parametrize(
    "value, expected",
    [("mae", "mae"), ("  MAE ", "mae"), ("Root Mean Squared Error", "rmse")],
)
def test_metric_key_known_names(value, expected):
    """Название метрики от LLM приводится к ключу реестра (регистр и пробелы не важны)."""
    assert metric_key(value) == expected


@pytest.mark.parametrize("value", [None, "", "   ", 5, ["mae"], "ndcg@10"])
def test_metric_key_unknown_or_garbage(value):
    """Пустое, не строка и метрика не из реестра дают None, а не исключение."""
    assert metric_key(value) is None


# ---------- build_messages ----------


def test_build_messages_has_no_leftover_placeholders():
    """В собранном промпте не остаётся '$' — все плейсхолдеры шаблона заполнены."""
    messages = build_messages("Метрика — MAE", TRAIN_COLS, TEST_COLS, None)
    assert messages is not None
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "$" not in user
    assert "Метрика — MAE" in user
    assert "y2" in user  # колонки train попали в промпт


def test_build_messages_long_text_and_many_columns_are_cut():
    """Огромный task.md и сотни колонок обрезаются: промпт остаётся компактным."""
    many = [f"c{i}" for i in range(300)]
    user = build_messages("слово " * 5000, many, many, None)[1]["content"]
    assert len(user) < 20000
    assert "и ещё" in user


def test_build_messages_reads_sample_submission_head(tmp_path):
    """Заголовок и первые строки sample_submission попадают в промпт; битый файл не ломает сборку."""
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("row,pred\n1,0.5\n2,0.1\n", encoding="utf-8")
    assert "row, pred" in build_messages("t", TRAIN_COLS, TEST_COLS, sample)[1]["content"]
    broken = tmp_path / "нет_такого.csv"
    assert build_messages("t", TRAIN_COLS, TEST_COLS, broken) is not None


# ---------- reconcile: пустой ответ ----------


@pytest.mark.parametrize("view", [None, {}])
def test_reconcile_empty_view(view):
    """Нет ответа LLM — нечего принимать и не о чем сообщать."""
    assert run(view) == ({}, [])


def test_reconcile_agrees_with_heuristic():
    """LLM согласна с разбором по данным: ничего не заменяем и расхождений в notes нет."""
    view = {"target": "y", "id_column": "id", "metric": "rmse"}
    assert run(view, guessed={"target", "id_column", "metric"}) == ({}, [])


# ---------- reconcile: target ----------


def test_reconcile_guessed_target_replaced_by_llm():
    """Target был лишь угадан, LLM называет колонку «только в train» — принимаем."""
    accepted, notes = run({"target": "y2"}, heur(target="y"), guessed={"target"})
    assert accepted == {"target": "y2"}
    assert notes == []


def test_reconcile_guessed_target_none_filled_by_llm():
    """Эвристика не нашла target вообще (None) — LLM заполняет его."""
    accepted, _ = run({"target": "y"}, heur(target=None), guessed={"target"})
    assert accepted == {"target": "y"}


def test_reconcile_confirmed_target_wins_over_llm():
    """Target подтверждён sample_submission (не угадан): LLM спорит — остаёмся на данных и пишем note."""
    accepted, notes = run({"target": "y2"}, heur(target="y"), guessed=set())
    assert "target" not in accepted
    assert len(notes) == 1
    assert notes[0].startswith(f"{PREFIX}: ")
    assert "y2" in notes[0] and "подтверждён данными" in notes[0]


@pytest.mark.parametrize("bad_target", ["нет_такой", "a"])
def test_reconcile_invalid_target_rejected_even_if_guessed(bad_target):
    """Колонки нет в train или она есть в test — это не target, даже если эвристика лишь гадала."""
    accepted, notes = run({"target": bad_target}, heur(target="y"), guessed={"target"})
    assert "target" not in accepted
    assert any(bad_target in n and "нет только в train" in n for n in notes)


def test_reconcile_multiple_targets_note():
    """Несколько таргетов: берём первый подходящий, а в notes предупреждаем, что предсказываем один."""
    accepted, notes = run({"target": ["y2", "y"]}, heur(target="y"), guessed=set())
    assert "target" not in accepted
    assert any("несколько таргетов" in n for n in notes)
    assert all(n.startswith(f"{PREFIX}: ") for n in notes)


# ---------- reconcile: id ----------


def test_reconcile_guessed_id_replaced_if_in_test():
    """id угадан, LLM называет колонку из test — принимаем."""
    accepted, notes = run({"id_column": "a"}, heur(id_column="id"), guessed={"id_column"})
    assert accepted == {"id_column": "a"}
    assert notes == []


def test_reconcile_id_not_in_test_rejected():
    """id нет в test — ответ по нему не получится собрать, поэтому не принимаем (но говорим об этом)."""
    accepted, notes = run({"id_column": "y"}, heur(id_column="id"), guessed={"id_column"})
    assert accepted == {}
    assert any("id по тексту" in n for n in notes)


def test_reconcile_confirmed_id_not_replaced():
    """id подтверждён sample_submission (не угадан) — LLM его не меняет."""
    accepted, notes = run({"id_column": "a"}, heur(id_column="id"), guessed={"target"})
    assert accepted == {}
    assert len(notes) == 1


# ---------- reconcile: метрика ----------


def test_reconcile_guessed_metric_accepted():
    """Метрику в тексте не нашли (угадана), LLM даёт валидную из реестра — принимаем."""
    accepted, notes = run({"metric": "MAE"}, heur(metric="rmse"), guessed={"metric"})
    assert accepted == {"metric": "mae"}
    assert notes == []


def test_reconcile_found_metric_wins_over_llm():
    """Метрику нашли в тексте по словарю, LLM спорит — остаётся поиск, в notes расхождение."""
    accepted, notes = run({"metric": "mae"}, heur(metric="rmse"), guessed=set())
    assert accepted == {}
    assert len(notes) == 1 and "оставлен поиск" in notes[0]


def test_reconcile_metric_not_in_registry():
    """Метрики нет в реестре (например ndcg@10) — ничего не принимаем, note «нет в реестре»."""
    accepted, notes = run({"metric": "ndcg@10"}, heur(metric="rmse"), guessed={"metric"})
    assert accepted == {}
    assert len(notes) == 1
    assert "нет в реестре" in notes[0] and "ndcg@10" in notes[0]


# ---------- reconcile: прочие поля ----------


def test_reconcile_time_and_group_columns_in_train_make_notes():
    """Колонки времени/групп из train попадают в notes (подсказка для валидации)."""
    _, notes = run({"time_column": "a", "group_column": "b"})
    assert len(notes) == 2
    assert any("колонка времени" in n and "a" in n for n in notes)
    assert any("колонка групп" in n and "b" in n for n in notes)


def test_reconcile_time_and_group_columns_missing_in_train_no_notes():
    """Выдуманная колонка времени/групп (нет в train) — молча игнорируем."""
    assert run({"time_column": "date", "group_column": "client"}) == ({}, [])


def test_reconcile_answer_format_and_constraints():
    """Формат ответа с вероятностями и условия из текста идут в notes; лишнее обрезается."""
    view = {"answer_format": "probability", "constraints": ["  не  округлять\nответ ", "", "x" * 500, "3", "4-е"]}
    _, notes = run(view)
    assert any("формат ответа" in n for n in notes)
    assert any("не округлять ответ" in n for n in notes)
    assert max(len(n) for n in notes) <= len(PREFIX) + 2 + 160 + 1
    assert len([n for n in notes if "формат" not in n]) <= 3  # не более трёх условий


def test_reconcile_garbage_fields_do_not_crash():
    """Странные типы в ответе (числа, словари вместо строк) не роняют сверку."""
    view = {"target": 5, "id_column": {"x": 1}, "metric": 3.14, "time_column": ["a"], "constraints": "строка"}
    accepted, notes = run(view, guessed={"target", "id_column", "metric"})
    assert accepted == {}
    assert isinstance(notes, list)


# ---------- ask_llm ----------


def ask(llm, sample=None):
    return ask_llm("Метрика — MAE", TRAIN_COLS, TEST_COLS, sample, llm)


def test_ask_llm_returns_dict():
    """Нормальный ответ-словарь возвращается как есть; запрос один, без разброса (temperature=0)."""
    llm = FakeLLM({"target": "y"})
    assert ask(llm) == {"target": "y"}
    assert llm.requests == 1
    assert llm.kwargs["temperature"] == 0.0
    assert llm.messages[0]["role"] == "system"


def test_ask_llm_passes_max_wait():
    """Разбор идёт до шага 0, поэтому ожидание ответа ограничено: max_wait передаётся в chat_json."""
    llm = FakeLLM({"target": "y"})
    ask(llm)
    assert llm.kwargs["max_wait"] == task_analyst.MAX_WAIT
    assert llm.kwargs["max_tokens"] == task_analyst.MAX_TOKENS


def test_ask_llm_disabled_by_env(monkeypatch):
    """AGENT_LLM_TASK=0 — LLM не спрашиваем вообще."""
    monkeypatch.setenv(ANALYST_ENV, "0")
    llm = FakeLLM({"target": "y"})
    assert ask(llm) is None
    assert llm.requests == 0


def test_ask_llm_unavailable():
    """LLM недоступна (нет ключа, лимит ₽) — None и ни одного запроса."""
    llm = FakeLLM({"target": "y"}, available=False)
    assert ask(llm) is None
    assert llm.requests == 0


def test_ask_llm_no_client():
    """Клиента нет совсем (llm=None) — None."""
    assert ask(None) is None


def test_ask_llm_exception_returns_none():
    """Исключение в chat_json (сеть, таймаут) не пробрасывается наверх."""
    llm = FakeLLM(RuntimeError("таймаут"))
    assert ask(llm) is None
    assert llm.requests == 1


def test_ask_llm_list_takes_first_dict():
    """Если модель вернула список, берём первый словарь из него."""
    assert ask(FakeLLM(["мусор", 3, {"target": "y"}, {"target": "z"}])) == {"target": "y"}


@pytest.mark.parametrize("garbage", ["просто строка", 42, None, [], ["a", 1]])
def test_ask_llm_garbage_returns_none(garbage):
    """Строка, число, пустой ответ и список без словарей — не ответ: None."""
    assert ask(FakeLLM(garbage)) is None


# ---------- parse_task: интеграция ----------


def write_task(
    folder, text="Предскажи y. Метрика — MAE.", train_extra=None, test_cols=("id", "a", "b"), n=40, n_test=10
):
    """Маленькая задача без sample_submission. train: id, a, b, y (+ train_extra); test: test_cols."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    train["y"] = train["a"] * 2 + rng.normal(scale=0.1, size=n)
    for name in train_extra or []:
        train[name] = rng.normal(size=n)
    test = pd.DataFrame({"id": range(n, n + n_test), "a": rng.normal(size=n_test), "b": rng.normal(size=n_test)})
    test["row_id"] = test["id"]
    train.to_csv(folder / "train.csv", index=False)
    test[list(test_cols)].to_csv(folder / "test.csv", index=False)
    (folder / "task.md").write_text(text, encoding="utf-8")
    return folder / "task.md"


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Ключа нет и аналитик включён по умолчанию: parse_task без llm создаёт недоступный клиент."""
    for var in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "LLM_MOCK", ANALYST_ENV):
        monkeypatch.delenv(var, raising=False)


def test_parse_task_ambiguous_target_fails_without_llm(tmp_path):
    """Две колонки «только в train» и нет словаря: без LLM target определить нельзя — ValueError."""
    task_md = write_task(tmp_path, train_extra=["y_leak"])
    with pytest.raises(ValueError, match="target"):
        parse_task(task_md)


def test_parse_task_llm_resolves_ambiguous_target(tmp_path):
    """С LLM, называющей target='y', тот же разбор проходит."""
    task_md = write_task(tmp_path, train_extra=["y_leak"])
    llm = FakeLLM({"target": "y", "id_column": "id", "metric": "mae"})
    spec = parse_task(task_md, llm=llm)
    assert spec.target == "y"
    assert spec.metric.name == "mae"
    assert llm.requests == 1


def test_parse_task_llm_names_leak_column_as_target(tmp_path):
    """LLM назвала не ту колонку, которой нет в train, — разбор падает честным ValueError, а не KeyError."""
    task_md = write_task(tmp_path, train_extra=["y_leak"])
    with pytest.raises(ValueError, match="target"):
        parse_task(task_md, llm=FakeLLM({"target": "нет_такой"}))


def test_parse_task_llm_refines_id_column(tmp_path):
    """Без sample_submission id угадан неверно (первая колонка test) — LLM уточняет, колонка есть в test."""
    task_md = write_task(tmp_path, test_cols=("a", "b", "row_id"))
    spec = parse_task(task_md, llm=FakeLLM({"target": "y", "id_column": "row_id"}))
    assert spec.id_column == "row_id"
    assert spec.submission_columns == ["row_id", "y"]


def test_parse_task_llm_id_not_in_test_is_ignored(tmp_path):
    """LLM назвала id, которого нет в test, — остаётся угаданный id, ответ собирается по существующей колонке."""
    task_md = write_task(tmp_path)
    spec = parse_task(task_md, llm=FakeLLM({"target": "y", "id_column": "ghost"}))
    assert spec.id_column == "id"
    assert spec.submission_columns == ["id", "y"]


def test_parse_task_llm_fills_missing_metric(tmp_path):
    """В тексте метрики нет — её называет LLM."""
    task_md = write_task(tmp_path, text="Предскажи значение y по признакам.")
    spec = parse_task(task_md, llm=FakeLLM({"target": "y", "metric": "mae"}))
    assert spec.metric.name == "mae"
    assert not spec.metric.greater_is_better


def test_parse_task_text_metric_beats_llm(tmp_path):
    """Метрика найдена в тексте (MAE), LLM предлагает другую — побеждает текст, расхождение в notes."""
    task_md = write_task(tmp_path)
    spec = parse_task(task_md, llm=FakeLLM({"target": "y", "metric": "rmse"}))
    assert spec.metric.name == "mae"
    assert any(n.startswith(f"{PREFIX}: ") and "оставлен поиск" in n for n in spec.notes)


def test_parse_task_unknown_metric_falls_back_to_default(tmp_path):
    """LLM назвала метрику не из реестра и в тексте её нет — берём метрику по умолчанию, не падаем."""
    task_md = write_task(tmp_path, text="Предскажи значение y.")
    spec = parse_task(task_md, llm=FakeLLM({"target": "y", "metric": "ndcg@10"}))
    assert spec.metric.name == "rmse"
    assert any("нет в реестре" in n for n in spec.notes)


def test_parse_task_llm_unavailable_same_as_no_llm(tmp_path):
    """LLM недоступна: разбор идентичен разбору без LLM, запросов нет."""
    task_md = write_task(tmp_path)
    llm = FakeLLM({"target": "y2", "metric": "rmse"}, available=False)
    with_fake = parse_task(task_md, llm=llm)
    without = parse_task(task_md)
    assert llm.requests == 0
    assert with_fake.model_dump() == without.model_dump()


def test_parse_task_garbage_or_error_from_llm_does_not_break(tmp_path):
    """Мусор или исключение от LLM — разбор остаётся детерминированным."""
    task_md = write_task(tmp_path)
    baseline = parse_task(task_md)
    for answer in ("не json", 7, RuntimeError("упала")):
        spec = parse_task(task_md, llm=FakeLLM(answer))
        assert (spec.target, spec.id_column, spec.metric.name) == (
            baseline.target,
            baseline.id_column,
            baseline.metric.name,
        )


def test_parse_task_disabled_by_env_never_asks_llm(tmp_path, monkeypatch):
    """AGENT_LLM_TASK=0: FakeLLM не вызывается ни разу, даже если без неё разбор не пройдёт бы иначе."""
    monkeypatch.setenv(ANALYST_ENV, "0")
    task_md = write_task(tmp_path)
    llm = FakeLLM({"target": "y"})
    spec = parse_task(task_md, llm=llm)
    assert llm.requests == 0
    assert spec.target == "y"


def test_parse_task_disabled_by_env_ambiguous_target_still_fails(tmp_path, monkeypatch):
    """Выключенный аналитик не подсказывает target: двусмысленная задача по-прежнему даёт ValueError."""
    monkeypatch.setenv(ANALYST_ENV, "0")
    task_md = write_task(tmp_path, train_extra=["y_leak"])
    llm = FakeLLM({"target": "y"})
    with pytest.raises(ValueError):
        parse_task(task_md, llm=llm)
    assert llm.requests == 0


def test_parse_task_analyst_notes_go_to_spec(tmp_path):
    """Заметки аналитика (колонка времени, условия, вероятности) попадают в spec.notes с префиксом."""
    task_md = write_task(tmp_path)
    view = {
        "target": "y",
        "time_column": "a",
        "group_column": "b",
        "answer_format": "probability",
        "constraints": ["ответ округлить до 2 знаков"],
    }
    spec = parse_task(task_md, llm=FakeLLM(view))
    analyst_notes = [n for n in spec.notes if n.startswith(f"{PREFIX}: ")]
    assert len(analyst_notes) == 4
    assert any("колонка времени" in n for n in analyst_notes)
    assert any("округлить" in n for n in analyst_notes)
