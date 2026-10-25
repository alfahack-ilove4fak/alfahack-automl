"""Точка входа для платформы Альфа Хак. Владелец — капитан.

Интерфейс как в baseline.py организаторов:
    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... python solution.py path/to/task.md
Результат: submission.csv рядом с task.md.

Стратегия «сначала надёжно, потом умно»:
  0. сразу пишем тривиальный ответ (среднее / самый частый класс) — файл есть всегда;
  1. обучаем надёжный бейзлайн и перезаписываем ответ. Бейзлайн не посчитался — лестница отступления (#171):
     обычная CV вместо выбранной схемы → облегчённый бейзлайн → модель без CV; тривиальный ответ — только если
     не обучилась ни одна модель;
  2. цикл AutoResearch улучшает модель, пока есть время; каждый новый лучший сразу сохраняется.

Защита от нулевого балла: ответ пишется атомарно и проверяется после каждой записи; сторож
(runtime/watchdog.py) по жёсткому дедлайну или сигналу остановки проверяет submission.csv и завершает процесс;
любая ошибка агента заканчивается кодом 0 и последним валидным ответом. Доли времени по этапам — runtime/budget.py.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import pandas as pd

from agent.hypotheses import baseline_hypothesis
from agent.loop import run_autoresearch
from agent.task_parser import find_task_files, parse_task, spec_for_target
from evaluation.metrics import METRICS
from evaluation.submission import per_class_columns, validate_submission, write_submission_atomic
from interfaces.schemas import (
    DataProfile,
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    RunBudget,
    TaskSpec,
    TaskType,
)
from ml.experiment import fit_predict_final, match_class_columns, run_experiment
from ml.profiling import load_data, profile_data
from ml.validation import choose_validation
from runtime.budget import StagePlan, ensure_final_reserve, make_budget, plan_stages
from runtime.log import setup_logging
from runtime.watchdog import Watchdog


def _class_share_answer(spec: TaskSpec, y, test, class_columns: list[str]):
    """Шаблон «колонка на класс» (#117): доли классов train.

    Колонка ↔ класс — по имени (match_class_columns: без регистра, без общей приставки «proba_<класс>», #238),
    иначе по порядку class_labels. Та же логика, что у финального ответа модели (ml/experiment._per_class_columns):
    иначе при срыве бейзлайна доли классов уходят в чужие колонки и logloss хуже тривиального ответа.
    """
    shares = y.astype(str).value_counts(normalize=True)
    labels = spec.class_labels or []
    matched = match_class_columns(class_columns, labels)
    values = []
    for i in range(len(class_columns)):
        label = matched[i] if matched is not None else (labels[i] if i < len(labels) else None)
        values.append(float(shares.get(label, 0.0)) if label is not None else 0.0)
    probs = pd.Series(values).clip(lower=1e-6)  # нулевая вероятность ломает logloss
    probs = probs / probs.sum()
    return test[[spec.id_column]].assign(**dict(zip(class_columns, probs, strict=True)))[spec.submission_columns]


def all_targets(spec: TaskSpec) -> list[str]:
    return [spec.target, *spec.extra_targets]


def join_targets(spec: TaskSpec, test, parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Склеить ответы по отдельным таргетам (id, таргет) в один по колонкам шаблона. Строки — в порядке test."""
    out = test[[spec.id_column]].copy()
    for part in parts:
        column = part.columns[-1]
        out[column] = part[column].to_numpy()
    return out[spec.submission_columns]


def fit_extra_targets(spec: TaskSpec, train, test) -> list[pd.DataFrame]:
    """Ответы по дополнительным таргетам (#174): LightGBM по умолчанию на каждый, без CV.

    Цикл AutoResearch улучшает только основной таргет; сбой по дополнительному — его тривиальный ответ.
    """
    log = setup_logging()
    parts = []
    for target in spec.extra_targets:
        sub = spec_for_target(spec, target, train[target])
        try:
            parts.append(fit_predict_final(sub, profile_data(sub, train, test), baseline_hypothesis(sub), train, test))
            log.info("Таргет %s (%s): модель обучена", target, sub.task_type.value)
        except Exception:  # noqa: BLE001 — один таргет не должен стоить ответа по остальным
            log.exception("Таргет %s: модель не обучилась — тривиальный ответ", target)
            parts.append(trivial_answer(sub, train, test))
    return parts


def to_template(answer: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    """Колонки ответа — как в шаблоне. Единственная колонка ответа может называться не как таргет (#276).

    train: `target`, шаблон: `id, prediction` — разбор находит target верно, а колонка ответа в шаблоне другая:
    без переименования `answer[spec.submission_columns]` падал с KeyError и сдавался шаблон из нулей.
    """
    answer_columns = [c for c in spec.submission_columns if c != spec.id_column]
    if len(answer_columns) == 1 and answer_columns[0] not in answer.columns and spec.target in answer.columns:
        answer = answer.rename(columns={spec.target: answer_columns[0]})
    return answer[spec.submission_columns]


def trivial_answer(spec: TaskSpec, train, test):
    """Ответ без модели: страховка на случай, если всё остальное упадёт."""
    if spec.extra_targets:  # несколько таргетов (#174): тривиальный ответ по каждому
        parts = [trivial_answer(spec_for_target(spec, t, train[t]), train, test) for t in all_targets(spec)]
        return join_targets(spec, test, parts)
    y = train[spec.target]
    class_columns = per_class_columns(spec)
    if class_columns and spec.task_type != TaskType.REGRESSION:
        return _class_share_answer(spec, y, test, class_columns)
    if spec.task_type == TaskType.REGRESSION:
        value = float(y.mean())
    elif METRICS[spec.metric.name].prediction == "proba" and spec.class_labels:
        value = float((y.astype(str) == spec.class_labels[-1]).mean())  # доля «положительного» класса
    else:
        value = str(y.astype(str).mode()[0])
    return to_template(test[[spec.id_column]].assign(**{spec.target: value}), spec)


LIGHT_ID = "baseline_light"
LIGHT_MAX_ROWS = 200_000  # облегчённый бейзлайн: CV на сэмпле — порядок качества тот же, время в разы меньше
LIGHT_PARAMS = {"n_folds": 3, "model_params": {"n_estimators": 500, "learning_rate": 0.1}}
PLAIN_SCHEMES = (None, "kfold", "stratified")


def baseline_ladder(spec: TaskSpec, profile: DataProfile, train, test, budget: RunBudget):
    """Бейзлайн с запасными ступенями (#151, #171). → (гипотеза, результат); результат без скора — CV не удалась.

    1. LightGBM в схеме валидации, похожей на скрытый test (choose_validation, ADR 0013) — один раз до цикла:
       acceptance не сравнивает разные схемы, поэтому бейзлайн обязан быть в той же схеме, что и гипотезы;
    2. та же модель в обычной CV — если выбранная схема не посчиталась (например, редкий класс во time-фолдах);
    3. облегчённая: 3 фолда, меньше деревьев, CV на сэмпле ≤ LIGHT_MAX_ROWS — если не хватило времени или памяти.
       После нехватки времени/памяти ступень 2 пропускается: та же модель на тех же строках не успеет и снова.
    """
    log = setup_logging()
    try:
        validation = choose_validation(spec, profile, train, test)
    except Exception:  # noqa: BLE001 — без выбранной схемы работаем в обычной CV
        log.exception("Схема валидации не выбрана — обычная CV")
        validation = None  # сам выбор схемы пишет её в лог с причиной
    base = baseline_hypothesis(spec)
    rungs = []
    if validation not in PLAIN_SCHEMES:
        rungs.append((base.model_copy(update={"params": {**base.params, "validation": validation}}), train))
    rungs.append((base, train))
    light = base.model_copy(update={"id": LIGHT_ID, "params": {**base.params, **LIGHT_PARAMS}})
    rungs.append((light, train.sample(n=min(len(train), LIGHT_MAX_ROWS), random_state=0)))
    result = None
    for hyp, rows in rungs:
        if hyp.id != LIGHT_ID and result is not None and _out_of_resources(result):
            continue  # полная CV не успела или не влезла в память — сразу облегчённая ступень
        result = run_experiment(spec, profile, hyp, rows, budget)
        if result.mean_score is not None:
            return hyp, result
        log.warning("Бейзлайн «%s» не посчитался (%s: %s) — следующая ступень", hyp.id, result.status, result.error)
    return light, result


def blind_answer(task_md: Path) -> pd.DataFrame | None:
    """Ответ без TaskSpec (#187): разбор задачи упал, образца ответа нет — лучше любой ответ, чем никакого.

    id — колонка `id` test, иначе первая. Ответ — колонки, которые есть только в train: упомянутые в task.md,
    иначе последняя. Значение — среднее (число) или самое частое (остальное). None — не из чего собрать.
    """
    files = find_task_files(task_md)
    if "train" not in files or "test" not in files:
        return None
    test = pd.read_csv(files["test"])
    extra = [c for c in pd.read_csv(files["train"], nrows=0).columns if c not in test.columns]
    if not extra or test.empty:
        return None
    text = task_md.read_text(encoding="utf-8", errors="replace")
    targets = [c for c in extra if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text)] or extra[-1:]
    y = pd.read_csv(files["train"], usecols=targets)
    out = test[["id" if "id" in test.columns else test.columns[0]]].copy()
    for col in targets:
        values = y[col].dropna()
        numeric = pd.api.types.is_numeric_dtype(values) and not values.empty
        out[col] = float(values.mean()) if numeric else (str(values.astype(str).mode()[0]) if len(values) else "")
    return out


def _out_of_resources(result: ExperimentResult) -> bool:
    """CV упала из-за времени или памяти (а не из-за схемы валидации) — повтор той же модели не поможет."""
    return result.status == ExperimentStatus.TIMEOUT or "memory" in (result.error or "").lower()


def write_fallback_answer(task_md: Path, where: str) -> bool:
    """Ответ без TaskSpec: образец формата от организаторов, иначе ответ без разбора задачи (#187). → записан ли.

    Пустой файл хуже: им закрываем и конец работы без ответа, и ожидание LLM-аналитика, когда разбор по данным
    не справился (#239) — пока он думает до 90 с, на диске уже что-то лежит.
    """
    log = setup_logging()
    path = task_md.parent / "submission.csv"
    sample = find_task_files(task_md).get("sample_submission")
    if sample is not None:
        shutil.copyfile(sample, path)
        log.error("%s: ответа не было — скопирован %s", where, sample.name)
        return True
    if (blind := blind_answer(task_md)) is not None:
        write_submission_atomic(blind, path)
        log.error("%s: ответа и образца нет — ответ без разбора задачи: %s", where, list(blind))
        return True
    log.error("%s: ответа нет, и образца sample_submission тоже", where)
    return False


class AnswerKeeper:
    """Хранит последний валидный ответ и в любой момент (в том числе из сторожа) восстанавливает submission.csv."""

    def __init__(self, task_md: Path) -> None:
        self.task_md = task_md
        self.path = task_md.parent / "submission.csv"
        self.spec: TaskSpec | None = None
        self.test: pd.DataFrame | None = None
        self.last_good: pd.DataFrame | None = None  # последний ответ, прошедший проверку
        self.lock = threading.Lock()  # держим на время записи: сторож не оборвёт её посередине

    def write(self, answer: pd.DataFrame) -> list[str]:
        """Записать ответ и проверить его. Невалидный ответ откатывается к последнему валидному. → список проблем."""
        assert self.spec is not None and self.test is not None
        with self.lock:
            write_submission_atomic(answer, self.path)
            report = validate_submission(self.path, self.spec, self.test)
            if report.ok:
                self.last_good = answer
            elif self.last_good is not None:
                write_submission_atomic(self.last_good, self.path)
            return report.problems

    def finalize(self, reason: str) -> None:
        """Последняя проверка: submission.csv есть и валиден, иначе — последний валидный или страховочный ответ.

        Не берёт `self.lock` — его уже держит сторож; из главного потока вызывается под `with keeper.lock`.
        """
        log = setup_logging()
        self.path.with_suffix(".tmp.csv").unlink(missing_ok=True)  # огрызок записи, оборванной убийством процесса
        try:
            if self.spec is not None and self.test is not None:
                report = validate_submission(self.path, self.spec, self.test)
                if report.ok:
                    log.info("Финал (%s): submission.csv валиден, %d строк", reason, report.n_rows)
                    return
                log.error("Финал (%s): submission.csv невалиден: %s — восстанавливаем", reason, report.problems)
                if self.last_good is not None:
                    write_submission_atomic(self.last_good, self.path)
                    return
            if not self.path.exists():
                write_fallback_answer(self.task_md, f"Финал ({reason})")
        except Exception:  # noqa: BLE001 — финализация не должна ронять процесс
            log.exception("Финал (%s): не удалось проверить или восстановить submission.csv", reason)


_SPEC_KEYS = ("target", "extra_targets", "id_column", "submission_columns", "task_type", "class_labels")


def spec_changed(old: TaskSpec, new: TaskSpec) -> bool:
    """Поменял ли аналитик то, от чего зависит ответ: таргеты, id, колонки ответа, тип задачи, метрику."""
    return old.metric.name != new.metric.name or any(getattr(old, key) != getattr(new, key) for key in _SPEC_KEYS)


def refine_with_analyst(task_md: Path, spec: TaskSpec, train, test, keeper: AnswerKeeper, log):
    """Второй разбор — с LLM-аналитиком, когда шаг 0 уже на диске (#239). Любой сбой — остаёмся на разборе по данным.

    Аналитик поменял target/id/метрику — перечитываем данные и перезаписываем шаг 0 по новому spec; ответ по нему
    не прошёл проверку — возвращаем прежний spec. → (spec, train, test), с которыми идём дальше.
    """
    try:
        refined = parse_task(task_md)
    except Exception:  # noqa: BLE001 — шаг 0 уже записан, разбор по данным надёжнее
        log.exception("Разбор с LLM-аналитиком упал — остаёмся на разборе по данным")
        return spec, train, test
    if not spec_changed(spec, refined):
        return spec, train, test
    old_spec, old_test = keeper.spec, keeper.test
    try:
        new_train, new_test = load_data(refined)
        keeper.spec, keeper.test = refined, new_test
        problems = keeper.write(trivial_answer(refined, new_train, new_test))
    except Exception:  # noqa: BLE001
        keeper.spec, keeper.test = old_spec, old_test
        log.exception("Не удалось пересчитать шаг 0 по разбору аналитика — остаёмся на разборе по данным")
        return spec, train, test
    if problems:
        keeper.spec, keeper.test = old_spec, old_test
        log.warning("Ответ по разбору аналитика не прошёл проверку (%s) — остаёмся на разборе по данным", problems)
        return spec, train, test
    log.info(
        "Аналитик уточнил разбор: target %s → %s, метрика %s → %s — шаг 0 перезаписан",
        spec.target, refined.target, spec.metric.name, refined.metric.name,
    )  # fmt: skip
    return refined, new_train, new_test


def run_pipeline(task_md: Path, budget: RunBudget, plan: StagePlan, keeper: AnswerKeeper, started: float) -> None:
    log = setup_logging()
    analyst_asked = False
    try:
        spec = parse_task(task_md, use_analyst=False)  # быстрый разбор по данным: шаг 0 не ждёт LLM (#239)
    except Exception as exc:  # noqa: BLE001 — по данным не разобрать (например, неоднозначный target): решает аналитик
        log.warning("Разбор по данным не справился (%s) — зовём LLM-аналитика до шага 0, как раньше", exc)
        try:
            write_fallback_answer(task_md, "До аналитика")  # аналитик ждёт до 90 с — пусть файл уже будет
        except Exception:  # noqa: BLE001 — страховка не должна мешать разбору
            log.exception("Страховочный ответ до аналитика не записан")
        spec = parse_task(task_md)
        analyst_asked = True
    keeper.spec = spec
    train, test = load_data(spec)
    keeper.test = test
    trivial = trivial_answer(spec, train, test)
    problems = keeper.write(trivial)
    log.info("Шаг 0: тривиальный ответ записан%s", f", но проверка: {problems}" if problems else "")
    if not analyst_asked:
        spec, train, test = refine_with_analyst(task_md, spec, train, test, keeper, log)

    def save(hyp: Hypothesis, res: ExperimentResult, profile: DataProfile) -> float:
        """Финальное обучение на всём train + запись ответа. Возвращает, сколько секунд это заняло."""
        began = time.time()
        try:
            answer = fit_predict_final(spec, profile, hyp, train, test)
            if spec.extra_targets:
                answer = join_targets(spec, test, [answer[[spec.id_column, spec.target]], *extra_answers])
            problems = keeper.write(to_template(answer, spec))
            score = "без CV" if res.mean_score is None else f"CV {res.metric_name}={res.mean_score:.5f}"
            log.info("Сохранён ответ «%s» (%s), проверка: %s", hyp.id, score, "OK" if not problems else problems)
        except Exception:  # noqa: BLE001 — старый валидный ответ остаётся на месте
            log.exception("Не удалось сохранить ответ гипотезы %s", hyp.id)
        return time.time() - began

    profile = profile_data(spec, train, test)
    # Дополнительные таргеты — один раз до бейзлайна: их ответ нужен в каждой записи модели
    extra_answers = fit_extra_targets(spec, train, test) if spec.extra_targets else []
    parsed_at = time.time()
    base, base_result = baseline_ladder(spec, profile, train, test, budget)
    log.info("Шаг 1: бейзлайн %s=%s ± %s", base_result.metric_name, base_result.mean_score, base_result.std_score)
    history = [(base, base_result)]
    if base_result.mean_score is None:
        # Ни одна CV не посчиталась: сравнивать гипотезы не с чем, но обученная модель почти всегда лучше среднего
        log.error("CV бейзлайна не удалась ни на одной ступени — финальная модель без CV, цикл пропускаем")
        save(base, base_result, profile)
        return
    if base.id == LIGHT_ID:
        # Облегчённая CV (3 фолда, сэмпл) несравнима с гипотезами цикла, а полная не успела — ресурсов мало
        log.warning("Бейзлайн посчитан только облегчённым — сохраняем его, цикл AutoResearch пропускаем")
        save(base, base_result, profile)
        return
    final_seconds = save(base, base_result, profile)
    if time.time() > plan.baseline_end:
        log.warning("Разбор + бейзлайн вышли за свою долю бюджета (%.0f с)", plan.baseline_end - started)
    reserve = ensure_final_reserve(budget, final_seconds)
    log.info(
        "Замер: загрузка+профиль %.1f с, CV бейзлайна %.0f с, финал %.0f с; AutoResearch до +%.0f с",
        parsed_at - started,
        base_result.fit_seconds,
        final_seconds,
        budget.total_seconds - reserve,
    )

    best = run_autoresearch(
        spec, profile, train, budget, history, (base, base_result), on_new_best=lambda h, r: save(h, r, profile)
    )
    log.info(
        "Готово за %.0f с. Лучшая гипотеза: %s (%s=%.5f), всего экспериментов: %d",
        time.time() - started,
        best[0].id,
        best[1].metric_name,
        best[1].mean_score,
        len(history),
    )


def main(argv: list[str]) -> int:
    started = time.time()
    run_dir = Path(os.environ["AGENT_RUN_DIR"]) if os.getenv("AGENT_RUN_DIR") else None
    log = setup_logging(run_dir)
    budget = make_budget(started)
    plan = plan_stages(budget)
    task_md = Path(argv[1] if len(argv) > 1 else "task.md")
    log.info("Старт: %s, бюджет %.0f с, RAM %.0f МБ", task_md, budget.total_seconds, budget.ram_limit_mb)
    log.info("План этапов: %s", plan.describe())

    keeper = AnswerKeeper(task_md)
    watchdog = Watchdog(plan.hard_deadline, keeper.finalize, lock=keeper.lock)
    watchdog.start()
    try:
        run_pipeline(task_md, budget, plan, keeper, started)
    except Exception:  # noqa: BLE001 — любая ошибка агента не должна стоить ответа
        log.exception("Агент упал — остаётся последний валидный ответ")
    finally:
        with keeper.lock:
            keeper.finalize("конец работы")
        watchdog.stop()
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):  # Windows/Git Bash: не UTF-8 → ✅ и кириллица роняют вывод
        _stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv))
