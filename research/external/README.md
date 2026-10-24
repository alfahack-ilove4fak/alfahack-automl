# Внешние задачи: независимая проверка агента на настоящих таблицах

Зачем: синтетические «чужие» задачи (`research/night`) делает один генератор — они ловят ошибки формата, но не
показывают, как агент справляется с живыми данными. Здесь — датасеты OpenML, собранные другими людьми:
AMLB (регрессия 269, классификация 271), OpenML-CTR23 (353), OpenML-CC18 (99), наборы Grinsztajn et al. (334–337).

**Это набор для гейта («стало ли хуже, есть ли нули»), а не для настройки.** Подгонять playbook под него нельзя.

## Быстрый старт

```bash
uv run python research/external/make_external_tasks.py --core              # 12 мелких задач → runs/night/external/
uv run python research/night/run_alien.py --set external --minutes 5       # ≈ 6 мин на задачу, без LLM
uv run python research/external/report.py                                  # таблица + сравнение с docs/external_baseline.json
uv run python research/external/report.py --update-baseline                # после сабмита — новая база
```

Все 30 задач: без `--core` (≈ 3 ч прогона). Одна задача: `--only churn`. Данные качаются через `sklearn.fetch_openml`
(нужна сеть и ≈ 1 ГБ в `~/scikit_learn_data`) и **в git не попадают** (`runs/` в `.gitignore`): у наборов разные лицензии.
В репозитории только `manifest.json` — список `data_id`, воспроизводимый `select_tasks.py`.

## Что в задачах

| Что варьируется | Как |
|---|---|
| тип и метрика | регрессия: rmse/mae/r2 · бинарная: roc_auc/logloss/pr_auc · мультикласс: f1_macro/accuracy/balanced_accuracy (по хэшу имени) |
| текст `task.md` | 3 стиля: английский, русский «как в банке», коротко; метрика словами |
| формат ответа | id с разными именами; без `sample_submission`; ответ без id вовсе (контракт 1.1.2); id со сдвигом |
| разбиение | 75/25 случайно (классификация — стратифицированно), seed = `data_id` |

Проверка независимая: `run_alien.py` считает метрику своей реализацией и сравнивает с тривиальным ответом и LightGBM
по умолчанию (формула организаторов — «балл к сильному бейзлайну»). `report.py` дополнительно проверяет, что агент
понял метрику и таргет так, как они записаны в `task.md`.

## Честные ограничения

- Разбиение случайное: **временные и групповые схемы здесь не проверяются**. Для них — BeyondArena
  ([HF TabArena/BeyondArena](https://huggingface.co/datasets/TabArena/BeyondArena), 142 датасета с временным и групповым
  разбиением): следующий шаг, пока не подключён.
- Часть датасетов могла попасть в обучение Qwen: выигрыш от LLM на них завышен — LLM-замеры тут не показательны.
- Баллы «к сильному» на таких данных малы (0–7): LightGBM по умолчанию уже силён. Важны **нули** и просадки, а не рост.
- Падения, видимые только на длинном прогоне с живой LLM (сабмит 22.10), этот набор не ловит.

## Источники

[AMLB](https://github.com/openml/automlbenchmark) (MIT) · [TabArena](https://github.com/AutoGluon/tabarena) (Apache-2.0) ·
[OpenML-CTR23](https://www.openml.org/s/353) · [OpenML-CC18](https://www.openml.org/s/99) ·
[Grinsztajn et al.](https://arxiv.org/abs/2207.08815) · [MLE-bench](https://github.com/openai/mle-bench) ·
[PMLB](https://github.com/EpistasisLab/pmlb) (MIT).
