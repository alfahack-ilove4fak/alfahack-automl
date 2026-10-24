"""Внешние задачи BeyondArena с временным разбиением: проверяем, что агент сам выбирает схему валидации. Владелец — research.

    uv run python research/external/make_beyond_tasks.py --select                          # HF → manifest_beyond.json
    uv run --with pyarrow==19.0.1 python research/external/make_beyond_tasks.py --core     # 5 задач → runs/night/beyond/
    uv run python research/night/run_alien.py --set beyond --minutes 5                     # прогон с независимой проверкой
    uv run python research/external/report.py --set beyond                                 # таблица + схема валидации

Зачем: во внешнем наборе OpenML (make_external_tasks.py) разбиение случайное, и выбор схемы валидации там не
проверяется. BeyondArena (HF TabArena/BeyondArena, arXiv 2606.30410) — 142 настоящих датасета, у которых кураторы
отметили колонку времени (time_on) или групп (group_on). Скрытые задачи вполне могут быть «test — следующий период»:
если агент возьмёт случайные фолды, оценка CV будет завышена, и он примет гипотезы, которые на test хуже.

Что делаем (формат задач и ответов — как в make_external_tasks.py, оттуда же тексты, метрики и форматы ответа):
  1. --select: из таблицы датасетов в README (ревизия REVISION) берём временные задачи до MAX_ROWS строк и
     MAX_FEATURES колонок, читаем task_metadata (target, time_on, метрика кураторов) → manifest_beyond.json;
  2. скачиваем dataset.parquet той же ревизии (кэш ~/.cache/alfahack/beyond, в git не кладём — лицензии разные);
  3. делим по времени: последние ≈ TEST_SHARE строк по time_on → test, одна дата не делится между train и test,
     строки без времени — в train. Колонка времени остаётся в train и test, как было бы на платформе;
  4. метрика — как у кураторов, если она есть у нас для этого типа задачи, иначе по хэшу имени (как в external);
  5. в половине задач task.md говорит, что test — более поздний период, в другой половине молчит: агент должен
     найти время сам. В meta — expected_validation = time:<колонка>; report.py --set beyond сверяет его с agent.log.
Групповые датасеты (group_on) — следующий шаг: здесь пропускаются.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_external_tasks as ext  # noqa: E402 — общие тексты, метрики и форматы ответа

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "night" / "beyond"
CACHE = Path.home() / ".cache" / "alfahack" / "beyond"
MANIFEST = Path(__file__).with_name("manifest_beyond.json")
REPO = "TabArena/BeyondArena"
REVISION = "2ecfe882ccfb814fc27c4de10a64ceefd5d7655c"  # данные 30.06.2026: ревизия зафиксирована ради воспроизводимости
TEST_SHARE = 0.25
MIN_TEST_SHARE = 0.10  # если на дате отсечки много строк и строгое «позже» оставило меньше — берём дату отсечки в test
MAX_ROWS, MAX_FEATURES, CORE_PER_TYPE = 100_000, 400, 2
PROBLEM_TYPES = {"Binary": "binary", "Multi": "multiclass", "Reg": "regression"}
CURATOR_METRICS = {"roc_auc": "roc_auc", "rmse": "rmse", "mae": "mae", "r2": "r2", "log_loss": "logloss",
                   "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}  # fmt: skip
HINT = (
    " Data are ordered in time: test.csv covers a later period than train.csv (column `{col}`).",
    " Данные упорядочены по времени: test.csv — более поздний период, чем train.csv (колонка `{col}`).",
)


def resolve_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"


# ------------------------------------------------------------------ отбор → manifest_beyond.json


def readme_rows(readme: str) -> list[dict]:
    """Таблица датасетов из README: | Dataset | Domain | Source | Year | Age | N | d | C | Prob. | Task | BibKey(s) |."""
    rows = []
    for line in readme.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 11 or cells[9] not in ("IID", "Temporal", "Grouped") or cells[8] not in PROBLEM_TYPES:
            continue
        rows.append(
            {"name": cells[0], "rows": int(cells[5].replace(",", "")), "features": int(cells[6].replace(",", "")),
             "task_type": PROBLEM_TYPES[cells[8]], "regime": cells[9].lower()}
        )  # fmt: skip
    return rows


def select() -> int:
    import httpx

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        readme = client.get(resolve_url("README.md")).text
        rows = [r for r in readme_rows(readme) if r["regime"] == "temporal"]
        print(f"временных датасетов в README: {len(rows)}")
        chosen = []
        for row in rows:
            if row["rows"] > MAX_ROWS or row["features"] > MAX_FEATURES:
                continue
            tree = client.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{row['name']}",
                              params={"recursive": "true"}).json()  # fmt: skip
            meta_path = next(
                f["path"] for f in tree if f["path"].endswith("task_metadata.predictive-ml-task-mold-v1.json")
            )
            folder = meta_path.rsplit("/", 1)[0]
            task = client.get(resolve_url(meta_path)).json()
            info = client.get(resolve_url(f"{folder}/dataset_metadata.dataset-mold-v1.json")).json()
            if not task.get("time_on"):
                continue
            chosen.append(
                {**{k: row[k] for k in ("name", "task_type", "rows", "features")}, "folder": folder,
                 "target": task["target_column_name"], "time_on": task["time_on"],
                 "curator_metric": task.get("objective_metric_name"), "license": info.get("license"),
                 "source": info.get("original_dataset_source_download_link")}
            )  # fmt: skip
    for kind in PROBLEM_TYPES.values():
        small = sorted((t for t in chosen if t["task_type"] == kind), key=lambda t: t["rows"])[:CORE_PER_TYPE]
        for t in small:
            t["core"] = True
    rule = (f"Temporal из README, ≤ {MAX_ROWS} строк и ≤ {MAX_FEATURES} колонок; "
            f"core — {CORE_PER_TYPE} самых маленьких каждого типа")  # fmt: skip
    payload = {"repo": REPO, "revision": REVISION, "rule": rule, "tasks": chosen}
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for t in chosen:
        print(f"  {'core' if t.get('core') else '    '} {t['name']:<38} {t['task_type']:<10} {t['rows']:>6}×{t['features']:<4} "
              f"time={t['time_on']} metric={t['curator_metric']} license={t['license']}")  # fmt: skip
    return 0


# ------------------------------------------------------------------ задачи


def load(entry: dict) -> pd.DataFrame:
    import httpx

    path = CACHE / entry["name"] / "dataset.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=600, follow_redirects=True) as client:
            response = client.get(resolve_url(f"{entry['folder']}/dataset.parquet"))
            response.raise_for_status()
        path.write_bytes(response.content)
    frame = pd.read_parquet(path)
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = frame[col].astype(object)
    frame.columns = [str(c) for c in frame.columns]
    return frame


def time_split(frame: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Последние ≈ TEST_SHARE строк по времени → test; одна дата целиком по одну сторону; без времени → train."""
    col = frame[time_col]
    t = col if pd.api.types.is_numeric_dtype(col) else pd.to_datetime(col, errors="coerce")  # год, номер периода
    cutoff = t.dropna().quantile(1 - TEST_SHARE)
    is_test = (t > cutoff).to_numpy()
    if is_test.mean() < MIN_TEST_SHARE:
        is_test = (t >= cutoff).to_numpy()
    train, test = frame.loc[~is_test].reset_index(drop=True), frame.loc[is_test].reset_index(drop=True)
    return train, test, str(cutoff)


def clean(frame: pd.DataFrame, target: str, kind: str) -> pd.DataFrame | None:
    """Как в make_external_tasks.make: без пропусков в target, у классов не меньше 10 строк."""
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    if kind == "regression":
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        frame = frame.dropna(subset=[target]).reset_index(drop=True)
        return frame if frame[target].nunique() >= 5 else None
    frame[target] = frame[target].astype(str)
    counts = frame[target].value_counts()
    frame = frame[frame[target].isin(counts[counts >= 10].index)].reset_index(drop=True)
    return frame if frame[target].nunique() >= 2 else None


def pick_metric(entry: dict) -> str:
    kind = entry["task_type"]
    ours = CURATOR_METRICS.get(entry.get("curator_metric") or "")
    if ours in ext.METRICS_BY_TYPE[kind]:
        return ours
    metrics = ext.METRICS_BY_TYPE[kind]
    return metrics[ext.h(entry["name"], "metric") % len(metrics)]


def make(entry: dict) -> str | None:
    kind, target, time_col = entry["task_type"], entry["target"], entry["time_on"]
    frame = clean(load(entry), target, kind)
    if frame is None or time_col not in frame.columns:
        return None
    train, test, cutoff = time_split(frame, time_col)
    if kind != "regression" and train[target].nunique() < 2 or len(test) < 50:
        return None
    if kind != "regression":  # классы, которых нет в train, агенту не предсказать — убираем их из test
        test = test[test[target].isin(set(train[target]))].reset_index(drop=True)
    metric = pick_metric(entry)
    positive = None
    if kind == "binary":
        shares = frame[target].value_counts(normalize=True)
        positive = sorted(frame[target].unique())[-1]
        if metric == "pr_auc" and shares.min() < 0.3:
            positive = shares.idxmin()
    name = ext.slug(entry["name"])
    fmt = ext.FORMATS[ext.h(entry["name"], "fmt") % len(ext.FORMATS)]
    style = ext.h(entry["name"], "style") % 3
    hint = ext.h(entry["name"], "hint") % 2 == 0
    id_col = None if fmt == "no_id_sample" else ext.ID_NAMES[ext.h(entry["name"], "id") % len(ext.ID_NAMES)]
    if id_col in frame.columns:
        id_col = None
    if id_col is None:
        fmt = "no_id_sample"
    shift = 100000 + ext.h(entry["name"], "shift") % 900000 if fmt == "id_sample_shifted" else 0
    answers = test[[target]].copy()
    test_x = test.drop(columns=[target])
    if id_col:
        ids = np.arange(len(test)) + shift + (7 if shift else 1)
        train.insert(0, id_col, np.arange(len(train)) + shift + len(test) + 100)
        test_x.insert(0, id_col, ids)
        answers.insert(0, id_col, ids)
    filler = {
        "regression": 0.0,
        "binary": 0.5 if metric in ext.PROBA else positive,
        "multiclass": sorted(train[target].unique())[0],
    }[kind]
    task_dir = OUT / name
    shutil.rmtree(task_dir, ignore_errors=True)
    task_dir.mkdir(parents=True)
    train.to_csv(task_dir / "train.csv", index=False)
    test_x.to_csv(task_dir / "test.csv", index=False)
    if fmt != "id_no_sample":
        answers.assign(**{target: filler}).to_csv(task_dir / "sample_submission.csv", index=False)
    text = ext.task_text(style, entry["name"], kind, metric, target, positive, fmt, id_col, frame)
    if hint:  # подсказка про время — отдельным абзацем в конце; язык — как у текста (стиль 0 — английский)
        text = text.rstrip("\n") + "\n\n" + HINT[0 if style == 0 else 1].format(col=time_col).strip() + "\n"
    (task_dir / "task.md").write_text(text, encoding="utf-8")
    ans = OUT / "_answers"
    ans.mkdir(parents=True, exist_ok=True)
    answers.to_csv(ans / f"{name}.csv", index=False)
    meta = {
        "metric": metric,
        "greater_is_better": ext.GREATER[metric],
        "task_type": kind,
        "id": id_col,
        "targets": [target],
        "submission_columns": ([id_col] if id_col else []) + [target],
        "positive": positive,
        "split": "temporal",
        "time_column": time_col,
        "cutoff": cutoff,
        "expected_validation": f"time:{time_col}",
        "hint_in_text": hint,
        "checks": f"BeyondArena {entry['name']} (temporal, отсечка {cutoff}); формат {fmt}, стиль {style}, "
        f"подсказка про время {'есть' if hint else 'нет'}",
        "source": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}/{entry['folder']}",
    }
    (ans / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {name}: {kind}, {metric}, {len(train)}+{len(test)} строк, время {time_col} > {cutoff}, формат {fmt}, "
          f"стиль {style}, подсказка {'да' if hint else 'нет'}", flush=True)  # fmt: skip
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", action="store_true", help="перебрать README на HF и записать manifest_beyond.json")
    parser.add_argument("--only", nargs="*", help="части имён датасетов")
    parser.add_argument("--core", action="store_true", help=f"только core: {CORE_PER_TYPE} мелких задачи каждого типа")
    args = parser.parse_args()
    if args.select:
        return select()
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
    if args.core:
        entries = [e for e in entries if e.get("core")]
    for entry in entries:
        if args.only and not any(o.lower() in entry["name"].lower() for o in args.only):
            continue
        try:
            if make(entry) is None:
                print(f"⏭ {entry['name']}: пропущен (нет колонки времени, мало классов или строк в test)")
        except Exception as exc:  # noqa: BLE001 — один датасет не должен ронять остальные
            print(f"💥 {entry['name']}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
