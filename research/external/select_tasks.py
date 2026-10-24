"""Отбор внешних задач из открытых наборов OpenML → research/external/manifest.json (коммитится, данные — нет).

    uv run python research/external/select_tasks.py              # перебрать наборы и записать manifest.json
    uv run python research/external/select_tasks.py --per-type 10

Наборы (все — настоящие табличные данные, независимые от нас и от синтетики research/night):
  269 AMLB Regression (33) · 271 AMLB Classification (71) · 353 OpenML-CTR23 регрессия (35) ·
  99 OpenML-CC18 классификация (72) · 334–337 Grinsztajn et al. (NeurIPS 2022): числовые/категориальные.
Фильтр: 1500–60000 строк (прогон в 5 минут), ≤ 200 признаков, ≥ 3 различных значений у регрессионного таргета,
не больше 40 классов. Дубли по имени убираем. Выбор детерминирован: сортировка по sha1(имя + seed) внутри типа,
но банковские темы (bank/credit/loan/churn/default/insurance) идут первыми — скрытые задачи, скорее всего, оттуда.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx

SUITES = {269: "AMLB-regression", 271: "AMLB-classification", 353: "OpenML-CTR23", 99: "OpenML-CC18"}
SUITES |= {334: "Grinsztajn-334", 335: "Grinsztajn-335", 336: "Grinsztajn-336", 337: "Grinsztajn-337"}
API = "https://www.openml.org/api/v1/json"
OUT = Path(__file__).with_name("manifest.json")
BANK_WORDS = ("bank", "credit", "loan", "churn", "default", "insurance", "fraud", "customer", "income", "adult")
CORE_PER_TYPE, CORE_MAX_ROWS = 4, 20000  # core — быстрый срез (≈ 12 задач)
MIN_ROWS, MAX_ROWS, MAX_FEATURES, MAX_CLASSES = 1500, 60000, 200, 40


def suite_ids(client: httpx.Client, suite: int) -> list[int]:
    data = client.get(f"{API}/study/{suite}").json()["study"]
    return [int(i) for i in data.get("data", {}).get("data_id", [])]


def qualities(client: httpx.Client, ids: list[int]) -> dict[int, dict]:
    out = {}
    for start in range(0, len(ids), 100):
        chunk = ",".join(map(str, ids[start : start + 100]))
        rows = client.get(f"{API}/data/list/data_id/{chunk}/limit/100").json()["data"]["dataset"]
        for row in rows:
            q = {x["name"]: x["value"] for x in row.get("quality", [])}
            out[int(row["did"])] = {"name": row["name"], "status": row.get("status"), **q}
    return out


def pick_key(name: str, seed: str) -> tuple[int, str]:
    bank = any(w in name.lower() for w in BANK_WORDS)
    return (0 if bank else 1, hashlib.sha1(f"{name}{seed}".encode()).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--per-type", type=int, default=10, help="сколько задач каждого типа: регрессия/бинарная/мульти"
    )
    parser.add_argument("--seed", default="alfahack-external-1")
    args = parser.parse_args()
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        member = {}
        for suite, label in SUITES.items():
            for did in suite_ids(client, suite):
                member.setdefault(did, []).append(label)
        info = qualities(client, sorted(member))
    candidates: dict[str, list] = {"regression": [], "binary": [], "multiclass": []}
    seen = set()
    for did, q in info.items():
        name = q["name"]
        n, f, k = (float(q.get(key, 0) or 0) for key in ("NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"))
        if not (MIN_ROWS <= n <= MAX_ROWS and 2 <= f <= MAX_FEATURES) or name.lower() in seen:
            continue
        if k == 0:
            kind = "regression"
        elif k == 2:
            kind = "binary"
        elif 3 <= k <= MAX_CLASSES:
            kind = "multiclass"
        else:
            continue
        seen.add(name.lower())
        candidates[kind].append(
            {"data_id": did, "name": name, "task_type": kind, "rows": int(n), "features": int(f), "classes": int(k),
             "suites": member[did]}
        )  # fmt: skip
    chosen = []
    for kind, items in candidates.items():
        items.sort(key=lambda it: pick_key(it["name"], args.seed))
        picked = items[: args.per_type]
        small = [it for it in picked if it["rows"] <= CORE_MAX_ROWS][:CORE_PER_TYPE]
        for it in picked:
            it["core"] = it in small  # быстрый срез для гейта: по CORE_PER_TYPE мелких задач каждого типа
        chosen += picked
        print(f"{kind}: кандидатов {len(items)}, взято {min(len(items), args.per_type)}")
    OUT.write_text(
        json.dumps({"seed": args.seed, "tasks": chosen}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    for it in chosen:
        print(
            f"  {it['data_id']:>6} {it['name']:<34} {it['task_type']:<10} {it['rows']:>6}×{it['features']:<4} {it['suites']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
