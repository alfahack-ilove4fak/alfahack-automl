"""#276: Macro F1 ответов на настоящем test multiclass по строкам с точной копией в train (метка — большинство копий).

uv run python research/night/copies_eval.py <папка с real/<версия>/multiclass/submission.csv> b6c6e0c 3fc774f
"""

import sys

import pandas as pd
from sklearn.metrics import f1_score

S = sys.argv[1]
P = "tasks/public/multiclass/"
tr = pd.read_csv(P + "train.csv")
te = pd.read_csv(P + "test.csv")
feats = [c for c in tr.columns if c not in ("id", "attack_cat")]
tr["g"] = pd.util.hash_pandas_object(tr[feats], index=False).values
te["g"] = pd.util.hash_pandas_object(te[feats], index=False).values
lab = tr.groupby("g").attack_cat.agg(lambda s: s.value_counts().index[0])
purity = tr.groupby("g").attack_cat.agg(lambda s: s.value_counts(normalize=True).iloc[0])
known = te[te.g.isin(lab.index)].copy()
known["y"] = known.g.map(lab)
known["purity"] = known.g.map(purity)
print(
    "строк test с копией в train",
    len(known),
    "из",
    len(te),
    "; из них с однозначной меткой",
    int((known.purity == 1).sum()),
)
cls = sorted(tr.attack_cat.unique())
for v in sys.argv[2:]:
    sub = pd.read_csv(f"{S}/real/{v}/multiclass/submission.csv")
    m = known.merge(sub, on="id", suffixes=("", "_pred"))
    pure = m[m.purity == 1]
    print(
        f'{v}: Macro F1 на строках с копией {f1_score(m.y, m.attack_cat, average="macro"):.4f},'
        f' только однозначные {f1_score(pure.y, pure.attack_cat, average="macro"):.4f},'
        f' accuracy {(m.y == m.attack_cat).mean():.4f}'
    )
    share = sub.attack_cat.value_counts(normalize=True).reindex(cls).fillna(0).round(3)
    print("  доли предсказанных классов на всём test:", share.to_dict())
print("доли классов (по меткам копий) на этих строках:", known.y.value_counts(normalize=True).round(3).to_dict())
