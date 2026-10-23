"""Одна большая задача: 3 млн строк × 55 признаков (регрессия, RMSE) — как ведёт себя бейзлайн на «много строк»."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
g = np.random.default_rng(sum(map(ord, "thenlaooo")) + 300)
n, nt, k = 3_000_000, 500_000, 50
X = g.normal(size=(n + nt, k)).astype("float32")
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(k)])
for j in range(5):
    df[f"c{j}"] = g.choice([f"v{i}" for i in range(30)], n + nt)
df["y"] = (3 * df["f0"] + np.sin(df["f1"] * 2) + df["f2"] * df["f3"] + (df["c0"] == "v1") * 2 + g.normal(size=n + nt) * 0.5).astype("float32")
df["id"] = np.arange(n + nt)
tr, te = df.iloc[:n], df.iloc[n:]
tr.to_csv(out / "train.csv", index=False); te.drop(columns="y").to_csv(out / "test.csv", index=False)
pd.DataFrame({"id": te["id"], "y": 0.0}).to_csv(out / "sample_submission.csv", index=False)
te[["id", "y"]].to_csv(out.parent / "big3m_answers.csv", index=False)
(out / "task.md").write_text("# Big\n\nPredict `y`. Metric: RMSE, lower is better.\n")
print("ok")
