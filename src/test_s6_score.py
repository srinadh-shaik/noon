"""Run: .venv/bin/python src/test_s6_score.py   (Stage 6 on synthetic features, seconds)"""
import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

import s6_score as s6


def test():
    d = Path(tempfile.mkdtemp())
    s6.S5W, s6.S6W, s6.REPORTS = d / "s5", d / "s6", d / "rep"
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.random(n) < 0.3
    df = pl.DataFrame({"s1": [f"S1-{i // 10}" for i in range(n)], "rec": [f"S2-{i}" for i in range(n)], "label": y,
                       "fold": (np.arange(n) // 10 % 5).astype(np.int8), "f1": y + rng.normal(0, 0.7, n),
                       "num_rel": rng.integers(0, 5, n).astype(np.int8)})
    for name in ("t", "tt"):
        (s6.S5W / name).mkdir(parents=True)
    df.write_parquet(s6.S5W / "t" / "features.parquet")
    df.drop("label", "fold").write_parquet(s6.S5W / "tt" / "features.parquet")
    s6.score("t")
    sc = pl.read_parquet(s6.S6W / "t" / "scores.parquet")
    rep = json.loads((s6.REPORTS / "stage6_t.json").read_text())
    assert sc.height == n and sc["p"].is_not_null().all() and sc.columns == ["s1", "rec", "label", "fold", "p"]
    assert len(list((s6.S6W / "t").glob("model_fold*.txt"))) == 5
    assert rep["oof_auc"] > 0.8 and abs(rep["shuffled_label_auc"] - 0.5) < 0.05, rep
    s6.predict("t", "tt")  # test mode: mean of the fold models, one p per pair
    pt = pl.read_parquet(s6.S6W / "tt" / "scores.parquet")
    assert pt.columns == ["s1", "rec", "p"] and pt.height == n
    assert abs(pt["p"].mean() - sc["p"].mean()) < 0.02  # same scale as OOF
    print("ok")


if __name__ == "__main__":
    test()
