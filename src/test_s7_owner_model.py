"""Run: .venv/bin/python src/test_s7_owner_model.py   (gate G5 owner-or-none model unit checks, seconds)"""
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

import gates
import s7_owner_model as om
import s8_decide


def claims(seed: int, n_rec: int = 3000) -> pl.DataFrame:
    """Records with 1-4 claims from 400 S1s; the owner is the top claim only when it leads the runner-up by > 0.15."""
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 5, n_rec)
    rec = np.repeat([f"r{seed}_{i}" for i in range(n_rec)], n)
    s1 = rng.integers(0, 400, len(rec))
    df = (pl.DataFrame({"s1": [f"S{x}" for x in s1], "rec": rec, "p": rng.random(len(rec))})
          .unique(["s1", "rec"]).with_columns(fold=(pl.col("s1").str.slice(1).cast(pl.Int32) % 5).cast(pl.Int8)))
    f = om.features(df)
    return f.with_columns(label=(pl.col("claim_rank") == 0) & (pl.col("gap_second") > 0.15)).select("s1", "rec", "p", "label", "fold")


def test():
    tmp = Path(tempfile.mkdtemp(prefix="g5_", dir="/home/dheeraj/.claude/jobs/8b1bc1a4/tmp/g5"))
    om.S5W, om.S6W, om.S7W, om.S8W = (tmp / s for s in ("s5", "s6", "s7", "s8"))
    s8_decide.S7W, gates.REPORTS = om.S7W, tmp / "reports"
    tr, te = claims(0), claims(1)
    for name, df in (("tr", tr), ("te", te.drop("label", "fold"))):
        (om.S6W / name).mkdir(parents=True)
        df.write_parquet(om.S6W / name / "scores.parquet")

    # features: record- and S1-relative, as specified
    f = om.features(pl.DataFrame({"s1": ["a", "b", "a"], "rec": ["r1", "r1", "r2"], "p": [0.9, 0.5, 0.3]}))
    r = f.filter(pl.col("s1") == "b").row(0, named=True)
    assert (r["claim_rank"], r["n_claims"], r["p_best"], r["p_second"], r["s1_rank"], r["s1_n"]) == (1, 2, 0.9, 0.5, 0, 1)
    assert abs(r["gap_best"] - 0.4) < 1e-12 and abs(r["p_minus_mean_others"] + 0.4) < 1e-12
    a2 = f.filter(pl.col("rec") == "r2").row(0, named=True)
    assert (a2["p_second"], a2["s1_rank"], a2["s1_n"], a2["s1_top"]) == (0.0, 1, 2, 0.9)

    # OOF never trains on the fold it predicts
    seen, real = [], om.train
    om.train = lambda d, c: (seen.append(set(d["fold"].unique())), real(d, c))[1]
    om.fit("tr")
    om.train = real
    assert len(seen) == 6 and all(len(s) == 4 for s in seen[:5]) and len(seen[5]) == 5  # 5 folds, then all for the model
    assert all(k not in s for k, s in zip(range(5), seen[:5]))

    q = pl.read_parquet(om.S7W / "tr" / "owner_q.parquet")
    assert q.height == om.prefilter(tr).height and q["q"].is_between(0, 1).all()  # same pair set as s8_decide
    assert q.group_by("rec").agg(pl.col("q").sum())["q"].max() <= 1 + 1e-9
    lab = q.join(tr, on=["s1", "rec"])
    assert lab.filter("label")["q"].mean() > 0.8 > 0.1 > lab.filter(~pl.col("label"))["q"].mean(), "OOF learns the pattern"

    # predict: the saved model on unseen claims keeps owners on top; on its own training claims too
    om.predict("te", "tr")
    qt = pl.read_parquet(om.S7W / "te" / "owner_q.parquet").join(te, on=["s1", "rec"])
    assert qt.group_by("rec").agg(pl.col("q").sum())["q"].max() <= 1 + 1e-9
    for d in (qt, pl.read_parquet(om.S7W / "tr" / "owner_q.parquet").join(tr, on=["s1", "rec"])):
        acc = (d.with_columns(hit=(pl.col("q") > 0.5) == pl.col("label")))["hit"].mean()
        assert acc > 0.95, acc

    # s8 --owner-model swaps p for q, and refuses a partial owner_q
    sc = s8_decide.owner_q(om.prefilter(tr.select("s1", "rec", "p", "label")), "tr")  # s8 pre-filters first
    assert sc.join(q, on=["s1", "rec"]).filter(pl.col("p") != pl.col("q")).is_empty()
    try:
        tr.head(1).select("s1", "rec", q=pl.lit(0.5)).write_parquet(om.S7W / "tr" / "owner_q.parquet")
        s8_decide.owner_q(tr.select("s1", "rec", "p"), "tr")
        raise AssertionError("partial owner_q accepted")
    except AssertionError as e:
        assert "misses" in str(e)

    # gate: best worst-case of each grid, then archi.md D0.6 epsilon
    def grid(name, a, b, bp):
        (om.S8W / name).mkdir(parents=True)
        pl.DataFrame({"delta": [-1.0, 0.0, 0.05], "tau1": [0.1] * 3, "tau2": [0.8] * 3,
                      "A": [0.99, a, a - 0.01], "B": [0.99, b, b], "Bp": [0.99, bp, bp]}).write_parquet(om.S8W / name / "grid.parquet")
    grid("g", 0.80, 0.78, 0.76), grid("g_owner", 0.803, 0.78, 0.763)
    assert om.gate("g")  # +0.003 in A and B', no loss in B; the delta -1 (ownership off) row is ignored
    grid("h", 0.80, 0.78, 0.76), grid("h_owner", 0.801, 0.78, 0.77)
    assert not om.gate("h")  # A gain below epsilon
    assert "G5" in (gates.REPORTS / "gates.md").read_text()
    print("s7_owner_model OK")


if __name__ == "__main__":
    test()
