"""Run: .venv/bin/python src/test_s78.py   (Stages 7-8 unit checks, seconds)"""
import itertools

import numpy as np
import polars as pl

from s0_harness import indexed, queries, score
from s7_ownership import checks, own
from s8_decide import World, expected_len, two_threshold
from test_s0_harness import toy


def brute_len(row: np.ndarray, miss: float) -> int:
    """Expected-F0.5 prefix length by enumerating every outcome."""
    m, best = len(row), []
    for n in range(m + 1):
        e = 0.0
        for o in itertools.product((0, 1), repeat=m):
            pr = np.prod([row[j] if o[j] else 1 - row[j] for j in range(m)])
            c, r = sum(o[:n]), sum(o[n:])
            e += pr * (1.25 * c / (n + 0.25 * (c + r + miss)) if n else (r == 0) * np.exp(-miss))
        best.append(e)
    return int(np.argmax(best))


def test():
    # Stage 7: best claim wins, a near tie abstains, never adds or raises
    df = pl.DataFrame({"s1": ["a", "b", "a", "b", "c"], "rec": ["r1", "r1", "r2", "r2", "r3"], "p": [0.9, 0.5, 0.60, 0.58, 0.4]})
    assert own(df, 0.0).select("s1", "rec").sort("rec").rows() == [("a", "r1"), ("a", "r2"), ("c", "r3")]
    assert own(df, 0.05).select("s1", "rec").sort("rec").rows() == [("a", "r1"), ("c", "r3")]  # r2: 0.60 vs 0.58 abstains
    tie = pl.DataFrame({"s1": ["b", "a"], "rec": ["r", "r"], "p": [0.7, 0.7]})
    assert own(tie, 0.0)["s1"].to_list() == ["a"] and own(tie, 0.01).is_empty()  # deterministic tie-break / abstain
    assert own(df, None).height == df.height
    ck = checks(df, own(df, 0.05))
    assert ck["max_owners"] == 1 and ck["pairs_not_in_input"] == 0 and ck["q_above_p"] == 0 and abs(ck["abstain_rate"] - 1 / 3) < 1e-4

    # Stage 8 fast scorer == the Stage 0 reference scorer, in every world, for random scores and thresholds
    s1, rec = toy()
    rng = np.random.default_rng(0)
    owned_pairs = rec.drop_nulls("owner").select(pl.col("owner").alias("s1"), "rec")
    wrong = rec.select("rec").sample(300, seed=1).with_columns(s1=pl.lit("S1-3"))
    cand = pl.concat([owned_pairs, wrong.select("s1", "rec")]).unique(["s1", "rec"]).sort("s1", "rec")
    cand = cand.with_columns(p=pl.Series(rng.random(cand.height)))
    cand = cand.join(rec.select("rec", "owner"), on="rec").with_columns(label=pl.col("owner") == pl.col("s1")).drop("owner")
    for w in ("A", "B", "Bp"):
        qs = queries(s1, w).select("s1", "k")
        d = cand.join(qs, on="s1", how="semi").join(indexed(rec, w).select("rec"), on="rec", how="semi")
        world = World(qs, own(d, 0.0))
        grid = world.grid()
        for t1, t2 in ((0.1, 0.5), (0.3, 0.3), (0.5, 0.9)):
            chosen = two_threshold(world.pairs, t1, t2)
            ref = score(chosen.select("s1", "rec"), s1, rec, w)["f05"].mean()
            assert abs(grid[(t1, t2)] - ref) < 1e-9, (w, t1, t2, grid[(t1, t2)], ref)
            assert abs(world.evaluate(chosen)[0].mean() - ref) < 1e-9

    # G6 expected-F0.5 prefix length == brute-force enumeration
    q = np.sort(rng.random((150, 5)) ** 2, 1)[:, ::-1].copy()
    q[:40, 3:] = 0
    for miss in (0.0, 0.3):
        got = expected_len(q, miss, chunk=37)
        want = np.array([brute_len(r, miss) for r in q])
        assert (got == want).all(), (miss, np.flatnonzero(got != want)[:5])
    assert expected_len(np.zeros((1, 4)), 0.0)[0] == 0  # nothing plausible -> empty
    assert expected_len(np.array([[0.99, 0.98, 0.97, 0.0]]), 0.0)[0] == 3
    print("ok")


if __name__ == "__main__":
    test()
