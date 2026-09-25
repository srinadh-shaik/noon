"""Run: .venv/bin/python src/test_s0_harness.py"""
import polars as pl

from s0_harness import build_worlds, indexed, queries, score


def toy():
    s1 = pl.DataFrame({"s1": [f"S1-{i}" for i in range(1, 2001)], "country": "US"})
    # S1-1: problem-statement example (truth S2-47, S3-812); S1-2: singleton; S1-3..: 3 copies each
    recs = [("S2-47", "S1-1"), ("S3-812", "S1-1"), ("S2-193", None)]
    recs += [(f"S{2 + j % 2}-{1000 * i + j}", f"S1-{i}") for i in range(3, 2001) for j in range(3)]
    rec = pl.DataFrame({"rec": [r for r, _ in recs], "owner": [o for _, o in recs]}, schema_overrides={"owner": pl.Utf8})
    return build_worlds(s1, rec.with_columns(country=pl.lit("US")))


def test():
    s1, rec = toy()
    pred = pl.DataFrame({"s1": ["S1-1"] * 3 + ["S1-2"], "rec": ["S2-47", "S2-193", "S3-812", "S2-47"]})
    f = dict(score(pred, s1, rec).select("s1", "f05").iter_rows())
    assert abs(f["S1-1"] - 0.714) < 1e-3, f["S1-1"]  # official example
    assert f["S1-2"] == 0.0  # any match on a singleton scores 0
    assert f["S1-3"] == 0.0  # empty answer on a non-singleton scores 0
    f = dict(score(pred.head(0), s1, rec).select("s1", "f05").iter_rows())
    assert f["S1-2"] == 1.0  # empty answer on a singleton scores 1

    # fold: every owned record shares its owner's fold
    j = rec.drop_nulls("owner").join(s1, left_on="owner", right_on="s1")
    assert (j["fold"] == j["fold_right"]).all()

    # B': each hidden owner keeps exactly one copy; visible owners keep all
    kept = indexed(rec, "Bp").drop_nulls("owner").group_by("owner").len()
    hidden = set(s1.filter(pl.col("hide_Bp"))["s1"])
    assert all((n == 1) == (o in hidden) for o, n in kept.iter_rows()), "B' keep rule broken"
    for w, share in (("B", 0.20), ("Bp", 0.34)):
        assert abs(1 - queries(s1, w).height / s1.height - share) < 0.04, w

    # oracle scores 1 in every world; a hidden S1's copy predicted for a query S1 counts as wrong
    truth = rec.drop_nulls("owner").select(pl.col("owner").alias("s1"), "rec")
    for w in ("A", "B", "Bp"):
        assert score(truth, s1, rec, w)["f05"].min() == 1.0, w
    h = next(iter(hidden))
    steal = truth.filter(pl.col("s1") == h).with_columns(s1=pl.lit("S1-3"))
    f3 = score(pl.concat([truth.filter(pl.col("s1") == "S1-3"), steal]), s1, rec, "Bp")
    assert f3.filter(pl.col("s1") == "S1-3")["f05"][0] < 1.0
    print("ok")


if __name__ == "__main__":
    test()
