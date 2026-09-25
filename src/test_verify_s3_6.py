"""Run: .venv/bin/python src/test_verify_s3_6.py   (Stage 3-6 verifier on synthetic artefacts, seconds)"""
import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

import verify_s3_6 as v
from s5_features import RELS


def world(d: Path, n_s1: int = 2000, per: int = 10) -> None:
    """Synthetic artefacts in the Stage 0-6 formats: every S1 owns its first 3 candidates; test has 3 countries."""
    rng = np.random.default_rng(0)
    for s in ("s0", "s1", "s4/tr", "s4/te", "s5/tr", "s5/te", "s6/tr", "s6/te", "reports"):
        (d / s).mkdir(parents=True, exist_ok=True)
    s1 = [f"S1-{i}" for i in range(n_s1)]
    fold = (np.arange(n_s1) % 5).astype(np.int8)
    pairs = pl.DataFrame({"s1": np.repeat(s1, per), "rec": [f"S{2 + j % 2}-{i * per + j}" for i in range(n_s1) for j in range(per)]})
    owned = pairs.with_columns(j=pl.int_range(pl.len()).over("s1")).filter(pl.col("j") < 3).drop("j")
    train_c = ["India" if i % 2 else "US" for i in range(n_s1)]
    pl.DataFrame({"s1": s1, "country": train_c, "k": 3, "fold": fold}).write_parquet(d / "s0/s1.parquet")
    owned.select("rec", owner="s1").write_parquet(d / "s0/rec.parquet")
    for sp, c in (("train", train_c), ("test", [("India", "US", "France")[i % 3] for i in range(n_s1)])):
        pl.DataFrame({"entity_id": s1, "country": c, "name_nonlatin": False, "addr_clean": "x"}).write_parquet(
            d / f"s1/{sp}_source1.parquet")
        recs = pairs.with_columns(country=pl.col("s1").replace_strict(dict(zip(s1, c))))
        for n in (2, 3):
            (recs.filter(pl.col("rec").str.starts_with(f"S{n}-")).select(entity_id="rec", country="country")
             .with_columns(name_nonlatin=pl.lit(False), addr_clean=pl.lit("x")).write_parquet(d / f"s1/{sp}_source{n}.parquet"))
    for m in ("tr", "te"):
        pairs.write_parquet(d / f"s4/{m}/pairs.parquet")
        pl.DataFrame({"s1": s1}).write_parquet(d / f"s4/{m}/s1.parquet")
    label = pairs.join(owned.with_columns(label=pl.lit(True)), on=["s1", "rec"], how="left")["label"].fill_null(False).to_numpy()
    true_rel = np.where(np.arange(pairs.height) % 3 == 0, RELS.index("one_none"), RELS.index("identical"))
    rel = np.where(label, true_rel, RELS.index("shared_near")).astype(np.int8)
    f = pairs.with_columns(name_jaccard=pl.lit(0.8), addr_jaccard=pl.lit(0.8), num_rel=pl.Series(rel),
                           f1=pl.Series(rng.normal(size=pairs.height)), v1_fwd_rank=pl.lit(None, pl.Int32))
    fold_of = pl.col("s1").replace_strict(dict(zip(s1, fold.tolist())), return_dtype=pl.Int8)
    f.with_columns(label=pl.Series(label), fold=fold_of).write_parquet(d / "s5/tr/features.parquet")
    f.write_parquet(d / "s5/te/features.parquet")
    p = np.clip(rng.random(pairs.height), 0.001, 0.999)
    y = rng.random(pairs.height) < p  # calibrated by construction
    pairs.with_columns(label=pl.Series(y), fold=fold_of, p=pl.Series(p)).write_parquet(d / "s6/tr/scores.parquet")
    pairs.with_columns(p=pl.Series(p)).write_parquet(d / "s6/te/scores.parquet")
    for k in range(5):
        (d / f"s6/tr/model_fold{k}.txt").write_text("tree\n")
    (d / "reports/stage6_tr.json").write_text(json.dumps({"shuffled_label_auc": 0.501, "oof_auc": 0.9,
                                                          "top_gain": {"num_rel": 0.5, "name_left_w_sum": 0.3}}))
    curve = [("V1", "fwd", c, "all", k, 0.9, 100) for c in ("India", "US") for k in v.KS]
    curve += [(vw, dr, c, "all", 20, 0.8, 100) for c in ("India", "US") for vw, dr in (("V1", "rev"), ("V2", "fwd"), ("V2", "rev"))]
    curve += [("V3", "key", "US", "all", 1, 0.42, 100), ("V3", "key", "India", "all", 1, 0.55, 100)]
    curve += [("unique_V1", "all", c, "all", 50, 0.01, 100) for c in ("India", "US")]
    curve += [("union", "all", c, s, 20, 0.99, 100) for c in ("India", "US") for s in ("all", "nonlatin_name")]
    pl.DataFrame(curve, schema=["view", "direction", "country", "slice", "K", "recall", "n"], orient="row").write_csv(
        d / "reports/recall_curve.csv")
    (d / "requirements.txt").write_text("lightgbm==4.7.0\n")


def run(d: Path) -> dict:
    v.ROOT, v.REPORTS = d, d / "reports"
    v.S0W, v.S1W, v.S4W, v.S5W, v.S6W = (d / s for s in ("s0", "s1", "s4", "s5", "s6"))
    v.SAMPLE_SHARE = 1.0
    for f in (d / "reports").glob("verify_stage*.json"):
        f.unlink()
    ok = {"3": v.stage3(["tr", "te"]), "4": v.stage4("tr", "te"), "5": v.stage5("tr", "te"), "6": v.stage6("tr", "te")}
    fails = {c["id"] for n in "3456" for c in json.loads((d / f"reports/verify_stage{n}.json").read_text())
             if c["level"] == "HARD" and not c["pass"]}
    return {"ok": ok, "fails": fails}


def test():
    d = Path(tempfile.mkdtemp())
    world(d)
    r = run(d)
    assert all(r["ok"].values()) and not r["fails"], r
    corrupt = {
        "V4.1": lambda: pl.concat([pl.read_parquet(d / "s4/tr/pairs.parquet")] * 2).write_parquet(d / "s4/tr/pairs.parquet"),
        "V4.2": lambda: pl.read_parquet(d / "s4/te/s1.parquet").head(10).write_parquet(d / "s4/te/s1.parquet"),
        "V4.6": lambda: pl.read_parquet(d / "s6/te/scores.parquet").head(100).write_parquet(d / "s6/te/scores.parquet"),
        "V5.2": lambda: pl.read_parquet(d / "s5/tr/features.parquet").with_columns(country_is_india=pl.lit(1)).write_parquet(
            d / "s5/tr/features.parquet"),
        "V5.5": lambda: pl.read_parquet(d / "s5/tr/features.parquet").with_columns(f1=pl.lit(float("inf"))).write_parquet(
            d / "s5/tr/features.parquet"),
        "V6.3": lambda: (d / "reports/stage6_tr.json").write_text(json.dumps({"shuffled_label_auc": 0.6, "top_gain": {}})),
    }
    for cid, break_it in corrupt.items():
        world(d)
        break_it()
        r = run(d)
        assert cid in r["fails"], (cid, r)
    print("ok")


if __name__ == "__main__":
    test()
