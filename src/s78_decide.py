"""Stages 7-8 — OWNERSHIP + DECIDE, tuned on out-of-fold scores in Worlds A, B and B'.

Usage:
  python src/s78_decide.py subset    # work/s6/subset/scores.parquet -> reports/first_score_subset.json

Stage 7: each record keeps only its best-scoring S1, and only if it beats the runner-up by delta (else nobody).
Stage 8: per S1, the top candidate is kept if q >= tau1, each further one if q >= tau2.
Thresholds are picked for the best worst-case macro F0.5 across the three worlds (archi.md Stage 8).
Subset limit: ownership sees only the subset's S1; rivals outside it enter through the Stage 5 competition features.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import REPORTS, WORLDS, indexed, load, queries, report, score  # noqa: E402
from s3_retrieve import log  # noqa: E402
from s4_candidates import subset_s1  # noqa: E402
from s5_features import S5W  # noqa: E402
from s6_score import S6W  # noqa: E402

TAU1 = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
TAU2 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
DELTA = [0.0, 0.05, 0.1, 0.2]


def own(df: pl.DataFrame, delta: float) -> pl.DataFrame:
    """q = p for the record's best S1 if it leads the runner-up by >= delta; every other claim is dropped."""
    return (df.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("rec"),
                            second=pl.col("p").sort(descending=True).get(1, null_on_oob=True).over("rec").fill_null(0.0))
            .filter((pl.col("r") == 1) & (pl.col("p") - pl.col("second") >= delta)).select("s1", "rec", q="p"))


def decide(owned: pl.DataFrame, tau1: float, tau2: float) -> pl.DataFrame:
    r = pl.col("q").rank("ordinal", descending=True).over("s1")
    return owned.filter(((r == 1) & (pl.col("q") >= tau1)) | ((r >= 2) & (pl.col("q") >= tau2))).select("s1", "rec")


def main(name: str) -> None:
    t0 = time.time()
    s1_all, rec_all = load()
    sub = s1_all.join(subset_s1(), on="s1", how="semi")
    sc = pl.read_parquet(S6W / name / "scores.parquet")
    feats = pl.read_parquet(S5W / name / "features.parquet", columns=["s1", "rec", "name_jaccard", "addr_jaccard"])
    per_world = {}
    for w in WORLDS:
        qs, ix = queries(sub, w).select("s1"), indexed(rec_all, w).select("rec")
        d = sc.join(qs, on="s1", how="semi").join(ix, on="rec", how="semi")
        f = lambda pred: score(pred, sub, rec_all, w)["f05"].mean()  # noqa: E731
        base = {"empty": f(d.select("s1", "rec").clear())}
        b0 = feats.join(qs, on="s1", how="semi").join(ix, on="rec", how="semi").filter(
            (pl.col("name_jaccard") >= 0.5) & (pl.col("addr_jaccard") >= 0.5)).with_columns(
            p=pl.col("name_jaccard") + pl.col("addr_jaccard"))
        base["B0_rule"] = f(own(b0, 0.0).select("s1", "rec"))  # V8.3 baseline: J >= 0.5 on both, then ownership
        base["oracle_candidates"] = f(d.filter("label").select("s1", "rec"))  # recall ceiling of the candidate set
        grid = {}
        for delta in DELTA:
            owned = own(d, delta)
            for t1, t2 in itertools.product(TAU1, TAU2):
                if t2 >= t1:
                    grid[(delta, t1, t2)] = f(decide(owned, t1, t2))
        per_world[w] = {"baselines": base, "grid": grid}
        best = max(grid, key=grid.get)
        log(f"World {w}: best {grid[best]:.4f} at delta={best[0]} tau1={best[1]} tau2={best[2]}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in base.items()), t0)
    robust = max(per_world["A"]["grid"], key=lambda k: min(per_world[w]["grid"][k] for w in WORLDS))
    delta, t1, t2 = robust
    rows = {"thresholds": {"delta": delta, "tau1": t1, "tau2": t2, "chosen_by": "best worst-case over A/B/Bp (OOF)"},
            "n_subset_s1": sub.height}
    for w in WORLDS:
        qs, ix = queries(sub, w).select("s1"), indexed(rec_all, w).select("rec")
        d = sc.join(qs, on="s1", how="semi").join(ix, on="rec", how="semi")
        per = score(decide(own(d, delta), t1, t2), sub, rec_all, w)
        rows[w] = {"macro_f05": per["f05"].mean(), **per_world[w]["baselines"],
                   "best_in_world": max(per_world[w]["grid"].values())}
        print(f"\nWorld {w} (robust thresholds)\n" + report(per))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"first_score_{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
    log(json.dumps(rows), t0)


if __name__ == "__main__":
    if sys.argv[1:2] == ["subset"]:
        main("subset")
    else:
        sys.exit(__doc__)
