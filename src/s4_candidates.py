"""Stage 4 — CANDIDATES: union of the Stage 3 views for a set of S1, with provenance and competition.

Usage:
  python src/s4_candidates.py subset|full|test   # subset = 50k train S1 (folds 1-4); full = all train S1;
                                                 # test = all test S1 -> work/s4/MODE/{s1,pairs,reverse}.parquet

pairs.parquet   one row per (s1, rec): forward/reverse rank and score per view, street-key flag
reverse.parquet every candidate record's reverse top-20 over ALL S1 of its country (the competition it faces)
Reverse search runs for every record of the country, so a pair found by reverse search alone is a candidate too
(the recall curve's union counts those pairs; V4.3 holds the Stage 4 ceiling to that curve).
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import ROOT, uniform  # noqa: E402
from s3_retrieve import COLS, FWD_K, REV_K, S0W, S1W, log, search_gpu, vectorizer, view_text  # noqa: E402
import s2_knowledge  # noqa: E402

S4W = ROOT / "work/s4"
N_SUBSET = 50_000
REV_CHUNK = 200_000  # records per reverse call; hits are filtered per chunk so memory tracks what is kept
# Depth of a record's reverse list that may ADD a pair no forward view found (user decision: 20).
# Measured on 300 test-France S1: depth 5 adds ~33 pairs/S1, 10 ~81, 20 ~178 (forward + key ~51);
# recall curve union@20: rev<=5 India 0.951 / US 0.988, rev<=20 0.967 / 0.993. Competition lists stay top-20.
REV_ADD_K = 20


def subset_s1() -> pl.DataFrame:
    s1 = pl.read_parquet(S0W / "s1.parquet", columns=["s1", "fold"]).filter(pl.col("fold") != 0)  # fold 0 trained V4
    return s1.filter(pl.Series(uniform(s1["s1"], 21) < N_SUBSET / s1.height)).select("s1")


def candidates(split: str, s1_ids: pl.DataFrame, out: Path) -> None:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    s1_ids.select("s1").write_parquet(out / "s1.parquet")  # the query set, incl. S1 that get no candidate
    s1_scan = pl.scan_parquet(S1W / f"{split}_source1.parquet").select(COLS)
    recs_scan = pl.scan_parquet([S1W / f"{split}_source{n}.parquet" for n in (2, 3)]).select(COLS)
    fwd, rev = [], []
    for country in sorted(s1_scan.select(pl.col("country").unique()).collect()["country"]):
        s1c = s1_scan.filter(pl.col("country") == country).collect()  # one country in RAM at a time (15 GB box)
        q = s1c.join(s1_ids.select(entity_id="s1"), on="entity_id", how="semi")
        if q.is_empty():
            continue
        r = recs_scan.filter(pl.col("country") == country).collect()
        index = {}
        for view in ("V1", "V2"):
            vec = vectorizer()
            x = vec.fit_transform(view_text(r, view))  # IDF from this split's own records
            index[view] = (x, vec.transform(view_text(s1c, view)))
            qi, xi, rank, sc = search_gpu(vec.transform(view_text(q, view)), x, FWD_K)
            fwd.append(pl.DataFrame({"s1": q["entity_id"].gather(qi), "rec": r["entity_id"].gather(xi),
                                     "view": view, "rank": rank.astype(np.int32), "score": sc}))
            log(f"{split}/{country} {view} forward: {q.height:,} S1", t0)
        key = s2_knowledge.candidate_pairs(q.select(s1="entity_id"), split)
        fwd.append(key.with_columns(view=pl.lit("V3"), rank=pl.lit(1, pl.Int32), score=pl.lit(1.0, pl.Float32)))
        cand_recs = pl.concat([f.select("rec") for f in fwd]).unique()
        is_cand = r.select(pl.col("entity_id").is_in(cand_recs["rec"].implode())).to_series().to_numpy()
        is_query = s1c.select(pl.col("entity_id").is_in(q["entity_id"].implode())).to_series().to_numpy()
        for view, (x, s) in index.items():
            # every record of the country searches every S1: a copy whose own top-20 names a query S1 is a
            # candidate even when no forward view or key found it. Kept: the full list of any record that is a
            # candidate or names a query S1 (the competition it faces); indices only, ids attached once.
            keep = []
            for a in range(0, x.shape[0], REV_CHUNK):
                qi, si, rank, sc = search_gpu(x[a:a + REV_CHUNK], s, REV_K)
                qi += a
                wanted = is_cand[a:a + REV_CHUNK].copy()
                wanted[qi[is_query[si] & (rank <= REV_ADD_K)] - a] = True
                m = wanted[qi - a]
                keep.append((qi[m], si[m], rank[m].astype(np.int32), sc[m]))
            qi, si, rank, sc = (np.concatenate(c) for c in zip(*keep))
            rev.append(pl.DataFrame({"rec": r["entity_id"].gather(qi), "s1": s1c["entity_id"].gather(si),
                                     "view": view, "rank": rank, "score": sc}))
            log(f"{split}/{country} {view} reverse: all {x.shape[0]:,} records, kept {len(np.unique(qi)):,}", t0)
        del index
    fwd, rev = pl.concat(fwd), pl.concat(rev)
    rev.write_parquet(out / "reverse.parquet")
    in_subset = rev.join(s1_ids, on="s1", how="semi")
    wide = lambda df, d: df.filter(pl.col("view") != "V3").pivot(  # noqa: E731
        "view", index=["s1", "rec"], values=["rank", "score"], aggregate_function="min").rename(
        lambda c: c if c in ("s1", "rec") else f"{c.split('_')[-1].lower()}_{d}_{c.split('_')[0]}")
    added = in_subset.filter(pl.col("rank") <= REV_ADD_K)  # reverse-only pairs enter up to this list depth
    pairs = (pl.concat([fwd.select("s1", "rec"), added.select("s1", "rec")]).unique()
             .join(wide(fwd, "fwd"), on=["s1", "rec"], how="left")
             .join(wide(in_subset, "rev"), on=["s1", "rec"], how="left")
             .join(fwd.filter(pl.col("view") == "V3").select("s1", "rec", key=pl.lit(True)), on=["s1", "rec"],
                   how="left").with_columns(pl.col("key").fill_null(False)))
    assert pairs.select(pl.struct("s1", "rec").is_unique().all()).item(), "V4.1: duplicate pair"
    assert pairs.filter(~pl.col("s1").str.starts_with("S1-") | pl.col("rec").str.starts_with("S1-")).is_empty()
    pairs.write_parquet(out / "pairs.parquet")
    log(f"{pairs.height:,} candidate pairs for {s1_ids.height:,} S1 ({pairs.height / s1_ids.height:.1f} per S1)", t0)


def all_s1(split: str) -> pl.DataFrame:
    return pl.read_parquet(S1W / f"{split}_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "subset":
        candidates("train", subset_s1(), S4W / "subset")
    elif mode == "full":   # World A: every train S1 (needs a large-RAM box; ~64 GB)
        candidates("train", all_s1("train"), S4W / "full")
    elif mode == "test":
        candidates("test", all_s1("test"), S4W / "test")
    else:
        sys.exit(__doc__)
