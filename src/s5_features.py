"""Stage 5 — EVIDENCE: features per candidate pair (archi.md F1-F8), using the Stage 1 fields and Stage 2 tables.

Usage:
  python src/s5_features.py subset|full|test   # work/s4/MODE/pairs.parquet -> work/s5/MODE/features.parquet
                                               # (test: no label/fold columns)
No feature encodes the country or the source file name (V5.2); counts come from the pair's own split.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

sys.path.insert(0, str(Path(__file__).parent))
from s0_harness import ROOT  # noqa: E402
from s2_knowledge import LEX, NUM, WORK as S2W, number_relation  # noqa: E402
from s3_retrieve import S0W, S1W, log  # noqa: E402
from s4_candidates import S4W  # noqa: E402

S5W = ROOT / "work/s5"
FIELDS = ["entity_id", "country", "name_roman", "name_alts", "name_tokens", "addr_clean", "addr_tokens", "numbers",
          "name_nonlatin", "addr_nonlatin"]
RELS = ["both_none", "one_none", "identical", "zero_pad", "letter", "fraction", "sub", "bis", "injected", "dropped",
        "truncation", "shared_near", "shared_far", "disjoint_near", "disjoint"]


def side_fields(split: str, ids: pl.DataFrame, sources) -> pl.DataFrame:
    return (pl.scan_parquet([S1W / f"{split}_source{n}.parquet" for n in sources]).select(FIELDS)
            .join(ids.lazy(), on="entity_id", how="semi").collect())


def token_features(p: pl.DataFrame, split: str, field: str, col: str) -> pl.DataFrame:
    """IDF-weighted overlap, alias-aware matching and learned leftover weights for one field."""
    words = pl.read_parquet(S2W / f"{split}_words.parquet").filter(pl.col("field") == field).select(
        "country", w="word", idf="idf", lw="leftover_w")
    aliases = pl.read_parquet(LEX / "aliases.parquet").filter(pl.col("field") == field).select("a", "b")
    noise = pl.read_parquet(LEX / "noise.parquet").filter(pl.col("field") == field).select(w="a")
    tok = pl.concat([p.select("pid", "country", w=pl.col(col), side=pl.lit(0)),         # 0 = record
                     p.select("pid", "country", w=pl.col(f"{col}_1"), side=pl.lit(1))]  # 1 = S1
                    ).explode("w", empty_as_null=True).drop_nulls("w").unique(["pid", "w", "side"])
    both = tok.group_by("pid", "w").agg(n=pl.len())
    left = tok.join(both.filter(pl.col("n") == 1), on=["pid", "w"], how="semi")
    # a record leftover and an S1 leftover that are learned aliases of each other count as matched
    matched = (left.filter(pl.col("side") == 0).join(aliases, left_on="w", right_on="a")
               .join(left.filter(pl.col("side") == 1).select("pid", b="w"), on=["pid", "b"], how="semi"))
    m = pl.concat([matched.select("pid", "w"), matched.select("pid", w="b")]).unique()
    left = left.join(m, on=["pid", "w"], how="anti").join(words, on=["country", "w"], how="left")
    if field == "addr":
        left = left.join(noise, on="w", how="anti")  # injected generator noise (cdp, po, box) is not disagreement
    shared = tok.filter(pl.col("side") == 0).join(both.filter(pl.col("n") == 2), on=["pid", "w"], how="semi").join(
        words, on=["country", "w"], how="left")
    agg_s = shared.group_by("pid").agg(**{f"{field}_shared_idf": pl.col("idf").sum(),
                                          f"{field}_shared_max_idf": pl.col("idf").max(),
                                          f"{field}_shared_n": pl.len()})
    extra = {"name_left_w_sum": pl.col("lw").sum(), "name_left_w_min": pl.col("lw").min()} if field == "name" else {}
    agg_l = left.group_by("pid").agg(**{f"{field}_left_idf": pl.col("idf").sum(), f"{field}_left_n": pl.len(),
                                        f"{field}_left_rec_n": (pl.col("side") == 0).sum(),
                                        f"{field}_left_s1_n": (pl.col("side") == 1).sum()}, **extra)
    agg_m = matched.group_by("pid").agg(**{f"{field}_alias_n": pl.len()})
    out = p.select("pid").join(agg_s, on="pid", how="left").join(agg_l, on="pid", how="left").join(
        agg_m, on="pid", how="left")
    counts = [c for c in out.columns if c.endswith("_n")]
    out = out.with_columns(pl.col(counts).fill_null(0),
                           pl.col(f"{field}_shared_idf", f"{field}_left_idf").fill_null(0.0))
    return out.with_columns(**{f"{field}_idf_overlap": pl.col(f"{field}_shared_idf")
                                / (pl.col(f"{field}_shared_idf") + pl.col(f"{field}_left_idf")).clip(1e-6)})


def fuzzy(a: pl.Series, b: pl.Series, scorer) -> np.ndarray:
    return cpdist(a.fill_null("").to_list(), b.fill_null("").to_list(), scorer=scorer, workers=-1).astype(np.float32)


def number_features(p: pl.DataFrame) -> pl.DataFrame:
    rel, shared, gap = [], [], []
    for a, b in zip(p["numbers_1"].to_list(), p["numbers"].to_list()):
        rel.append(RELS.index(number_relation(a, b)))
        va, vb = {n["v"] for n in a}, {n["v"] for n in b}
        shared.append(len(va & vb))
        da, db = va - vb, vb - va
        gap.append(min(abs(x - y) for x in da for y in db) if da and db else -1)
    return pl.DataFrame({"pid": p["pid"], "num_rel": rel, "num_shared": shared, "num_gap": gap},
                        schema_overrides={"num_rel": pl.Int8, "num_gap": pl.Int64})


def competition(p: pl.DataFrame, rev: pl.DataFrame) -> pl.DataFrame:
    """How contested the record is: its best rival score per view, and how many S1 claim it nearly as well."""
    feats = []
    for view in ("V1", "V2"):
        best = rev.filter(pl.col("view") == view).group_by("rec").agg(
            best=pl.col("score").max(), second=pl.col("score").sort(descending=True).get(1, null_on_oob=True),
            near=(pl.col("score") >= pl.col("score").max() - 0.05).sum())
        vv = view.lower()
        own = pl.coalesce(f"{vv}_rev_score", f"{vv}_fwd_score")
        feats.append(p.select("pid", "rec", own=own).join(best, on="rec", how="left").select(
            "pid", **{f"{vv}_rec_best": pl.col("best"), f"{vv}_rec_second": pl.col("second"),
                      f"{vv}_n_claims": pl.col("near"), f"{vv}_gap_to_best": pl.col("best") - pl.col("own")}))
    return feats[0].join(feats[1], on="pid")


def features(split: str, name: str) -> None:
    t0 = time.time()
    pairs = pl.read_parquet(S4W / name / "pairs.parquet").with_row_index("pid")
    rev = pl.read_parquet(S4W / name / "reverse.parquet")
    s1f = side_fields(split, pairs.select(entity_id="s1").unique(), (1,))
    recf = side_fields(split, pairs.select(entity_id="rec").unique(), (2, 3))
    p = (pairs.join(s1f.rename({c: f"{c}_1" for c in FIELDS if c != "entity_id"} | {"entity_id": "s1"}), on="s1")
         .join(recf.rename({"entity_id": "rec"}), on="rec").drop("country_1"))
    log(f"{p.height:,} pairs joined to their fields", t0)
    jac = lambda a, b: pl.col(a).list.set_intersection(b).list.len() / pl.col(a).list.set_union(b).list.len().clip(1)  # noqa: E731
    base = p.select(
        "pid", "s1", "rec",
        name_jaccard=jac("name_tokens", "name_tokens_1"), addr_jaccard=jac("addr_tokens", "addr_tokens_1"),
        name_len_ratio=pl.col("name_tokens").list.len() / pl.col("name_tokens_1").list.len().clip(1),
        addr_len_ratio=pl.col("addr_tokens").list.len() / pl.col("addr_tokens_1").list.len().clip(1),
        rec_addr_empty=pl.col("addr_clean") == "", s1_addr_empty=pl.col("addr_clean_1") == "",
        rec_no_numbers=pl.col("numbers").list.len() == 0, s1_no_numbers=pl.col("numbers_1").list.len() == 0,
        rec_nonlatin_name="name_nonlatin", rec_nonlatin_addr="addr_nonlatin",
        rec_has_alts=pl.col("name_alts").list.len() > 0, rec_is_s2=pl.col("rec").str.starts_with("S2-"),
        n_views=pl.sum_horizontal(pl.col("v1_fwd_rank", "v2_fwd_rank", "v1_rev_rank", "v2_rev_rank").is_not_null(),
                                  "key"),
        **{c: pl.col(c) for c in ("v1_fwd_rank", "v1_fwd_score", "v2_fwd_rank", "v2_fwd_score", "v1_rev_rank",
                                   "v1_rev_score", "v2_rev_rank", "v2_rev_score", "key")},
    ).with_columns(
        name_tsr=fuzzy(p["name_roman"], p["name_roman_1"], fuzz.token_set_ratio),
        name_ratio=fuzzy(p["name_roman"], p["name_roman_1"], fuzz.ratio),
        name_partial=fuzzy(p["name_roman"], p["name_roman_1"], fuzz.partial_ratio),
        addr_tsr=fuzzy(p["addr_clean"], p["addr_clean_1"], fuzz.token_set_ratio),
        addr_ratio=fuzzy(p["addr_clean"], p["addr_clean_1"], fuzz.ratio),
    )
    alts = p.select("pid", alt="name_alts", s1n="name_roman_1").explode("alt", empty_as_null=True).drop_nulls("alt")
    alts = alts.with_columns(s=fuzzy(alts["alt"], alts["s1n"], fuzz.token_set_ratio)).group_by("pid").agg(
        alt_tsr_max=pl.col("s").max())
    log("similarities", t0)
    counts = pl.read_parquet(S2W / f"{split}_records.parquet").select(
        "entity_id", "name_n_s1_pct", "name_local_n_s1", "coloc_n_s1_pct", "name_rec_per100k")
    out = (base.join(alts, on="pid", how="left")
           .join(token_features(p, split, "name", "name_tokens"), on="pid")
           .join(token_features(p, split, "addr", "addr_tokens"), on="pid")
           .join(number_features(p), on="pid")
           .join(competition(pairs, rev), on="pid")
           .join(counts.rename(lambda c: c if c == "entity_id" else f"s1_{c}"), left_on="s1", right_on="entity_id",
                 how="left")
           .join(counts.rename(lambda c: c if c == "entity_id" else f"rec_{c}"), left_on="rec", right_on="entity_id",
                 how="left"))
    if split == "train":
        owner = pl.read_parquet(S0W / "rec.parquet", columns=["rec", "owner"])
        folds = pl.read_parquet(S0W / "s1.parquet", columns=["s1", "fold"])
        out = (out.join(owner, on="rec", how="left")
               .with_columns(label=(pl.col("owner") == pl.col("s1")).fill_null(False)).drop("owner")
               .join(folds, on="s1"))
    (S5W / name).mkdir(parents=True, exist_ok=True)
    out.drop("pid").write_parquet(S5W / name / "features.parquet")
    log(f"{out.height:,} rows x {out.width} columns -> work/s5/{name}/features.parquet", t0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode in ("subset", "full", "test"):
        features("test" if mode == "test" else "train", mode)
    else:
        sys.exit(__doc__)
